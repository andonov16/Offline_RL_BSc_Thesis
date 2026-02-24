from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from typing import List, Tuple
import os
import optuna
import pandas as pd
import numpy as np

from src.datasets import BCDataset
from src.normalization import NormalizationModule
from src.tuning.bc_objective import BCObjectiveTorch


def prepare_data(df: pd.DataFrame,
                 selected_features: List[str],
                 norm_script: NormalizationModule = None) -> Tuple[torch.Tensor, torch.Tensor]:
    x = df.drop(columns=['action'])

    all_features_names = x.columns
    selected_features_ids = [i for i, x in enumerate(all_features_names) if x in selected_features]
    x = torch.tensor(x.values.astype('float32'), dtype=torch.float32)

    # apply normalization if exists
    if norm_script is not None:
        x = norm_script(x)

    # select only desired features (columns)
    x = x[:, selected_features_ids]
    y = torch.tensor(df['action'].values, dtype=torch.long)

    return x, y

def conduct_bc_experiment(dataset_name: str = 'final_policy',
                       norm_technique_name: str = 'raw',
                       norm_technique_script: NormalizationModule = None,
                       selected_features: List[str] = None,
                       output_model_name: str = None,
                       train_df: pd.DataFrame = None,
                       valid_df: pd.DataFrame = None,
                       experiments_config: dict = None) -> None:
    import random
    seed = experiments_config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print('='*100)
    if experiments_config['experiment']['stratified_train_set_ratio'] != 1:
        train_df_stratified, _ = train_test_split(
            train_df,
            stratify=train_df['action'],
            train_size=experiments_config['experiment']['stratified_train_set_ratio'],
            random_state=experiments_config['experiment']['seed']
        )
    else:
        train_df_stratified = train_df

    X_train, y_train = prepare_data(df=train_df_stratified,
                                    selected_features=selected_features,
                                    norm_script=norm_technique_script)
    X_valid, y_valid = prepare_data(df=valid_df,
                                    selected_features=selected_features,
                                    norm_script=norm_technique_script)

    train_dataset = BCDataset(states=X_train,
                              actions=y_train)
    valid_dataset = BCDataset(states=X_valid,
                              actions=y_valid)

    base_log_dir = os.path.abspath(experiments_config['runtime']['log_dir'])
    log_dir = os.path.join(base_log_dir, dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    storage = f'sqlite:///{os.path.join(log_dir, f'BC_{norm_technique_name.lower().replace(' ', '_')}.db')}'

    num_workers = 0

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=experiments_config['experiment']['batch_size'],
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  persistent_workers = False,)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=experiments_config['experiment']['batch_size'],
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  persistent_workers = False,)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=1
    )

    objective = BCObjectiveTorch(
        train_loader=train_dataloader,
        eval_loader=valid_dataloader,
        model_dir=os.path.join(experiments_config['runtime']['best_model_dir'], f'{dataset_name}/'),
        logs_dir=os.path.join(experiments_config['runtime']['log_dir'], f'{dataset_name}/'),
        dataset_name=dataset_name,
        model_name=output_model_name,
        device=device,
        num_features=len(selected_features),
        config=experiments_config,
        max_epochs=experiments_config['experiment']['max_epochs']
    )

    # Load or create study
    study_name = f'bc_{dataset_name}_{norm_technique_name.lower().replace(" ", "_")}_data_study'
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage,
        pruner=pruner,
        load_if_exists=True
    )

    # Determine how many trials are already done
    existing_trials = [t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]]
    n_existing = len(existing_trials)
    n_target = experiments_config['experiment']['n_optuna_trials']
    n_remaining = max(n_target - n_existing, 0)

    if n_remaining == 0:
        print(f"Study '{study_name}' already has {n_existing}/{n_target} trials. Skipping optimization.")
    else:
        print(f"Study '{study_name}' already has {n_existing} trials. Running {n_remaining} more...")
        study.optimize(objective, n_trials=n_remaining)

    del X_train, X_valid, y_train, y_valid
