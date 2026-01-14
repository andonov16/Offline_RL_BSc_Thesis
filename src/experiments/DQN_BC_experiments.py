import torch
import os
import random
import optuna
import pandas as pd
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader

from src.datasets import DQNReplayMemoryDataset
from src.normalization import NormalizationModule
from src.tuning.DQN_BC_objective import DQNBCObjectiveTorch


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


def conduct_dqn_bc_experiment(dataset_name: str = 'final_policy',
                       norm_technique_name: str = 'raw',
                       norm_technique_script: NormalizationModule | None = None,
                       generative_model_script: torch.nn.Module | None = None,
                       selected_features: List[str] = None,
                       output_model_name: str = None,
                       train_df: pd.DataFrame = None,
                       dones: torch.Tensor = None,
                       experiments_config: dict = None) -> None:
    seed = experiments_config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print('='*100)
    X_train, y_train = prepare_data(df=train_df,
                                    selected_features=selected_features,
                                    norm_script=norm_technique_script)

    train_dataset = DQNReplayMemoryDataset(
            states_rewards_next_states_tensor=X_train,
            actions_tensor=y_train,
            dones_tensor=dones,
    )

    base_log_dir = os.path.abspath(experiments_config['runtime']['log_dir'])
    log_dir = os.path.join(base_log_dir, dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    storage = f'sqlite:///{os.path.join(log_dir, f'DQN_BC_{norm_technique_name.lower().replace(' ', '_')}.db')}'

    num_workers = max(os.cpu_count()-2, 2)
    pref_factor = 2

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=experiments_config['experiment']['mini_batch_size'],
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  persistent_workers = True,
                                  prefetch_factor=pref_factor)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=1
    )

    objective = DQNBCObjectiveTorch(
        train_loader=train_dataloader,
        device=device,
        dataset_name=dataset_name,
        model_name=output_model_name,
        model_dir=os.path.join(experiments_config['runtime']['best_model_dir'], f'{dataset_name}/'),
        logs_dir=os.path.join(experiments_config['runtime']['log_dir'], f'{dataset_name}/'),
        max_num_training_iters=experiments_config['experiment']['max_num_training_iters'],
        early_stopping_criterion_iters=experiments_config['experiment']['early_stopping_criterion_iters'],
        gamma=float(experiments_config['experiment']['gamma']),
        generative_model=generative_model_script,
        num_features=8, # the dimentionality of a single state
        config=experiments_config,
    )

    # Load or create study
    study_name = f'dqn_bc_{dataset_name}_{norm_technique_name.lower().replace(" ", "_")}_data_study'
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