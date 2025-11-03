from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from copy import deepcopy
import torch
from typing import List, Tuple
import os
import yaml
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
    with open('../config/bc_experiments_config.yaml', 'r') as f:
        bc_experiments_config = yaml.safe_load(f)

    import random
    seed = experiments_config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Phase 1:
    print('='*100)
    print('Phase 1:')
    if bc_experiments_config['experiment']['stratified_train_set_ratio'] != 1:
        train_df_stratified, _ = train_test_split(
            train_df,
            stratify=train_df['action'],
            train_size=bc_experiments_config['experiment']['stratified_train_set_ratio'],
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

    train_dataset = BCDataset(states=X_train, actions=y_train)
    valid_dataset = BCDataset(states=X_valid, actions=y_valid)

    base_log_dir = os.path.abspath(experiments_config['runtime']['log_dir'])
    log_dir = os.path.join(base_log_dir, dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    storage = f'sqlite:///{os.path.join(log_dir, f'BC_{norm_technique_name.lower().replace(' ', '_')}.db')}'

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=bc_experiments_config['experiment']['batch_size_phase_1'],
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=os.cpu_count(),
                                  persistent_workers = True,
                                  drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=bc_experiments_config['experiment']['batch_size_phase_1'],
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=os.cpu_count(),
                                  persistent_workers = True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=1
    )

    objective = BCObjectiveTorch(
        train_loader=train_dataloader,
        eval_loader=valid_dataloader,
        model_dir=os.path.join(bc_experiments_config['runtime']['best_model_dir'], f'{dataset_name}/'),
        model_name=output_model_name + '_stratified',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_features=X_train.shape[1],
        config=bc_experiments_config,
        max_epochs=bc_experiments_config['experiment']['max_epochs_phase_1']
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
    n_target = bc_experiments_config['experiment']['n_optuna_trials']
    n_remaining = max(n_target - n_existing, 0)

    if n_remaining == 0:
        print(f"Study '{study_name}' already has {n_existing}/{n_target} trials. Skipping optimization.")
    else:
        print(f"Study '{study_name}' already has {n_existing} trials. Running {n_remaining} more...")
        study.optimize(objective, n_trials=n_remaining)

    del X_train, X_valid, y_train, y_valid

    # Phase 2
    print('=' * 100)
    print('Phase 2:')

    X_train_valid, y_train_valid = prepare_data(
        df=pd.concat([train_df, valid_df], ignore_index=True),
        selected_features=selected_features,
        norm_script=norm_technique_script
    )
    train_dataset = BCDataset(states=X_train_valid, actions=y_train_valid)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=bc_experiments_config['experiment']['batch_size_phase_2'],
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        drop_last=True
    )

    valid_dataset = BCDataset(states=X_train_valid, actions=y_train_valid)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=bc_experiments_config['experiment']['batch_size_phase_2'],
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=os.cpu_count(),
                                  persistent_workers = True)


    # Load Phase 1 study (read-only)
    phase1_study_name = f"bc_{dataset_name}_{norm_technique_name.lower().replace(' ', '_')}_data_study"
    phase1_study = optuna.load_study(
        study_name=phase1_study_name,
        storage=storage
    )

    # --- Create a new Optuna study for Phase 2 results ---
    refined_storage = f"sqlite:///{os.path.join(log_dir, f'BC_{norm_technique_name.lower().replace(' ', '_')}_refined.db')}"
    refined_study_name = f"{phase1_study_name}_refined"
    refined_study = optuna.create_study(
        study_name=refined_study_name,
        direction="minimize",
        storage=refined_storage,
        load_if_exists=True,
    )



    # Sort trials by score and pick top-K
    top_k = 5
    completed_trials = [t for t in phase1_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_k]
    print(len(top_trials))

    print(f"Refining top {top_k} configurations on full dataset...")
    existing_params = [t.params for t in refined_study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    for i, trial in enumerate(top_trials):
        params = trial.params

        print(f"\n> Phase 2 – Retraining trial #{trial.number} (rank {i+1}) with params: {params}")

        if any(params == p for p in existing_params):
            print(f"Skipping retraining for trial #{trial.number} (already in refined study).")
            continue

        # Update config for deterministic training
        refined_config = deepcopy(bc_experiments_config)

        # Overwrite search space with fixed parameters
        for k, v in params.items():
            refined_config['search_space'][k] = {'type': 'fixed', 'value': v}

        # Create a new objective for full-data retraining
        objective_full = BCObjectiveTorch(
            train_loader=train_dataloader,
            eval_loader=valid_dataloader,
            model_dir=os.path.join(bc_experiments_config['runtime']['best_model_dir'], f"{dataset_name}/"),
            model_name=f"{output_model_name}_refined",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_features=X_train_valid.shape[1],
            config=refined_config,
            max_epochs=refined_config['experiment']['max_epochs_phase_2']
        )

        # Reset best-loss tracker for Phase 2
        objective_full.overall_best_loss = float('inf')

        # Train deterministically (no Optuna optimization)
        final_score = objective_full(optuna.trial.FixedTrial(params))
        print(f'Finished retraining (rank {i+1}) — Final loss: {final_score:.4f}')


        # --- Save this retraining result in the refined Optuna DB ---
        refined_study.add_trial(
            optuna.trial.create_trial(
                params=params,
                value=final_score,
                state=optuna.trial.TrialState.COMPLETE,
                distributions=study.trials[0].distributions
            )
        )

        del objective_full
        torch.cuda.empty_cache()

    print('=' * 100)
    print(f'Two-phase optimization complete. Phase 2 results saved to: {refined_storage}')