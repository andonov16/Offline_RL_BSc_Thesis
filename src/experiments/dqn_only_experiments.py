import random
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.normalization import NormalizationModule
from src.datasets import DQNReplayMemoryDataset


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


def conduct_DQN_only_experiment(dataset_name: str = 'final_policy',
                       norm_technique_name: str = 'raw',
                       norm_technique_script: NormalizationModule = None,
                       selected_features: List[str] = None,
                       output_model_name: str = None,
                       train_df: pd.DataFrame = None,
                       experiments_config: dict = None) -> None:
    seed = experiments_config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
