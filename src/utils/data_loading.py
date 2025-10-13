import numpy as np
import os
import pandas as pd
from typing import Tuple

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.datasets import BCDataset


def load_data(data_path: str = '../data/replay_buffer.npz') -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Loads data from a saved file in NumPy .npz format.

    Accepts:
        - 'data_path': str the path to the data stored in a file format .npz or .npy

    Returns:
        - `observations` (np.array): space observation vectors at each timestep. Shape: (1000000, 8), dtype: float32
        - `next_observations`(np.array): the next state reached after taking
        the given action at each timestep. Shape: (1000000, 8), dtype: float32
        - `actions` (np.array): the action taken at each timestep. Shape: (1000000, 1), dtype: int64
        - `rewards` (np.array): the reward received at each timestep. Shape: (1000000, 1), dtype: float32
        - `dones`(np.array): indicates whether the episode terminated at each timestep
        either due to crashing the lander or landing successfully. Shape: (1000000, 1), dtype: float32
    """
    with np.load(data_path) as data:
        observations = data['arr_0']
        next_observations = data['arr_1']
        actions = data['arr_2']
        rewards = data['arr_3']
        dones = data['arr_4']

        # remove the 'empty' dimensions
        observations = np.squeeze(observations, axis=1)
        next_observations = np.squeeze(next_observations, axis=1)
        actions = np.squeeze(actions, axis=2)
        
    return observations, next_observations, actions, rewards, dones


def load_data_as_df(observations: np.array, next_observations: np.array, actions: np.array,
                rewards: np.array, dones) -> pd.DataFrame:
    """
    Organizes the data into a pd.DataFrame

    Accepts:
        - `observations` (np.array): space observation vectors at each timestep. Shape: (1000000, 8), dtype: float32
        - `next_observations`(np.array): the next state reached after taking
        the given action at each timestep. Shape: (1000000, 8), dtype: float32
        - `actions` (np.array): the action taken at each timestep. Shape: (1000000, 1), dtype: int64
        - `rewards` (np.array): the reward received at each timestep. Shape: (1000000, 1), dtype: float32
        - `dones`(np.array): indicates whether the episode terminated at each timestep
        either due to crashing the lander or landing successfully. Shape: (1000000, 1), dtype: float32

    Returns:
        - `pd.Dataframe`: the dataset organized into a pd.DataFrame
    """
    data = {
        'X': observations[:, 0],
        'Y': observations[:, 1],
        'lv_X': observations[:, 2],
        'lv_Y': observations[:, 3],
        'angle': observations[:, 4],
        'angular_velocity': observations[:, 5],
        'leg_1': observations[:, 6],
        'leg_2': observations[:, 7],
        'action': actions.flatten(),
        'reward': rewards.flatten(),
        'done': dones.flatten(),
        'next_X': next_observations[:, 0],
        'next_Y': next_observations[:, 1],
        'next_lv_X': next_observations[:, 2],
        'next_lv_Y': next_observations[:, 3],
        'next_angle': next_observations[:, 4],
        'next_angular_velocity': next_observations[:, 5],
        'next_leg_1': next_observations[:, 6],
        'next_leg_2': next_observations[:, 7],
    }
    data_df = pd.DataFrame(data)
    
    for column in data_df.columns:
        if column in ['done', 'leg_1', 'leg_2', 'next_leg_1', 'next_leg_2']:
            data_df[column] = data_df[column].astype('bool')
        elif 'action' in column:
            data_df[column] = data_df[column].astype('int')
        else:
            data_df[column] = data_df[column].astype('float64')
    
    return data_df


def get_BC_data_loaders(observations: np.array, actions: np.array,
                        train: float = 0.70,
                        test: float = 0.15,
                        validation: float = 0.15,
                        batch_size: int = 32,
                        seed: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the data into Training, Validation and Test subsets and returns the DataLoader classes

    Accepts:
        - `observations` (np.array): space observation vectors at each timestep. Shape: (1000000, 8), dtype: float32
        - `actions` (np.array): the action taken at each timestep. Shape: (1000000, 1), dtype: int64
        - `train` (float): proportion of the data to be used for training
        - `test` (float):  proportion of the data to be used for final testing
        - `validation` (float):  proportion of the data to be used for validation

    Returns:
        - `Tuple[DataLoader, DataLoader, DataLoader]`: train_loader, test_loader, valid_loader
    """

    # ensures that the data split is in the correct format
    assert abs(train + test + validation - 1.0) < 1e-5, 'Data splits must add up to 1.'

    np.random.seed(seed)
    test_to_valid_ratio = test / (test + validation)

    observations_train, observations_test_valid, actions_train, actions_test_valid = train_test_split(
        observations, actions, test_size=(1 - train), random_state=seed)
    observations_test, observations_valid, actions_test, actions_valid = train_test_split(
        observations, actions, test_size=test_to_valid_ratio, random_state=seed)

    train_dataset = BCDataset(observations_train, actions_train)
    test_dataset = BCDataset(observations_test, actions_test)
    valid_dataset = BCDataset(observations_valid, actions_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader


def get_legs_df(df: pd.DataFrame) -> pd.DataFrame:
    leg_1, leg_2 = df[['leg_1']], df[['leg_2']]
    both_legs = (leg_1.to_numpy() & leg_2.to_numpy()).flatten()
    both_legs = pd.Series(both_legs).reset_index(drop=True)

    result_df = pd.concat([leg_1, leg_2, both_legs], axis=1)
    result_df.columns = ['leg_1', 'leg_2', 'both_legs']
    return result_df
