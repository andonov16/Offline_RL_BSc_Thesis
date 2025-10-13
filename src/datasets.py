import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset class used for training the Behaviour Cloning (BC)
class BCDataset(Dataset):
    def __init__(self, observations: np.array, actions: np.array):
        self.observations, self.actions = torch.tensor(observations), torch.tensor(actions)
        self.__size__ = len(self.actions)

    def __len__(self):
        return self.__size__

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]
