import torch
from torch.utils.data import Dataset


# Dataset class used for training the Behaviour Cloning (BC)
class BCDataset(Dataset):
    def __init__(self, states: torch.Tensor,
                 actions: torch.Tensor):
        self.states, self.actions = states, actions
        if not isinstance(states, torch.Tensor):
            self.states = torch.tensor(self.states)
        if not isinstance(actions, torch.Tensor):
            self.actions = torch.tensor(self.actions)
        self.__size__ = len(self.actions)

    def __len__(self):
        return self.__size__

    def __getitem__(self, index):
        return self.states[index], self.actions[index]
