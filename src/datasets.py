import torch
from torch.utils.data import Dataset, TensorDataset


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


class DQNReplayMemoryDataset(TensorDataset):
    def __init__(self, states_rewards_next_states_tensor: torch.Tensor,
                 dones_tensor: torch.Tensor,
                 actions_tensor: torch.Tensor,
                 device: torch.device):
        states = states_rewards_next_states_tensor[:, 0:8].float().contiguous().to(device)
        rewards = states_rewards_next_states_tensor[:, 8].float().contiguous().to(device)
        next_states = states_rewards_next_states_tensor[:, 9:17].float().contiguous().to(device)
        actions = actions_tensor.long().to(device)
        dones = dones_tensor.float().to(device)

        super().__init__(states, actions, rewards, next_states, dones)