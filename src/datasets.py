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


class DQNReplayMemoryDataset(Dataset):
    def __init__(self,
                 states_rewards_next_states_tensor: torch.Tensor,
                 dones_tensor: torch.Tensor,
                 actions_tensor: torch.Tensor):

        # states_rewards_next_states_tensor = [state(8), reward(1), next_state(8)]
        self.states = states_rewards_next_states_tensor[:, 0:8]
        self.rewards = states_rewards_next_states_tensor[:, 8]
        self.next_states = states_rewards_next_states_tensor[:, 9:17]

        self.dones = dones_tensor.bool()
        self.actions = actions_tensor.long()

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )