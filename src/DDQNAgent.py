import torch


# BC agent used only during the loss calculation as a regularization term??
class DDQNAgent(torch.nn.Module):
    def __init__(self,
                 Q_net_online: torch.nn.Module,
                 Q_net_target: torch.nn.Module,):
        super(DDQNAgent, self).__init__()
        self.Q_net_online = Q_net_online
        self.Q_net_target = Q_net_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # after training in the live env. only Q_online is used
        self.Q_net_online(x)