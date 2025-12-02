import torch

class QNetwork(torch.nn.Module):
    def __init__(self,
                 input_neurons: int,
                 hidden_neurons: int = 128,
                 num_hidden_layers: int = 2,
                 out_neurons: int = 4,
                 dropout: float = 0.0):
        super(QNetwork, self).__init__()

        layers = [torch.nn.Linear(input_neurons, hidden_neurons),
                  # Apply LayerNorm only once after the first input layer?
                  torch.nn.LayerNorm(hidden_neurons),
                  torch.nn.GELU()]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(torch.nn.GELU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_neurons, out_neurons))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # outputs [Q(s,0), Q(s,1), Q(s,2), Q(s,3)]
        return self.network(x)