import torch
from typing import Type


# Behaviour Cloning class: FNN model
class BC(torch.nn.Module):
    def __init__(self,
                 input_neurons: int,
                 hidden_neurons: int,
                 num_hidden_layers: int,
                 out_neurons: int,
                 activation_function: Type[torch.nn.Module] = torch.nn.ReLU,
                 dropout: float = 0):
        super(BC, self).__init__()

        # Add the first (input) layer + activation function
        layers = [torch.nn.Linear(input_neurons, hidden_neurons),
                  torch.nn.BatchNorm1d(hidden_neurons),
                  activation_function()]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        # Add the hidden layers
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(torch.nn.BatchNorm1d(hidden_neurons))
            layers.append(activation_function())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_neurons, out_neurons))

        # Combine the layers into a container
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))


if __name__ == "__main__":
    x_dummy = torch.tensor([0.45, 23, 11.43, 3, 5], dtype=torch.float32)
    x_dummy = torch.unsqueeze(x_dummy, dim=0)

    test_agent = BC(
        input_neurons=x_dummy.shape[-1],
        hidden_neurons=16,
        num_hidden_layers=2,
        out_neurons=4,
        dropout=0.1
    )
    test_agent.eval()

    output = test_agent(x_dummy)
    print(output)
    print("Output shape:", output.shape)
    output = test_agent.get_action_probs(x_dummy)
    print(output)
    print("Action probs. shape:", output.shape)

