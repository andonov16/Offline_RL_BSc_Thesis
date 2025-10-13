import torch


# Behaviour Cloning class: FNN model
class BC(torch.nn.Module):
    def __init__(self, input_neurons: int,
                 hidden_neurons: int,
                 num_hidden_layers: int,
                 out_neurons: int,
                 activation_function: torch.nn.Module = torch.nn.ReLU()):
        super(BC, self).__init__()

        # Add the first (input) layer + activation function
        layers = [torch.nn.Linear(input_neurons, hidden_neurons),
                  activation_function]

        # Add the hidden layers
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation_function)

        layers.append(torch.nn.Linear(hidden_neurons, out_neurons))

        # Combine the layers into a container
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
