import torch
from typing import Type

# Behaviour Cloning FNN model
class BC(torch.nn.Module):
    def __init__(self,
                 input_neurons: int,
                 hidden_neurons: int = 128,
                 num_hidden_layers: int = 2,
                 out_neurons: int = 4,
                 dropout: float = 0.0):
        super(BC, self).__init__()

        layers = [torch.nn.Linear(input_neurons, hidden_neurons),
                  torch.nn.LayerNorm(hidden_neurons),
                  torch.nn.GELU()]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(torch.nn.LayerNorm(hidden_neurons))
            layers.append(torch.nn.GELU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_neurons, out_neurons))
        self.network = torch.nn.Sequential(*layers)

        # He/Kaiming initialization for stability
        for m in self.network:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

    def get_action(self, x: torch.Tensor) -> int:
        return torch.argmax(self.get_action_probs(x), dim=-1)


if __name__ == "__main__":
    # Dummy input (Lunar Lander has 8 state features)
    x_dummy = torch.tensor([0.45, 23, 11.43, 3, 5, -0.2, 0.5, 0.1], dtype=torch.float32)
    x_dummy = x_dummy.unsqueeze(0)

    test_agent = BC(
        input_neurons=x_dummy.shape[-1],
        hidden_neurons=128,
        num_hidden_layers=2,
        out_neurons=4,
        dropout=0.1,
        activation_function=torch.nn.GELU
    )
    test_agent.eval()

    output = test_agent(x_dummy)
    print("Raw logits:", output)
    print("Output shape:", output.shape)

    probs = test_agent.get_action_probs(x_dummy)
    print("Action probabilities:", probs)
    print("Probabilities shape:", probs.shape)

    action = test_agent.get_action(x_dummy)
    print("Selected action:", action.item())