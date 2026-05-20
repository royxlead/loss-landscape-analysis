from __future__ import annotations

import torch
import torch.nn as nn


class SmallNet(nn.Module):
    """Fixed MLP architecture for all experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.reset_weights(seed=42)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_weights_flat(self) -> torch.Tensor:
        """Concatenate all model parameters into a single 1D tensor."""
        return torch.cat([param.view(-1) for param in self.parameters()])

    def reset_weights(self, seed: int = 42) -> None:
        """Reinitialize all linear layers with deterministic Xavier-uniform init."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


def get_fresh_model() -> SmallNet:
    """Create a new deterministically initialized model."""
    model = SmallNet()
    model.reset_weights(seed=42)
    return model
