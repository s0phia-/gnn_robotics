import torch.nn as nn
import torch
import numpy as np
from skrl.models.torch import DeterministicMixin
from skrl.models.torch import Model


class FeedForward(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device: torch.device,
                 hidden_dim: int = 64,
                 hidden_layers: int = 3):
        """
        Feed forward Neural Network
        :param in_dim: dimensions of input to network
        :param out_dim: dimensions of output of network
        """
        super().__init__()
        self.layers = [nn.Linear(in_dim, hidden_dim, device=device), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim, device=device))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.layers(x)


class SKRLFeedForward(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        DeterministicMixin.__init__(self)
        Model.__init__(self, observation_space, action_space, device)
        self.network = FeedForward(
            in_dim=observation_space.shape[0] - 1,
            out_dim=1,
            device=device,
            **kwargs
        )

    def compute(self, inputs, role=""):
        states = inputs["states"]
        if states.dim() == 1:
            states = states[:-1]
        else:
            states = states[:, :-1]
        value = self.network(states)
        return value, {}
