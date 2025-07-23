import torch.nn as nn
import torch
import numpy as np
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


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
        nn.Module.__init__(self)
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


class SKRLFeedForward(GaussianMixin, DeterministicMixin, Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 hidden_dim=64,
                 hidden_layers=3,
                 clip_actions=False,
                 clip_log_std=True,
                 min_log_std=-20,
                 max_log_std=2,
                 reduction="sum",
                 **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        in_dim = observation_space.shape[0] - 1

        self.policy_network = FeedForward(in_dim=in_dim,
                                          out_dim=self.num_actions,
                                          device=device)
        self.value_network = FeedForward(in_dim=in_dim,
                                         out_dim=1,
                                         device=device)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        inputs = inputs["states"][..., :-1]
        if role == "policy":
            return self.policy_network(inputs), self.log_std_parameter, {}
        elif role == "value":
            return self.value_network(inputs), {}
