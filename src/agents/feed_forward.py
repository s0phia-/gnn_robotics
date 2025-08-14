import torch.nn as nn
import torch
import numpy as np


class FeedForward(nn.Module):
    def __init__(self,
                 hidden_shape: list,
                 out_dim: int,
                 device: torch.device):
        """
        Feed forward Neural Network
        :param in_dim: dimensions of input to network
        :param out_dim: dimensions of output of network
        """
        nn.Module.__init__(self)
        self.hidden_shape = hidden_shape
        self.device = device
        self.out_dim = out_dim
        self._layers = {}

    def _get_layer(self, in_dim: int):
        if in_dim not in self._layers:
            self.layers = [nn.Linear(in_dim, self.hidden_shape[0], device=self.device), nn.ReLU()]
            for i in range(len(self.hidden_shape) - 1):
                self.layers.append(nn.Linear(self.hidden_shape[i], self.hidden_shape[i + 1], device=self.device))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(self.hidden_shape[-1], self.out_dim, device=self.device))
        return self._layers[in_dim]

    def forward(self, x: torch.Tensor, in_dim: int) -> torch.Tensor:
        layers = self._get_layer(in_dim)
        layers = nn.Sequential(*layers)
        return layers(x)
