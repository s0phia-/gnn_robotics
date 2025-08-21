import torch.nn as nn
import torch
from torch_geometric.data import Data


class FeedForward(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_shape: list,
                 out_dim: int,
                 device: torch.device):
        """
        Feed forward Neural Network
        :param in_dim: dimensions of input to network
        :param out_dim: dimensions of output of network
        """
        nn.Module.__init__(self)

        self.layers = [nn.Linear(in_dim, hidden_shape[0], device=device), nn.ReLU()]
        for i in range(len(hidden_shape) - 1):
            self.layers.append(nn.Linear(hidden_shape[i], hidden_shape[i + 1], device=device))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_shape[-1], out_dim, device=device))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, data: Data) -> torch.Tensor:
        if data.batch is not None:
            batch_size = data.batch.max().item() + 1
        else:
            batch_size = 1
        flattened = data.x.view(batch_size, -1)
        flattened = flattened.squeeze()
        return self.layers(flattened)
