import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class Encoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 device: torch.device):
        """
        An encoder network, part one of the NerveNet Message Passing GNN architecture.
        :param in_dim:
        :param hidden_dim:
        :param device:
        """
        super().__init__()
        self.layers = [nn.Linear(in_dim, hidden_dim, device=device), nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.layers(x)


class GGNN_layer(MessagePassing):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device):
        """
        Message passing graph neural network, used between an encoder and decoder in NerveNet.
        :param message_hidden_layers:
        :param message_hidden_dim:
        :param update_hidden_layers:
        :param update_hidden_dim:
        :param device:
        """
        super().__init__(aggr='mean')

        # construct message function
        self.message_layers = [nn.Linear(in_dim * 2, hidden_dim, device=device), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            self.message_layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            self.message_layers.append(nn.Tanh())
        self.message_layers.append(nn.Linear(hidden_dim, out_dim, device=device))
        self.message_function = nn.Sequential(*self.message_layers)

        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim, hidden_size=out_dim, device=device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        msg = torch.cat([x_i, x_j], dim=-1)
        return self.message_function(msg)

    def update(self, aggr_out, x):
        return self.update_function(aggr_out, x)


class Decoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device):
        """
        A decoder network, part four of the NerveNet Message Passing GNN architecture.
        :param out_dim:
        :param hidden_dim:
        :param hidden_layers:
        :param device:
        """
        super().__init__()
        self.layers = [nn.Linear(in_dim, hidden_dim, device=device), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_dim, out_dim, device=device))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.layers(x)


class MessagePassingGNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device: torch.device,
                 propagation_steps: int = 3,  # 4, 5, 6
                 hidden_node_dim: int = 32,  # 64, 128
                 decoder_and_message_layers: int = 2,
                 decoder_and_message_hidden_dim: int = 64,  # 128, 256
                 ):
        """
        Message passing GNN architecture.
        see https://openreview.net/forum?id=S1sqHMZCb
        :param in_dim: dimensions of input to network
        :param out_dim: dimensions of output of network
        """
        super().__init__()
        self.propagation_steps = propagation_steps

        self.encoder = Encoder(in_dim=in_dim,
                               hidden_dim=hidden_node_dim,
                               device=device).to(device)

        self.middle = nn.ModuleList()
        for _ in range(propagation_steps):
            self.middle.append(GGNN_layer(in_dim=hidden_node_dim,
                                          out_dim=hidden_node_dim,
                                          hidden_dim=decoder_and_message_hidden_dim,
                                          hidden_layers=decoder_and_message_layers,
                                          device=device).to(device))

        self.decoder = Decoder(out_dim=out_dim,
                               in_dim=hidden_node_dim,
                               hidden_dim=decoder_and_message_hidden_dim,
                               hidden_layers=decoder_and_message_layers,
                               device=device).to(device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.encoder(x=x)
        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_index=edge_index)
        x = self.decoder(x=x)
        return x


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
