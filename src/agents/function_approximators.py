import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch


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
        self.layers = [nn.Linear(in_dim, hidden_dim, device=device), nn.Tanh()]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.layers(x)


class Gnnlayer(MessagePassing):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device,
                 aggregator_type: str = 'mean'):
        """
        Message passing graph neural network, used between an encoder and decoder in NerveNet.
        :param message_hidden_layers:
        :param message_hidden_dim:
        :param update_hidden_layers:
        :param update_hidden_dim:
        :param device:
        """
        super().__init__(aggr=aggregator_type)

        # construct message function
        self.message_function = self._build_mlp(in_dim * 2, hidden_dim, out_dim, hidden_layers, device)

        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim, hidden_size=out_dim, device=device)

    @staticmethod
    def _build_mlp(in_dim, hidden_dim, out_dim, hidden_layers, device):
        layers = [nn.Linear(in_dim, hidden_dim, device=device), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim, device=device))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) #add self-loops (the so on prop the message form the node is considered)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """
        Message passing involves aggregating information from neighboring nodes.
        For a given edge (i, j), where i is the target node and j is the source node:
        
        - x_i: Features of the target node i.
        - x_j: Features of the source node j.
        
        The message function computes a message m_ij as:
        
        m_ij = f_message([x_i || x_j])
        
        where:
        - [x_i || x_j] denotes the concatenation of x_i and x_j.
        - f_message is a neural network (here, self.message_function).
        
        These messages are then aggregated by mean for all neighbors of node i.
        """
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
                 num_nodes: int,
                 action_dim: int,
                 mask: list,
                 device: torch.device,
                 **kwargs
                 ):
        """
        Message passing GNN architecture.
        see https://openreview.net/forum?id=S1sqHMZCb
        :param in_dim: dimensions of input to network
        :param out_dim: dimensions of output of network
        """
        super().__init__()
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.num_nodes = num_nodes
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.node_feature_dim = in_dim

        self.encoder = Encoder(in_dim=in_dim,
                               hidden_dim=self.hidden_node_dim,
                               device=device).to(device)

        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(Gnnlayer(in_dim=self.hidden_node_dim,
                                          out_dim=self.hidden_node_dim,
                                          hidden_dim=self.decoder_and_message_hidden_dim,
                                          hidden_layers=self.decoder_and_message_layers,
                                          device=device))

        self.decoder = Decoder(out_dim=action_dim,
                               in_dim=self.hidden_node_dim,
                               hidden_dim=self.decoder_and_message_hidden_dim,
                               hidden_layers=self.decoder_and_message_layers,
                               device=device).to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        x = self.encoder(x=x)
        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_index=edge_index)
        
        x = self.decoder(x=x)
        x = x.view(-1, self.num_nodes)
        x = x.squeeze(0)
        
        if batch is not None:  # deals with when batch of graphs
            num_graphs = batch.max().item() + 1
            mask = self.mask.view(1, -1).repeat(num_graphs, 1).view(-1)
            x = x.view(-1)[mask]
            x = x.view(num_graphs, x.shape[0] // num_graphs)
            return x
        
        else:  # Single graph case
            x = x[self.mask]
            return x


def make_graph(obs, num_nodes, edge_index):
    """
    make a pyg graph
    """
    x = obs.view(num_nodes, -1)
    return Data(x=x, edge_index=edge_index)


def make_graph_batch(obs_batch, num_nodes, edge_index):
    data_list = []
    for obs in obs_batch:
        x = obs.view(num_nodes, -1)
        graph = Data(x=x, edge_index=edge_index)
        data_list.append(graph)

    return Batch.from_data_list(data_list)


def graph_to_action(graph):
    """
    convert graph to action
    """
    x = graph.x
    x = x.view(-1)
    x = x.unsqueeze(0)
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