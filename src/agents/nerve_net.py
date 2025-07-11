import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from skrl.models.torch import Model, GaussianMixin


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
        self.device = device

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
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
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
        nn.Module.__init__(self)
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.num_nodes = num_nodes
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.node_feature_dim = in_dim
        self.device = device

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
        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float, device=self.device)
                print(f"DEBUG: states shape: {data.shape}")
                print(f"DEBUG: states type: {type(data)}")

            if data.dim() == 1:  # single observation
                obs_part = data[:135]
                print(f"DEBUG: obs_part shape: {obs_part.shape}")
                print(f"DEBUG: obs_part elements: {obs_part.numel()}")
                edge_part = data[135:]  # todo
                edge_index = edge_part.view(2, 16).long()
                x = obs_part.view(self.num_nodes, -1)
                batch = None
            else:  # Batch of observations
                batch_size = data.shape[0]
                print(f"DEBUG: batch_size: {batch_size}")
                print(f"DEBUG: self.num_nodes: {self.num_nodes}")
                obs_part = data[:, :135]   # todo
                print(f"DEBUG: obs_part shape: {obs_part.shape}")
                print(f"DEBUG: obs_part elements: {obs_part.numel()}")
                edge_part = data[:, 135:]
                edge_indices = edge_part.view(batch_size, 2, 16).long()
                x = obs_part.view(batch_size * self.num_nodes, -1)
                batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes)

                edge_index_batch = []
                for i in range(batch_size):
                    offset = i * self.num_nodes
                    edge_index_batch.append(edge_indices[i] + offset)
                edge_index = torch.cat(edge_index_batch, dim=1)
        else:
            x, edge_index = data.x, data.edge_index
            batch = data.batch

        x = self.encoder(x=x)

        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_index=edge_index)

        x = self.decoder(x=x)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x = x.view(-1, self.num_nodes)
        x = x.squeeze(0)

        if batch is not None:
            batch_size = batch.max().item() + 1
            x = x.view(batch_size, self.num_nodes)
            x = x[:, self.mask]
            return x

        else:
            x = x[self.mask]
            return x


# CHANGE: New SKRL-compatible class that inherits from your original
class SKRLMessagePassingGNN(MessagePassingGNN, Model, GaussianMixin):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True)

        MessagePassingGNN.__init__(
            self,
            in_dim=kwargs['in_dim'],
            num_nodes=kwargs['num_nodes'],
            action_dim=1,
            mask=kwargs['mask'],
            device=device,
            **{k: v for k, v in kwargs.items() if k not in ['in_dim', 'num_nodes', 'mask']}
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(action_space.shape[0], device=device))

    def compute(self, inputs, role=""):
        states = inputs["states"]
        mean = self.forward(states)
        log_std = self.log_std_parameter.expand_as(mean)
        return mean, log_std, {}

    def act(self, inputs, role=""):
        mean, log_std, outputs = self.compute(inputs, role)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        return actions, log_prob, outputs


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


def states_to_graph(states, num_nodes, edge_index):
    """
    Convert SKRL states tensor to PyTorch Geometric Data format
    You need to implement this based on how your observations are structured
    """
    if states.dim() == 1:
        # Single observation
        return make_graph(states, num_nodes, edge_index)
    else:
        # Batch of observations
        return make_graph_batch(states, num_nodes, edge_index)
