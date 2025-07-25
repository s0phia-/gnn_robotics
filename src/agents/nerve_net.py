import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


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
                 hidden_shape: list,
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
        self.message_function = self._build_mlp(in_dim * 2, hidden_shape, out_dim, device)

        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim, hidden_size=out_dim, device=device)

    @staticmethod
    def _build_mlp(in_dim, hidden_shape, out_dim, device):
        layers = [nn.Linear(in_dim, hidden_shape[0], device=device), nn.Tanh()]
        for i in range(len(hidden_shape) - 1):
            layers.append(nn.Linear(hidden_shape[i], hidden_shape[i+1], device=device))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_shape[-1], out_dim, device=device))
        return nn.Sequential(*layers)

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
                 hidden_shape: list,
                 device: torch.device):
        """
        A decoder network, part four of the NerveNet Message Passing GNN architecture.
        :param out_dim:
        :param hidden_dim:
        :param hidden_layers:
        :param device:
        """
        super().__init__()

        self.layers = [nn.Linear(in_dim, hidden_shape[0], device=device), nn.Tanh()]
        for i in range(len(hidden_shape) - 1):
            self.layers.append(nn.Linear(hidden_shape[i], hidden_shape[i + 1], device=device))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_shape[-1], out_dim, device=device))
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
        self.node_feature_dim = in_dim
        self.device = device

        self.encoder = Encoder(in_dim=in_dim,
                               hidden_dim=self.node_representation_dim,
                               device=device).to(device)

        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(Gnnlayer(in_dim=self.node_representation_dim,
                                        out_dim=self.node_representation_dim,
                                        hidden_shape=self.network_shape,
                                        device=device))

        self.decoder = Decoder(out_dim=action_dim,
                               in_dim=self.node_representation_dim,
                               hidden_shape=self.network_shape,
                               device=device).to(device)

    def make_graph(self, obs):
        """
        make a pyg graph
        """
        env_idx = int(obs[-1])
        obs = obs[:-1]
        graph_data = getattr(self, f"graph_info_{env_idx}", None)
        num_nodes = graph_data['num_nodes']
        edge_idx = graph_data['edge_idx']
        actuator_mask = graph_data['actuator_mask']

        x = obs.view(num_nodes, -1)
        mask = torch.tensor(actuator_mask, dtype=torch.bool)
        return Data(x=x, edge_index=edge_idx), mask

    def make_graph_batch(self, obs_batch):
        env_idx = int(obs_batch[0][-1])
        graph_data = getattr(self, f"graph_info_{env_idx}", None)
        num_nodes = graph_data['num_nodes']
        edge_idx = graph_data['edge_idx']
        actuator_mask = graph_data['actuator_mask']

        data_list = []
        for obs in obs_batch:
            obs = obs[:-1]
            x = obs.view(num_nodes, -1)
            graph = Data(x=x, edge_index=edge_idx)
            data_list.append(graph)
        return Batch.from_data_list(data_list), actuator_mask

    def forward(self, data):
        if isinstance(data, Data):  # if a pytorch geometric object
            x, edge_index = data.x, data.edge_index
            batch = data.batch
        else:  # assume np array or torch tensor
            data = torch.tensor(data, dtype=torch.float, device=self.device)
            if data.dim() == 1:  # single observation
                data, mask = self.make_graph(data)
                x, edge_index = data.x, data.edge_index
                batch = None
            else:  # Batch of observations
                data, mask = self.make_graph_batch(data)
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
            x = x[:, mask]
            return x

        else:
            x = x[mask]
            return x


class SkrlNerveNet(GaussianMixin, DeterministicMixin, Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 num_nodes,
                 device,
                 clip_actions=False,
                 clip_log_std=True,
                 min_log_std=-20,
                 max_log_std=2,
                 reduction="sum",
                 **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        total_dim = observation_space.shape[0] - 1
        per_node_dim = total_dim // num_nodes

        self.policy_network = MessagePassingGNN(in_dim=per_node_dim,
                                                num_nodes=num_nodes,
                                                action_dim=1,
                                                device=device,
                                                ** {k: v for k, v in kwargs.items() if k not in
                                                    ['in_dim', 'num_nodes', 'mask']},)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        inputs = inputs["states"]
        if role == "policy":
            return self.policy_network(inputs), self.log_std_parameter, {}
