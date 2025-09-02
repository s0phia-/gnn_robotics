import torch
from torch_scatter import scatter_mean
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


class Encoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 device: torch.device):
        """
        An encoder network, part one of the NerveNet Message Passing GNN architecture.
        :param hidden_dim:
        :param device:
        """
        nn.Module.__init__(self)
        self.device = device
        self.hidden_dim = hidden_dim
        self._layers = {}

    def _get_layer(self, in_dim: int):
        if in_dim not in self._layers:
            layer = nn.Linear(in_dim, self.hidden_dim, device=self.device)
            for param in layer.parameters():
                param.requires_grad = False
            self._layers[in_dim] = [layer, nn.Tanh()]
        return self._layers[in_dim]

    def forward(self, x: torch.Tensor, in_dim: int) -> torch.Tensor:
        layers = self._get_layer(in_dim)
        layers = nn.Sequential(*layers)
        return layers(x)


class Gnnlayer(MessagePassing):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_shape: list,
                 device: torch.device,
                 aggregator_type: str = 'mean'):
        """
        Message passing graph neural network, used between an encoder and decoder in NerveNet.
        :param in_dim:
        :param out_dim:
        :param hidden_shape:
        :param device:
        :param aggregator_type:
        """
        super().__init__(aggr=aggregator_type)
        self.device = device
        # construct message function
        self.message_function = self._build_mlp(in_dim * 2, hidden_shape, out_dim, device)
        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim, hidden_size=out_dim, device=device)

    def _build_mlp(self, in_dim, hidden_shape, out_dim, device):
        layers = [nn.Linear(in_dim, hidden_shape[0], device=device), nn.Tanh()]
        for i in range(len(hidden_shape) - 1):
            layers.append(nn.Linear(hidden_shape[i], hidden_shape[i+1], device=device))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_shape[-1], out_dim, device=device))
        network_layers = nn.Sequential(*layers)
        # self._init_weights(network_layers)
        return network_layers

    @staticmethod
    def _init_weights(network_layers, method="orthogonal"):
        if method == "orthogonal":
            init_ftn = nn.init.orthogonal_
        for layer in network_layers:
            if isinstance(layer, nn.Linear):
                init_ftn(layer.weight)
                nn.init.zeros_(layer.bias)

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
                 device: torch.device,
                 network_type: str = 'actor'):
        """
        A decoder network, part four of the NerveNet Message Passing GNN architecture.
        :param in_dim:
        :param out_dim:
        :param hidden_shape:
        :param device:
        """
        super().__init__()
        self.network_type = network_type
        self.actor_layers = self._build_mlp(in_dim, hidden_shape, out_dim, device)
        self.critic_layers = self._build_mlp(in_dim, hidden_shape, 1, device)

    def _build_mlp(self, in_dim, hidden_shape, out_dim, device):
        layers = [nn.Linear(in_dim, hidden_shape[0], device=device), nn.Tanh()]
        for i in range(len(hidden_shape) - 1):
            layers.append(nn.Linear(hidden_shape[i], hidden_shape[i+1], device=device))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_shape[-1], out_dim, device=device))
        layers = nn.Sequential(*layers)
        # self._init_weights(layers)
        return layers

    @staticmethod
    def _init_weights(network_layers, method="orthogonal"):
        if method == "orthogonal":
            init_ftn = nn.init.orthogonal_
        for layer in network_layers:
            if isinstance(layer, nn.Linear):
                init_ftn(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask, batch, batch_size):
        if self.network_type == 'actor':
            return self.forward_actor(x, batch_size=batch_size, mask=mask)
        else:
            return self.forward_critic(x, batch=batch)

    def forward_actor(self, x: torch.Tensor, batch_size, mask):
        x = self.actor_layers(x)
        x = x[mask]
        x = x.view(batch_size, -1)
        if batch_size == 1:
            x = x.squeeze()
        return x

    def forward_critic(self, x: torch.Tensor, batch=None):
        output = self.critic_layers(x)
        if batch is None:
            return output.mean(dim=0)
        else:
            return scatter_mean(output, batch, dim=0)


class MessagePassingGNN(nn.Module):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 **kwargs
                 ):
        """
        Message passing GNN architecture.
        see https://openreview.net/forum?id=S1sqHMZCb
        :param device:
        """
        nn.Module.__init__(self)
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.device = device
        self.encoder = Encoder(hidden_dim=self.node_hidden_size,
                               device=device).to(device)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(Gnnlayer(in_dim=self.node_hidden_size,
                                        out_dim=self.node_hidden_size,
                                        hidden_shape=self.network_shape,
                                        device=device))
        self.decoder = Decoder(in_dim=self.node_hidden_size,
                               out_dim=1,
                               hidden_shape=self.network_shape,
                               network_type=network_type,
                               device=device).to(device)

    def forward(self, data: Data):
        x, edge_index, mask, num_nodes, batch, node_dim = (data.x, data.edge_index, data.mask, data.num_nodes,
                                                           data.batch, data.node_dim)
        if batch is None:  # not a batch
            batch_size = 1
            x = self.encoder(x=x, in_dim=node_dim)
        else:  # a batch
            batch_size = batch.max().item() + 1
            x = self.encoder(x, node_dim[0].item())
        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_index=edge_index)
        x = self.decoder(x=x, batch=batch, batch_size=batch_size, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x
