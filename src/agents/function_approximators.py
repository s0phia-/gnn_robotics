from typing import Callable
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch_geometric.data


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
                #  edge_index: torch.Tensor,
                 actuator_mapping: Callable,
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
        self.actuator_mapping = actuator_mapping
        self.num_nodes = num_nodes
        self.node_feature_dim = in_dim
        # self.register_buffer('edge_index', edge_index)

        self.encoder = Encoder(in_dim=in_dim,
                               hidden_dim=self.hidden_node_dim,
                               device=device).to(device)

        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GGNN_layer(in_dim=self.hidden_node_dim,
                                          out_dim=self.hidden_node_dim,
                                          hidden_dim=self.decoder_and_message_hidden_dim,
                                          hidden_layers=self.decoder_and_message_layers,
                                          device=device))
        print('decodor set up')

        self.decoder = Decoder(out_dim=action_dim,
                               in_dim=self.hidden_node_dim,
                               hidden_dim=self.decoder_and_message_hidden_dim,
                               hidden_layers=self.decoder_and_message_layers,
                               device=device).to(device)
        

    def forward(self, data:torch_geometric.data.Data):
        x, edge_index = data.x, data.edge_index

        x = self.encoder(x=x)

        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_index=edge_index)
        
        x = self.decoder(x=x)
        print('shape after message passing : ',x.shape)
        x = x.view(-1, self.num_nodes)
        return x.squeeze(0)


def make_graph(obs,num_nodes,edge_index):
    """
    make a pyg graph
    """
    x = torch.tensor(obs, dtype=torch.float).view(num_nodes, -1)
    print('nodes shape : ',x.shape)
    return torch_geometric.data.Data(x=x, edge_index=edge_index)

def graph_to_action(graph):
    """
    convert graph to action
    """
    x = graph.x
    x = x.view(-1)
    x = x.unsqueeze(0)
    return x

    # def forward(self, x: torch.Tensor):
    #     batch_size = 1
    #     if x.dim() > 1:
    #         batch_size = x.size(0)
    #         x = x.view(batch_size, self.num_nodes, self.node_feature_dim)
    #         x = x.reshape(batch_size * self.num_nodes, self.node_feature_dim)
    #     else:
    #         x = x.view(self.num_nodes, self.node_feature_dim)

    #     x = self.encoder(x=x)

    #     if batch_size > 1:
    #         edge_indices_batched = []
    #         for i in range(batch_size):
    #             offset = i * self.num_nodes
    #             edge_indices_batched.append(self.edge_index + offset)
    #         batched_edge_index = torch.cat(edge_indices_batched, dim=1)
    #     else:
    #         batched_edge_index = self.edge_index

    #     for i in range(self.propagation_steps):
    #         x = self.middle[i](x=x, edge_index=batched_edge_index)

    #     x = self.decoder(x=x)
    #     x = x.squeeze(-1)

    #     if batch_size > 1:
    #         x = x.view(batch_size, self.num_nodes)
    #         actuator_outputs = []
    #         for i in range(batch_size):
    #             actuator_outputs.append(self.actuator_mapping(x[i]))
    #         return torch.stack(actuator_outputs)
    #     else:
    #         return self.actuator_mapping(x)


class MultiEdgeGGNN_layer(GGNN_layer):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, device):
        super().__init__(in_dim, out_dim, hidden_dim, hidden_layers, device)

        # Add a separate message function for global edges
        self.global_message_layers = [nn.Linear(in_dim * 2, hidden_dim, device=device), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            self.global_message_layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            self.global_message_layers.append(nn.Tanh())
        self.global_message_layers.append(nn.Linear(hidden_dim, out_dim, device=device))
        self.global_message_function = nn.Sequential(*self.global_message_layers)

    def message(self, x_i, x_j, edge_type=None):
        msg = torch.cat([x_i, x_j], dim=-1)
        # Use appropriate message function based on edge type
        global_msg = self.global_message_function(msg)
        orig_msg = self.message_function(msg)
        # Select messages based on edge type
        mask_global = (edge_type == 1).view(-1, 1)
        mask_orig = (edge_type == 0).view(-1, 1)
        return mask_orig * orig_msg + mask_global * global_msg


class MultiEdgeTypeGNN(MessagePassingGNN):
    def __init__(self, in_dim, num_nodes, edge_index, actuator_mapping, device, **kwargs):
        super().__init__(in_dim, num_nodes, edge_index, actuator_mapping, device, **kwargs)

        # Create fully connected edges
        fc_edges = self._create_fully_connected_edges(num_nodes).to(device)
        self.register_buffer('fc_edge_index', fc_edges)

        # Replace middle layers with multi-edge version
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(MultiEdgeGGNN_layer(
                in_dim=self.hidden_node_dim,
                out_dim=self.hidden_node_dim,
                hidden_dim=self.decoder_and_message_hidden_dim,
                hidden_layers=self.decoder_and_message_layers,
                device=device
            ))

    def _create_fully_connected_edges(self, num_nodes):
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long)

    def forward(self, x):
        batch_size = 1
        if x.dim() > 1:
            batch_size = x.size(0)
            x = x.view(batch_size, self.num_nodes, self.node_feature_dim)
            x = x.reshape(batch_size * self.num_nodes, self.node_feature_dim)
        else:
            x = x.view(self.num_nodes, self.node_feature_dim)

        x = self.encoder(x=x)

        if batch_size > 1:
            orig_edges_batched = [self.edge_index + i * self.num_nodes for i in range(batch_size)]
            orig_edge_index = torch.cat(orig_edges_batched, dim=1)
            fc_edges_batched = [self.fc_edge_index + i * self.num_nodes for i in range(batch_size)]
            fc_edge_index = torch.cat(fc_edges_batched, dim=1)
        else:
            orig_edge_index = self.edge_index
            fc_edge_index = self.fc_edge_index

        combined_edge_index = torch.cat([orig_edge_index, fc_edge_index], dim=1)
        edge_type = torch.cat([
            torch.zeros(orig_edge_index.size(1), device=x.device),
            torch.ones(fc_edge_index.size(1), device=x.device)
        ])

        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_index=combined_edge_index, edge_type=edge_type)

        x = self.decoder(x=x)
        x = x.squeeze(-1)

        if batch_size > 1:
            x = x.view(batch_size, self.num_nodes)
            return torch.stack([self.actuator_mapping(x[i]) for i in range(batch_size)])
        else:
            return self.actuator_mapping(x)


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
