from src.agents.nerve_net import MessagePassingGNN, Gnnlayer
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops


class Method5Gnn(MessagePassingGNN):
    def __init__(self,
                 in_dim: int,
                 num_nodes: int,
                 action_dim: int,
                 mask: list,
                 device: torch.device,
                 max_neighbours: int,
                 **kwargs
                 ):
        super().__init__(in_dim, num_nodes, action_dim, mask, device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GnnLayerNoPooling(in_dim=self.hidden_node_dim,
                                                 out_dim=self.hidden_node_dim,
                                                 hidden_dim=self.decoder_and_message_hidden_dim,
                                                 hidden_layers=self.decoder_and_message_layers,
                                                 device=device,
                                                 max_neighbours=max_neighbours))


class GnnLayerNoPooling(Gnnlayer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device,
                 max_neighbours: int,):
        super().__init__(in_dim, out_dim, hidden_dim, hidden_layers, device)

        self.max_neighbors = max_neighbours
        self.update_function = nn.GRUCell(input_size=int(out_dim*(max_neighbours+1)), hidden_size=out_dim, device=device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        msgs = self._get_messages(x, edge_index)
        node_msgs = self._concatenate_messages_with_padding(msgs, edge_index, x.size(0))
        return self.update_function(node_msgs, x)

    def _get_messages(self, x, edge_index):
        row, col = edge_index
        return self.message(x[row], x[col])

    def _concatenate_messages_with_padding(self, msgs, edge_index, num_nodes):
        out_dim = msgs.size(1)
        target_nodes = edge_index[0]

        node_msgs = torch.zeros(num_nodes, int(out_dim * (self.max_neighbors + 1)), device=self.device)
        for node_id in range(num_nodes):
            mask = (target_nodes == node_id)
            node_messages = msgs[mask]

            if node_messages.size(0) > 0:
                flattened_msgs = node_messages.flatten()
                node_msgs[node_id, :len(flattened_msgs)] = flattened_msgs
        return node_msgs
