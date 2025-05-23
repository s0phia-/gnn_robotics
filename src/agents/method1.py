from src.agents.function_approximators import Gnnlayer
from src.agents.method2 import *
from torch_geometric.utils import scatter


class Method1Gnn(Method2Gnn):
    def __init__(self,
                 in_dim: int,
                 num_nodes: int,
                 action_dim: int,
                 mask: list,
                 device: torch.device,
                 **kwargs
                 ):
        super().__init__(in_dim, num_nodes, action_dim, mask, device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GnnLayerDoubleMessage(in_dim=self.hidden_node_dim,
                                                     out_dim=self.hidden_node_dim,
                                                     hidden_dim=self.decoder_and_message_hidden_dim,
                                                     hidden_layers=self.decoder_and_message_layers,
                                                     device=device))


class GnnLayerDoubleMessage(Gnnlayer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device,
                 aggregator_type: str = 'mean'):
        """
        Message passing GNN layer with two edge types, each aggregated separately and then combined in an update
        function which now takes the form h_{t+1} = U(h_t, agg1, agg2) where agg1 and agg2 are the separately aggregated
        messages.
        :param message_hidden_layers:
        :param message_hidden_dim:
        :param update_hidden_layers:
        :param update_hidden_dim:
        :param device:
        """
        super().__init__(in_dim, out_dim, hidden_dim, hidden_layers, device, aggregator_type)

        # construct message functions
        self.message_function_type1 = self._build_mlp(in_dim * 2, hidden_dim, out_dim, hidden_layers, device)
        self.message_function_type2 = self._build_mlp(in_dim * 2, hidden_dim, out_dim, hidden_layers, device)

    def forward(self, x: torch.Tensor, edge_index_type1: torch.Tensor, edge_index_type2: torch.Tensor):
        self.current_edge_type = 1
        msg_type1 = self._get_messages(x, edge_index_type1)
        self.current_edge_type = 2
        msg_type2 = self._get_messages(x, edge_index_type2)

        # Combine edge indices and messages
        combined_edge_index = torch.cat([edge_index_type1, edge_index_type2], dim=1)
        combined_messages = torch.cat([msg_type1, msg_type2], dim=0)

        # Aggregate combined messages
        aggr_out = scatter(combined_messages, combined_edge_index[1], dim=0, reduce=self.aggr, dim_size=x.size(0))

        return self.update_function(aggr_out, x)

    def _get_messages(self, x, edge_index):
        """Get raw messages for given edge index without aggregation"""
        row, col = edge_index
        return self.message(x[row], x[col])

    def message(self, x_i, x_j):
        msg = torch.cat([x_i, x_j], dim=-1)
        if self.current_edge_type == 1:
            return self.message_function_type1(msg)
        else:  # edge_type == 2
            return self.message_function_type2(msg)
