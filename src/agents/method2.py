from src.agents.function_approximators import *
from torch_geometric.utils import dense_to_sparse


class Method2Gnn(MessagePassingGNN):
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
            self.middle.append(GnnLayerDoubleAgg(in_dim=self.hidden_node_dim,
                                                 out_dim=self.hidden_node_dim,
                                                 hidden_dim=self.decoder_and_message_hidden_dim,
                                                 hidden_layers=self.decoder_and_message_layers,
                                                 device=device))


class GnnLayerDoubleAgg(Gnnlayer):
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
        self.message_function_type1 = self._build_mlp(in_dim * 2, hidden_dim, out_dim * 2, hidden_layers, device)
        self.message_function_type2 = self._build_mlp(in_dim * 2, hidden_dim, out_dim * 2, hidden_layers, device)

        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim*2, hidden_size=out_dim, device=device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        agg_type1 = self.propagate(edge_index, x=x, edge_type=1)

        edge_index_fc, _ = dense_to_sparse(torch.ones(len(x), len(x), device=self.device))
        agg_type2 = self.propagate(edge_index_fc, x=x, edge_type=2)

        combined_agg = torch.cat([agg_type1, agg_type2], dim=1)  # concatenate the aggregated messages
        updated_features = self.update_function(combined_agg, x)
        return updated_features

    def message(self, x_i, x_j, edge_type):
        msg = torch.cat([x_i, x_j], dim=-1)
        if edge_type == 1:
            return self.message_function_type1(msg)
        if edge_type == 2:
            return self.message_function_type2(msg)
