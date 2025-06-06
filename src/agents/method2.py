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

    def forward(self, data):
        x_morph, edge_index_morph = data.x, data.edge_index
        n = data.num_nodes

        batch = data.batch
        if batch is not None:
            edge_index_fc = torch.cat(
                [torch.combinations(torch.where(batch == i)[0], 2).flip(1).repeat_interleave(2, dim=0).reshape(2, -1)
                 for i in range(batch.max().item() + 1)], dim=1)
        else:
            adj = torch.ones(n, n, device=data.edge_index.device) - torch.eye(n, device=data.edge_index.device)
            edge_index_fc, _ = dense_to_sparse(adj)

            del adj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        x = self.encoder(x=x_morph)

        for i in range(self.propagation_steps):
            x = self.middle[i](x=x,
                               edge_index_type1=edge_index_morph,
                               edge_index_type2=edge_index_fc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        self.current_edge_type = None

    def forward(self, x: torch.Tensor, edge_index_type1: torch.Tensor, edge_index_type2: torch.Tensor):
        self.current_edge_type = 1
        agg_type1 = self.propagate(edge_index_type1, x=x)

        self.current_edge_type = 2
        agg_type2 = self.propagate(edge_index_type2, x=x)

        combined_agg = torch.cat([agg_type1, agg_type2], dim=1)  # concatenate the aggregated messages
        updated_features = self.update_function(combined_agg, x)

        return updated_features

    def message(self, x_i, x_j):
        msg = torch.cat([x_i, x_j], dim=-1)
        if self.current_edge_type == 1:
            return self.message_function_type1(msg)
        else:  # edge_type == 2
            return self.message_function_type2(msg)
