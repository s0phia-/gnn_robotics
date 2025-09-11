from src.agents.nerve_net import *


class Method2Gnn(MessagePassingGNN):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 **kwargs
                 ):
        MessagePassingGNN.__init__(self, device=device, network_type=network_type, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GnnLayerDoubleAgg(in_dim=self.node_hidden_size,
                                                 out_dim=self.node_hidden_size,
                                                 hidden_shape=self.network_shape,
                                                 device=device,
                                                 morph_weight=self.morphology_fc_ratio))

    def forward(self, data):
        x, edge_idx, mask, num_nodes, batch, node_dim, edge_type = (data.x, data.edge_index, data.mask, data.num_nodes,
                                                                    data.batch, data.node_dim, data.edge_attr)

        if batch is None:
            batch_size = 1
            x = self.encoder(x=x, in_dim=node_dim)
        else:
            batch_size = batch.max().item() + 1
            x = self.encoder(x, node_dim[0].item())  # todo

        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_idx=edge_idx, edge_type=edge_type)

        x = self.decoder(x=x, batch=batch, batch_size=batch_size, mask=mask)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x


class GnnLayerDoubleAgg(Gnnlayer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_shape: list,
                 device: torch.device,
                 aggregator_type: str = 'mean',
                 morph_weight: float = .5, ):
        """
        Message passing GNN layer with two edge types, each aggregated separately and then combined in an update
        function which now takes the form h_{t+1} = U(h_t, agg1, agg2) where agg1 and agg2 are the separately aggregated
        messages.
        :param in_dim: input dimensions
        :param out_dim: output dimensions
        :param hidden_shape: hidden dimensions
        :param device:
        :param aggregator_type: aggregation function for GNN. Examples: mean, sum
        :param morph_weight: morphology weighting. Fully connected weighting will be 1-morph_weight
        """
        Gnnlayer.__init__(self, in_dim, out_dim, hidden_shape, device, aggregator_type)
        self.morph_weight = morph_weight

        # construct message functions
        self.message_function_type0 = self._build_mlp(in_dim * 2, hidden_shape, out_dim * 2, device)
        self.message_function_type1 = self._build_mlp(in_dim * 2, hidden_shape, out_dim * 2, device)

        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim * 2, hidden_size=out_dim, device=device)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:

        edge_morph = edge_idx[:, edge_type[:, 0].bool()]
        edge_morph, _ = add_self_loops(edge_morph, num_nodes=x.size(0))
        agg_type0 = self.propagate(edge_morph, x=x, edge_type=0)

        edge_fc = edge_idx[:, edge_type[:, 1].bool()]
        agg_type1 = self.propagate(edge_fc, x=x, edge_type=1)

        combined_agg = torch.cat([agg_type0, agg_type1], dim=1)
        updated_features = self.update_function(combined_agg, x)
        return updated_features

    def message(self, x_i, x_j, edge_type):
        msg = torch.cat([x_i, x_j], dim=-1)
        if edge_type == 0:  # morphology respecting
            return self.message_function_type0(msg) * self.morph_weight
        if edge_type == 1:  # fully connected
            return self.message_function_type1(msg) * (1 - self.morph_weight)
