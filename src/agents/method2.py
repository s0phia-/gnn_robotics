from src.agents.nerve_net import *
from torch_geometric.utils import dense_to_sparse


class Method2Gnn(MessagePassingGNN):
    def __init__(self,
                 in_dim: int,
                 num_nodes: int,
                 action_dim: int,
                 device: torch.device,
                 **kwargs
                 ):
        super().__init__(in_dim, num_nodes, action_dim, device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GnnLayerDoubleAgg(in_dim=self.hidden_node_dim,
                                                 out_dim=self.hidden_node_dim,
                                                 hidden_dim=self.decoder_and_message_hidden_dim,
                                                 hidden_layers=self.decoder_and_message_layers,
                                                 device=device,
                                                 morph_weight=self.morphology_fc_ratio))

    def forward(self, data):
        if isinstance(data, Data):  # if a pytorch geometric object
            x, edge_index = data.x, data.edge_index
            batch = data.batch
        else:  # assume np array or torch tensor
            data = torch.tensor(data, dtype=torch.float, device=self.device)
            if data.dim() == 1:  # single observation
                data, mask = self.make_graph(data)
                x, edge_morph = data.x, data.edge_index
                batch = None
            else:  # Batch of observations
                data, mask = self.make_graph_batch(data)
                x, edge_morph = data.x, data.edge_index
                batch = data.batch

        x = self.encoder(x=x)

        if batch is None:
            edge_fc, _ = dense_to_sparse(torch.ones(len(x), len(x), device=self.device))
        else:
            batch_ids = torch.unique(batch)
            edges = []
            for batch_id in batch_ids:
                nodes = torch.where(batch == batch_id)[0]
                edges.append(torch.cartesian_prod(nodes, nodes).T)
            edge_fc = torch.cat(edges, dim=1)

        for i in range(self.propagation_steps):
            x = self.middle[i](x=x, edge_morph=edge_morph, edge_fc=edge_fc)

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


class GnnLayerDoubleAgg(Gnnlayer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device,
                 aggregator_type: str = 'mean',
                 morph_weight: float = .5,):
        """
        Message passing GNN layer with two edge types, each aggregated separately and then combined in an update
        function which now takes the form h_{t+1} = U(h_t, agg1, agg2) where agg1 and agg2 are the separately aggregated
        messages.
        :param message_hidden_layers:
        :param message_hidden_dim:
        :param update_hidden_layers:
        :param update_hidden_dim:
        :param device:
        :param morph_weight: morphology weighting. Fully connected weighting will be 1-morph_weight
        """
        super().__init__(in_dim, out_dim, hidden_dim, hidden_layers, device, aggregator_type)
        self.morph_weight = morph_weight

        # construct message functions
        self.message_function_type1 = self._build_mlp(in_dim * 2, hidden_dim, out_dim * 2, hidden_layers, device)
        self.message_function_type2 = self._build_mlp(in_dim * 2, hidden_dim, out_dim * 2, hidden_layers, device)

        # construct update function
        self.update_function = nn.GRUCell(input_size=out_dim*2, hidden_size=out_dim, device=device)

    def forward(self, x: torch.Tensor, edge_morph: torch.Tensor, edge_fc: torch.Tensor) -> torch.Tensor:

        edge_morph, _ = add_self_loops(edge_morph, num_nodes=x.size(0))
        agg_type1 = self.propagate(edge_morph, x=x, edge_type=1)

        agg_type2 = self.propagate(edge_fc, x=x, edge_type=2)

        combined_agg = torch.cat([agg_type1, agg_type2], dim=1)
        updated_features = self.update_function(combined_agg, x)
        return updated_features

    def message(self, x_i, x_j, edge_type):
        msg = torch.cat([x_i, x_j], dim=-1)
        if edge_type == 1:  # morphology respecting
            return self.message_function_type1(msg)*self.morph_weight
        if edge_type == 2:  # fully connected
            return self.message_function_type2(msg)*(1-self.morph_weight)


class SKRLMethod2GNN(Method2Gnn, SKRLMixin):
    def __init__(self, observation_space, action_space, device, **kwargs):

        SKRLMixin.__init__(self, observation_space, action_space, device, **kwargs)
        Method2Gnn.__init__(
            self,
            in_dim=kwargs['in_dim'],
            num_nodes=kwargs['num_nodes'],
            action_dim=1,
            device=device,
            **{k: v for k, v in kwargs.items() if k not in ['in_dim', 'num_nodes', 'mask']},
        )
        self._last_distribution = None
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space.shape[0], device=device))