from src.agents.nerve_net import *
from torch_geometric.utils import dense_to_sparse

from GNN_Layers.EGAT import EGAT


class EGATMethod(MessagePassingGNN):
    def __init__(self,
                 in_dim: int,
                 action_dim: int,
                 device: torch.device,
                 node_message: str = 'x_j',
                 **kwargs
                 ):
        MessagePassingGNN.__init__(in_dim=in_dim, action_dim=, device=device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(EGAT(node_in_channels=self.hidden_node_dim,
                                    node_out_channels=self.hidden_node_dim,
                                    edge_in_channels=self.hidden_edge_dim,
                                    edge_out_channels=self.hidden_edge_dim,
                                    heads=self.num_heads,
                                    negative_slope=self.negative_slope,
                                    dropout=self.dropout,
                                    ).to(device))

    def forward(self, data):
        data = torch.tensor(data, dtype=torch.float, device=self.device)
        if data.dim() == 1:  # single observation
            data = self.make_graph(data)
            x, edge_idx_morph, mask, num_nodes = data.x, data.edge_index, data.mask, data.num_nodes
            batch = None

            edge_idx_fc, _ = dense_to_sparse(torch.ones(len(x), len(x), device=self.device))
        else:  # Batch of observations
            data = self.make_graph_batch(data)
            x, edge_idx_morph, mask, num_nodes = data.x, data.edge_index, data.mask, data.num_nodes
            batch = data.batch

            batch_ids = torch.unique(batch)
            edges = []
            for batch_id in batch_ids:
                nodes = torch.where(batch == batch_id)[0]
                edges.append(torch.cartesian_prod(nodes, nodes).T)
            edge_idx_fc = torch.cat(edges, dim=1)

        x = self.encoder(x=x)

        edge_attr_combined = torch.zeros(len(edge_idx_fc[0]),2)
        edge_attr_combined[:,-1]=1

        for idx_morf, src_morf in enumerate(edge_idx_morph[0]):
            for idx_fc, src_fc in enumerate(edge_idx_fc[0]):
                if src_morf == src_fc:
                    if edge_idx_morph[1][idx_morf] == edge_idx_fc[1][idx_fc]:
                        edge_attr_combined[idx_fc, 0] = 1
                        break 

        edge_index_combined = edge_idx_fc
        
        out = {'x': x, 'edge_attr': edge_attr_combined, 'edge_index': edge_index_combined}

        for i in range(self.propagation_steps - 1):
            out = self.middle[i](x=out['x'],
                                 edge_index=out['edge_index'],
                                 edge_attr=out['edge_attr'])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x = out['x']

        x = self.decoder(x=x)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if batch is not None:
            x = x[mask]
            batch_size = batch.max().item() + 1
            x = x.view(batch_size, -1)
            return x

        else:
            x = x.view(-1, self.num_nodes)
            x = x.squeeze(0)
            x = x[mask]
            return x


class SkrlEGAT(GaussianMixin, DeterministicMixin, Model):
    def __init__(self,
                 observation_space,
                 action_space,
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

        self.policy_network = EGATMethod(in_dim=per_node_dim,
                                          action_dim=1,
                                          device=device,
                                          **{k: v for k, v in kwargs.items() if k not in
                                             ['in_dim', 'num_nodes', 'mask']}, )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        inputs = inputs["states"]
        if role == "policy":
            return self.policy_network(inputs), self.log_std_parameter, {}
