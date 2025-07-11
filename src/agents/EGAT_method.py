from src.agents.function_approximators import *
from torch_geometric.utils import dense_to_sparse

from GNN_Layers.EGAT import EGAT

class GAT_Method(MessagePassingGNN):
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
            self.middle.append(EGAT(node_in_channels=self.hidden_node_dim,
                                   node_out_channels=self.hidden_node_dim,
                                   edge_in_channels=self.hidden_edge_dim,
                                   edge_out_channels=self.hidden_edge_dim,
                                   heads=self.num_heads,
                                   negative_slope=self.negative_slope,
                                   dropout=self.dropout,
                                   ).to(device))
            
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
        
        edge_index_combined = torch.cat([edge_index_morph, edge_index_fc], dim=1)

        edge_attr_morph_zero = torch.zeros(len(edge_index_morph[0]), 1, device=edge_index_morph.device)
        edge_attr_morph_one = torch.ones_like(edge_attr_morph_zero) 
        edge_attr_morph = torch.cat([edge_attr_morph_zero, edge_attr_morph_one], dim=1) 

        edge_attr_fc_zero = torch.zeros(edge_index_fc.shape[0], 1, device=edge_index_morph.device)
        edge_attr_fc_one = torch.ones_like(edge_attr_fc_zero)
        edge_attr_fc = torch.cat([edge_attr_fc_zero, edge_attr_fc_one], dim=1)

        edge_index_combined = torch.cat([edge_index_morph, edge_index_fc], dim=1)
        edge_attr_combined = torch.cat([edge_attr_morph, edge_attr_fc], dim=0)
        x = self.encoder(x=x_morph)
        out = {'x': x, 'edge_attr': edge_attr_combined}
        for i in range(self.propagation_steps-1):
            out = self.middle[i](x=out['x'],
                               edge_index=edge_index_combined,
                               edge_attr=out['edge_attr'])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        x = out['x']

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
