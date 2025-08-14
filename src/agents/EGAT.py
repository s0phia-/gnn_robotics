from src.agents.nerve_net import *
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.utils import softmax
from src.agents.message_utils import MessagePass


class EGAT(MessagePassing):
    """"
    Edge-Featured Graph Attention Network

    my implementation of the EGAT proposed in doi:10.1007/978-3-030-86362-3_21

    """

    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 heads=2,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.0,
                 edge_contribution=0.5,
                 node_message='x_j',
                 edge_message=None,
                 **kwargs):
        super().__init__(aggr='add')

        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.heads = heads
        self.concat = concat
        kwargs['heads'] = heads
        self.node_message = MessagePass(node_message,
                                        node_out_channels,
                                        node_out_channels,
                                        edge_out_channels,
                                        edge_out_channels,
                                        **kwargs)

        # Node method initializations
        self.nm_src_lin = Linear(node_in_channels, heads * node_out_channels, bias=False)
        self.nm_dst_lin = Linear(node_in_channels, heads * node_out_channels, bias=False)
        self.nm_edge_lin = Linear(edge_in_channels, heads * edge_out_channels, bias=False)
        self.nm_node_att = Parameter(torch.Tensor(1, heads, 2 * node_out_channels + edge_out_channels))
        self.nm_bias = Parameter(torch.Tensor(heads * node_out_channels if concat else node_out_channels))

        # Edge method initializations
        self.em_src_lin = Linear(node_in_channels, heads * node_out_channels, bias=False)
        self.em_dst_lin = Linear(node_in_channels, heads * node_out_channels, bias=False)
        self.em_edge_lin = Linear(edge_in_channels, heads * edge_out_channels, bias=False)
        self.em_edge_att = Parameter(torch.Tensor(1, heads, 2 * node_out_channels + edge_out_channels))
        self.em_bias = Parameter(torch.Tensor(edge_out_channels))

        # Edge updater
        self.edge_update_mlp = nn.Sequential(
            Linear(2 * node_out_channels + 2 * edge_out_channels, edge_out_channels),
            nn.ReLU(),
            Linear(edge_out_channels, edge_out_channels)
        )

        self.edge_message = edge_message

        self.reset_parameters()

    def reset_parameters(self):
        # node method
        torch.nn.init.xavier_uniform_(self.nm_src_lin.weight)
        torch.nn.init.xavier_uniform_(self.nm_dst_lin.weight)
        torch.nn.init.xavier_uniform_(self.nm_edge_lin.weight)
        torch.nn.init.xavier_uniform_(self.nm_node_att)
        torch.nn.init.zeros_(self.nm_bias)

        # edge method
        torch.nn.init.xavier_uniform_(self.em_src_lin.weight)
        torch.nn.init.xavier_uniform_(self.em_dst_lin.weight)
        torch.nn.init.xavier_uniform_(self.em_edge_lin.weight)
        torch.nn.init.xavier_uniform_(self.em_edge_att)
        torch.nn.init.zeros_(self.em_bias)

        # Initialize MLP weights
        for layer in self.edge_update_mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_attr):
        # Node updates
        nm_src_x = self.nm_src_lin(x)
        nm_dst_x = self.nm_dst_lin(x)
        nm_x = (nm_src_x, nm_dst_x)
        nm_edge_attr = self.nm_edge_lin(edge_attr)

        node_out = self.propagate(edge_index, x=nm_x, edge_attr=nm_edge_attr)

        # Edge updates
        em_src_x = self.em_src_lin(x)
        em_dst_x = self.em_dst_lin(x)
        em_x = (em_src_x, em_dst_x)
        em_edge_attr = self.nm_edge_lin(edge_attr)

        edge_out = self.edge_updater(edge_index, x=em_x, edge_attr=em_edge_attr)

        return {'x': node_out, 'edge_attr': edge_out, 'edge_index': edge_index}

    def message(self, x_i, x_j, index, edge_attr, ptr, size_i):
        x_j = x_j.view(-1, self.heads, self.node_out_channels)
        x_i = x_i.view(-1, self.heads, self.node_out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.edge_out_channels)

        # Attention computation
        alpha = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = (alpha * self.nm_node_att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        alpha = alpha.view(-1, self.heads, 1)

        # Apply message function and attention
        message_out = self.node_message(x_i, x_j, edge_attr)
        x_j = (alpha * message_out).view(-1, self.heads * self.node_out_channels)

        return x_j

    def update(self, aggr_out):
        if self.concat:
            return aggr_out + self.nm_bias
        else:
            aggr_out = aggr_out.view(-1, self.heads, self.node_out_channels)
            return aggr_out.mean(dim=1) + self.nm_bias

    def edge_update(self, x_i, x_j, edge_attr, index, ptr, size_i):
        """
        Edge update function that computes edge features based on node features and edge attributes.
        """
        x_j = x_j.view(-1, self.heads, self.node_out_channels)
        x_i = x_i.view(-1, self.heads, self.node_out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.edge_out_channels)
        print(f'training : {self.training}')
        # Compute attention for edges
        beta = torch.cat([x_i, x_j, edge_attr], dim=-1)
        beta = (beta * self.em_edge_att).sum(dim=-1)
        beta = F.leaky_relu(beta, self.negative_slope)
        beta = softmax(beta, index, ptr, num_nodes=size_i)
        # beta = torch.sigmoid(beta)
        beta = F.dropout(beta, p=self.dropout, training=self.training)
        beta = beta.view(-1, self.heads, 1)

        # Apply attention to edge features
        attended_edge_attr = (beta * edge_attr).view(-1, self.heads * self.edge_out_channels)
        attended_edge_attr = attended_edge_attr.view(-1, self.heads, self.edge_out_channels)
        attended_edge_attr = attended_edge_attr.mean(dim=1)

        # Combine node and edge information
        x_i_mean = x_i.mean(dim=1)
        x_j_mean = x_j.mean(dim=1)
        edge_attr_mean = edge_attr.mean(dim=1)

        edge_info_combined = torch.cat([x_i_mean, x_j_mean, attended_edge_attr, edge_attr_mean], dim=-1)
        edge_out = self.edge_update_mlp(edge_info_combined)

        return edge_out


class EGATMethod(MessagePassingGNN):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 node_message: str = 'x_j',
                 **kwargs
                 ):
        MessagePassingGNN.__init__(self, network_type=network_type, device=device, **kwargs)
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

    def forward(self, data: Data):
        x, edge_idx_morph, mask, num_nodes, batch, node_dim = (data.x, data.edge_index, data.mask, data.num_nodes,
                                                               data.batch, data.node_dim)
        if batch is None:
            edge_idx_fc, _ = dense_to_sparse(torch.ones(len(x), len(x), device=self.device))
            batch_size = 1

            x = self.encoder(x=x, in_dim=node_dim)
        else:
            batch_size = batch.max().item() + 1
            # make fully connected edges
            batch_ids = torch.unique(batch)
            edges = []
            for batch_id in batch_ids:
                nodes = torch.where(batch == batch_id)[0]
                edges.append(torch.cartesian_prod(nodes, nodes).T)
            edge_idx_fc = torch.cat(edges, dim=1)

            x = self.encoder(x, node_dim[0].item())

        # make edge indices
        edge_attr_combined = torch.zeros(len(edge_idx_fc[0]), 2)
        edge_attr_combined[:, -1] = 1
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
        x = out['x']

        x = self.decoder(x=x, batch=batch, batch_size=batch_size, mask=mask)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return x
