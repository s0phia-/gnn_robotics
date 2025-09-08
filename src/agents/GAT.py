from src.agents.nerve_net import *
from torch_geometric.nn import GATConv
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.typing import OptTensor
import torch.nn.functional as F
from torch_geometric.utils import softmax


class GATTwoEdgeTypes(MessagePassingGNN):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 **kwargs
                 ):
        MessagePassingGNN.__init__(self, network_type=network_type, device=device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GATConv(
                in_channels=self.node_hidden_size,
                out_channels=self.node_hidden_size,
                heads=2,
                edge_dim=2,
                fill_value=0.5,
                concat=False,
            ).to(device))

    def forward(self, data: Data):
        x, edge_idx, mask, num_nodes, batch, node_dim, edge_attr = (data.x, data.edge_index, data.mask, data.num_nodes,
                                                                    data.batch, data.node_dim, data.edge_attr)
        if batch is None:
            batch_size = 1
            x = self.encoder(x=x, in_dim=node_dim)
        else:
            batch_size = batch.max().item() + 1
            x = self.encoder(x, node_dim[0].item())  # todo

        for i in range(self.propagation_steps - 1):
            x = self.middle[i](x=x, edge_index=edge_idx, edge_attr=edge_attr)

        x = self.decoder(x=x, batch=batch, batch_size=batch_size, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x


class GATMorphology(MessagePassingGNN):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 **kwargs
                 ):
        MessagePassingGNN.__init__(self, network_type=network_type, device=device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GATConv(
                in_channels=self.node_hidden_size,
                out_channels=self.node_hidden_size,
                heads=2,
                concat=False,
            ).to(device))

    def forward(self, data: Data):
        x, edge_idx, mask, num_nodes, batch, node_dim, edge_attr = (data.x, data.edge_index, data.mask, data.num_nodes,
                                                                    data.batch, data.node_dim, data.edge_attr)
        edge_attr_mask = [attr[0] == 1 for attr in edge_attr]
        filtered_edges = []
        for i, edge_list in enumerate(edge_idx):
            filtered_edges.append([edge_list[j] for j, keep in enumerate(edge_attr_mask) if keep])

        if batch is None:
            batch_size = 1
            x = self.encoder(x=x, in_dim=node_dim)
        else:
            batch_size = batch.max().item() + 1
            x = self.encoder(x, node_dim[0].item())  # todo

        for i in range(self.propagation_steps - 1):
            x = self.middle[i](x=x, edge_index=edge_idx)

        x = self.decoder(x=x, batch=batch, batch_size=batch_size, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x


class GATFullyConnected(MessagePassingGNN):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 **kwargs
                 ):
        MessagePassingGNN.__init__(self, network_type=network_type, device=device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GATConv(
                in_channels=self.node_hidden_size,
                out_channels=self.node_hidden_size,
                heads=2,
                concat=False,
            ).to(device))

    def forward(self, data: Data):
        x, edge_idx, mask, num_nodes, batch, node_dim, edge_attr = (data.x, data.edge_index, data.mask, data.num_nodes,
                                                                    data.batch, data.node_dim, data.edge_attr)
        if batch is None:
            batch_size = 1
            x = self.encoder(x=x, in_dim=node_dim)
        else:
            batch_size = batch.max().item() + 1
            x = self.encoder(x, node_dim[0].item())  # todo

        for i in range(self.propagation_steps - 1):
            x = self.middle[i](x=x, edge_index=edge_idx)

        x = self.decoder(x=x, batch=batch, batch_size=batch_size, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x


class RegularizedGAT(GATConv):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 bias: bool = True,
                 residual: bool = False,
                 edge_type_1_regularization: float = 0.5,
                 **kwargs):
        GATConv.__init__(self,
                         in_channels,
                         out_channels,
                         heads,
                         concat,
                         negative_slope,
                         dropout,
                         add_self_loops,
                         edge_dim,
                         fill_value,
                         bias,
                         residual,
                         **kwargs)
        self.edge_type_1_regularization = edge_type_1_regularization

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
