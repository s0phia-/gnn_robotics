from src.agents.nerve_net import *
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.utils import softmax


class GAT(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=2,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.0,
                 add_self_loops=False,
                 contribution=0.5):
        """
        Graph Attention Network (GAT) layer.
        Parameters:
        - in_channels (int): Number of input features per node.
        - out_channels (int): Number of output features per node.
        - heads (int): Number of attention heads.
        - concat (bool): Whether to concatenate the output of multiple heads.
        - negative_slope (float): Negative slope for LeakyReLU activation.
        - dropout (float): Dropout rate for attention coefficients.
        - add_self_loops (bool): Whether to add self-loops to the input graph.
        - contribution (float): Contribution factor for edge features (not used in this implementation).
        """
        super(GAT, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # node method initialisation
        self.src_lin = Linear(in_channels, heads * out_channels,
                              bias=False)  # changed: replaced separate src/dst transformations with one
        self.dst_lin = Linear(in_channels, heads * out_channels,
                              bias=False)  # changed: replaced separate src/dst transformations with one
        self.att = Parameter(
            torch.Tensor(1, heads, 2 * out_channels))  # changed: unified attention mechanism with concat of x_i and x_j
        self.bias = Parameter(torch.Tensor(out_channels)) if concat else Parameter(
            torch.Tensor(out_channels))  # added: bias term depending on concat

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.src_lin.weight)  # changed: match new linear layer name
        torch.nn.init.xavier_uniform_(self.dst_lin.weight)
        torch.nn.init.xavier_uniform_(self.att)  # changed: match new attention parameter
        torch.nn.init.zeros_(self.bias)  # added: initialize bias

    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        src_x = self.src_lin(x)  # changed: apply single linear transform and reshape

        dst_x = self.dst_lin(x)  # .view(-1,H,C)  # changed: apply single linear transform and reshape
        x = (src_x, dst_x)  # changed: use tuple for source and target embeddings

        return self.propagate(edge_index, x=x)  # unchanged

    def message(self, x_i, x_j, index, ptr, size_i):  # changed: added x_i and x_j for attention
        x_j = x_j.view(-1, self.heads, self.out_channels)  # changed: reshape to (N, heads, out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)  # changed: reshape to (N, heads, out_channels)

        alpha = torch.cat([x_i, x_j], dim=-1)  # added: concatenate source and target embeddings
        alpha = (alpha * self.att)
        alpha = alpha.sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # unchanged
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)  # changed: apply softmax over neighbors
        alpha = F.dropout(alpha, p=self.dropout)  # , training=self.training)  # unchanged
        alpha = alpha.view(-1, self.heads, 1)

        print("alpha shape after softmax ", alpha.shape)
        print("x_j shape before reshape ", x_j.shape)
        x_j = (alpha * x_j).view(-1, self.heads * self.out_channels)
        # x_j = x_j.view(-1, self.heads*self.out_channels)  # changed: apply attention weights to target embeddings
        return x_j

    def update(self, aggr_out):  # added: update function to apply bias and head aggregation
        aggr_out = aggr_out.view(-1, self.heads, self.out_channels)

        return aggr_out.mean(dim=1) + self.bias  # added: average heads and apply bias


class GAT_Method(MessagePassingGNN):
    def __init__(self,
                 in_dim: int,
                 num_nodes: int,
                 action_dim: int,
                 device: torch.device,
                 **kwargs
                 ):
        MessagePassingGNN.__init__(in_dim, num_nodes, action_dim, device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GAT(in_channels=self.hidden_node_dim,
                                   out_channels=self.hidden_node_dim,
                                   heads=self.num_heads,
                                   concat=True,
                                   negative_slope=self.negative_slope,
                                   dropout=self.dropout,
                                   add_self_loops=False,
                                   contribution=0.5).to(device))
