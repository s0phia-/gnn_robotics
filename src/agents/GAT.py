# %%
import matplotlib.pyplot as plt

import networkx as nx

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from torch_geometric.utils import from_networkx


class GAT(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 heads=2, 
                 concat=True, 
                 negative_slope=0.2, 
                 dropout=0.0,
                 add_self_loops=False):
        super(GAT, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.src_lin = Linear(in_channels, heads * out_channels, bias=False)  # changed: replaced separate src/dst transformations with one
        self.dst_lin = Linear(in_channels, heads * out_channels, bias=False)  # changed: replaced separate src/dst transformations with one


        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))  # changed: unified attention mechanism with concat of x_i and x_j

        self.bias = Parameter(torch.Tensor(out_channels)) if concat else Parameter(torch.Tensor(out_channels))  # added: bias term depending on concat
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.src_lin.weight)  # changed: match new linear layer name
        torch.nn.init.xavier_uniform_(self.dst_lin.weight)
        torch.nn.init.xavier_uniform_(self.att)  # changed: match new attention parameter
        torch.nn.init.zeros_(self.bias)  # added: initialize bias

    def forward(self, x, edge_index):
        src_x = self.src_lin(x)  # changed: apply single linear transform and reshape

        dst_x = self.dst_lin(x) #.view(-1,H,C)  # changed: apply single linear transform and reshape

        x = (src_x, dst_x)  # changed: use tuple for source and target embeddings


        # print("edge_index shape ",edge_index.shape)
        return self.propagate(edge_index, x=x)  # unchanged
    
    def message(self, x_i,x_j, index, ptr, size_i):  # changed: added x_i and x_j for attention
        x_j = x_j.view(-1,self.heads,self.out_channels)  # changed: reshape to (N, heads, out_channels)
        x_i = x_i.view(-1,self.heads,self.out_channels)  # changed: reshape to (N, heads, out_channels)

        alpha = torch.cat([x_i,x_j], dim=-1)  # added: concatenate source and target embeddings
        alpha = (alpha * self.att)

        alpha = alpha.sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)  # unchanged

        alpha = softmax(alpha, index, ptr, num_nodes=size_i)  # changed: apply softmax over neighbors

        alpha = F.dropout(alpha, p=self.dropout)#, training=self.training)  # unchanged

        alpha = alpha.view(-1, self.heads, 1)

        x_j = x_j.view(-1, self.heads*self.out_channels)  # changed: apply attention weights to target embeddings
        return  x_j # unchanged

    def update(self, aggr_out):  # added: update function to apply bias and head aggregation
        aggr_out = aggr_out.view(-1, self.heads, self.out_channels)

        return aggr_out.mean(dim=1) + self.bias  # added: average heads and apply bias


def visualize_graph_and_output(graph, output, title="Graph Visualization"):
    # Create a NetworkX graph from the edge index
    nx_graph = nx.Graph()
    edge_index = graph.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        nx_graph.add_edge(edge_index[0, i], edge_index[1, i], weight=1.0)

    # Get node positions
    pos = nx.spring_layout(nx_graph, seed=42)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)

    # Add node embeddings as labels
    for i, (x, y) in pos.items():
        plt.text(x, y + 0.1, f"{output[i].detach().numpy()}", fontsize=8, ha="center", color="red")

    plt.title(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create a simple graph
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 0]], dtype=torch.long)  # Edges in COO format
    x = torch.tensor([[0, 1], [1, 1], [2, 1]], dtype=torch.float)  # Node features

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # Instantiate the GAT model
    in_channels = x.size(1)
    out_channels = 4
    heads = 1
    gat = GAT(in_channels, out_channels, heads=heads)
    gat_2 = GAT(out_channels, out_channels, heads=heads)

    # Forward pass
    out = gat(data.x, data.edge_index)
    print("Output node embeddings:")
    print(out)

    visualize_graph_and_output(data, out, title="GAT Output Visualization")

    out = gat_2(out, data.edge_index)
    # print("Output node embeddings:")
    # print(out)
    visualize_graph_and_output(data, out, title="GAT Output Visualization")

# %%



        

