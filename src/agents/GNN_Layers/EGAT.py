# %%
"""
Edge-Feature Graph Attention Network (GAT) implementation using PyTorch Geometric.

based on the EGAT proposed in https://doi.org/10.1007/978-3-030-86362-3_21

by Ferdinand Krammer"""
import networkx as nx

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, softmax, dropout, degree
from torch_geometric.utils import from_networkx


class EGAT(MessagePassing):
    """"
    Edge-Featured Graph Attention Network 

    my implementation of the EGAT proposed in doi:10.1007/978-3-030-86362-2_21
    
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
                 add_self_loops=False,
                 edge_contribution=0.5):
        super(EGAT, self).__init__(aggr='add')
        # general params initialisation
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels

        self.heads = heads
        self.concat = concat

        self.negative_slope = negative_slope
        self.dropout = dropout

        self.add_self_loops = False 
        self.edge_congribution = edge_contribution

        # mode method initialisations
        self.nm_src_lin = Linear(node_in_channels, heads * node_out_channels, bias=False) 
        self.nm_dst_lin = Linear(node_in_channels, heads * node_out_channels, bias=False) 
        self.nm_edge_lin = Linear(edge_in_channels, heads * edge_out_channels, bias=False)  
        self.nm_node_att = Parameter(torch.Tensor(1, heads, 2 * node_out_channels + edge_out_channels)) 
        self.nm_bias = Parameter(torch.Tensor(node_out_channels)) if concat else Parameter(torch.Tensor(node_out_channels))  # added: bias term depending on concat

        # edge method intitialisations
        self.em_src_lin = Linear(node_in_channels, heads * node_out_channels, bias=False) 
        self.em_dst_lin = Linear(node_in_channels, heads * node_out_channels, bias=False) 
        self.em_edge_lin = Linear(edge_in_channels, heads * edge_out_channels, bias=False)  
        self.em_edge_att = Parameter(torch.Tensor(1, heads, 2 * node_out_channels + edge_out_channels)) 
        self.em_bias = Parameter(torch.Tensor(node_out_channels)) if concat else Parameter(torch.Tensor(node_out_channels))  # added: bias term depending on concat

        # edge updater
        self.edge_update_mlp = nn.Sequential(
            Linear(node_out_channels*2+edge_out_channels,edge_in_channels,edge_out_channels),)
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

    def forward(self, x, edge_index, edge_attr):
        nm_src_x = self.nm_src_lin(x)  
        nm_dst_x = self.nm_dst_lin(x) 
        nm_x = (nm_src_x, nm_dst_x)
        nm_edge_attr = self.nm_edge_lin(edge_attr)

        out = self.propagate(edge_index, x=nm_x, edge_attr=nm_edge_attr)

        em_src_x = self.em_src_lin(x)  
        em_dst_x = self.em_dst_lin(x)
        em_x = (em_src_x, em_dst_x)
        em_edge_attr = self.em_edge_lin(edge_attr)

        edge_out = self.edge_updater(edge_index, x=em_x, edge_attr=em_edge_attr)

        return {'x': out, 'edge_attr': edge_out}
    
    def message(self, x_i,x_j, index, edge_attr, ptr, size_i):  
        x_j = x_j.view(-1,self.heads,self.node_out_channels)  
        x_i = x_i.view(-1,self.heads,self.node_out_channels)  
        edge_attr = edge_attr.view(-1,self.heads,self.edge_out_channels)  

        alpha = torch.cat([x_i,x_j,edge_attr], dim=-1) 
        alpha = (alpha * self.nm_node_att).sum(dim=-1)   
        alpha = F.leaky_relu(alpha, self.negative_slope)  
        alpha = softmax(alpha, index, ptr, num_nodes=size_i) 
        alpha = F.dropout(alpha, p=self.dropout)

        alpha = alpha.view(-1, self.heads, 1)
        x_j = (alpha * x_j).view(-1, self.heads * self.node_out_channels)
        return  x_j # unchanged

    def update(self, aggr_out):  
        # if self.concat:
        #     return aggr_out.view(-1, self.heads * self.out_channels) + self.bias  # added: concat heads and apply bias
        # else:
        aggr_out = aggr_out.view(-1, self.heads, self.node_out_channels)
        return aggr_out.mean(dim=1) + self.nm_bias  # added: average heads and apply bias

    def edge_update(self, x_i,x_j, index, edge_attr, ptr, size_i):  
        x_j = x_j.view(-1,self.heads,self.node_out_channels)
        x_i = x_i.view(-1,self.heads,self.node_out_channels)
        edge_attr = edge_attr.view(-1,self.heads,self.edge_out_channels) 

        beta = torch.cat([x_i,x_j,edge_attr], dim=-1)  # added: concatenate source and target embeddings
        beta = (beta * self.em_edge_att).sum(dim=-1)   # added: compute attention scores
        beta = F.leaky_relu(beta, self.negative_slope)  # unchanged
        beta = softmax(beta, index, ptr, num_nodes=size_i)  # changed: apply softmax over neighbors
        beta = F.dropout(beta, p=self.dropout)#, training=self.training)  # unchanged
        beta = beta.view(-1, self.heads, 1)

        aggr_edge_feats = (beta * edge_attr).view(-1, self.heads * self.edge_out_channels)
        aggr_edge_feats = aggr_edge_feats.view(-1, self.heads, self.edge_out_channels)
        aggr_edge_feats = aggr_edge_feats.mean(dim=1) 

        edge_info_combined = torch.cat([x_i.mean(-1), 
                                        x_j.mean(-1),
                                        aggr_edge_feats,
                                        edge_attr.mean(dim=1)], dim=-1)
        edge_attr = self.edge_update_mlp(edge_info_combined)  # apply edge update MLP
        return edge_attr

import matplotlib.pyplot as plt

def visualize_graph_and_output(graph, output, title="Graph Visualization"):
    # Create a NetworkX graph from the edge index
    nx_graph = nx.Graph()
    edge_index = graph.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        print(f"Edge {i}: {output['edge_attr'][i].detach().numpy()}")
        nx_graph.add_edge(edge_index[0, i], edge_index[1, i], weight=output['edge_attr'][i][0].item())


    # Get node positions
    pos = nx.spring_layout(nx_graph, seed=42)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    # # Add edge weights as labels

    edge_labels = nx.get_edge_attributes(nx_graph, 'weight')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_color='black')


    # Add node embeddings as labels
    for i, (x, y) in pos.items():
        plt.text(x, y + 0.1, f"{output['x'][i].detach().numpy()}", fontsize=8, ha="center", color="red")

    plt.title(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create a simple graph
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 0]], dtype=torch.long)  # Edges in COO format
    x = torch.tensor([[0, 1], [1, 1], [2, 1]], dtype=torch.float)  # Node features
    # Add edge weights
    edge_weight = torch.tensor([[0.5, 0.2,0.6], [0.8, 0.1,0.5], [0.3, 0.7,0.6], [0.9, 0.4,0.3]], dtype=torch.float)  # Example edge weights
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    # Instantiate the EGAT model
    node_in_channels = x.size(1)
    edge_in_channels = edge_weight.size(1)

    node_out_channels = 4
    edge_out_channels = 4
    heads = 2
    gat = EGAT_tunedContribution(node_in_channels,
                                 node_out_channels,
                                 edge_in_channels,
                                 edge_out_channels,
                                 heads=heads)

    # Forward pass
    out = gat(data.x, data.edge_index,data.edge_attr)
    print("Output node embeddings:")
    print(out)

    visualize_graph_and_output(data, out, title="EGAT Output Visualization")
# %%
