import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 nn_layers = 2,
                 hidden_nodes = 64,
                 dropout = 0.2,
                 activation = 'prelu',
                 final_layer_activation = True):
        super(MLP, self).__init__()
        activation_fn_dict = {'relu':nn.ReLU(),
                              'elu':nn.ELU(),
                              'prelu':nn.PReLU(),
                              'leakeyrelu':nn.LeakyReLU()
                              }
        activation_fn = activation_fn_dict.get(activation.lower(),None)

        if activation_fn == None:
            raise ValueError(f'actviattion function used for message passing dose not exist {activation}')
        
        if nn_layers < 1:
            raise ValueError("nn_layer must be at least 1")
        if nn_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels,out_channels),
                activation_fn,
                nn.Dropout(dropout)
            )
        else:
            layers = []
            layers.append(nn.Linear(in_channels, hidden_nodes))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            for _ in range(nn_layers - 2):
                layers.append(nn.Linear(hidden_nodes, hidden_nodes))
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_nodes, out_channels))
            if final_layer_activation:
                layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*layers)

    def forward(self,input):
        return self.mlp(input)

class x_j(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super(x_j, self).__init__()

    def forward(self, x_i, x_j,edge_attr=None):
        return x_j
    
class NN_concat_xj_xi(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super().__init__()     
        nn_layers = kwargs.get('nn_layers', 2)
        hidden_nodes = kwargs.get('hidden_nodes', None)
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'prelu')

        if nn_layers < 1:
            raise ValueError("nn_layer must be at least 1")
        if hidden_nodes is None:
            hidden_nodes = node_out_channels * node_out_channels

        self.mlp = MLP(in_channels=node_in_channels * 2,
                      out_channels=node_out_channels,
                      nn_layers=nn_layers,
                      hidden_nodes=hidden_nodes,
                      dropout=dropout,
                      activation=activation)

    def forward(self, x_i, x_j,edge_attr=None):
        batch_size, heads, node_channels = x_i.size(0), x_i.size(1), x_i.size(2)

        concat = torch.cat([x_i, x_j], dim=-1)
        concat = concat.view(batch_size * heads, node_channels * 2)
        output = self.mlp(concat)
        output = output.view(batch_size, heads, node_channels)
        return output

class NN_concat_xj_xi_eij(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super().__init__()
        nn_layers = kwargs.get('nn_layers', 2)
        hidden_nodes = kwargs.get('hidden_nodes', None)
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'prelu')

        if hidden_nodes is None:
            hidden_nodes = node_out_channels * node_out_channels
        
        self.mlp = MLP(in_channels=node_in_channels * 2 + edge_in_channels,
                      out_channels=node_out_channels,
                      nn_layers=nn_layers,
                      hidden_nodes=hidden_nodes,
                      dropout=dropout,
                      activation=activation)
            
    def forward(self, x_i, x_j,edge_attr=None):
        if edge_attr is None:
            raise ValueError("Edge attributes must be provided for NN_concat_xj_xi_eij")
        batch_size, heads, node_channels = x_i.size(0), x_i.size(1), x_i.size(2)
        if edge_attr is None:
            edge_attr = torch.zeros((batch_size, heads, 0), device=x_i.device)

        edge_channels = edge_attr.size(-1)

        # x_i = x_i.view(batch_size, heads * node_channels)
        # x_j = x_j.view(batch_size, heads * node_channels)
        # edge_attr = edge_attr.view(batch_size, heads, edge_channels)

        concat = torch.cat([x_i, x_j,edge_attr], dim=-1)

        concat = concat.view(batch_size * heads, node_channels * 2 + edge_channels)

        output = self.mlp(concat)

        output = output.view(batch_size, heads, node_channels)
        return output
    
class node_diff(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super(node_diff,self).__init__()

    def forward(self, x_i, x_j,edge_attr=None):
        diff = x_i - x_j
        return diff
    
class E_NN_Conv(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super(E_NN_Conv,self).__init__()
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        nn_layer = kwargs.get('nn_layer', 2)
        hidden_nodes = kwargs.get('hidden_nodes', None)
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'prelu')
        final_layer_activation = kwargs.get('final_layer_activation', False)


        if hidden_nodes is None:
            hidden_nodes = node_out_channels * node_out_channels
        
        self.mlp = MLP(in_channels=edge_in_channels,
                      out_channels=node_out_channels * node_out_channels,
                      nn_layers=nn_layer,
                      hidden_nodes=hidden_nodes,
                      dropout=dropout,
                      activation=activation,
                      final_layer_activation=final_layer_activation)

    def forward(self, x_i, x_j, edge_attr):
        if edge_attr is None:
            raise ValueError("Edge attributes must be provided for E_NN_Conv")
        edge_weights = self.mlp(edge_attr)
        # print(edge_weights.shape, x_j.shape)

        edge_weights = edge_weights.view(-1, self.node_out_channels, self.node_out_channels)
        x_j = torch.matmul(x_j.unsqueeze(1), edge_weights).squeeze(1)
        # print(x_j.shape)

        return x_j
    

        # return torch.matmul(edge_weights, x_j)
    
class E_NN_Conv_diff(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super(E_NN_Conv_diff,self).__init__()
        nn_layer = kwargs.get('nn_layer', 2)
        hidden_nodes = kwargs.get('hidden_nodes', None)
        dropout = kwargs.get('dropout', 0.1)
        activation=kwargs.get('activation', 'prelu')
        final_layer_activation = kwargs.get('final_layer_activation', False)


        if hidden_nodes is None:
            hidden_nodes = node_out_channels * node_out_channels
        
        self.mlp = MLP(in_channels=edge_in_channels,
                      out_channels=node_out_channels * node_out_channels,
                      nn_layers=nn_layer,
                      hidden_nodes=hidden_nodes,
                      dropout=dropout,
                      activation=activation,
                      final_layer_activation=final_layer_activation)

    def forward(self, x_i, x_j, edge_attr= None):
        if edge_attr is None:
            raise ValueError("Edge attributes must be provided for E_NN_Conv_diff")
        edge_weights = self.mlp(edge_attr)
        diff = x_i - x_j
        return torch.matmul(edge_weights, diff)
    
class NN_Conv(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = None,
                 edge_out_channels = None,
                 **kwargs):
        super(NN_Conv,self).__init__()
        nn_layer = kwargs.get('nn_layer', 2)
        hidden_nodes = kwargs.get('hidden_nodes', None)
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'prelu')

        if hidden_nodes is None:
            hidden_nodes = node_out_channels * node_out_channels
        
        self.mlp = MLP(in_channels=node_in_channels,
                      out_channels=node_out_channels,
                      nn_layers=nn_layer,
                      hidden_nodes=hidden_nodes,
                      dropout=dropout,
                      activation=activation)

    def forward(self, x_i, x_j, edge_attr=None):
        return self.mlp(x_j)

class NN_Conv_diff(nn.Module):
    def __init__(self, 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = None,
                 edge_out_channels = None,
                 **kwargs):
        super(NN_Conv_diff, self).__init__()
        nn_layer = kwargs.get('nn_layer', 2)
        hidden_nodes = kwargs.get('hidden_nodes', None)
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'prelu')

        if hidden_nodes is None:
            hidden_nodes = node_out_channels * node_out_channels
        
        self.mlp = MLP(in_channels=node_in_channels,
                      out_channels=node_out_channels,
                      nn_layers=nn_layer,
                      hidden_nodes=hidden_nodes,
                      dropout=dropout,
                      activation=activation)

    def forward(self, x_i, x_j, edge_attr=None):
        diff = x_i - x_j
        return self.mlp(diff)
    
class message_pass(nn.Module):
    def __init__(self,message,                 
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels = 0,
                 edge_out_channels = 0,
                 **kwargs):
        super(message_pass, self).__init__()

        if message == 'x_j':
            self.message_fn = x_j(node_in_channels,
                             node_out_channels,
                             edge_in_channels,
                             edge_out_channels)
            
        elif message == 'NN(x_j||x_i)':
            self.message_fn = NN_concat_xj_xi(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)
            
        elif message == 'NN(x_j||x_i||e_ij)':
            self.message_fn = NN_concat_xj_xi_eij(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)
            
        elif message == 'x_i - x_j':
            self.message_fn = node_diff(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)
            
        elif message == 'W_eij(x_j)':
            raise TypeError('W_eij(x_j) is not supported in message passing, needs the W_eij to be implemented')
            self.message_fn = E_NN_Conv(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)
            
        elif message == 'W_eij(x_i - x_j)':
            raise TypeError('W_eij(x_j) is not supported in message passing, needs the W_eij to be implemented')
            self.message_fn = E_NN_Conv_diff(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)  
        elif message == 'NN(x_j)':
            self.message_fn = NN_Conv(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)
        elif message == 'NN(x_i - x_j)':
            self.message_fn = NN_Conv_diff(node_in_channels,
                                  node_out_channels,
                                  edge_in_channels,
                                  edge_out_channels,
                                  **kwargs)
        
    def forward(self,x_i, x_j,edge_attr = None):
        return self.message_fn(x_i, x_j,edge_attr)
        