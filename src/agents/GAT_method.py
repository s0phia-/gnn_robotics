from src.agents.function_approximators import *
from torch_geometric.utils import dense_to_sparse

from GNN_Layers.GAT import GAT

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
            self.middle.append(GAT(in_channels=self.hidden_node_dim,
                                   out_channels=self.hidden_node_dim,
                                   heads=self.num_heads,
                                   concat=True,
                                   negative_slope=self.negative_slope,
                                   dropout=self.dropout,
                                   add_self_loops=False,
                                   contribution=0.5).to(device))
            

            # self.middle.append(GnnLayerDoubleAgg(in_dim=self.hidden_node_dim,
            #                                      out_dim=self.hidden_node_dim,
            #                                      hidden_dim=self.decoder_and_message_hidden_dim,
            #                                      hidden_layers=self.decoder_and_message_layers,
            #                                      device=device))

