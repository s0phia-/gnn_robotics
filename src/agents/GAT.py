from src.agents.nerve_net import *
from torch_geometric.nn import GATConv


class GATMethod(MessagePassingGNN):
    def __init__(self,
                 device: torch.device,
                 network_type: str,
                 node_message: str = 'x_j',
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
        x, edge_idx, mask, num_nodes, batch, node_dim, edge_attr = (data.x, data.edge_index, data.mask,
                                                                    data.num_nodes, data.batch, data.node_dim, data.edge_attr)
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
