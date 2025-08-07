from src.agents.nerve_net import *
from torch_geometric.utils import dense_to_sparse

from GNN_Layers.GAT import GAT

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


class SkrlGat(GaussianMixin, DeterministicMixin, Model):
    def __init__(self,
                 observation_space,
                 action_space,
                 num_nodes,
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

        self.policy_network = GAT_Method(in_dim=per_node_dim,
                                         action_dim=1,
                                         num_nodes=num_nodes,
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
