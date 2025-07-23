from src.agents.method2 import Method2Gnn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from src.agents.nerve_net import Gnnlayer, add_self_loops
from torch_geometric.utils import scatter
import torch
import torch.nn as nn


class Method1Gnn(Method2Gnn):
    def __init__(self,
                 in_dim: int,
                 num_nodes: int,
                 action_dim: int,
                 device: torch.device,
                 **kwargs
                 ):
        super().__init__(in_dim, num_nodes, action_dim, device, **kwargs)
        self.middle = nn.ModuleList()
        for _ in range(self.propagation_steps):
            self.middle.append(GnnLayerDoubleMessage(in_dim=self.node_representation_dim,
                                                     out_dim=self.node_representation_dim,
                                                     hidden_dim=self.hidden_dim,
                                                     hidden_layers=self.hidden_layers,
                                                     device=device,
                                                     morph_weight=self.morphology_fc_ratio))


class GnnLayerDoubleMessage(Gnnlayer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 device: torch.device,
                 aggregator_type: str = 'mean',
                 morph_weight: float = .5, ):
        """
        Message passing GNN layer with two edge types, each aggregated separately and then combined in an update
        function which now takes the form h_{t+1} = U(h_t, agg1, agg2) where agg1 and agg2 are the separately aggregated
        messages.
        :param message_hidden_layers:
        :param message_hidden_dim:
        :param update_hidden_layers:
        :param update_hidden_dim:
        :param device:
        :param morph_weight: morphology weighting. Fully connected weighting will be 1-morph_weight
        """
        super().__init__(in_dim, out_dim, hidden_dim, hidden_layers, device, aggregator_type)
        self.morph_weight = morph_weight

        # construct message functions
        self.message_function_type1 = self._build_mlp(in_dim * 2, hidden_dim, out_dim, hidden_layers, device)
        self.message_function_type2 = self._build_mlp(in_dim * 2, hidden_dim, out_dim, hidden_layers, device)

    def forward(self, x: torch.Tensor, edge_morph: torch.Tensor, edge_fc: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of the message passing GNN layer.
        Adds self loops, and collects messages along the FC edges and morphology respecting edges.
        The messages are weighted according to self.morph_weight, with FC messages being weighted by
        (1-self.morph_weight).
        Finally, messages are propagated along edges and input to the update function.
        """
        edge_morph, _ = add_self_loops(edge_morph, num_nodes=x.size(0))
        msg_morph = self._get_messages(x, edge_morph, 1)
        msg_fc = self._get_messages(x, edge_fc, 2)

        combined_edge_index = torch.cat([edge_morph, edge_fc], dim=1)
        combined_messages = torch.cat([msg_morph, msg_fc], dim=0)

        aggr_out = scatter(combined_messages, combined_edge_index[1], dim=0, reduce=self.aggr, dim_size=x.size(0))

        return self.update_function(aggr_out, x)

    def _get_messages(self, x, edge_index, edge_type):
        """
        Get raw messages for given edge index without aggregation
        """
        row, col = edge_index
        return self.message(x[row], x[col], edge_type)

    def message(self, x_i, x_j, edge_type):
        """
        Calculate messages. There are different message functions for different edge types.
        """
        msg = torch.cat([x_i, x_j], dim=-1)
        if edge_type == 1:  # morph
            return self.message_function_type1(msg) * self.morph_weight
        if edge_type == 2:  # FC
            return self.message_function_type2(msg) * (1 - self.morph_weight)


class SkrlMethod1(GaussianMixin, DeterministicMixin, Model):
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

        self.policy_network = Method1Gnn(in_dim=per_node_dim,
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
