import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin


class SKRLMixin(Model, GaussianMixin):
    def __init__(self, observation_space, action_space, device, **kwargs):
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True)

    def compute(self, inputs, role=""):
        states = inputs["states"]
        mean = self.forward(states)
        log_std = self.log_std_parameter.expand_as(mean)
        return mean, log_std, {}

    def act(self, inputs, role=""):
        mean, log_std, outputs = self.compute(inputs, role)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        self._last_distribution = distribution
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        return actions, log_prob, outputs

    def distribution(self, role=""):
        if hasattr(self, '_last_distribution'):
            return self._last_distribution
        else:
            dummy_mean = torch.zeros(self.action_space.shape[0], device=self.device)
            dummy_std = torch.ones(self.action_space.shape[0], device=self.device)
            return torch.distributions.Normal(dummy_mean, dummy_std)
