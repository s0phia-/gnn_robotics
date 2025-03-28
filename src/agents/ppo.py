from src.utils.logger_config import logger
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import gymnasium as gym
from src.agents.function_approximators import FeedForward


class PPO:
    def __init__(self, device, **kwargs):
        # extract parameters
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # set seeds
        torch.manual_seed(self.seed)

        # set up environment
        self.env = gym.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # initialise actor and critic networks
        self.actor = FeedForward(self.obs_dim, self.action_dim, device)
        self.critic = FeedForward(self.obs_dim, self.action_dim, device)

        # initialise optimiser for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def learn(self):
        iters = 0
        t = 0
        while t < self.total_timesteps:

            # perform a rollout
            batch_obvs, batch_actions, batch_log_probs, batch_reward_to_go, batch_lens = self.rollout()

            # keep track of time!
            t += self.timesteps_per_batch
            iters += 1
            logger("Iteration {}.".format(iters))

            # find advantage, normalize
            advantage_unnormalized = batch_reward_to_go - self.calculate_value(batch_obvs, batch_actions)
            advantage = (advantage_unnormalized-np.mean(advantage_unnormalized))/(np.std(advantage_unnormalized)+1e-8)



    def rollout(self):
        pass