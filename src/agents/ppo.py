from src.utils.logger_config import logger
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import gymnasium as gym
from src.agents.function_approximators import FeedForward, MessagePassingGNN


class PPO:
    """
    Implementation of Proximal Policy Optimization.
    Schulman, John, et al. "Proximal policy optimization algorithms."
    """
    def __init__(self, actor, device, env, **kwargs):
        # extract parameters
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # set seeds
        torch.manual_seed(self.seed)

        # set up environment
        self.env = env
        self.device = device
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # initialise actor and critic networks
        self.actor = actor
        self.critic = FeedForward(self.obs_dim, 1, device)

        # initialise optimiser for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=float(self.lr))
        self.critic_optim = Adam(self.critic.parameters(), lr=float(self.lr))

        # initialise covariance matrix
        self.cov_mat = torch.eye(self.action_dim) * 0.5

    def learn(self):
        """
        PPO learning step. Nice description and pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        iters = 0
        t = 0
        while t < int(self.total_timesteps):

            # perform a rollout
            batch_obs, batch_actions, batch_log_probs, batch_reward_to_go, batch_lens = self.rollout()

            # keep track of time!
            t += self.timesteps_per_batch
            iters += 1

            # find advantage, normalize
            advantage_unnormalized = batch_reward_to_go - self.get_value(batch_obs).detach()
            advantage = (advantage_unnormalized - advantage_unnormalized.mean()) / (advantage_unnormalized.std() + 1e-8)

            # loop to update network
            for _ in range(self.n_updates_per_iter):

                vv = self.get_value(batch_obs)
                log_probs = self.get_action_log_probs(batch_obs, batch_actions)
                action_prob_ratio = torch.exp(log_probs - batch_log_probs)

                # calculate losses
                surr_loss_1 = action_prob_ratio * advantage
                surr_loss_2 = torch.clamp(action_prob_ratio, 1-self.clip_value, 1+self.clip_value) * advantage
                actor_loss = (-torch.min(surr_loss_1, surr_loss_2)).mean()
                critic_loss = nn.MSELoss()(vv, batch_reward_to_go)

                # backprop actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # backprop critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            logger.info("Iteration {} loss {}.".format(iters, critic_loss.item()))
            if iters % self.save_model_freq == 0:
                torch.save(self.actor.state_dict(), './../checkpoints/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './../checkpoints/ppo_critic.pth')

    def rollout(self):
        """
        Collect batch of experiences.
        :return: batch_observations: observations experienced in batch
                 batch_actions: actions taken in batch
                 batch_log_probabilities: log-prob of each action in batch
                 batch_rewards-to-gos: reward-to-go at each timestep
                 batch_lengths: length of each episode in batch
        """

        # collect and save experience
        batch_observations = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_lens = []
        t = 0
        while t < self.timesteps_per_batch:
            episode_rewards = []
            obs = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            for ep_t in range(self.max_episodic_timesteps):
                t += 1
                batch_observations.append(obs)
                action, log_prob = self.get_action(obs, calculate_log_probs=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                episode_rewards.append(reward)
                if terminated or truncated:
                    break
            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)

        # get experiences in shape, calc rewards to go
        batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards_to_gos = self.get_reward_to_go(batch_rewards)

        return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_gos, batch_lens

    def get_action(self, obs, calculate_log_probs=False):
        """
        find optimal action for given observation.
        :param calculate_log_probs: whether to return the log probability of the action
        :param obs:observation to get action for
        :return: action, log probability of action (optional)
        """
        # create an action distribution
        mean_action = self.actor(obs)
        dist = MultivariateNormal(mean_action, self.cov_mat)

        # sample action, find log prob of action
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if calculate_log_probs:
            return action.detach().numpy(), log_prob
        return action.detach().numpy()

    def get_reward_to_go(self, rewards):
        """
        Compute reward to go based on rewards.
        :param rewards: rewards (in a batch)
        :return: reward-to-go per timestep
        """
        rewards_to_go = []
        for episode_rewards in reversed(rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = self.gamma * discounted_reward + reward
                rewards_to_go.insert(0, discounted_reward)
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float)
        return rewards_to_go

    def get_value(self, obs):
        """
        calculate value function for given observation.
        :param obs: observation to calculate value for
        :return: observation values
        """
        return self.critic(obs).squeeze()

    def get_action_log_probs(self, obs, actions):
        """
        calculate value function for given observation.
        :param obs: observation actions are chosen in
        :param actions: actions to calculate log probability for
        :return: log probabilities of actions
        """
        dist = MultivariateNormal(self.actor(obs), self.cov_mat)
        log_probs = dist.log_prob(actions)
        return log_probs

    def demo(self):
        env = self.env
        obs = env.reset()
        for _ in range(100):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action = self.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
