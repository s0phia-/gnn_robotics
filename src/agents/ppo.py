from src.utils.logger_config import get_logger
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from time import sleep
from src.agents.function_approximators import make_graph, make_graph_batch, FeedForward


class PPO:
    """
    Implementation of Proximal Policy Optimization.
    Schulman, John, et al. "Proximal policy optimization algorithms."
    """
    def __init__(self, actor, device, env, **kwargs):
        # extract parameters
        self.__dict__.update((k, v) for k, v in kwargs.items())

        self.graph_info = kwargs['graph_info']

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
        self.cov_mat = torch.eye(self.action_dim, device=self.device) * 0.5

        # set up file paths
        self.results_dir = f"{self.run_dir}/results/"
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_dir = f"{self.run_dir}/checkpoints/{self.run_id}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(run_id=self.run_id, run_dir=self.run_dir)

    def learn(self):
        """
        PPO learning step. Nice description and pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        iters = int(0)
        total_iters = int(self.total_timesteps / self.timesteps_per_batch)
        t = 0
        rewards_history = []
        while t < int(self.total_timesteps):

            # perform a rollout
            batch_obs, batch_actions, batch_log_probs, batch_reward_to_go, batch_lens, batch_rewards = self.rollout()

            # Calculate average reward per episode in this batch
            avg_ep_reward = sum([sum(ep_rewards) for ep_rewards in batch_rewards]) / len(batch_rewards)
            rewards_history.append([iters, avg_ep_reward])

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
                self.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optim.step()

                # backprop critic network
                self.critic_optim.zero_grad(set_to_none=True)
                critic_loss.backward()
                self.critic_optim.step()

            self.logger.info("Iteration {}/{} loss {}.".format(iters, total_iters, critic_loss.item()))
            if iters % self.save_model_freq == 0:
                # track rewards
                np.savetxt(f"{self.results_dir}/{self.run_id}.csv", rewards_history,
                           delimiter=',', header='iteration,reward', comments='')
                # save model
                torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/ppo_actor.pth")
                torch.save(self.critic.state_dict(), f"{self.checkpoint_dir}/ppo_critic.pth")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def rollout(self):
        """
        Collect batch of experiences.
        :return: batch_observations: observations experienced in batch
                 batch_actions: actions taken in batch
                 batch_log_probabilities: log-prob of each action in batch
                 batch_rewards-to-gos: reward-to-go at each timestep
                 batch_lengths: length of each episode in batch
        """
        batch_observations = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_lens = []
        t = 0
        while t < self.timesteps_per_batch:
            episode_rewards = []
            obs = self.env.reset()
            for ep_t in range(self.max_episodic_timesteps):
                t += 1
                batch_observations.append(obs)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, log_prob = self.get_action(obs_tensor, calculate_log_probs=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.cpu().item())  # If log_prob is a scalar tensor
                episode_rewards.append(reward)
                if terminated or truncated:
                    break
            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)
        batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float, device=self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rewards_to_gos = self.get_reward_to_go(batch_rewards)
        return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_gos, batch_lens, batch_rewards

    def get_action(self, obs, calculate_log_probs=False):
        """
        find optimal action for given observation.
        :param calculate_log_probs: whether to return the log probability of the action
        :param obs:observation to get action for
        :return: action, log probability of action (optional)
        """
        self.num_nodes = obs.shape[0]
        graph = make_graph(obs, self.graph_info['num_nodes'],edge_index=self.graph_info['edge_idx'])
        mean_action = self.actor(graph)
        dist = MultivariateNormal(mean_action, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_cpu = action.cpu()
        if calculate_log_probs:
            return action_cpu.detach().numpy(), log_prob
        return action_cpu.detach().numpy()

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
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float, device=self.device)
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
        graph_batch = make_graph_batch(obs, self.graph_info['num_nodes'],edge_index=self.graph_info['edge_idx'])
        batch_action = self.actor(graph_batch)
        dist = MultivariateNormal(batch_action, self.cov_mat)
        log_probs = dist.log_prob(actions)

        return log_probs

    def demo(self, actor_path, critic_path):
        self.load_actor(actor_path, self.device)
        self.load_critic(critic_path, self.device)
        self.actor.eval()
        self.critic.eval()
        env = self.env
        obs = env.reset()
        for _ in range(100):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action = self.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            sleep(.1)

    def load_actor(self, actor_path, device):
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

    def load_critic(self, critic_path, device):
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
