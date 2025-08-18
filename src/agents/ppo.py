from src.utils.logger_config import get_logger
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch_geometric.data import Data, Batch
from src.agents import FeedForward


class PPO:
    """
    Implementation of Proximal Policy Optimization.
    Schulman, John, et al. "Proximal policy optimization algorithms."
    """

    def __init__(self, actor, critic, device, env, **kwargs):
        # extract parameters
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # set seeds
        torch.manual_seed(self.seed)

        # set up environment
        self.env = env
        self.device = device

        # initialise actor and critic networks
        self.actor = actor
        self.critic = critic

        # initialise optimiser for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=float(self.lr))
        self.critic_optim = Adam(self.critic.parameters(), lr=float(self.lr))

        # create covariance matrix depending on action size
        self.cov_mat = lambda x: torch.eye(x, device=self.device) * 0.5

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

            # Calculate the average reward per episode in this batch
            avg_ep_reward = sum([sum(ep_rewards).item() for ep_rewards in batch_rewards]) / len(batch_rewards)
            rewards_history.append([iters, avg_ep_reward])

            # keep track of time!
            t += self.timesteps_per_batch
            iters += 1

            if self.advantage_method == "unnormalized":
                advantage = batch_reward_to_go - self.get_value(batch_obs).detach()
            if self.advantage_method == "normalized":
                advantage = batch_reward_to_go - self.get_value(batch_obs).detach()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            if self.advantage_method == "gae":
                next_value = self.get_value(batch_obs).detach()
                advantages = torch.zeros_like(batch_rewards).to(self.device)

            # loop to update network
            for _ in range(self.n_updates_per_iter):
                vv = self.get_value(batch_obs)
                log_probs = self.get_action_log_probs(batch_obs, batch_actions)
                action_prob_ratio = torch.exp(log_probs - batch_log_probs)

                # calculate losses
                surr_loss_1 = action_prob_ratio * advantage
                surr_loss_2 = torch.clamp(action_prob_ratio, 1 - self.clip_value, 1 + self.clip_value) * advantage
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
            obs, info = self.env.reset()
            for ep_t in range(self.max_episodic_timesteps):
                t += 1
                graph = self.make_graph(obs, info)
                batch_observations.append(graph)
                action, log_prob = self.get_action(graph, calculate_log_probs=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.cpu().item())
                episode_rewards.append(reward)
                if terminated or truncated:
                    break
            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)
        batch_observations = self.make_graph_batch(batch_observations)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        # if self.advantage_method == "gae":
        #     return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_gos, batch_lens, batch_reward
        batch_rewards_to_gos = self.get_reward_to_go(batch_rewards)
        return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_gos, batch_lens, batch_rewards

    def get_action(self, obs, calculate_log_probs=False):
        """
        find optimal action for given observation.
        :param calculate_log_probs: whether to return the log probability of the action
        :param obs:observation to get action for
        :return: action, log probability of action (optional)
        """
        mean_action = self.actor(obs)
        dist = MultivariateNormal(mean_action, self.cov_mat(len(mean_action)))
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

    # def get_gae_values(self, rewards, values, dones, lengths):
    #     for ep_rewards, ep_values, ep_dones, ep_length in reversed(zip(rewards, values, dones, lengths)):
    #         advantages = torch.zeros_like(rewards)
    #         last_gae_lam = 0
    #         for t in range(ep_length):
    #             if t == ep_length - 1:
    #                 nextnonterminal = 1.0 - ep_dones[t]
    #                 nextvalues = 0
    #             else:
    #                 nextnonterminal = 1.0 - ep_dones[t+1]
    #                 nextvalues = ep_values[t+1]
    #             delta = ep_rewards[t] + self.gamma * nextvalues * nextnonterminal - ep_values[t]
    #             advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lam
    #         ep_returns = advantages + ep_values
    #
    #
    #
    #             for t in reversed(range(num_steps)):
    #         if t == num_steps - 1:
    #             nextnonterminal = 1.0 - next_done
    #             nextvalues = next_value
    #         else:
    #             nextnonterminal = 1.0 - dones[t + 1]
    #             nextvalues = values[t + 1]
    #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

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
        batch_action = self.actor(obs)
        dist = MultivariateNormal(batch_action, self.cov_mat(len(batch_action[0])))
        log_probs = dist.log_prob(actions)
        return log_probs

    def make_graph(self, obs, info):
        num_nodes, edge_idx, mask = info['num_nodes'], info['edge_idx'], info['mask']
        node_dim = int(len(obs) / num_nodes)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        x = obs.view(num_nodes, -1)
        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        edge_idx = torch.tensor(edge_idx, device=self.device)
        return Data(x=x, edge_index=edge_idx, mask=mask, num_nodes=num_nodes, node_dim=node_dim)

    @staticmethod
    def make_graph_batch(obs_batch):
        return Batch.from_data_list(obs_batch)

    def load_actor(self, actor_path, device):
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

    def load_critic(self, critic_path, device):
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
