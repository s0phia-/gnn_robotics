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

    def __init__(self, actor, device, env, **kwargs):
        # extract parameters
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # set seeds
        torch.manual_seed(self.seed)

        # set up environment
        self.env = env
        self.device = device
        self.obs_dim = self.env.observation_space.shape[0]

        # initialise actor and critic networks
        self.actor = actor
        self.critic = FeedForward(self.obs_dim, 1, device)  # todo

        # initialise optimiser for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=float(self.lr))
        self.critic_optim = Adam(self.critic.parameters(), lr=float(self.lr))

        # create covariance matrix depending on action size
        self.cov_mat = lambda x: np.eye(x, device=self.device) * 0.5

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
                graph = create_graph(obs, info)
                batch_observations.append(graph)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, log_prob = self.get_action(obs_tensor, calculate_log_probs=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.cpu().item())  # If log_prob is a scalar tensor
                episode_rewards.append(reward)
                if terminated or truncated:
                    graph = create_graph(obs, info)
                    batch_observations.append(graph)
                    break
            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)
        # batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float, device=self.device)
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
        graph = make_graph(obs, self.graph_info['num_nodes'], edge_index=self.graph_info['edge_idx'])
        mean_action = self.actor(graph)
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
        graph_batch = self.make_graph_batch(obs,
                                            num_nodes=self.graph_info['num_nodes'],
                                            edge_index=self.graph_info['edge_idx'],
                                            mask=self.graph_info['mask'])
        batch_action = self.actor(graph_batch)
        dist = MultivariateNormal(batch_action, self.cov_mat(len(batch_action[0])))
        log_probs = dist.log_prob(actions)

        return log_probs

    def make_graph(self, obs, num_nodes, edge_idx, mask):
        node_dim = obs / num_nodes
        x = obs.view(num_nodes, -1)
        mask = torch.tensor(mask, dtype=torch.bool)
        return Data(x=x, edge_index=edge_idx, mask=mask, num_nodes=num_nodes, node_dim=node_dim)

    def make_graph_batch(self, obs_batch, num_nodes, edge_idx, mask):
        data_list = []
        for i, obs in enumerate(obs_batch):
            graph = self.make_graph(obs, num_nodes[i], edge_idx[i], mask[i])
            data_list.append(graph)
        return Batch.from_data_list(data_list)

    def load_actor(self, actor_path, device):
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

    def load_critic(self, critic_path, device):
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
