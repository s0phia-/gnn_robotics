from src.utils.logger_config import get_logger
import numpy as np
import os
import itertools
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


class PPO:
    """
    Implementation of Proximal Policy Optimization.
    Schulman, John, et al. "Proximal policy optimization algorithms."
    """
    def __init__(self, actor, critic, device, **kwargs):
        # extract parameters
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # set seeds
        torch.manual_seed(self.seed)

        # set up environment
        self.device = device

        # initialise actor and critic networks
        self.actor = actor
        self.critic = critic

        # initialise optimiser for actor and critic
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor.parameters(), self.critic.parameters()), lr=self.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # create covariance matrix depending on action size
        self.cov_mat = lambda x: torch.eye(x, device=self.device) * 0.5
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

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
        t = 0
        rewards_history = []
        while t < int(self.total_timesteps):
            # perform a rollout
            b_obs, b_actions, b_log_probs, b_rtgs, b_lens, b_rewards, b_dones, b_last_obs = self.rollout()

            # Calculate the average reward per episode in this batch
            avg_ep_reward = sum([sum(ep_rewards).item() for ep_rewards in b_rewards]) / len(b_rewards)
            rewards_history.append([iters, avg_ep_reward])

            # keep track of time!
            t += self.batch_size
            iters += 1

            if self.advantage_method == "simple":
                b_advantages = b_rtgs - self.get_value(b_obs).detach()
            if self.advantage_method == "gae":
                values = self.get_value(b_obs).detach()
                b_advantages = self.compute_gae(b_rewards, b_dones, values, b_last_obs)

            for epoch in range(self.update_epochs):
                b_inds = torch.randperm(self.batch_size, device=self.device)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    vv = self.get_value(b_obs)
                    log_probs = self.get_action_log_probs(b_obs, b_actions)
                    act_prob_ratio = torch.exp(log_probs - b_log_probs)

                    mb_advantages = b_advantages[mb_inds]
                    if self.normalize_advantage:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # calculate losses
                    surr_loss_1 = act_prob_ratio * mb_advantages
                    surr_loss_2 = torch.clamp(act_prob_ratio, 1 - self.clip_value, 1 + self.clip_value) * mb_advantages
                    actor_loss = (-torch.min(surr_loss_1, surr_loss_2)).mean()
                    critic_loss = nn.MSELoss()(b_rtgs, vv)

                    # backprop actor network
                    self.optimizer.zero_grad()
                    self.scaler.scale(actor_loss + critic_loss).backward()

                    if self.grad_clip_value > 0:
                        self.scaler.unscale_(self.optimizer)
                        if self.actor is self.critic:
                            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(self.actor.parameters(), self.critic.parameters()),
                                self._grad_norm_clip
                            )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()


            self.logger.info("Iteration {} loss {}.".format(iters, critic_loss.item()))
            if iters % self.save_model_freq == 0:
                # track rewards
                np.savetxt(f"{self.results_dir}{self.run_id}.csv", rewards_history,
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
        b_observations = []
        b_actions = []
        b_log_probs = []
        b_rewards = []
        b_lens = []
        b_dones = []
        b_last_obs = []
        t = 0
        while t < self.batch_size:
            episode_rewards = []
            obs, info = self.env.reset()
            for ep_t in range(self.max_episodic_timesteps):
                t += 1
                graph = self.make_graph(obs, info)
                b_observations.append(graph)
                action, log_prob = self.get_action(graph, calculate_log_probs=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                b_actions.append(action)
                b_log_probs.append(log_prob.cpu().item())
                b_dones.append(terminated or truncated)
                episode_rewards.append(reward)
                if terminated or truncated:
                    last_obs = obs
                    break
            b_lens.append(len(episode_rewards))
            b_rewards.append(episode_rewards)
            b_last_obs.append(last_obs)
        b_observations = self.make_graph_batch(b_observations)
        b_actions = torch.tensor(np.array(b_actions), dtype=torch.float, device=self.device)
        b_log_probs = torch.tensor(b_log_probs, dtype=torch.float, device=self.device)
        b_dones = torch.tensor(np.array(b_dones), dtype=torch.bool, device=self.device)
        b_rtgs = self.get_reward_to_go(b_rewards)
        b_last_obs = torch.tensor(np.array(b_last_obs), dtype=torch.float, device=self.device)
        return b_observations, b_actions, b_log_probs, b_rtgs, b_lens, b_rewards, b_dones, b_last_obs

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

    def compute_gae(self,
                    rewards: torch.Tensor,
                    dones: torch.Tensor,
                    values: torch.Tensor,
                    last_value: torch.Tensor,
                    ) -> torch.Tensor:
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]
        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else last_value
            advantage = (
                    rewards[i]
                    - values[i]
                    + self.gamma * not_dones[i] * (next_values + self.gae_lambda * advantage)
            )
            advantages[i] = advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    @staticmethod
    def compute_kl_divergence(old_log_probs, new_log_probs):
        kl = torch.mean(torch.exp(old_log_probs) * (old_log_probs - new_log_probs))
        return kl

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
        num_nodes, morph_edges, mask = info['num_nodes'], info['edge_idx'], info['mask']
        node_dim = int(len(obs) / num_nodes)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        x = obs.view(num_nodes, -1)
        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        # get morph and fc edges
        fc_edges, _ = dense_to_sparse(torch.ones(len(x), len(x)))
        morph_edges = torch.tensor(morph_edges, device=self.device)
        # combine edges and create labels
        edges, inverse_idx = torch.unique(torch.cat([fc_edges, morph_edges], dim=1), dim=1, return_inverse=True)
        morph_labels = torch.zeros(fc_edges.shape[1]).scatter_(0, inverse_idx[fc_edges.shape[1]:], 1)
        edge_labels = torch.stack([morph_labels, torch.ones_like(morph_labels)], dim=1)
        return Data(x=x, edge_index=edges, edge_attr=edge_labels, mask=mask, num_nodes=num_nodes, node_dim=node_dim)

    @staticmethod
    def make_graph_batch(obs_batch):
        return Batch.from_data_list(obs_batch)

    def load_actor(self, actor_path, device):
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

    def load_critic(self, critic_path, device):
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
