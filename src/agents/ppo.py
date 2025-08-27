from src.utils.logger_config import get_logger
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import itertools
from torch_geometric.data import Data, Batch


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
        self.device = device
        self.env = env

        # initialise actor and critic networks
        self.actor = actor
        self.critic = critic

        # initialise optimisers
        if self.opt_together:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.actor.parameters(), self.critic.parameters()), lr=self.learning_rate)
        else:
            self.actor_optim = Adam(self.actor.parameters(), lr=float(self.learning_rate))
            self.critic_optim = Adam(self.critic.parameters(), lr=float(self.learning_rate))
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.mixed_precision)

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
            b_obs, b_actions, b_log_probs, b_gae, b_returns, b_avg_reward, b_values = self.rollout()
            rewards_history.append([iters, b_avg_reward])
            batch_size = len(b_obs)
            t += batch_size
            iters += 1
            if self.normalize_advantage:  # need to decide where to normalize
                b_advantages = (b_gae - b_gae.mean()) / (b_gae.std() + 1e-8)
            for epoch in range(self.update_epochs):
                mb_indices_list = torch.chunk(torch.randperm(batch_size-1, device=self.device), self.num_minibatches)
                for mb_inds in mb_indices_list:
                    mb_obs = self.make_graph_batch([b_obs[i] for i in mb_inds.cpu().numpy()])
                    mb_actions = torch.index_select(b_actions, 0, mb_inds)
                    mb_old_log_probs = torch.index_select(b_log_probs, 0, mb_inds)
                    mb_advantages = torch.index_select(b_advantages, 0, mb_inds)
                    mb_returns = torch.index_select(b_returns, 0, mb_inds)

                    mb_values = self.critic(mb_obs)
                    mb_new_log_probs = self.get_action_log_probs(mb_obs, mb_actions)

                    act_prob_ratio = torch.exp(mb_new_log_probs - mb_old_log_probs.detach())
                    surr_loss_1 = act_prob_ratio * mb_advantages
                    surr_loss_2 = torch.clamp(act_prob_ratio, 1-self.clip_value, 1+self.clip_value) * mb_advantages
                    actor_loss = (-torch.min(surr_loss_1, surr_loss_2)).mean()
                    critic_loss = nn.MSELoss()(mb_returns.detach(), mb_values)

                    if self.opt_together:
                        self.optimizer.zero_grad(set_to_none=True)

                        self.scaler.scale(actor_loss + critic_loss).backward()
                        if self.grad_clip_value > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(itertools.chain(self.actor.parameters(),
                                                                     self.critic.parameters()), self.grad_clip_value)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        self.actor_optim.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        if self.grad_clip_value > 0:
                            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_value)
                        self.scaler.step(self.actor_optim)

                        if self.actor is not self.critic:
                            self.critic_optim.zero_grad(set_to_none=True)
                            critic_loss.backward()
                            if self.grad_clip_value > 0:
                                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_value)
                            self.scaler.step(self.critic_optim)

            self.logger.info("Iteration {} loss {}.".format(iters, critic_loss.item()))
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
        b_obs = []
        b_actions = []
        b_log_probs = []
        b_values = []
        b_gae = []
        b_total_reward = 0
        for _ in range(self.rollouts):
            obs, info = self.env.reset()
            ep_obs = []
            ep_dones = []
            ep_rewards = []
            while True:
                graph = self.make_graph(obs, info)
                ep_obs.append(graph)
                action, log_prob = self.get_action(graph, calculate_log_probs=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                b_total_reward += reward
                ep_dones.append(terminated or truncated)
                b_actions.append(action)
                b_log_probs.append(log_prob)
                ep_rewards.append(reward)
                if terminated or truncated:
                    last_value = self.get_value(self.make_graph(obs, info))
                    break
            b_obs.extend(ep_obs)
            ep_obs = (self.make_graph_batch(ep_obs))
            # calculate GAE
            ep_value = self.get_value(ep_obs)
            b_values.extend(ep_value)
            rr = torch.tensor(ep_rewards, dtype=torch.float, device=self.device)
            not_dones = torch.tensor(np.array(ep_dones), dtype=torch.bool, device=self.device).logical_not()
            curr_adv = 0
            for i in reversed(range(rr.shape[0])):
                next_values = ep_value[i + 1] if i < rr.shape[0] - 1 else last_value
                curr_adv = rr[i] - ep_value[i] + self.gamma * not_dones[i] * (next_values + self.gae_lambda * curr_adv)
                b_gae.append(curr_adv)
        b_obs = self.make_graph_batch(b_obs)
        b_actions = torch.tensor(b_actions, dtype=torch.float, device=self.device)
        b_log_probs = torch.tensor(b_log_probs, dtype=torch.float, device=self.device)
        b_gae = torch.tensor(b_gae, dtype=torch.float, device=self.device)
        b_returns = b_gae.flatten() + torch.tensor(b_values, device=self.device)
        b_returns = torch.tensor(b_returns, dtype=torch.float, device=self.device)
        b_avg_reward = b_total_reward / len(b_returns)
        b_gae = (b_gae - b_gae.mean()) / (b_gae.std() + 1e-8)
        return b_obs, b_actions, b_log_probs, b_gae, b_returns, b_avg_reward, b_values

    # def compute_gae(self,
    #                 b_rewards: torch.Tensor,
    #                 b_dones: torch.Tensor,
    #                 b_values: torch.Tensor,
    #                 b_last_obs: torch.Tensor,
    #                 ) -> torch.Tensor:
    #     """
    #     :param b_rewards: rewards
    #     :param b_dones: dones
    #     :param b_obs: observations
    #     :param b_last_obs: last observations
    #     :return: GAE advantage
    #     """
    #     vv = torch.tensor(b_values, dtype=torch.float, device=self.device)
    #     print(vv.shape)
    #     last_value = self.get_value(self.make_graph_batch(b_last_obs)).detach()
    #     rr = torch.tensor(b_rewards, dtype=torch.float, device=self.device)
    #     advantage = 0
    #     advantages = torch.zeros_like(rr)
    #     not_dones = torch.tensor(np.array(b_dones), dtype=torch.bool, device=self.device).logical_not()
    #     memory_size = rr.shape[0]
    #     print(memory_size)
    #     for i in reversed(range(memory_size)):
    #         next_values = vv[i + 1] if i < memory_size - 1 else last_value
    #         print(rr[i], vv[i], self.gamma, not_dones[i], next_values, self.gae_lambda, advantage)
    #         advantage = rr[i] - vv[i] + self.gamma * not_dones[i] * (next_values + self.gae_lambda * advantage)
    #         advantages[i] = advantage
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #     returns = advantages + vv
    #     return advantages, returns, vv

    def get_action(self, obs, calculate_log_probs=False):
        """
        find optimal action for given observation.
        :param calculate_log_probs: whether to return the log probability of the action
        :param obs:observation to get action for
        :return: action, log probability of action (optional)
        """
        mean_action = self.actor(obs)
        cov_mat = torch.eye(len(mean_action), device=self.device) * 0.5

        dist = MultivariateNormal(mean_action, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_cpu = action.cpu()
        if calculate_log_probs:
            return action_cpu.numpy(), log_prob
        return action_cpu.numpy()

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
        cov_mat = torch.eye(len(batch_action[0]), device=self.device) * 0.5  # todo
        dist = MultivariateNormal(batch_action, cov_mat)
        log_probs = dist.log_prob(actions)
        return log_probs

    def make_graph(self, obs, info):
        num_nodes, edge_idx, mask = info['num_nodes'], info['edge_idx'], info['mask']
        node_dim = int(len(obs) / num_nodes)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        x = obs.view(num_nodes, -1)
        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        return Data(x=x, edge_index=edge_idx, mask=mask, num_nodes=num_nodes, node_dim=node_dim)

    @staticmethod
    def make_graph_batch(obs_batch):
        return Batch.from_data_list(obs_batch)

    def load_actor(self, actor_path, device):
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

    def load_critic(self, critic_path, device):
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))