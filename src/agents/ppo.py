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

            # perform a rollout
            b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_avg_reward = self.rollout()

            # Calculate the average reward per episode in this batch
            rewards_history.append([int(iters), float(b_avg_reward)])

            # keep track of time!
            t += len(b_obs)
            iters += 1

            if self.normalize_advantage:  # need to decide where to normalize
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            for _ in range(self.update_epochs):

                b_values = self.get_value(b_obs)
                new_log_probs = self.get_action_log_probs(b_obs, b_actions)
                action_prob_ratio = torch.exp(new_log_probs - b_log_probs.detach())

                # calculate losses
                surr_loss_1 = action_prob_ratio * b_advantages
                surr_loss_2 = torch.clamp(action_prob_ratio, 1-self.clip_value, 1+self.clip_value) * b_advantages
                actor_loss = (-torch.min(surr_loss_1, surr_loss_2)).mean()
                critic_loss = nn.MSELoss()(b_returns.detach(), b_values)

                print(f"b_values grad_fn: {b_values.grad_fn}")
                print(f"actor_loss grad_fn: {actor_loss.grad_fn}")
                print(f"b_values requires_grad: {b_values.requires_grad}")
                print(f"new_log_probs requires_grad: {new_log_probs.requires_grad}")
                print(f"b_log_probs requires_grad: {b_log_probs.requires_grad}")

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
        batch_obs = np.empty([self.parallel_envs, self.batch_size], dtype=object)
        batch_actions = np.empty([self.parallel_envs, self.batch_size], dtype=object)
        batch_log_probs = np.empty([self.parallel_envs, self.batch_size], dtype=float)
        batch_values = np.empty([self.parallel_envs, self.batch_size], dtype=float)
        batch_ep_returns = np.empty([self.parallel_envs, self.batch_size], dtype=object)
        timestep = 0

        while True:
            ep_timestep = 0
            obs, info = self.env.reset()
            ep_obs = np.empty([self.parallel_envs, self.max_episodic_timesteps], dtype=object)
            ep_dones = np.empty([self.parallel_envs, self.max_episodic_timesteps], dtype=object)
            ep_rewards = np.empty([self.parallel_envs, self.max_episodic_timesteps], dtype=object)
            ep_cum_rewards = np.empty([self.parallel_envs])
            while timestep < self.batch_size:
                graph = self.make_graph(obs, info)
                ep_obs[np.arange(self.parallel_envs), ep_timestep] = graph
                action, log_prob = self.get_action(graph, calculate_log_probs=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_rewards[np.arange(self.parallel_envs), ep_timestep] = reward
                ep_cum_rewards[np.arange(self.parallel_envs)] += reward
                ep_dones[np.arange(self.parallel_envs), ep_timestep] = done
                batch_actions[np.arange(self.parallel_envs), timestep] = action
                batch_log_probs[np.arange(self.parallel_envs), timestep] = log_prob
                timestep += 1
                ep_timestep += 1
                if done:
                    last_value = self.get_value(self.make_graph_batch(self.make_graph(obs, info))).detach()
                    batch_ep_returns[np.arange(self.parallel_envs), timestep] = ep_cum_rewards
                    print(batch_ep_returns)
                    break
            batch_obs.extend(ep_obs)
            ep_obs = self.make_graph_batch(ep_obs)
            ep_values = self.get_value(ep_obs).detach()
            batch_values.extend(ep_values)
            batch_gae, batch_returns = self.calculate_gae(ep_rewards, ep_values, ep_dones, last_value)
            batch_obs = self.make_graph_batch(batch_obs)
            batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float, device=self.device)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
            return batch_obs, batch_actions, batch_log_probs, batch_gae, batch_returns, np.mean(batch_ep_returns)

    def calculate_gae(self, rewards, values, dones, last_value):
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        advantages = torch.zeros_like(rewards).detach()
        gae = 0
        num_steps = rewards.shape[0]
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_nonterminal = 0
                next_value = last_value
            else:
                next_nonterminal = 1.0 - int(dones[t + 1])
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            advantages[t] = gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
        returns = advantages + values
        return advantages.flatten(), returns.flatten()

    def get_action(self, batch_obs, calculate_log_probs=True):
        """
        find optimal action for given a list of observations
        :param calculate_log_probs: whether to return the log probability of the action
        :param obs:observation to get action for
        :return: action, log probability of action (optional)
        """
        actions = []
        log_probs = [] if calculate_log_probs else None

        for obs in batch_obs:
            mean_action = self.actor(obs)

            cov_mat = torch.eye(len(mean_action), device=self.device) * 0.5
            dist = MultivariateNormal(mean_action, cov_mat)
            action = dist.sample()
            action_cpu = action.cpu().detach().numpy()
            actions.append(action_cpu)

            if calculate_log_probs:
                log_prob = dist.log_prob(action).cpu().detach().numpy()
                log_probs.append(log_prob)

        if calculate_log_probs:
            return actions, log_probs
        return actions

        # actions = []
        # log_probs = [] if calculate_log_probs else None
        # for mean_action in mean_actions:
        #     print(obs)
        #     cov_mat = torch.eye(len(mean_action), device=self.device) * 0.5
        #     dist = MultivariateNormal(mean_action, cov_mat)
        #     action = dist.sample()
        #     action_cpu = action.cpu().detach.numpy()
        #     actions.append(action_cpu)
        #     if calculate_log_probs:
        #         log_prob = dist.log_prob(action)
        #         log_probs.append(log_prob)
        # if calculate_log_probs:
        #     return actions, log_probs
        # return actions

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
        graph_list = []
        for _obs, _info in zip(obs, info):
            num_nodes, edge_idx, mask = _info['num_nodes'], _info['edge_idx'], _info['mask']
            node_dim = int(len(_obs) / num_nodes)
            _obs = torch.tensor(_obs, dtype=torch.float32, device=self.device)
            x = _obs.view(num_nodes, -1)
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
            x = Data(x=x, edge_index=edge_idx, mask=mask, num_nodes=num_nodes, node_dim=node_dim)
            graph_list.append(x)
        return graph_list

    @staticmethod
    def make_graph_batch(obs_batch):
        return Batch.from_data_list(obs_batch)

    def load_actor(self, actor_path, device):
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))

    def load_critic(self, critic_path, device):
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
