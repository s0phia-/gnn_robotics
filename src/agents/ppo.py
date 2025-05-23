from src.utils.logger_config import get_logger
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from time import sleep
import gc
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

        # Memory management settings
        self.memory_log_freq = 5  # Log memory usage every N iterations
        self.memory_cleanup_freq = 2  # Clean memory every N iterations

    def learn(self):
        """
        PPO learning step. Nice description and pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        iters = int(0)
        t = 0
        rewards_history = []

        if torch.cuda.is_available():
            self.logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB allocated, "
                             f"{torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB reserved")

        while t < int(self.total_timesteps):
            if torch.cuda.is_available() and iters % self.memory_log_freq == 0:
                self.logger.info(
                    f"Iteration {iters} - Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB allocated")

            # perform a rollout
            batch_obs, batch_actions, batch_log_probs, batch_reward_to_go, batch_lens, batch_rewards = self.rollout()

            # Calculate average reward per episode in this batch
            avg_ep_reward = sum([sum(ep_rewards) for ep_rewards in batch_rewards]) / len(batch_rewards)
            rewards_history.append([iters, avg_ep_reward])

            # keep track of time!
            t += self.timesteps_per_batch
            iters += 1

            # find advantage, normalize
            with torch.no_grad():
                advantage_unnormalized = batch_reward_to_go - self.get_value(batch_obs).detach()
                advantage = (advantage_unnormalized - advantage_unnormalized.mean()) / (advantage_unnormalized.std() + 1e-8)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # loop to update network
            for update_iter in range(self.n_updates_per_iter):

                with torch.set_grad_enabled(True):
                    log_probs = self.get_action_log_probs(batch_obs, batch_actions)
                    action_prob_ratio = torch.exp(log_probs - batch_log_probs)

                    # Calculate surrogate losses
                    surr_loss_1 = action_prob_ratio * advantage
                    surr_loss_2 = torch.clamp(action_prob_ratio, 1 - self.clip_value, 1 + self.clip_value) * advantage
                    actor_loss = (-torch.min(surr_loss_1, surr_loss_2)).mean()

                    # Backprop actor network
                    self.actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    self.actor_optim.step()
                del log_probs, action_prob_ratio, surr_loss_1, surr_loss_2, actor_loss

                with torch.set_grad_enabled(True):
                    value_preds = self.get_value(batch_obs)
                    critic_loss = nn.MSELoss()(value_preds, batch_reward_to_go)

                    self.critic_optim.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    self.critic_optim.step()
                if update_iter % 3 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.logger.info(f"Iteration {iters} - Loss: {critic_loss.item():.4f}, Avg reward: {avg_ep_reward:.4f}")

            # Perform memory cleanup
            if iters % self.memory_cleanup_freq == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Save results and model periodically
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
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            for ep_t in range(self.max_episodic_timesteps):
                t += 1
                batch_observations.append(obs.clone())
                with torch.no_grad():
                    action, log_prob = self.get_action(obs, calculate_log_probs=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.cpu().item())
                episode_rewards.append(reward)
                if terminated or truncated:
                    break
                if ep_t > 0 and ep_t % 1000 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            batch_lens.append(len(episode_rewards))
            batch_rewards.append(episode_rewards)
            if t % 1000 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        observations_array = np.array([o.cpu().numpy() for o in batch_observations])
        batch_observations = torch.tensor(observations_array, dtype=torch.float, device=self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rewards_to_gos = self.get_reward_to_go(batch_rewards)
        del observations_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        log_prob = dist.log_prob(action) if calculate_log_probs else None
        action_np = mean_action.cpu().detach().numpy()
        if calculate_log_probs:
            return action_np, log_prob.detach()
        return action_np

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
        with torch.set_grad_enabled(self.critic.training):
            return self.critic(obs).squeeze()

    def get_action_log_probs(self, obs, actions):
        """
        calculate value function for given observation.
        :param obs: observation actions are chosen in
        :param actions: actions to calculate log probability for
        :return: log probabilities of actions
        """
        with torch.set_grad_enabled(True):
            graph_batch = make_graph_batch(obs, self.graph_info['num_nodes'], edge_index=self.graph_info['edge_idx'])
            batch_action = self.actor(graph_batch)
            dist = MultivariateNormal(batch_action, self.cov_mat)
            log_probs = dist.log_prob(actions)
        return log_probs

    def demo(self, actor_path, critic_path):
        """
        Run a demo of the trained model.
        """
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.eval()
        self.critic.eval()
        env = self.env
        obs = env.reset()
        for _ in range(100):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action = self.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            sleep(.1)
            if terminated or truncated:
                obs = env.reset()
