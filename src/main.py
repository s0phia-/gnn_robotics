import yaml
import torch
from argparse import ArgumentParser
import os
from src.environments.mujoco_parser import MujocoParser
import gymnasium as gym
from src.agents.function_approximators import MessagePassingGNN
from src.agents.ppo import PPO



def main(hyperparams):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YAML hyperparameters
    with open(f'utils/hyperparameters.yaml', 'r') as f:
        hparam = yaml.safe_load(f)

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam[k] = v

    envs = MujocoParser(**hparam).envs_train
    # num_actions = int(envs.action_space.shape)
    for env in envs:
        print(env)
        env.reset()
        env.render()

    ### testing PPO ###
    # actor = MessagePassingGNN(in_dim=9, out_dim=8, env=envs[0], device=device, **hparam)
    # model = PPO(device, actor, **hparam)
    # model.learn()




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=int(1e7),
        help='Total number of time-steps to train model for.'
    )
    hparam = parser.parse_args()

    main(hparam)
