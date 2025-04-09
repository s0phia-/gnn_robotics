import yaml
import torch
from argparse import ArgumentParser
import os
from src.environments.mujoco_utils import MujocoParser
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

    x = MujocoParser(**hparam)
    print(x.envs_train.action_space)

    # ### testing GNN ###
    # num_nodes = 10
    # num_features = 16
    #
    # # Random node features
    # x = torch.randn(num_nodes, num_features)
    #
    # # Random edges (just for demonstration)
    # edge_index = torch.tensor([
    #     [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0],  # Source nodes
    #     [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0, 9]  # Target nodes
    # ], dtype=torch.long)
    #
    # model = MessagePassingGNN(in_dim=16, out_dim=10, device=device, **hparam)
    # output = model(x, edge_index)
    # print(output)

    ### testing PPO ###
    # actor = MessagePassingGNN(in_dim, out_dim, device, **hparam)
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
