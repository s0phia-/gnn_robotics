import yaml
import torch
from argparse import ArgumentParser
from src.utils.misc_utils import create_edges, create_actuator_mapping
from src.environments.mujoco_parser import MujocoParser
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

    for env in envs:
        edges = create_edges(env)
        actuator_mapping = create_actuator_mapping(env)
        env.reset()
        actor = MessagePassingGNN(in_dim=15, num_nodes=9, edge_index=edges, actuator_mapping=actuator_mapping,
                                  device=device, **hparam)
        model = PPO(actor=actor, device=device, env=env, **hparam)
        model.learn()
        model.demo()


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
