import torch
import yaml
import torch
from argparse import ArgumentParser
from agents.ppo import PPO


def main(hyperparams):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YAML hyperparameters
    with open(f'utils/hyperparameters.yaml', 'r') as f:
        hparam = yaml.safe_load(f)

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam[k] = v

    # set seeds
    PPO(device, **hparam)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save the model.'
    )
    hparam = parser.parse_args()

    main(hparam)
