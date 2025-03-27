import yaml
from argparse import ArgumentParser
from agents.ppo import PPO


def main(hyperparams):
    # Load YAML hyperparameters
    with open(f'./src/hyperparam/{hyperparams.config_file}', 'r') as f:
        hparam = yaml.safe_load(f)

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam[k] = v

    PPO(**hparam)


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
