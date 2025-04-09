import yaml
import torch
from argparse import ArgumentParser
from time import sleep
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

    def get_action_joint_mapping(model):
        """Return a mapping from action indices to joint names"""
        mapping = {}

        # Get names using the generic id2name function instead
        for i in range(model.nu):  # nu is the number of actuators
            actuator_id = i
            # Get the joint ID that this actuator controls
            joint_id = model.actuator_trnid[actuator_id, 0]

            # Use the generic id2name function with 'joint' type
            try:
                joint_name = model.id2name(joint_id, 'joint')
                mapping[i] = joint_name
            except Exception as e:
                # Fallback if id2name doesn't work
                mapping[i] = f"joint_{joint_id}"
        print(mapping)
        return mapping

    # num_actions = int(envs.action_space.shape)
    for env in envs:
        env.reset()
        get_action_joint_mapping(env.unwrapped.model)

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
