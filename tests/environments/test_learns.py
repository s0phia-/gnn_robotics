"""
Test to make sure that a basic version is actually learning
"""


from src.main import run
import os
from pathlib import Path
from src.environments.mujoco_parser import MujocoParser
from src.utils import load_hparams, load_agent_and_env, set_run_id, get_logger, plot_rewards_with_seeds
import src 


hparams = {
    "env_name": "swimmer",
    "max_episodic_timesteps": 1000,
    "population_size": 4,
    "enable_features": True,
    "lr": "3e-4",
    "gamma": 0.95,
    "clip_value": 0.1,
    "n_updates_per_iter": 5,
    "timesteps_per_batch": 2048,
    "total_timesteps": 100_000,
    "propagation_steps": 3,
    "hidden_node_dim": 64,
    "decoder_and_message_layers": 2,
    "decoder_and_message_hidden_dim": 64,
    "morphology_fc_ratio": .5,
    "method": "NerveNet",
    "seed": 6,
    "save_model_freq": 10,
    # "xml_path": "environments/assets",
    "env_dir": "./environments",
    "base_modular_env_path": "./environments/ModularEnv.py",
    "data_dir": "./results",
    "run_id": 6,
    "run_dir": "../runs/test_run",
}

def test_swimmer_env() -> None:
    # if the cwd is not the src directory, change it to the src directory
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    # run the main function with swimmer env
    hparams["env_name"] = "swimmer"
    # MujocoParser(**hparams)
    run(hparams)
    plot_rewards_with_seeds(f'{hparams["run_dir"]}/results')

if __name__ == "__main__":
    test_swimmer_env()
