from src.main import run
import src
import os
from pathlib import Path

hparams = {
    "env_name": "ant",
    "max_episodic_timesteps": 1000,
    "population_size": 1,
    "enable_features": True,
    "lr": "3e-4",
    "gamma": 0.95,
    "clip_value": 0.1,
    "n_updates_per_iter": 10,
    "timesteps_per_batch": 2048,
    "total_timesteps": 10000,
    "propagation_steps": 3,
    "hidden_node_dim": 64,
    "decoder_and_message_layers": 2,
    "decoder_and_message_hidden_dim": 64,
    "seed": 6,
    "save_model_freq": 10,
    "xml_path": "environments/assets",
    "env_dir": "./environments",
    "base_modular_env_path": "./environments/ModularEnv.py",
    "data_dir": "./results",
    "run_id": 6,
    "run_dir": "../runs/run_20250515_114811",
}


def test_main() -> None:
    # if the cwd is not the src directory, change it to the src directory
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    run(hparams)
    # DELETE: temp assertion
    assert True
