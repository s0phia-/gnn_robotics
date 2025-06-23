from src.main import run
import os
from pathlib import Path
from src.environments.mujoco_parser import MujocoParser
import src 


hparams = {
    "env_name": "hopper",
    "max_episodic_timesteps": 100,
    "population_size": 1,
    "enable_features": True,
    "lr": "3e-4",
    "gamma": 0.95,
    "clip_value": 0.1,
    "n_updates_per_iter": 10,
    "timesteps_per_batch": 2048,
    "total_timesteps": 1000,
    "propagation_steps": 3,
    "hidden_node_dim": 64,
    "decoder_and_message_layers": 2,
    "decoder_and_message_hidden_dim": 64,
    "morphology_fc_ratio": .5,
    "method": "method2",
    "seed": 6,
    "save_model_freq": 10000000000,
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

def test_hopper_env() -> None:
    # if the cwd is not the src directory, change it to the src directory
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    # run the main function with hopper env
    hparams["env_name"] = "hopper"
    # MujocoParser(**hparams)
    run(hparams)

def test_ant_env() -> None:
    # if the cwd is not the src directory, change it to the src directory
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    # run the main function with swimmer env
    hparams["env_name"] = "ant"
    # MujocoParser(**hparams)
    run(hparams)

def test_method1() -> None:
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    hparams["method"] = "method1"
    run(hparams)

def test_method2() -> None:
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    hparams["method"] = "method2"
    run(hparams)

def test_nervenet() -> None:
    if os.getcwd() != str(Path(src.__path__[0]).resolve()):
        os.chdir(str(Path(src.__path__[0]).resolve()))
    hparams["method"] = "NerveNet"
    run(hparams)
