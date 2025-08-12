import shutil
import os
import datetime
import yaml
import itertools
from copy import deepcopy
from src.utils.logger_config import get_logger
from src.agents import PPO, Method1Gnn, Method2Gnn, NerveNet


def load_hparams(yaml_hparam_path, num_seeds=5):
    """
    :param yaml_hparam_path: path to YAML hyperparameters
    :param num_seeds: number of different seeds to use
    """
    with open(yaml_hparam_path, 'r') as f:
        hparam = yaml.safe_load(f)
    run_dir = f"../runs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    yaml_filename = os.path.basename(yaml_hparam_path)
    shutil.copy2(yaml_hparam_path, os.path.join(run_dir, yaml_filename))
    if hparam['load_run_path'] is None:
        os.makedirs(f"{run_dir}/logs", exist_ok=True)
        os.makedirs(f"{run_dir}/results", exist_ok=True)
    else:
        load_dir = f'../runs/{hparam["load_run_path"]}'
        shutil.copytree(f'{load_dir}/logs', f"{run_dir}/logs")
        shutil.copytree(f'{load_dir}/results', f"{run_dir}/results")

    base_seed = hparam.get('seed', 0)
    seeds = [base_seed + i * 100 for i in range(num_seeds)]

    test_params = {k: v for k, v in hparam.items() if isinstance(v, list) and len(v) > 1}
    base_params = {k: v[0] if isinstance(v, list) and len(v) == 1 else v
                   for k, v in hparam.items() if k not in test_params}
    param_names = list(test_params.keys())
    param_values = list(test_params.values())
    all_combinations = []

    for combination in itertools.product(*param_values):
        for seed in seeds:
            hparams = deepcopy(base_params)
            for i, param_name in enumerate(param_names):
                hparams[param_name] = combination[i]
            hparams['seed'] = seed
            run_id = ",".join([f"{param_name}-{combination[i]}" for i, param_name in enumerate(param_names)])
            run_id += f",seed-{seed}"
            hparams['run_id'] = run_id
            hparams['run_dir'] = run_dir
            all_combinations.append(hparams)
    return all_combinations


def load_env(hparam, device):
    from src.environments.mujoco_parser import MujocoParser
    env_setup = MujocoParser(**hparam)
    env, node_dim, num_nodes = env_setup.envs_train[0], env_setup.limb_obs_size, env_setup.num_nodes
    print(f"{env=}, {node_dim=}, {num_nodes=}")
    env.reset()
    return env


def load_agent_and_env(hparam, device):
    env = load_env(hparam, device)
    method = hparam['method']
    if method == "method1":
        agent = Method1Gnn
    elif method == "method2":
        agent = Method2Gnn
    elif method == "NerveNet":
        agent = NerveNet
    else:
        raise ValueError(f"Method {method} not implemented")
    actor = agent(device=device,
                  network_type='actor',
                  **hparam)
    critic = agent(device=device,
                   network_type='critic',
                   **hparam)
    return actor, critic, env


def run_worker(args):
    """Worker function that handles both GPU and CPU cases"""
    import torch
    if len(args) == 2:  # GPU
        hparam, gpu_id = args
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
    else:  # CPU
        hparam = args
        device = torch.device('cpu')

    def run(hparam, device):
        logger = get_logger()
        logger.info(f"Starting run with parameters: {hparam['run_id']} on {device}")
        if device.type == 'cuda':
            logger.info(f"Current GPU in run(): {torch.cuda.current_device()}")
        actor, critic, env = load_agent_and_env(hparam, device)
        agent = PPO(actor=actor,
                    critic=critic,
                    device=device,
                    env=env, **hparam)
        agent.learn()

    return run(hparam, device)
