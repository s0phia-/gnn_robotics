import shutil
import os
import datetime
import yaml
import itertools
import torch
from copy import deepcopy
from src.environments.mujoco_parser import MujocoParser, create_edges, check_actuators
from src.agents import *


def load_hparams(yaml_hparam_path, num_seeds=5):
    """
    :param yaml_hparam_path: path to YAML hyperparameters
    :param num_seeds: number of different seeds to use
    """
    run_dir = f"../runs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{run_dir}/results", exist_ok=True)

    yaml_filename = os.path.basename(yaml_hparam_path)
    shutil.copy2(yaml_hparam_path, os.path.join(run_dir, yaml_filename))

    with open(yaml_hparam_path, 'r') as f:
        hparam = yaml.safe_load(f)

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


def load_agent_and_env(hparam):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparam['method']
    if method == "method1":
        agent = Method1Gnn
    elif method == "method2":
        agent = Method2Gnn
    elif method == "method5":
        agent = Method5Gnn
    elif method == "NerveNet":
        agent = NerveNet
    else:
        raise ValueError(f"Method {method} not implemented")
    env_setup = MujocoParser(**hparam)
    env, node_dim, num_nodes = env_setup.envs_train[0], env_setup.limb_obs_size, env_setup.num_nodes
    print(f"{env=}, {node_dim=}, {num_nodes=}")
    edges = create_edges(env, device)

    from torch_geometric.utils import degree
    in_degree = degree(edges[1], num_nodes=num_nodes)
    max_in = in_degree.max().item()
    actuator_mask = check_actuators(env)
    env.reset()
    hparam['graph_info'] = {'edge_idx': edges, 'num_nodes': num_nodes, 'node_dim': node_dim}
    actor = agent(in_dim=node_dim,
                  num_nodes=num_nodes,
                  edge_index=edges,
                  action_dim=1,
                  mask=actuator_mask,
                  device=device,
                  max_neighbours=max_in,
                  **hparam)
    return actor, env, device
