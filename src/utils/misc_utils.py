import torch
import os
import datetime
import yaml
import itertools
from copy import deepcopy
from src.environments.mujoco_parser import get_graph_structure, get_graph_joints, get_motor_joints



def create_edges(env):
    parent_list = get_graph_structure(env.unwrapped.xml)
    edges = []
    for i, j in enumerate(parent_list):
        if j != -1:
            edges.append([i, j])
            edges.append([j, i])
    if edges:
        return torch.tensor(edges, dtype=torch.long).t()
    else:
        return torch.zeros((2, 0), dtype=torch.long)


def create_actuator_mapping(env):
    joint_list = get_graph_joints(env.unwrapped.xml)
    actuator_list = get_motor_joints(env.unwrapped.xml)
    mapping_dict = {}

    for node_idx, (_, joint_name) in enumerate(joint_list):
        if joint_name in actuator_list:
            actuator_idx = actuator_list.index(joint_name)
            mapping_dict[node_idx] = actuator_idx

    def actuator_mapping(node_outputs):
        actuator_actions = torch.zeros(len(actuator_list), device=node_outputs.device)
        for node_idx, actuator_idx in mapping_dict.items():
            actuator_actions[actuator_idx] = node_outputs[node_idx]
        return actuator_actions

    return actuator_mapping


def load_hparams(yaml_hparam_path):
    """
    :param yaml_hparam_path: path to YAML hyperparameters
    """
    run_dir = f"../runs/run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
    os.makedirs(run_dir, exist_ok=True)

    # Create the directory structure
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{run_dir}/results", exist_ok=True)

    with open(yaml_hparam_path, 'r') as f:
        hparam = yaml.safe_load(f)
    test_params = {k: v for k, v in hparam.items() if isinstance(v, list) and len(v) > 1}
    base_params = {k: v[0] if isinstance(v, list) and len(v) == 1 else v
                   for k, v in hparam.items() if k not in test_params}
    param_names = list(test_params.keys())
    param_values = list(test_params.values())
    all_combinations = []

    for combination in itertools.product(*param_values):
        hparams = deepcopy(base_params)
        for i, param_name in enumerate(param_names):
            hparams[param_name] = combination[i]
        run_id = "_".join([f"{param_name}-{combination[i]}" for i, param_name in enumerate(param_names)])
        hparams['run_id'] = run_id
        hparams['run_dir'] = run_dir
        all_combinations.append(hparams)
    return all_combinations
