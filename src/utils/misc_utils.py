import torch
import os
import datetime
import yaml
import itertools
from copy import deepcopy


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
    with open(yaml_hparam_path, 'r') as f:
        hparam = yaml.safe_load(f)

    base_seed = hparam.get('seed', 0)
    seeds = [base_seed + i*100 for i in range(num_seeds)]

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
            run_id += f"_seed-{seed}"
            hparams['run_id'] = run_id
            hparams['run_dir'] = run_dir
            all_combinations.append(hparams)
    return all_combinations
