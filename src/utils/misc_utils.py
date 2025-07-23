import shutil
import os
import datetime
import yaml
import itertools
from torch_geometric.utils import degree
from copy import deepcopy
from src.environments.mujoco_parser import MujocoParser, create_edges, check_actuators
from src.agents import SKRLFeedForward, SkrlNerveNet, SkrlMethod1, SkrlMethod2


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

    hparam['env_mapping'] = {env_name: idx for idx, env_name in enumerate(hparam['env_name'])}

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
    env_setup = MujocoParser(**hparam)
    env, node_dim, num_nodes = env_setup.envs_train[0], env_setup.limb_obs_size, env_setup.num_nodes
    print(f"{env=}, {node_dim=}, {num_nodes=}")
    edges = create_edges(env, device)
    actuator_mask = check_actuators(env)
    env.reset()
    env_idx = hparam['env_mapping'][hparam['env_name']]
    hparam[f'graph_info_{env_idx}'] = {'edge_idx': edges, 'num_nodes': num_nodes, 'node_dim': node_dim,
                                       'actuator_mask': actuator_mask}
    return env


def load_agent_and_env(hparam, device):
    env = load_env(hparam, device)
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
    env_idx = hparam['env_mapping'][hparam['env_name']]
    graph_info = hparam[f'graph_info_{env_idx}']
    edges = graph_info['edge_idx']
    in_degree = degree(edges[1], num_nodes=graph_info['num_nodes'])
    max_in = in_degree.max().item()
    actor = agent(in_dim=graph_info['num_nodes'],
                  num_nodes=graph_info['num_nodes'],
                  edge_index=edges,
                  action_dim=1,
                  mask=graph_info['actuator_mask'],
                  device=device,
                  max_neighbours=max_in,
                  **hparam)
    return actor, env


def load_skrl_agent_and_env(hparam, device):
    env = load_env(hparam, device)
    method = hparam['method']
    if method == "method1":
        agent = SkrlMethod1
    elif method == "NerveNet":
        agent = SkrlNerveNet
    elif method == "method2":
        agent = SkrlMethod2
    elif method == "MLP":
        agent = SKRLFeedForward
    else:
        raise ValueError(f"Method {method} not implemented")
    if not hasattr(env, 'num_agents'):
        env.num_agents = 1
    if not hasattr(env, 'num_envs'):
        env.num_envs = 1
    env_idx = hparam['env_mapping'][hparam['env_name']]
    graph_info = hparam[f'graph_info_{env_idx}']
    models = {"policy": agent(observation_space=env.observation_space,
                              action_space=env.action_space,
                              device=device,
                              num_nodes=graph_info['num_nodes'],
                              **hparam),
              "value": SKRLFeedForward(observation_space=env.observation_space,
                                       action_space=env.action_space,
                                       device=device,
                                       hidden_dim=hparam.get('decoder_and_message_hidden_dim', 64),
                                       hidden_layers=hparam.get('decoder_and_message_layers', 3)
                                       )}
    return models, env


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
    from src.utils import load_skrl_agent_and_env, set_run_id, get_logger
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer

    def run(hparam, device):
        set_run_id(hparam['run_id'])
        logger = get_logger()
        logger.info(f"Starting run with parameters: {hparam['run_id']} on {device}")
        if device.type == 'cuda':
            logger.info(f"Current GPU in run(): {torch.cuda.current_device()}")
        models, env = load_skrl_agent_and_env(hparam, device)

        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg.update(hparam["ppo"])
        cfg.update(hparam["trainer"])
        cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
        # cfg["learning_rate_scheduler"] = KLAdaptiveRL
        cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        cfg.update({"experiment": {"directory": hparam.get("run_dir", "runs"),
                                   "experiment_name": hparam.get("run_id", "")}})

        memory = RandomMemory(
            memory_size=cfg["rollouts"],
            num_envs=1,
            device=device
        )
        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

        trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent)
        trainer.train()

    return run(hparam, device)
