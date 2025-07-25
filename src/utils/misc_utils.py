import shutil
import os
import datetime
import yaml
import itertools
# from tensorboard.plugins.hparams import api as hp
from copy import deepcopy


def load_hparams(yaml_hparam_path, num_seeds=5, experiment_name=None):
    with open(yaml_hparam_path, 'r') as f:
        hparam = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_run_dir = f"../runs/{experiment_name or f'exp_{timestamp}'}"

    os.makedirs(f"{base_run_dir}/configs", exist_ok=True)
    shutil.copy2(yaml_hparam_path, f"{base_run_dir}/configs/{os.path.basename(yaml_hparam_path)}")

    base_seed = hparam.get('seed', 42)
    seeds = [base_seed + i * 100 for i in range(num_seeds)]
    hparam['env_mapping'] = {env_name: idx for idx, env_name in enumerate(hparam['env_name'])}

    # Flatten nested parameters for sweeping
    def flatten_params(params, prefix=''):
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, key))
            else:
                flat[key] = v
        return flat
    
    def unflatten_params(flat_params):
        nested = {}
        for key, value in flat_params.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested
    
    flat_params = flatten_params(hparam)
    test_params = {k: v for k, v in flat_params.items() if isinstance(v, list) and len(v) > 1}
    base_params = {k: v[0] if isinstance(v, list) else v
                   for k, v in flat_params.items() if k not in test_params}

    all_combinations = []

    if test_params:
        param_names = list(test_params.keys())
        param_values = list(test_params.values())

        for combination in itertools.product(*param_values):
            for seed in seeds:
                hparams = deepcopy(base_params)
                for i, param_name in enumerate(param_names):
                    hparams[param_name] = combination[i]

                # Reconstruct nested structure from flattened params
                hparams = unflatten_params(hparams)

                # Only create combo name if multiple parameters are being swept
                if len(param_names) > 1:
                    combo_name = "_".join([f"{param_names[i]}-{combination[i]}" for i in range(len(param_names))])
                    tensorboard_dir = f"{base_run_dir}/ppo/{hparams.get('env_name', 'unknown')}/{hparams.get('method', 'unknown')}/{combo_name}/seed_{seed:03d}"
                    experiment_name = f"{hparams.get('method', 'unknown')}_{hparams.get('env_name', 'unknown')}_{combo_name}"
                else:
                    # Single parameter sweep - no combo folder needed
                    tensorboard_dir = f"{base_run_dir}/ppo/{hparams.get('env_name', 'unknown')}/{hparams.get('method', 'unknown')}/seed_{seed:03d}"
                    experiment_name = f"{hparams.get('method', 'unknown')}_{hparams.get('env_name', 'unknown')}"

                hparams.update({
                    'seed': seed,
                    'tensorboard_dir': tensorboard_dir,
                    'experiment_name': experiment_name,
                    'run_id': f"{hparams.get('method', 'unknown')}_seed-{seed}"
                })
                hparams['skrl_config'] = create_skrl_config(hparams)
                all_combinations.append(hparams)
    else:
        for seed in seeds:
            hparams = deepcopy(base_params)
            # Reconstruct nested structure from flattened params
            hparams = unflatten_params(hparams)
            method_name = hparams.get('method', 'unknown')
            env_name = hparams.get('env_name', 'unknown')

            hparams.update({
                'seed': seed,
                'tensorboard_dir': f"{base_run_dir}/ppo/{env_name}/{method_name}/seed_{seed:03d}",
                'experiment_name': f"{method_name}_{env_name}",
                'run_id': f"{method_name}_{env_name}_seed-{seed}"
            })
            hparams['skrl_config'] = create_skrl_config(hparams)
            all_combinations.append(hparams)

    return all_combinations


# CHANGE: Added standalone function to replace lambda (for multiprocessing compatibility)
def reward_shaper(rewards, timestep, timesteps):
    return rewards * 0.01


def create_skrl_config(hparam):
    from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
    cfg = PPO_DEFAULT_CONFIG.copy()
    
    # Handle flattened parameters
    ppo_params = {}
    trainer_params = {}
    for key, value in hparam.items():
        if key.startswith("ppo."):
            ppo_params[key[4:]] = value  # Remove "ppo." prefix
        elif key.startswith("trainer."):
            trainer_params[key[8:]] = value  # Remove "trainer." prefix
    
    # Also handle legacy nested format
    if "ppo" in hparam:
        ppo_params.update(hparam["ppo"])
    if "trainer" in hparam:
        trainer_params.update(hparam["trainer"])
    
    cfg.update(ppo_params)
    cfg.update(trainer_params)
    cfg["rewards_shaper"] = reward_shaper  # CHANGE: Use function instead of lambda
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["experiment"] = {
        "directory": hparam["tensorboard_dir"],
        "experiment_name": hparam["experiment_name"],
        "write_interval": hparam.get("write_interval", 100),
        "checkpoint_interval": hparam.get("checkpoint_interval", 1000),
    }
    return cfg


def load_env(hparam, device):
    from src.environments.mujoco_parser import MujocoParser, create_edges, check_actuators
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
    from torch_geometric.utils import degree
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
    from src.agents import SKRLFeedForward, SkrlNerveNet, SkrlMethod1, SkrlMethod2
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

        cfg = hparam['skrl_config']

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
