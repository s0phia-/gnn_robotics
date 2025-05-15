import torch
import os
import multiprocessing as mp
from src.utils.misc_utils import load_hparams
from src.environments.mujoco_parser import MujocoParser, create_edges, check_actuators
from src.agents.method2 import Method2Gnn
from src.agents.ppo import PPO
from src.utils.logger_config import set_run_id, get_logger
from src.utils.analyse_data import plot_rewards_with_seeds
from torch.profiler import profile, record_function, ProfilerActivity


def run(hparam):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_run_id(hparam['run_id'])
    logger = get_logger()
    logger.info(f"Starting run with parameters: {hparam['run_id']}")
    env_setup = MujocoParser(**hparam)
    env, node_dim, num_nodes = env_setup.envs_train[0], env_setup.limb_obs_size, env_setup.num_nodes
    edges = create_edges(env, device)
    actuator_mask = check_actuators(env)
    env.reset()
    hparam['graph_info'] = {'edge_idx': edges, 'num_nodes': num_nodes, 'node_dim': node_dim}
    actor = Method2Gnn(in_dim=node_dim,
                              num_nodes=num_nodes, 
                              edge_index=edges, 
                              action_dim=1,
                              mask=actuator_mask,
                              device=device, **hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)


    # Set up profiler with both CPU and CUDA activities
    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)

    # Run with profiler
    with profile(
            activities=profiler_activities,
            record_shapes=True,
            profile_memory=True,  # Enable memory profiling
            with_stack=True,  # Record stack trace for each operation
            with_flops=True,  # Estimate FLOPS for each operation
    ) as prof:
        with record_function("model_training"):
            model.learn()

    # Print profiler results sorted by different metrics
    print("\n--- Profiler Results (sorted by CPU time) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    if torch.cuda.is_available():
        print("\n--- Profiler Results (sorted by CUDA time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print("\n--- Profiler Results (sorted by CUDA memory usage) ---")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

        print("\n--- Profiler Results (sorted by CUDA memory allocation size) ---")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))


def view_model_demo(model_path, hparam):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_run_id(hparam['run_id'])
    env = MujocoParser(**hparam).envs_train[0]
    actuator_mask = check_actuators(env)
    edges = create_edges(env)
    env.reset()
    actor = MessagePassingGNN(in_dim=15, num_nodes=9, edge_index=edges, device=device, mask=actuator_mask, **hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.demo(actor_path=os.path.join(f'{model_path}','ppo_actor.pth'), critic_path=os.path.join(f'{model_path}','ppo_critic.pth'))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__))) # ensure the script is run from the correct directory
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)
    # if torch.cuda.is_available():
    #     mp.set_start_method('spawn', force=True)
    # pool = mp.Pool(processes=min(mp.cpu_count(), len(hparams)))
    # results = pool.map(run, hparams)
    # pool.close()
    # pool.join()
    run(hparams[0])
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')

    # view_model_demo(f'../runs/{hparams[0]["run_id"]}/checkpoints/propagation_steps-4_seed-6', hparams[0])
