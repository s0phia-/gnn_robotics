import torch
import os,sys
import multiprocessing as mp
from src.utils.misc_utils import load_hparams
from src.environments.mujoco_parser import MujocoParser, create_edges, check_actuators
from src.agents.function_approximators import MessagePassingGNN
from src.agents.ppo import PPO
from src.utils.logger_config import set_run_id, get_logger
from src.utils.analyse_data import plot_rewards_with_seeds


def run(hparam):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_run_id(hparam['run_id'])
    logger = get_logger()
    logger.info(f"Starting run with parameters: {hparam['run_id']}")
    env = MujocoParser(**hparam).envs_train[0]
    edges = create_edges(env, device)
    actuator_mask = check_actuators(env)
    env.reset()
    node_dim = 15
    num_nodes = 9  
    hparam['graph_info'] = {'edge_idx': edges, 'num_nodes': num_nodes, 'node_dim': node_dim}
    actor = MessagePassingGNN(in_dim=node_dim, 
                              num_nodes=num_nodes, 
                              edge_index=edges, 
                              action_dim=1,
                              mask=actuator_mask,
                              device=device, **hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.learn()


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
    hparams = load_hparams(os.path.join('utils','hyperparameters.yaml'), num_seeds=1)
    # if torch.cuda.is_available():
    #     mp.set_start_method('spawn', force=True)
    # pool = mp.Pool(processes=min(mp.cpu_count(), len(hparams)))
    # results = pool.map(run, hparams)
    # pool.close()
    # pool.join()
    run(hparams[0])
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results', hparams)
    # view_model_demo(f'../runs/{hparams[0]["run_id"]}/checkpoints/propagation_steps-4_seed-6', hparams[0])
