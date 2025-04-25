import torch
import multiprocessing as mp
from src.utils.misc_utils import  load_hparams
from src.environments.mujoco_parser import MujocoParser, create_edges, create_actuator_mapping
from src.agents.function_approximators import MessagePassingGNN
from src.agents.ppo import PPO
from src.utils.logger_config import set_run_id, get_logger
from src.utils.analyse_data import plot_rewards_with_seeds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(hparam):
    set_run_id(hparam['run_id'])
    logger = get_logger()
    logger.info(f"Starting run with parameters: {hparam['run_id']}")
    env = MujocoParser(**hparam).envs_train[0]
    edges = create_edges(env)
    actuator_mapping = create_actuator_mapping(env)
    env.reset()
    actor = MessagePassingGNN(in_dim=15, num_nodes=9, edge_index=edges, actuator_mapping=actuator_mapping,
                              device=device, **hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.learn()


def view_model_demo(model_path, hparam):
    set_run_id(hparam['run_id'])
    env = MujocoParser(**hparam).envs_train[0]
    actuator_mapping = create_actuator_mapping(env)
    edges = create_edges(env)
    env.reset()
    actor = MessagePassingGNN(in_dim=15, num_nodes=9, edge_index=edges, actuator_mapping=actuator_mapping,
                              device=device, **hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.demo(actor_path=f'{model_path}/ppo_actor.pth', critic_path=f'{model_path}/ppo_critic.pth')


if __name__ == '__main__':
    hparams = load_hparams('utils/hyperparameters.yaml', num_seeds=3)
    # pool = mp.Pool(processes=min(mp.cpu_count(), len(hparams)))
    # results = pool.map(run, hparams)
    # pool.close()
    # pool.join()
    # plot_rewards_with_seeds(f'../runs/run_20250422_172544/results', hparams)
    view_model_demo('../runs/run_20250422_172544/checkpoints/propagation_steps-3_hidden_node_dim'
                    '-64_decoder_and_message_hidden_dim-64_seed-106', hparams[0])
