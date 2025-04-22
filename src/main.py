import torch
import multiprocessing as mp
from src.utils.misc_utils import create_edges, create_actuator_mapping, load_hparams
from src.environments.mujoco_parser import MujocoParser
from src.agents.function_approximators import MessagePassingGNN
from src.agents.ppo import PPO
from src.utils.logger_config import set_run_id, get_logger

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
    # model.demo()


if __name__ == '__main__':

    hparams = load_hparams('utils/hyperparameters.yaml')
    pool = mp.Pool(processes=min(mp.cpu_count(), len(hparams)))
    results = pool.map(run, hparams)
    pool.close()
    pool.join()
