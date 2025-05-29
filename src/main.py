import os
import torch
import multiprocessing as mp
from src.utils.misc_utils import load_hparams, load_agent_and_env
from src.agents.ppo import PPO
from src.utils.logger_config import set_run_id, get_logger
from src.utils.analyse_data import plot_rewards_with_seeds

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run(hparam):
    set_run_id(hparam['run_id'])
    logger = get_logger()
    logger.info(f"Starting run with parameters: {hparam['run_id']}")
    actor, env, device = load_agent_and_env(hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.learn()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # ensure the script is run from the correct directory
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)
    # if torch.cuda.is_available():
    #     mp.set_start_method('spawn', force=True)
    #     pool = mp.Pool(processes=min(4, len(hparams)))
    #     results = pool.map(run, hparams)
    #     pool.close()
    #     pool.join()
    # else:
    for h in hparams:
        run(h)
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')
    # plot_rewards_with_seeds(f'../runs/run_20250522_175111/results')
