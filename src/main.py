import os
import torch
import multiprocessing as mp
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.utils import load_hparams, load_agent_and_env, set_run_id, get_logger, plot_rewards_with_seeds
from src.agents.ppo import PPO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run(hparam):
    set_run_id(hparam['run_id'])
    logger = get_logger()
    logger.info(f"Starting run with parameters: {hparam['run_id']}")
    actor, env, device = load_agent_and_env(hparam)
    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.learn()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(processes=min(4, len(hparams)))
        results = pool.map(run, hparams)
        pool.close()
        pool.join()
    else:
        run(hparams[0])
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')
    # plot_rewards_with_seeds(f'../runs/change ppo, ftn_approx/results')
