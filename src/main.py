import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")
print(f"Contents of project root: {os.listdir(project_root)}")

# Check if src directory exists and what's in it
src_path = os.path.join(project_root, 'src')
if os.path.exists(src_path):
    print(f"Contents of src: {os.listdir(src_path)}")
    utils_path = os.path.join(src_path, 'utils')
    if os.path.exists(utils_path):
        print(f"Contents of src/utils: {os.listdir(utils_path)}")
    else:
        print("src/utils directory does not exist!")
else:
    print("src directory does not exist!")

# Now try the import
from src.utils.misc_utils import load_hparams, load_agent_and_env


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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(processes=min(6, len(hparams)))
        results = pool.map(run, hparams)
        pool.close()
        pool.join()
    # plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')
    # plot_rewards_with_seeds(f'../runs/best')
