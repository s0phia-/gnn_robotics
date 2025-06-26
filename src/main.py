import os
import multiprocessing as mp
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set this before any torch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.utils import load_hparams, load_agent_and_env, set_run_id, get_logger, plot_rewards_with_seeds
from src.agents.ppo import PPO


def run(hparam, device):
    set_run_id(hparam['run_id'])
    logger = get_logger()
    logger.info(f"Starting run with parameters: {hparam['run_id']} on {device}")
    logger.info(f"Current GPU in run(): {torch.cuda.current_device()}")

    actor, env = load_agent_and_env(hparam, device)

    for name, param in actor.named_parameters():
        if param.device != device:
            logger.warning(f"Parameter {name} is on {param.device}, expected {device}")
        break

    model = PPO(actor=actor, device=device, env=env, **hparam)
    model.learn()


def run_with_gpu(args):
    hparam, gpu_id = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Process GPU {gpu_id}: Available GPUs = {torch.cuda.device_count()}")
    return run(hparam, device)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import torch
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        num_gpus = torch.cuda.device_count()
        gpu_assignments = [(hparam, i % num_gpus) for i, hparam in enumerate(hparams)]
        pool = mp.Pool(processes=min(num_gpus, len(hparams)))
        results = pool.map(run_with_gpu, gpu_assignments)
        pool.close()
        pool.join()
    else:
        from src.utils import load_agent_and_env, set_run_id, get_logger
        from src.agents.ppo import PPO
        run(hparams[0])
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')


# def run_with_gpu(args):
#     hparam, gpu_id = args
#     device = torch.device(f'cuda:{gpu_id}')
#     return run(hparam, device)
#
#
# def run(hparam, device):
#     set_run_id(hparam['run_id'])
#     logger = get_logger()
#     logger.info(f"Starting run with parameters: {hparam['run_id']} on {device}")
#     actor, env = load_agent_and_env(hparam, device)
#     model = PPO(actor=actor, device=device, env=env, **hparam)
#     model.learn()
#
#
if __name__ == '__main__':
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)
#     if torch.cuda.is_available():
#         mp.set_start_method('spawn', force=True)
#         num_gpus = torch.cuda.device_count()
#
#         # Assign each experiment to a GPU
#         gpu_assignments = [(hparam, i % num_gpus) for i, hparam in enumerate(hparams)]
#
#         pool = mp.Pool(processes=min(num_gpus, len(hparams)))
#         results = pool.map(run_with_gpu, gpu_assignments)
#         pool.close()
#         pool.join()
#     else:
#         run(hparams[0])
#     plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')
    plot_rewards_with_seeds(f'../runs/ac/results')
