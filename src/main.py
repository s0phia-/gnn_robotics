import os
import multiprocessing as mp
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_worker(args):
    """Worker function that handles both GPU and CPU cases"""
    if len(args) == 2:  # GPU
        hparam, gpu_id = args
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:  # CPU
        hparam = args
        gpu_id = None

    import torch
    from src.utils import load_agent_and_env, set_run_id, get_logger
    from src.agents.ppo import PPO

    def run(hparam, device):
        set_run_id(hparam['run_id'])
        logger = get_logger()
        logger.info(f"Starting run with parameters: {hparam['run_id']} on {device}")
        if device.type == 'cuda':
            logger.info(f"Current GPU in run(): {torch.cuda.current_device()}")

        actor, env = load_agent_and_env(hparam, device)

        for name, param in actor.named_parameters():
            if param.device != device:
                logger.warning(f"Parameter {name} is on {param.device}, expected {device}")
            break

        model = PPO(actor=actor, device=device, env=env, **hparam)
        model.learn()

    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device('cuda:0')
        print(f"Process GPU {gpu_id}: Available GPUs = {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print(f"Process running on CPU")

    return run(hparam, device)


if __name__ == '__main__':
    import torch
    from src.utils import load_hparams, plot_rewards_with_seeds

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=5)

    if torch.cuda.is_available():  # GPU
        mp.set_start_method('spawn', force=True)
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        gpu_assignments = [(hparam, i % num_gpus) for i, hparam in enumerate(hparams)]
        pool = mp.Pool(processes=min(num_gpus, len(hparams)))
        results = pool.map(run_worker, gpu_assignments)
        pool.close()
        pool.join()
    else:  # CPU
        print("CUDA not available, running on CPU")
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(processes=min(4, len(hparams)))  # Use 4 CPU cores
        results = pool.map(run_worker, hparams)
        pool.close()
        pool.join()

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
# if __name__ == '__main__':
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
#     plot_rewards_with_seeds(f'../runs/ac/results')
