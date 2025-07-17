import os
import torch.multiprocessing as mp
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_worker(args):
    """Worker function that handles both GPU and CPU cases"""
    import torch
    if len(args) == 2:  # GPU
        hparam, gpu_id = args
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)  # Set the specific GPU
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

        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg.update(hparam["ppo"])
        cfg.update(hparam["trainer"])
        cfg.update({"experiment": {"directory": hparam.get("run_dir", "runs"),
                                   "experiment_name": hparam.get("run_id", "")}})
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
            device=device
        )
        trainer_cfg = {
            "value_preprocessor": None,
            "timesteps": hparam.get("timesteps", 1000000),
            "headless": hparam.get("headless", True)
        }
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
        trainer.train()

    return run(hparam, device)


if __name__ == '__main__':
    import torch
    from src.utils import load_hparams, plot_rewards_with_seeds
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=3)
    # if torch.cuda.is_available():  # GPU
    #     mp.set_start_method('spawn', force=True)
    #     mp.set_sharing_strategy('file_system')
    #     num_gpus = torch.cuda.device_count()
    #     print(f"Using {num_gpus} GPUs")
    #     gpu_assignments = [(hparam, i % num_gpus) for i, hparam in enumerate(hparams)]
    #     with mp.Pool(processes=min(num_gpus, len(hparams))) as pool:
    #         results = pool.map(run_worker, gpu_assignments)
    # else:  # CPU
    print("CUDA not available, running on CPU")
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    with mp.Pool(processes=min(10, len(hparams))) as pool:
        results = pool.map(run_worker, hparams)
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')
    # # plot_rewards_with_seeds('../runs/reg_plot/results')
