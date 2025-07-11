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
        cfg.update({
            # "state_preprocessor": None,
            "rollouts": hparam.get("rollouts", 2048),
            "learning_epochs": hparam.get("learning_epochs", 8),
            "mini_batches": hparam.get("mini_batches", 32),
            "learning_rate": hparam.get("learning_rate", 3e-4),
            "discount_factor": hparam.get("discount_factor", 0.99),
            "lambda": hparam.get("lambda", 0.95),
            "clip_predicted_values": hparam.get("clip_predicted_values", False),
            "entropy_loss_scale": hparam.get("entropy_loss_scale", 0.0),
            "value_loss_scale": hparam.get("value_loss_scale", 1.0),
            "kl_threshold": hparam.get("kl_threshold", 0),
            "experiment": {
                "directory": hparam.get("run_dir", "runs"),
                "experiment_name": hparam.get("run_id", ""),
                "write_interval": hparam.get("write_interval", 250),
                "checkpoint_interval": hparam.get("checkpoint_interval", 1000),
            }
        })
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
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=3)
    print(hparams[0])

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
        # print("CUDA not available, running on CPU")
        # mp.set_start_method('spawn', force=True)
        # pool = mp.Pool(processes=min(3, len(hparams)))
        # results = pool.map(run_worker, hparams)
        # pool.close()
        # pool.join()
        run_worker(hparams[0])
    plot_rewards_with_seeds(f'{hparams[0]["run_dir"]}/results')
    # # plot_rewards_with_seeds('../runs/reg_plot/results')
