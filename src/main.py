##########################################################################
### main run script -- change params in src/utils/hyperparameters.yaml ###
##########################################################################

import os
import torch
import torch.multiprocessing as mp
import sys
from src.utils.run_utils import run_worker

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == '__main__':
    from src.utils import load_hparams
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    hparams = load_hparams(os.path.join('utils', 'hyperparameters.yaml'), num_seeds=1)
    # hparams = [{'env_name': ['ant', 'hopper'], 'max_episodic_timesteps': 2048, 'population_size': 4, 'enable_features': True, 'learning_rate': 0.0003, 'gamma': 0.99, 'clip_value': 0.5, 'update_epochs': 8, 'total_timesteps': 1000000, 'mixed_precision': False, 'grad_clip_value': 0.5, 'rollouts': 8, 'normalize_advantage': True, 'advantage_method': 'gae', 'gae_lambda': 0.95, 'opt_together': True, 'method': 'method1', 'propagation_steps': 3, 'hidden_node_dim': 64, 'network_shape': [64, 64, 64], 'decoder_and_message_layers': 2, 'decoder_and_message_hidden_dim': 64, 'morphology_fc_ratio': 0, 'seed': 5, 'save_model_freq': 10, 'xml_path': 'environments/assets', 'env_dir': './environments', 'base_modular_env_path': './environments/modular_env.py', 'data_dir': './results', 'load_run_path': None, 'run_id': ',seed-5', 'run_dir': '../runs/run_20250901_192750'}]
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')

    gpu = False  # torch.cuda.is_available()
    if gpu:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        gpu_assignments = [(hparam, i % num_gpus) for i, hparam in enumerate(hparams)]
        with mp.Pool(processes=min(num_gpus, len(hparams))) as pool:
            results = pool.map(run_worker, gpu_assignments)
    else:  # CPU
        print("CUDA not available, running on CPU")
        with mp.Pool(processes=min(1, len(hparams))) as pool:
            results = pool.map(run_worker, hparams)
