##########################################################################
### main run script -- change params in src/utils/hyperparameters.yaml ###
##########################################################################

import os
import torch.multiprocessing as mp
import sys
from src.utils.misc_utils import run_worker


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == '__main__':
    import torch
    from src.utils import load_hparams
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
    with mp.Pool(processes=min(5, len(hparams))) as pool:
        results = pool.map(run_worker, hparams)
