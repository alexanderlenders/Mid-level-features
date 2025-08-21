import random
import numpy as np
import torch
import lightning as L

def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

    # Set internal precision
    torch.set_float32_matmul_precision("high")

    # Set random seeds
    L.seed_everything(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
