"""Common utility functions."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, benchmark: bool = False) -> None:
    """Set random seeds for reproducibility across supported libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if benchmark:
        print("Benchmark mode enabled. Deterministic mode disabled.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("Deterministic mode enabled. Benchmark mode disabled.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
