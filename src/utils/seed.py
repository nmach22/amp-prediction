"""
Reproducibility helper – set all random seeds in one call.
"""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and (optionally) PyTorch.

    Args:
        seed: The integer seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

