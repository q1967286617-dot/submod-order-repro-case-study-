# src/utils/repro.py
import os
import random
import numpy as np

def set_global_seed(seed: int) -> np.random.Generator:
    """
    Set seeds for reproducibility and return a numpy Generator.
    Use the returned rng everywhere instead of np.random.* global calls.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy RNG, avoid using directly in your code
    rng = np.random.default_rng(seed)  # modern RNG
    return rng