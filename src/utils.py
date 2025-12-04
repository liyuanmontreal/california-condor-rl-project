# src/utils.py
import os
import json
import random
import time
from contextlib import contextmanager

import numpy as np


# =========================================================
# 1. Set random seed
# =========================================================
def set_seed(seed: int = 42):
    """
    Set global random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# 2. Linear schedule (for RL learning rate decay)
# =========================================================
def linear_schedule(initial_value: float):
    """
    Returns a function that linearly decays the value from initial_value to 0.
    Useful in RL (e.g., exploration rate).
    """
    def schedule(progress_remaining: float):
        # progress_remaining goes from 1 (start) → 0 (end)
        return progress_remaining * initial_value
    return schedule


# =========================================================
# 3. Directory utilities
# =========================================================
def ensure_dir(path: str):
    """
    Create directory if not exists.
    """
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


# =========================================================
# 4. JSON helpers
# =========================================================
def save_json(data, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# 5. Timing tool
# =========================================================
@contextmanager
def timer(name: str = "Elapsed"):
    start = time.time()
    yield
    end = time.time()
    print(f"[TIMER] {name}: {end - start:.3f} sec")


# =========================================================
# 6. Soft clipping for stability
# =========================================================
def soft_clip(value, min_v, max_v):
    """
    Clip value but allow float, np arrays.
    """
    return np.minimum(np.maximum(value, min_v), max_v)


# =========================================================
# 7. Pretty print configuration dicts
# =========================================================
def pretty_print_cfg(cfg, indent=0):
    """
    Nicely print nested dict (for YAML config inspection).
    """
    pad = "  " * indent
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, dict):
                print(f"{pad}{k}:")
                pretty_print_cfg(v, indent + 1)
            else:
                print(f"{pad}{k}: {v}")
    else:
        print(cfg)


# =========================================================
# 8. Colored logging for terminal output
# =========================================================
def colored_log(msg, color="blue"):
    """
    Print colored text in terminal.
    """
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m",
    }
    c = COLORS.get(color, COLORS["blue"])
    print(f"{c}{msg}{COLORS['end']}")
