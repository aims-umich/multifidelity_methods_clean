#helpers3f.py

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
import torch
rng = np.random.default_rng(42)  # Global NumPy random generator with fixed seed for reproducibility.
torch.manual_seed(42)  # Fix PyTorch RNG seed for reproducible NN training.

from helpers_2f import (
    MLP,
    MLP_lin,
    train_torch_regressor,
    uniform_samples,
    r2_score,
    parity_and_residual_plots,
    plot_loss_curves
)

#========================
#3-fidelity dataset type
#========================

@dataclass
class Dataset:
    """Container for paired low/medium/high-fidelity training data and domain bounds."""
 
    Xl: np.ndarray  # Low-fidelity input locations (n_low, D).
    yl: np.ndarray  # Low-fidelity outputs (n_low, 1).
    Xm: np.ndarray  # Medium-fidelity input locations (n_medium, D).
    ym: np.ndarray  # Medium-fidelity outputs (n_medium, 1).
    Xh: np.ndarray  # High-fidelity input locations (n_high, D).
    yh: np.ndarray  # High-fidelity outputs (n_high, 1).
    bounds: np.ndarray  # (D,2) array of box bounds for sampling/test.

#========================
#3-fidelity dataset maker
#========================

def make_dataset(low_f: Callable, medium_f: Callable, high_f: Callable, bounds: np.ndarray,
                 n_low=200, n_medium=100, n_high=40,
                 seed=0,
                 noise_low=0.0, noise_medium=0.0, noise_high=0.0) -> Dataset:
    """Sample low/med/high-fidelity inputs, evaluate functions, optionally add noise, and pack into Dataset."""
    bounds = np.asarray(bounds)

    #sample inputs (independent design for now; can switch to nested later)
    Xl = uniform_samples(bounds, n_low,   seed=seed)
    Xm = uniform_samples(bounds, n_medium, seed=seed+1)
    Xh = uniform_samples(bounds, n_high,  seed=seed+2)

    #evaluate functions
    yl = low_f(Xl).reshape(-1, 1)
    ym = medium_f(Xm).reshape(-1, 1)
    yh = high_f(Xh).reshape(-1, 1)

    #optional noise
    if noise_low > 0:
        yl = yl + noise_low * rng.standard_normal(size=yl.shape)
    if noise_medium > 0:
        ym = ym + noise_medium * rng.standard_normal(size=ym.shape)
    if noise_high > 0:
        yh = yh + noise_high * rng.standard_normal(size=yh.shape)

    #pack into Dataset
    return Dataset(
        Xl=Xl, yl=yl,
        Xm=Xm, ym=ym,
        Xh=Xh, yh=yh,
        bounds=bounds,
    )

