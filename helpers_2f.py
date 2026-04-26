#Imports
import numpy as np  # Numerical computing (arrays, random numbers, vectorized ops).
from dataclasses import dataclass  # Lightweight class containers for typed data.
from typing import Callable, Tuple, Dict  # Type hints for readability and tooling.
from tqdm import trange  # Progress-bar iterator for training loops.
import pandas as pd
from sklearn.metrics import r2_score as sk_r2_score

# --- Benchmarks (mf2) ---
import mf2  # pip install mf2  # Benchmark functions providing low/high fidelity variants.

# --- Traditional ML ---
from sklearn.gaussian_process import GaussianProcessRegressor  # GP regressor implementation.
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel  # Common GP kernels.
from sklearn.preprocessing import StandardScaler  # Feature scaling/standardization.
from sklearn.metrics import mean_squared_error  # RMSE computation utility (via MSE).
from sklearn.model_selection import train_test_split as _sk_train_test_split  # Robust splitter from sklearn.

# --- Torch for NNs ---
import torch  # PyTorch main package (tensors, device management).
import torch.nn as nn  # Neural network layers and modules.
import torch.optim as optim  # Optimizers (Adam, SGD, etc.).
import torch.nn.init as init


# --- Plots ---
import matplotlib.pyplot as plt  # Plotting (figures for parity/residuals and inputs).
from pathlib import Path  # Filesystem paths (create figs directory, save images).

rng = np.random.default_rng(42)  # Global NumPy random generator with fixed seed for reproducibility.
torch.manual_seed(42)  # Fix PyTorch RNG seed for reproducible NN training.

# -----------------------
# Device & thread management
# -----------------------

def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU. Single source of truth for all methods."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def limit_cpu_threads(n_threads: int = 4) -> None:
    """
    Limit CPU parallelism so the script plays nicely on a shared cluster.
    Controls PyTorch intra/inter-op threads AND the underlying BLAS/OpenMP
    threads used by NumPy and sklearn.

    Call this once at the top of any entry-point script, e.g.:
        from helpers_2f import limit_cpu_threads
        limit_cpu_threads(4)
    """
    import os
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)

# -----------------------
# Utilities
# -----------------------

def train_test_split(X, y, test_ratio=0.2, seed=0):
    """Split arrays X, y into train/test using sklearn's robust splitter (shuffle + reproducible)."""
    # Use sklearn splitter
    X_train, X_test, y_train, y_test = _sk_train_test_split(
        X, y, test_size=test_ratio, random_state=seed, shuffle=True
    )
    return X_train, X_test, y_train, y_test  # Return split arrays.

def uniform_samples(bounds: np.ndarray, n: int, seed=0):
    """Uniform random samples within [low, high] per dimension (simpler version)."""
    rng_local = np.random.default_rng(seed)  # Local RNG for reproducibility.
    low, high = bounds[:, 0], bounds[:, 1]  # Extract per-dimension bounds.
    #this is same as np.random.uniform but with fixed seeding using the variable rng_local
    return rng_local.uniform(low, high, size=(n, bounds.shape[0]))

# def r2_score(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     if ss_tot == 0:
#         return 1.0 if ss_res == 0 else -np.inf
#     return 1 - ss_res / ss_tot

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return sk_r2_score(y_true, y_pred)

@dataclass
class Dataset:
    """Container for paired low/high-fidelity training data and domain bounds."""
    #this is good for us to access all data with one variable/container

    Xl: np.ndarray  # Low-fidelity input locations (n_low, D).
    yl: np.ndarray  # Low-fidelity outputs (n_low, 1).
    Xh: np.ndarray  # High-fidelity input locations (n_high, D).
    yh: np.ndarray  # High-fidelity outputs (n_high, 1).
    bounds: np.ndarray  # (D,2) array of box bounds for sampling/test.

# -----------------------
# Data generation
# -----------------------

def make_dataset(low_f: Callable, high_f: Callable, bounds: np.ndarray,
                 n_low=200, n_high=40, seed=0, noise_low=0.0, noise_high=0.0) -> Dataset:
    """Sample low/high-fidelity inputs, evaluate functions, optionally add noise, and pack into Dataset."""
    Xl = uniform_samples(bounds, n_low, seed=seed)  # Sample low-fidelity input locations.
    Xh = uniform_samples(bounds, n_high, seed=seed+1)  # Sample high-fidelity input locations (different seed).
    yl = low_f(Xl).reshape(-1, 1)  # Evaluate low-fidelity outputs and make column vector.
    yh = high_f(Xh).reshape(-1, 1)  # Evaluate high-fidelity outputs and make column vector.

    if noise_low > 0:
        #this line is only used if noise is added (NA so far for us)
        yl = yl + noise_low * rng.standard_normal(size=yl.shape)  # Add Gaussian noise to low-fid outputs.
    if noise_high > 0:
        #this line is only used if noise is added (NA so far for us)
        yh = yh + noise_high * rng.standard_normal(size=yh.shape)  # Add Gaussian noise to high-fid outputs.

    return Dataset(Xl=Xl, yl=yl, Xh=Xh, yh=yh, bounds=bounds)  # Return structured dataset using the container class above

# -----------------------
# Visualization helpers
# -----------------------

def parity_and_residual_plots(func_name: str, ytrue: np.ndarray, preds: Dict[str, np.ndarray], out_dir="figs"):
    """
    Creates parity plots (y_pred vs y_true) and residual histograms for each model.
    Saves figures under ./figs/<func_name>_parity.png and _residuals.png
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.
    n_models = len(preds)  # Number of models running
    # Parity plots
    plt.figure(figsize=(4*n_models, 4))  # New figure for parity with 1×n_models subplots
    for i, (model_name, yhat) in enumerate(preds.items(), start=1):  # Iterate over model predictions.
        plt.subplot(1, n_models, i)  # Select subplot i.
        plt.scatter(ytrue, yhat, s=12, alpha=0.7)  # Scatter y_true vs y_pred.
        mn = min(np.min(ytrue), np.min(yhat))  # Lower bound for y=x line.
        mx = max(np.max(ytrue), np.max(yhat))  # Upper bound for y=x line.
        plt.plot([mn, mx], [mn, mx], linestyle="--")  # Reference line y = x.
        plt.xlabel("y_true (high-fid)")  # X-axis label.
        plt.ylabel("y_pred")  # Y-axis label.
        plt.title(model_name)  # Subplot title with model name.
    plt.suptitle(f"{func_name}: Parity (Pred vs Target)")  # Figure title for function.
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle.
    plt.savefig(Path(out_dir) / f"{func_name.replace(' ','_')}_parity.png", dpi=300)  # Save parity figure.
    # plt.show()  # Optional interactive display (commented for scripts).

    # Residual histograms
    plt.figure(figsize=(4*n_models, 4))  # New figure for residuals.
    for i, (model_name, yhat) in enumerate(preds.items(), start=1):  # Iterate over predictions again.
        plt.subplot(1, n_models, i)  # Subplot i.
        resid = (yhat - ytrue).ravel()  # Residuals flattened.
        plt.hist(resid, bins=30, alpha=0.9)  # Histogram of residuals.
        plt.xlabel("Residual (y_pred - y_true)")  # X-axis label.
        plt.ylabel("Count")  # Y-axis label.
        plt.title(model_name)  # Subplot title.
    plt.suptitle(f"{func_name}: Residuals")  # Figure-level title.
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Improve layout.
    plt.savefig(Path(out_dir) / f"{func_name.replace(' ','_')}_residuals.png", dpi=150)  # Save residuals figure.
    # plt.show()  # Optional display.

def plot_branin_training_inputs(Xl: np.ndarray, Xh: np.ndarray, out_dir="figs"):
    """
    Quick look at training input locations for Branin (2D only).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.
    plt.figure(figsize=(5,5))  # New square figure.
    plt.scatter(Xl[:,0], Xl[:,1], s=10, alpha=0.6, label="Low-fid X")  # Plot low-fidelity input points.
    plt.scatter(Xh[:,0], Xh[:,1], s=18, alpha=0.9, marker="x", label="High-fid X")  # Plot high-fidelity input points.
    plt.xlabel("x1")  # X label.
    plt.ylabel("x2")  # Y label.
    plt.title("Branin training inputs: low vs high")  # Title.
    plt.legend()  # Show legend distinguishing low/high sets.
    plt.tight_layout()  # Compact layout.
    plt.savefig(Path(out_dir) / "Branin_training_inputs.png", dpi=150)  # Save figure.
    # plt.show()  # Optional display.

def plot_loss_curves(loss_dict, title, out_dir="figs"):
    """
    loss_dict: {label: loss_history}
      where loss_history is a list or 1D array
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for label, losses in loss_dict.items():
        if losses is None:
            continue
        plt.plot(losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{title.replace(' ', '_')}_loss.png", dpi=150)
    plt.close()


# -----------------------
# Machine learning helpers
# -----------------------

class MLP(nn.Module):
    """Generic fully-connected MLP with ReLU activations."""
    def __init__(self, in_dim, hidden=(64,64), out_dim=1):
        super().__init__()  # Initialize base nn.Module.
        layers = []  # List to accumulate Linear/ReLU layers.
        d = in_dim  # Current input dimensionality.
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]  # Add a dense layer and ReLU.
            d = h  # Update current dimension.
        layers += [nn.Linear(d, out_dim)]  # Final linear layer to out_dim.
        self.net = nn.Sequential(*layers)  # Wrap as a sequential model.

    def forward(self, x):
        return self.net(x)  # Forward pass through the MLP.


def train_torch_regressor(model, X, y, lr=1e-3, epochs=500, weight_decay=0, verbose=False,
                          device: torch.device = None):
    """Train a PyTorch regression model with Adam on MSE loss.

    Args:
        device: torch.device to use. Defaults to get_device() (GPU if available, else CPU).
                Pass torch.device('cpu') to force CPU regardless of GPU availability.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loss_history = []

    for ep in trange(epochs, disable=not verbose):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    return model, loss_history

class MLP_lin(nn.Module):
    """Generic fully-connected MLP with ReLU activations."""
    def __init__(self, in_dim, hidden=(64,64), out_dim=1):
        super().__init__()  # Initialize base nn.Module.
        layers = []  # List to accumulate Linear/ReLU layers.
        d = in_dim  # Current input dimensionality.
        for h in hidden:
            layers += [nn.Linear(d, h)]  # Add a linear layer only
            d = h  # Update current dimension.
        layers += [nn.Linear(d, out_dim)]  # Final linear layer to out_dim.
        self.net = nn.Sequential(*layers)  # Wrap as a sequential model.
    def forward(self, x):
        return self.net(x)  # Forward pass through the MLP.
    
