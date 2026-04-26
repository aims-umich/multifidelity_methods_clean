# --- Core ---
import numpy as np
from typing import Callable, Dict
import time

# --- Metrics ---
from sklearn.metrics import mean_squared_error

# --- Helpers ---
from helpers_3f import (
    make_dataset,
    parity_and_residual_plots,
    r2_score,
    plot_loss_curves
)

# --- Methods ---
from threefid_methods.GPmimic_3f import GPmimic3f
from threefid_methods.MFNN_intermediate_3f import MFNN_Intermediate3f
from threefid_methods.MFNN_flag_3f import MFNN_Flag3f

# -----------------------
# Define hyperparameters
# -----------------------

# GPmimic3f
epochs_gpmimic3f = 2000
lr_gpmimic3f = 1e-3
hidden_gpmimic3f = (128, 128, 128, 128)
w_h_gpmimic3f = 0.5
w_m_gpmimic3f = 0.3
w_l_gpmimic3f = 0.2
lam_gpmimic3f = 1e-4

# MFNN-Intermediate3f
epochs_intermediate3f = 2000
lr_intermediate3f = 1e-3
hidden_intermediate3f = (128, 128, 128, 128)
wd_intermediate3f = 0.0
w_hf_intermediate3f = 0.7
w_mf_intermediate3f = 0.2
w_lf_intermediate3f = 0.1
lam_intermediate3f = 1e-3

# MFNN-Flag3f
epochs_flag3f = 2000
lr_flag3f = 1e-3
hidden_flag3f = (128, 128, 128, 128)
wd_flag3f = 0.0

# -----------------------
# Orchestration
# -----------------------

def evaluate_on_test(high_func: Callable, bounds: np.ndarray, model, seed=123, test_n=200):
    """
    Sample a test set over bounds, evaluate high-fidelity target and model predictions.
    Returns: rmse, Xtest, y_true, y_pred
    """
    bounds = np.asarray(bounds)
    D = bounds.shape[0]

    rng_local = np.random.default_rng(seed)
    X_unit = rng_local.random((test_n, D))
    Xtest = bounds[:, 0] + X_unit * (bounds[:, 1] - bounds[:, 0])

    y_true = high_func(Xtest)
    y_pred = model.predict(Xtest)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse, Xtest, y_true, y_pred


def run_for_function(name: str,
                     low_f: Callable, medium_f: Callable, high_f: Callable, bounds: np.ndarray,
                     n_low=200, n_medium=100, n_high=40, seed=0) -> Dict[str, Dict[str, float]]:

    print(f"\n=== {name} ===")

    # 1) build data
    data = make_dataset(
        low_f, medium_f, high_f, bounds,
        n_low=n_low, n_medium=n_medium, n_high=n_high,
        seed=seed
    )
    D = data.bounds.shape[0]

    # 2) GPmimic3f
    gpm3f = GPmimic3f(
        x_dim=D,
        hidden=hidden_gpmimic3f,
        lr=lr_gpmimic3f,
        epochs=epochs_gpmimic3f,
        w_h=w_h_gpmimic3f,
        w_m=w_m_gpmimic3f,
        w_l=w_l_gpmimic3f,
        lam=lam_gpmimic3f,
        verbose=False,
    )
    start = time.perf_counter()
    gpm3f.fit(data.Xl, data.yl, data.Xm, data.ym, data.Xh, data.yh)

    if hasattr(gpm3f, "loss_history"):
        plot_loss_curves(
            {"total": gpm3f.loss_history},
            title=f"{name}_GPmimic3f",
            out_dir="figs",
        )

    time_gpm3f = time.perf_counter() - start

    rmse_gpm3f, Xtest, ytrue, yhat_gpm3f = evaluate_on_test(high_f, data.bounds, gpm3f)

    # 3) MFNN-Intermediate3f
    mfnn_inter3f = MFNN_Intermediate3f(
        x_dim=D,
        hidden=hidden_intermediate3f,
        lr=lr_intermediate3f,
        epochs=epochs_intermediate3f,
        wd=wd_intermediate3f,
        w_hf=w_hf_intermediate3f,
        w_mf=w_mf_intermediate3f,
        w_lf=w_lf_intermediate3f,
        lam=lam_intermediate3f,
        verbose=False,
    )
    start = time.perf_counter()
    mfnn_inter3f.fit(data.Xl, data.yl, data.Xm, data.ym, data.Xh, data.yh)

    if hasattr(mfnn_inter3f, "loss_history"):
        plot_loss_curves(
            {"total": mfnn_inter3f.loss_history},
            title=f"{name}_MFNN_Intermediate3f",
            out_dir="figs",
        )

    time_inter3f = time.perf_counter() - start

    rmse_inter3f, _, _, yhat_inter3f = evaluate_on_test(high_f, data.bounds, mfnn_inter3f)
    
    # 4) MFNN-Flag3f
    mfnn_flag3f = MFNN_Flag3f(
        x_dim=D,
        hidden=hidden_flag3f,
        lr=lr_flag3f,
        epochs=epochs_flag3f,
        wd=wd_flag3f,
        verbose=False,
    )
    start = time.perf_counter()
    mfnn_flag3f.fit(data.Xl, data.yl, data.Xm, data.ym, data.Xh, data.yh)

    if hasattr(mfnn_flag3f, "loss_history"):
        plot_loss_curves(
            {"total": mfnn_flag3f.loss_history},
            title=f"{name}_MFNN_Flag3f",
            out_dir="figs",
        )

    time_flag3f = time.perf_counter() - start

    rmse_flag3f, _, _, yhat_flag3f = evaluate_on_test(high_f, data.bounds, mfnn_flag3f)

    # 5) plots
    preds = {
        "GPmimic3f": yhat_gpm3f,
        "MFNN-Intermediate3f": yhat_inter3f,
        "MFNN-Flag3f": yhat_flag3f,
    }
    parity_and_residual_plots(name, ytrue, preds, out_dir="figs")

    # 5) R² scores
    r2_gpm3f = r2_score(ytrue, yhat_gpm3f)
    r2_inter3f = r2_score(ytrue, yhat_inter3f)
    r2_flag3f = r2_score(ytrue, yhat_flag3f)

    results = {
        "GPmimic3f": {"RMSE": rmse_gpm3f, "R2": r2_gpm3f, "Time_sec": time_gpm3f},
        "MFNN-Intermediate3f": {"RMSE": rmse_inter3f, "R2": r2_inter3f, "Time_sec": time_inter3f},
        "MFNN-Flag3f": {"RMSE": rmse_flag3f, "R2": r2_flag3f, "Time_sec": time_flag3f},
    }

    for model, vals in results.items():
        print(f"{model:22s} -> RMSE: {vals['RMSE']:.4f}, R²: {vals['R2']:.4f}")

    return results