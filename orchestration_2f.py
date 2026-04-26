#orchastration_2f.py
#Functions necessary to orchastrate all of the two fidelity methods with all bi-fidelity benchmarks imported from mf2

#Imports
import numpy as np
from typing import Callable, Dict  # Type hints for readability and tooling.
import time

# --- Benchmarks (mf2) ---
import mf2 # Benchmark functions providing low/high fidelity variants.

# --- Traditional ML ---
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel

# --- Helpers ---
from helpers_2f import  uniform_samples, r2_score, Dataset, make_dataset, parity_and_residual_plots, plot_branin_training_inputs, plot_loss_curves

# --- Methods ---
from twofid_methods.GPmimic_2f import GPmimic
from twofid_methods.MFGP_2f import MFGP
from twofid_methods.MFNN_delta_2f import MFNN_Delta
from twofid_methods.MFNN_flag_2f import MFNN_Flag
from twofid_methods.MFNN_intermediate_2f import MFNN_Intermediate
from twofid_methods.MFNN_threestep_2f import MFNN_3step
from twofid_methods.MFNN_twostep_2f import MFNN_2step

# -----------------------
# Define hyperparameters
# -----------------------
# MFNN-Delta
epochs_delta = 2000
lr_delta = 1e-3
hidden_delta = (64, 64, 64, 64)
wd_delta = 0.0

# MFNN-Flag
epochs_flag = 2000
lr_flag = 1e-3
hidden_flag = (128, 128, 128, 128)
wd_flag = 0.0

# MFNN-TwoStep
epochs_two_step = 2000
lr_two_step = 1e-3
hidden_two_step = (64, 64, 64, 64)
wd_two_step = 0.0

# MFNN-ThreeStep
epochs_three_step = 2000
lr_three_step = 1e-3
hidden_low_three_step = (128, 128, 128, 128)
hidden_lin_three_step = (128,)
hidden_high_three_step = (128,)
wd_three_step = 0.0

# GPmimic
epochs_gpmimic = 2000
lr_gpmimic = 1e-3
hidden_gpmimic = (128, 128, 128, 128)
alpha_gpmimic = 0.05
lam_gpmimic = 0.0005

# MFNN-Intermediate
epochs_intermediate = 2000
lr_intermediate = 1e-3
hidden_intermediate = (128, 128, 128)
hf_hidden_intermediate = (128,)
wd_intermediate = 0.0
alpha_intermediate = 0.05
lam_intermediate = 0.0005

# -----------------------
# Orchestration
# -----------------------

def evaluate_on_test(dataset: Dataset, model_name: str, model, seed=123, test_n=200):
    """Create a test set over bounds, get high-fid targets, predict with model, and compute RMSE."""
    # Build a common test set over domain to assess high-fidelity error
    Xtest = uniform_samples(dataset.bounds, test_n, seed=seed)  # Sample test inputs.
    ytrue = mf_high_eval(dataset, Xtest)  # Compute true high-fidelity outputs on test.

    if model_name == "MF-GP":
        ypred = model.predict(Xtest)  # Use GP model's predict.
    elif model_name in ("MF-NN-Delta", "MFNN-Flag", "MFNN-TwoStep", "MFNN-ThreeStep", "GPmimic","MFNN-Intermediate"):
        ypred = model.predict(Xtest)  # Use NN models' predict.
    else:
        raise ValueError("Unknown model name")  # Guard for typos/unsupported.

    rmse = np.sqrt(mean_squared_error(ytrue, ypred))  # Compute RMSE on high-fidelity targets.
    return rmse, Xtest, ytrue, ypred  # Return metrics and arrays for plotting.

def mf_high_eval(dataset: Dataset, X: np.ndarray):
    """Route to the correct mf2 high-fidelity function based on dataset bounds."""
   # Select by domain to call the correct mf2 high-fidelity function
    D = dataset.bounds.shape[0]  # Dimensionality inferred from bounds rows.
    if D == 2 and np.allclose(dataset.bounds, np.array([[-5.0,10.0],[0.0,15.0]])):
        return mf2.branin.high(X).reshape(-1,1)  # Branin high-fidelity.
    if D == 1 and np.allclose(dataset.bounds, np.array([[0.0,1.0]])):
        return mf2.forrester.high(X).reshape(-1,1)  # Forrester high-fidelity.
    if D == 6 and np.allclose(dataset.bounds, np.array([[0.0,1.0]] * 6, dtype=float)):
        return mf2.hartmann6.high(X).reshape(-1,1)  # <-- using hartmann6 as requested
    if D == 2 and np.allclose(dataset.bounds, np.array([[-10.0, 10.0], [-10.0, 10.0]])):
        return mf2.booth.high(X).reshape(-1,1)  # Booth high-fidelity.
    if D == 4 and np.allclose(dataset.bounds, np.array([
        [1e-08, 1.0],
        [0.0,   1.0],
        [0.0,   1.0],
        [0.0,   1.0]
    ], dtype=float)):
        return mf2.park91a.high(X).reshape(-1,1)
    if D == 8 and np.allclose(dataset.bounds, np.array([
        [0.05,   0.15],
        [100.0,  50000.0],
        [63070., 115600.],
        [990.,   1110.],
        [63.1,   116.0],
        [700.,   820.],
        [1120.,  1680.],
        [9855.,  12045.]
    ], dtype=float)):
        return mf2.borehole.high(X).reshape(-1,1)


    raise ValueError("Unknown dataset bounds; cannot pick high-fidelity function.")  # If bounds don’t match known sets.

def run_for_function(name: str,
                     low_f: Callable, high_f: Callable, bounds: np.ndarray,
                     n_low=200, n_high=40, seed=0) -> Dict[str, float]:
    """Train all three models on a given benchmark, evaluate, plot, and return RMSEs."""
    print(f"\n=== {name} ===")  # Section header for this benchmark.

    data = make_dataset(low_f, high_f, bounds, n_low=n_low, n_high=n_high, seed=seed)  # Generate paired datasets.
    times = {}

    # ---- Train models ----
    # MF-GP
    D = bounds.shape[0]  # Input dimensionality to set kernel length_scale shapes.

    # Updated kernels with looser bounds and non-trivial noise floors to reduce bound/convergence warnings.
    ker_low = (
        C(1.0, (1e-5, 1e5))
        * Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )  # Low GP kernel.
    ker_res = (
        C(1.0, (1e-5, 1e5))
        * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )  # Residual GP kernel.

    mf_gp = MFGP(kernel_low=ker_low, kernel_res=ker_res)  # Create MF-GP model.
    
    start = time.perf_counter()
    mf_gp.fit(data.Xl, data.yl, data.Xh, data.yh)  # Fit MF-GP on datasets.
    times["MF-GP"] = time.perf_counter() - start


    rmse_gp, Xtest, ytrue, yhat_gp = evaluate_on_test(data, "MF-GP", mf_gp)  # Evaluate MF-GP and collect predictions.

    #Two-step 
    ts_delta = MFNN_2step(
    x_dim=D,
    hidden=hidden_two_step,
    lr=lr_two_step,
    epochs=epochs_two_step,
    wd=wd_two_step,
    verbose=False
    )
    start = time.perf_counter()
    ts_delta.fit(data.Xl, data.yl, data.Xh, data.yh)
    times["MFNN-TwoStep"] = time.perf_counter() - start


    losses = {}
    if hasattr(ts_delta, "low_loss_history"):
        losses["low"] = ts_delta.low_loss_history
    if hasattr(ts_delta, "delta_loss_history"):
        losses["high"] = ts_delta.delta_loss_history

    if losses:
        plot_loss_curves(
            losses,
            title=f"{name}_MFNN_2step",
            out_dir="figs"
        )


    rmse_ts_delta, _, _, yhat_ts_delta = evaluate_on_test(data, "MFNN-TwoStep", ts_delta)

    #Three-step
    three_step = MFNN_3step(
    x_dim=D,
    hidden_low=hidden_low_three_step,
    hidden_lin=hidden_lin_three_step,
    hidden_high=hidden_high_three_step,
    lr=lr_three_step,
    epochs=epochs_three_step,
    wd=wd_three_step,
    verbose=False
    )
   
    start = time.perf_counter()
    three_step.fit(data.Xl, data.yl, data.Xh, data.yh)
    times["MFNN-ThreeStep"] = time.perf_counter() - start


    losses = {}
    if hasattr(three_step, "low_loss_history"):
        losses["low"] = three_step.low_loss_history
    if hasattr(three_step, "lin_loss_history"):
        losses["lin"] = three_step.lin_loss_history
    if hasattr(three_step, "high_loss_history"):
        losses["high"] = three_step.high_loss_history

    if losses:
        plot_loss_curves(
            losses,
            title=f"{name}_MFNN_3step",
            out_dir="figs"
        )

    rmse_three_step, _, _, yhat_three_step = evaluate_on_test(data, "MFNN-ThreeStep", three_step)

    #GPmimic
    gpm = GPmimic(
    x_dim=D,
    hidden=hidden_gpmimic,
    lr=lr_gpmimic,
    epochs=epochs_gpmimic,
    alpha=alpha_gpmimic,
    lam=lam_gpmimic,
    verbose=False
    )
    
    start = time.perf_counter()
    gpm.fit(data.Xl, data.yl, data.Xh, data.yh)
    times["GPmimic"] = time.perf_counter() - start

    if hasattr(gpm, "loss_history"):
        plot_loss_curves(
            {"total": gpm.loss_history},
            title=f"{name}_GPmimic",
            out_dir="figs"
        )

    rmse_gpm, _, _, yhat_gpm = evaluate_on_test(data, "GPmimic", gpm)


    # MFNN_Delta
    mfnn_delta = MFNN_Delta(
    x_dim=D,
    hidden=hidden_delta,
    lr=lr_delta,
    epochs=epochs_delta,
    wd=wd_delta,
    verbose=False
    )
    start = time.perf_counter()
    mfnn_delta.fit(data.Xl, data.yl, data.Xh, data.yh)
    times["MF-NN-Delta"] = time.perf_counter() - start


    losses = {}
    if hasattr(mfnn_delta, "low_loss_history"):
        losses["low"] = mfnn_delta.low_loss_history
    if hasattr(mfnn_delta, "delta_loss_history"):
        losses["delta"] = mfnn_delta.delta_loss_history

    if losses:
        plot_loss_curves(
            losses,
            title=f"{name}_MFNN_Delta",
            out_dir="figs"
        )
    
    rmse_delta, _, _, yhat_delta = evaluate_on_test(data, "MF-NN-Delta", mfnn_delta)  # Evaluate delta NN.

    # Fidelity-flag NN
    mfnn_flag = MFNN_Flag(
    x_dim=D,
    hidden=hidden_flag,
    lr=lr_flag,
    epochs=epochs_flag,
    wd=wd_flag,
    verbose=False
    )
    # show_model_architecture(mfnn_flag, input_dim=D, name="MFNN-Flag")
    start = time.perf_counter()
    mfnn_flag.fit(data.Xl, data.yl, data.Xh, data.yh)
    times["MFNN-Flag"] = time.perf_counter() - start


    if hasattr(mfnn_flag, "loss_history"):
        plot_loss_curves(
            {"total": mfnn_flag.loss_history},
            title=f"{name}_MFNN_Flag",
            out_dir="figs"
        )

    rmse_flag, _, _, yhat_flag = evaluate_on_test(data, "MFNN-Flag", mfnn_flag)  # Evaluate flag NN.

    #Intermediate
    mfnn_inter = MFNN_Intermediate(
    x_dim=D,
    hidden=hidden_intermediate,
    hf_hidden=hf_hidden_intermediate,
    lr=lr_intermediate,
    epochs=epochs_intermediate,
    wd=wd_intermediate,
    alpha=alpha_intermediate,
    lam=lam_intermediate,
    verbose=False
    )

    start = time.perf_counter()
    mfnn_inter.fit(data.Xl, data.yl, data.Xh, data.yh)
    times["MFNN-Intermediate"] = time.perf_counter() - start


    if hasattr(mfnn_inter, "loss_history"):
        plot_loss_curves(
            {"total": mfnn_inter.loss_history},
            title=f"{name}_MFNN_Intermediate",
            out_dir="figs"
        )

    rmse_inter, _, _, yhat_inter = evaluate_on_test(data, "MFNN-Intermediate", mfnn_inter)

    # Visualizations for this function
    preds = {"MF-GP": yhat_gp,"GPmimic": yhat_gpm, "MFNN-Flag": yhat_flag, "MF-NN-Delta": yhat_delta, "MFNN-TwoStep": yhat_ts_delta, "MFNN-ThreeStep": yhat_three_step, "MFNN-Intermediate": yhat_inter} # Bundle predictions for plotting.
    parity_and_residual_plots(name, ytrue, preds, out_dir="figs")  # Save parity and residual plots.

    #For Branin (2D), also visualize training inputs
    if "Branin" in name and data.Xl.shape[1] == 2:  # Only for 2D case.
        plot_branin_training_inputs(data.Xl, data.Xh, out_dir="figs")  # Save scatter of input locations.

    r2_gp = r2_score(ytrue, yhat_gp)
    r2_ts = r2_score(ytrue, yhat_ts_delta)
    r2_delta = r2_score(ytrue, yhat_delta)
    r2_3s = r2_score(ytrue,yhat_three_step)
    r2_flag = r2_score(ytrue, yhat_flag)
    r2_gpm = r2_score(ytrue, yhat_gpm)
    r2_inter = r2_score(ytrue, yhat_inter)

    results = {
        "MF-GP": {"RMSE": rmse_gp, "R2": r2_gp, "Time_sec": times["MF-GP"]},
        "MF-NN-Delta": {"RMSE": rmse_delta, "R2": r2_delta, "Time_sec": times["MF-NN-Delta"]},
        "MFNN-TwoStep": {"RMSE": rmse_ts_delta, "R2": r2_ts, "Time_sec": times["MFNN-TwoStep"]},
        "MFNN-ThreeStep": {"RMSE": rmse_three_step, "R2": r2_3s, "Time_sec": times["MFNN-ThreeStep"]},
        "MFNN-Flag": {"RMSE": rmse_flag, "R2": r2_flag, "Time_sec": times["MFNN-Flag"]},
        "GPmimic": {"RMSE": rmse_gpm, "R2": r2_gpm, "Time_sec": times["GPmimic"]},
        "MFNN-Intermediate": {"RMSE": rmse_inter, "R2": r2_inter, "Time_sec": times["MFNN-Intermediate"]}
   }

    print(f"\n=== {name} ===")
    for model, vals in results.items():
        print(f"{model:12s} -> RMSE: {vals['RMSE']:.4f}, R²: {vals['R2']:.4f}")

    return results  # Return RMSE mapping to caller.