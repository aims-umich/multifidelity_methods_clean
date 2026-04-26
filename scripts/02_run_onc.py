"""
02_run_onc.py
Run all 2f and 3f methods on ONC real data across sample size combinations.
Converted from MF_ONC_experiments_fixed.ipynb.

Usage:
    python scripts/02_run_onc.py
    python scripts/02_run_onc.py --fidelity 2f --out outputs/results/onc_results.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers_2f import limit_cpu_threads, r2_score

from twofid_methods.GPmimic_2f import GPmimic
from twofid_methods.MFGP_2f import MFGP
from twofid_methods.MFNN_delta_2f import MFNN_Delta
from twofid_methods.MFNN_flag_2f import MFNN_Flag
from twofid_methods.MFNN_intermediate_2f import MFNN_Intermediate

from threefid_methods.GPmimic_3f import GPmimic3f
from threefid_methods.MFNN_flag_3f import MFNN_Flag3f
from threefid_methods.MFNN_intermediate_3f import MFNN_Intermediate3f

from src.datasets import (
    load_onc_data,
    make_onc_dataset_2f_lf_hf,
    make_onc_dataset_2f_lf_mf,
    make_onc_dataset_2f_mf_hf,
    make_onc_dataset_3f,
)


# -----------------------
# Model runners
# -----------------------

def run_2f_models(dataset, Xtest, ytest, hp):
    """Train all 2f models on a dataset and return results dict."""
    D = dataset.Xl.shape[1]
    results = {}
    h = hp["twofid"]

    # MF-GP
    ker_low = (C(1.0, (1e-5, 1e5))
               * Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3), nu=2.5)
               + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
    ker_res = (C(1.0, (1e-5, 1e5))
               * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3))
               + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
    gp = MFGP(kernel_low=ker_low, kernel_res=ker_res)
    t0 = time.perf_counter()
    gp.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    yhat = gp.predict(Xtest)
    results["MF-GP"] = _metrics(ytest, yhat, t)

    # GPmimic
    g = h["gpmimic"]
    gpm = GPmimic(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                  epochs=g["epochs"], alpha=g["alpha"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    gpm.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["GPmimic"] = _metrics(ytest, gpm.predict(Xtest), t)

    # MFNN-Delta
    g = h["delta"]
    mdelta = MFNN_Delta(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                        epochs=g["epochs"], wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    mdelta.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["MFNN-Delta"] = _metrics(ytest, mdelta.predict(Xtest), t)

    # MFNN-Flag
    g = h["flag"]
    mflag = MFNN_Flag(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                      epochs=g["epochs"], wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    mflag.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["MFNN-Flag"] = _metrics(ytest, mflag.predict(Xtest), t)

    # MFNN-Intermediate
    g = h["intermediate"]
    minter = MFNN_Intermediate(x_dim=D, hidden=tuple(g["hidden"]),
                               hf_hidden=tuple(g["hf_hidden"]),
                               lr=g["lr"], epochs=g["epochs"], wd=g["wd"],
                               alpha=g["alpha"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    minter.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["MFNN-Intermediate"] = _metrics(ytest, minter.predict(Xtest), t)

    return results


def run_3f_models(dataset, Xtest, ytest, hp):
    """Train all 3f models on a dataset and return results dict."""
    D = dataset.Xl.shape[1]
    results = {}
    h = hp["threefid"]

    # GPmimic3f
    g = h["gpmimic3f"]
    gpm = GPmimic3f(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                    epochs=g["epochs"], w_h=g["w_h"], w_m=g["w_m"],
                    w_l=g["w_l"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    gpm.fit(dataset.Xl, dataset.yl, dataset.Xm, dataset.ym, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["GPmimic3f"] = _metrics(ytest, gpm.predict(Xtest), t)

    # MFNN-Flag3f
    g = h["flag3f"]
    mflag = MFNN_Flag3f(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                        epochs=g["epochs"], wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    mflag.fit(dataset.Xl, dataset.yl, dataset.Xm, dataset.ym, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["MFNN-Flag3f"] = _metrics(ytest, mflag.predict(Xtest), t)

    # MFNN-Intermediate3f
    g = h["intermediate3f"]
    minter = MFNN_Intermediate3f(x_dim=D, hidden=tuple(g["hidden"]),
                                  lr=g["lr"], epochs=g["epochs"], wd=g["wd"],
                                  w_hf=g["w_hf"], w_mf=g["w_mf"], w_lf=g["w_lf"],
                                  lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    minter.fit(dataset.Xl, dataset.yl, dataset.Xm, dataset.ym, dataset.Xh, dataset.yh)
    t = time.perf_counter() - t0
    results["MFNN-Intermediate3f"] = _metrics(ytest, minter.predict(Xtest), t)

    return results


def _metrics(ytrue, ypred, time_sec):
    rmse = float(np.sqrt(mean_squared_error(ytrue, ypred)))
    r2   = float(r2_score(ytrue, ypred))
    return {"RMSE": rmse, "R2": r2, "Time_sec": time_sec}


# -----------------------
# Experiment sweep
# -----------------------

def sweep_onc(cfg, data, fidelity="2f"):
    """Run the full sample-size sweep and return a DataFrame of results."""
    exp  = cfg["experiments"]
    cols = cfg["columns"]
    rows = []

    input_groups = {
        "AllInputs":    cols["x_all"],
        "Temp":         cols["x_temp"],
        "Temp_HTC":     cols["x_temp_htc"],
    }
    outputs = {
        "ONC":    cols["y_onc"],
        "Tafter": cols["y_tafter"],
    }

    for out_name, y_col in outputs.items():
        for inp_name, x_cols in input_groups.items():
            for n_lf in exp["lf_sizes"]:
                for n_hf in exp["hf_sizes"]:
                    for seed in exp["seeds"]:
                        case = f"{inp_name}_to_{out_name}_LF{n_lf}_HF{n_hf}_seed{seed}"
                        print(f"  {case}", flush=True)

                        try:
                            if fidelity == "2f":
                                for combo, builder in [
                                    ("LF_HF", lambda: make_onc_dataset_2f_lf_hf(
                                        data, x_cols, y_col, n_lf, n_hf, seed)),
                                    ("LF_MF", lambda: make_onc_dataset_2f_lf_mf(
                                        data, x_cols, y_col, n_lf, n_hf, seed)),
                                    ("MF_HF", lambda: make_onc_dataset_2f_mf_hf(
                                        data, x_cols, y_col, n_lf, n_hf, seed)),
                                ]:
                                    dataset, (Xtest, ytest) = builder()
                                    metrics = run_2f_models(dataset, Xtest, ytest, cfg)
                                    for model, vals in metrics.items():
                                        rows.append({
                                            "Case": f"{case}_{combo}",
                                            "Combo": combo, "Fidelity": "2F",
                                            "Model": model, "Output": out_name,
                                            "Inputs": inp_name, "n_lf": n_lf, "n_hf": n_hf,
                                            "seed": seed, **vals,
                                        })
                            else:  # 3f
                                dataset, (Xtest, ytest) = make_onc_dataset_3f(
                                    data, x_cols, y_col, n_lf, n_hf, n_hf, seed)
                                metrics = run_3f_models(dataset, Xtest, ytest, cfg)
                                for model, vals in metrics.items():
                                    rows.append({
                                        "Case": case, "Fidelity": "3F",
                                        "Model": model, "Output": out_name,
                                        "Inputs": inp_name, "n_lf": n_lf, "n_hf": n_hf,
                                        "seed": seed, **vals,
                                    })
                        except Exception as e:
                            print(f"    ERROR: {e}", flush=True)

    return pd.DataFrame(rows)


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONC experiments")
    parser.add_argument("--config",   default="configs/onc_config.yaml")
    parser.add_argument("--fidelity", default="both", choices=["2f", "3f", "both"])
    parser.add_argument("--out",      default="outputs/results/onc_results.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    limit_cpu_threads(cfg.get("cpu_threads", 4))

    data = load_onc_data(ROOT, cfg)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.fidelity in ("2f", "both"):
        print("\n=== Running 2-fidelity ONC sweep ===")
        frames.append(sweep_onc(cfg, data, fidelity="2f"))
    if args.fidelity in ("3f", "both"):
        print("\n=== Running 3-fidelity ONC sweep ===")
        frames.append(sweep_onc(cfg, data, fidelity="3f"))

    df = pd.concat(frames).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
