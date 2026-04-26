"""
02_run_onc.py
Run all 2f and 3f methods on ONC data using cost-matched budgets from Table 4 of the paper.
Hyperparams read from configs/onc_config.yaml (populated by 01_tune_onc.py).
Final runs use 2000 epochs (paper Section 3).

Usage:
    python scripts/02_run_onc.py
    python scripts/02_run_onc.py --fidelity 2f
    
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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
# Cost-matched budgets from Table 4 in paper
# Each entry: (n_lf, n_mf, n_hf, total_budget)
# -----------------------
BUDGETS_LF_HF = [
    (200,  0,  25, 300),
    (400,  0,  50, 600),
    (800,  0, 100, 1200),
    (1000, 0, 200, 1800),
]
BUDGETS_LF_MF = [
    (200, 50,  0, 300),
    (400, 100, 0, 600),
    (800, 200, 0, 1200),
    (1000, 400, 0, 1800),
]
BUDGETS_MF_HF = [
    (0, 100,  25, 300),
    (0, 200,  50, 600),
    (0, 400, 100, 1200),
    (0, 500, 200, 1800),
]
BUDGETS_3F = [
    (150,  50, 12, 298),
    (300, 100, 25, 600),
    (600, 200, 50, 1200),
    (1000, 200, 100, 1800),
]

FINAL_EPOCHS = 2000  # paper uses 2000 epochs for final runs


def _metrics(ytrue, ypred, time_sec):
    rmse = float(np.sqrt(mean_squared_error(ytrue, ypred)))
    r2   = float(r2_score(ytrue, ypred))
    return {"RMSE": rmse, "R2": r2, "Time_sec": time_sec}


def run_2f_models(dataset, Xtest, ytest, hp):
    D = dataset.Xl.shape[1]
    results = {}
    h = hp["twofid"]

    # MF-GP (no epochs, uses sklearn)
    ker_low = (C(1.0, (1e-5, 1e5))
               * Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3), nu=2.5)
               + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
    ker_res = (C(1.0, (1e-5, 1e5))
               * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3))
               + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
    gp = MFGP(kernel_low=ker_low, kernel_res=ker_res)
    t0 = time.perf_counter()
    gp.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    results["MF-GP"] = _metrics(ytest, gp.predict(Xtest), time.perf_counter() - t0)

    # GPmimic
    g = h["gpmimic"]
    m = GPmimic(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                epochs=FINAL_EPOCHS, alpha=g["alpha"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    results["GPmimic"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    # MFNN-Delta
    g = h["delta"]
    m = MFNN_Delta(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                   epochs=FINAL_EPOCHS, wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    results["MF-NN-Delta"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    # MFNN-Flag
    g = h["flag"]
    m = MFNN_Flag(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                  epochs=FINAL_EPOCHS, wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    results["MFNN-Flag"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    # MFNN-Intermediate
    g = h["intermediate"]
    m = MFNN_Intermediate(x_dim=D, hidden=tuple(g["hidden"]),
                          hf_hidden=tuple(g["hf_hidden"]),
                          lr=g["lr"], epochs=FINAL_EPOCHS, wd=g["wd"],
                          alpha=g["alpha"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
    results["MFNN-Intermediate"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    return results


def run_3f_models(dataset, Xtest, ytest, hp):
    D = dataset.Xl.shape[1]
    results = {}
    h = hp["threefid"]

    # GPmimic3f
    g = h["gpmimic3f"]
    m = GPmimic3f(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                  epochs=FINAL_EPOCHS, w_h=g["w_h"], w_m=g["w_m"],
                  w_l=g["w_l"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xm, dataset.ym, dataset.Xh, dataset.yh)
    results["GPmimic3f"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    # MFNN-Flag3f
    g = h["flag3f"]
    m = MFNN_Flag3f(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                    epochs=FINAL_EPOCHS, wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xm, dataset.ym, dataset.Xh, dataset.yh)
    results["MFNN-Flag3f"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    # MFNN-Intermediate3f
    g = h["intermediate3f"]
    m = MFNN_Intermediate3f(x_dim=D, hidden=tuple(g["hidden"]),
                             lr=g["lr"], epochs=FINAL_EPOCHS, wd=g["wd"],
                             w_hf=g["w_hf"], w_mf=g["w_mf"], w_lf=g["w_lf"],
                             lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(dataset.Xl, dataset.yl, dataset.Xm, dataset.ym, dataset.Xh, dataset.yh)
    results["MFNN-Intermediate3f"] = _metrics(ytest, m.predict(Xtest), time.perf_counter() - t0)

    return results


def sweep_onc(cfg, data, fidelity="2f"):
    """Run cost-matched budget sweep matching Table 4 of paper."""
    cols = cfg["columns"]
    rows = []

    input_groups = {
        "AllInputs":   cols["x_all"],
        "Temp":        cols["x_temp"],        # dominant for ONC
        "Temp_HTC":    cols["x_temp_htc"],    # dominant for Tafter
        "Others":      cols["x_others"],      # non-dominant
    }
    outputs = {
        "ONC":    cols["y_onc"],
        "Tafter": cols["y_tafter"],
    }

    for out_name, y_col in outputs.items():
        for inp_name, x_cols in input_groups.items():

            if fidelity == "2f":
                # LF+HF
                for n_lf, _, n_hf, budget in BUDGETS_LF_HF:
                    label = f"{inp_name}_to_{out_name}_LF{n_lf}_HF{n_hf}"
                    print(f"  {label}", flush=True)
                    try:
                        ds, (Xt, yt) = make_onc_dataset_2f_lf_hf(
                            data, x_cols, y_col, n_lf, n_hf, seed=0)
                        res = run_2f_models(ds, Xt, yt, cfg)
                        for model, vals in res.items():
                            rows.append({"Case": label, "Combo": "LFHF",
                                "Fidelity": "2F", "Model": model,
                                "Output": out_name, "Inputs": inp_name,
                                "LF": n_lf, "HF": n_hf, "TotalBudget": budget,
                                **vals})
                    except Exception as e:
                        print(f"    ERROR: {e}", flush=True)

                # LF+MF
                for n_lf, n_mf, _, budget in BUDGETS_LF_MF:
                    label = f"{inp_name}_to_{out_name}_LF{n_lf}_MF{n_mf}"
                    print(f"  {label}", flush=True)
                    try:
                        ds, (Xt, yt) = make_onc_dataset_2f_lf_mf(
                            data, x_cols, y_col, n_lf, n_mf, seed=0)
                        res = run_2f_models(ds, Xt, yt, cfg)
                        for model, vals in res.items():
                            rows.append({"Case": label, "Combo": "LFMF",
                                "Fidelity": "2F", "Model": model,
                                "Output": out_name, "Inputs": inp_name,
                                "LF": n_lf, "MF": n_mf, "TotalBudget": budget,
                                **vals})
                    except Exception as e:
                        print(f"    ERROR: {e}", flush=True)

                # MF+HF
                for _, n_mf, n_hf, budget in BUDGETS_MF_HF:
                    label = f"{inp_name}_to_{out_name}_MF{n_mf}_HF{n_hf}"
                    print(f"  {label}", flush=True)
                    try:
                        ds, (Xt, yt) = make_onc_dataset_2f_mf_hf(
                            data, x_cols, y_col, n_mf, n_hf, seed=0)
                        res = run_2f_models(ds, Xt, yt, cfg)
                        for model, vals in res.items():
                            rows.append({"Case": label, "Combo": "MFHF",
                                "Fidelity": "2F", "Model": model,
                                "Output": out_name, "Inputs": inp_name,
                                "MF": n_mf, "HF": n_hf, "TotalBudget": budget,
                                **vals})
                    except Exception as e:
                        print(f"    ERROR: {e}", flush=True)

            else:  # 3f
                for n_lf, n_mf, n_hf, budget in BUDGETS_3F:
                    label = f"{inp_name}_to_{out_name}_LF{n_lf}_MF{n_mf}_HF{n_hf}"
                    print(f"  {label}", flush=True)
                    try:
                        ds, (Xt, yt) = make_onc_dataset_3f(
                            data, x_cols, y_col, n_lf, n_mf, n_hf, seed=0)
                        res = run_3f_models(ds, Xt, yt, cfg)
                        for model, vals in res.items():
                            rows.append({"Case": label, "Combo": "LFMFHF",
                                "Fidelity": "3F", "Model": model,
                                "Output": out_name, "Inputs": inp_name,
                                "LF": n_lf, "MF": n_mf, "HF": n_hf,
                                "TotalBudget": budget, **vals})
                    except Exception as e:
                        print(f"    ERROR: {e}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
