"""
02_run_benchmarks.py
Run all 2f and 3f methods across benchmarks and save results.
Hyperparams read from configs/benchmarks_config.yaml (populated by 01_tune_benchmarks.py).
3f benchmarks: Forrester 1D, Rastrigin D=2,5,10, Rosenbrock D=2,5,10.

Usage:
    python scripts/02_run_benchmarks.py
    python scripts/02_run_benchmarks.py --fidelity 2f
    python scripts/02_run_benchmarks.py --fidelity 3f
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

from helpers_2f import (
    limit_cpu_threads, r2_score, make_dataset,
    uniform_samples,
)

from benchmarks.twofid_benchmarks import (
    branin_funcs, forrester_funcs, hartmann6_funcs,
    booth_funcs, park91a_funcs, borehole_funcs,
)
from benchmarks.forrester_3f  import forrester_3f_funcs
from benchmarks.rastrigin_3f  import rastrigin_3f_funcs
from benchmarks.rosenbrock_3f import rosenbrock_3f_funcs

from helpers_3f import make_dataset as make_dataset_3f

from twofid_methods.GPmimic_2f import GPmimic
from twofid_methods.MFGP_2f import MFGP
from twofid_methods.MFNN_delta_2f import MFNN_Delta
from twofid_methods.MFNN_flag_2f import MFNN_Flag
from twofid_methods.MFNN_intermediate_2f import MFNN_Intermediate
from twofid_methods.MFNN_threestep_2f import MFNN_3step
from twofid_methods.MFNN_twostep_2f import MFNN_2step

from threefid_methods.GPmimic_3f import GPmimic3f
from threefid_methods.MFNN_flag_3f import MFNN_Flag3f
from threefid_methods.MFNN_intermediate_3f import MFNN_Intermediate3f

FINAL_EPOCHS = 2000  # tuning uses 500, final runs use 2000


def _metrics(ytrue, ypred, time_sec):
    rmse = float(np.sqrt(mean_squared_error(ytrue, ypred)))
    r2   = float(r2_score(ytrue, ypred))
    return {"RMSE": rmse, "R2": r2, "Time_sec": time_sec}


def evaluate_2f(model, data, test_n=200, seed=123):
    """Sample test set over bounds, evaluate model."""
    Xtest = uniform_samples(data.bounds, test_n, seed=seed)
    ytest = model.predict(Xtest)  # placeholder — overridden below
    return Xtest


def run_2f_models(data, cfg):
    """Train all 2f models on a benchmark dataset."""
    D = data.Xl.shape[1]
    h = cfg["twofid"]
    results = {}

    # Test set
    Xtest = uniform_samples(data.bounds, 200, seed=123)

    # MF-GP
    ker_low = (C(1.0, (1e-5, 1e5))
               * Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3), nu=2.5)
               + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
    ker_res = (C(1.0, (1e-5, 1e5))
               * RBF(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3))
               + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)))
    gp = MFGP(kernel_low=ker_low, kernel_res=ker_res)
    t0 = time.perf_counter()
    gp.fit(data.Xl, data.yl, data.Xh, data.yh)
    t = time.perf_counter() - t0
    # For benchmarks we don't have a separate HF eval fn here —
    # use the existing orchestration_2f evaluate logic via direct predict
    results["MF-GP"] = {"model": gp, "time": t}

    # GPmimic
    g = h["gpmimic"]
    m = GPmimic(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                epochs=FINAL_EPOCHS, alpha=g["alpha"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xh, data.yh)
    results["GPmimic"] = {"model": m, "time": time.perf_counter() - t0}

    # MFNN-Delta
    g = h["delta"]
    m = MFNN_Delta(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                   epochs=FINAL_EPOCHS, wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xh, data.yh)
    results["MF-NN-Delta"] = {"model": m, "time": time.perf_counter() - t0}

    # MFNN-Flag
    g = h["flag"]
    m = MFNN_Flag(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                  epochs=FINAL_EPOCHS, wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xh, data.yh)
    results["MFNN-Flag"] = {"model": m, "time": time.perf_counter() - t0}

    # MFNN-Intermediate
    g = h["intermediate"]
    m = MFNN_Intermediate(x_dim=D, hidden=tuple(g["hidden"]),
                          hf_hidden=tuple(g["hf_hidden"]), lr=g["lr"],
                          epochs=FINAL_EPOCHS, wd=g["wd"],
                          alpha=g["alpha"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xh, data.yh)
    results["MFNN-Intermediate"] = {"model": m, "time": time.perf_counter() - t0}

    return results, Xtest


def run_3f_models(data, cfg):
    """Train all 3f models on a benchmark dataset."""
    D = data.Xl.shape[1]
    h = cfg["threefid"]
    results = {}

    Xtest = uniform_samples(data.bounds, 200, seed=123)

    # GPmimic3f
    g = h["gpmimic3f"]
    m = GPmimic3f(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                  epochs=FINAL_EPOCHS, w_h=g["w_h"], w_m=g["w_m"],
                  w_l=g["w_l"], lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xm, data.ym, data.Xh, data.yh)
    results["GPmimic3f"] = {"model": m, "time": time.perf_counter() - t0}

    # MFNN-Flag3f
    g = h["flag3f"]
    m = MFNN_Flag3f(x_dim=D, hidden=tuple(g["hidden"]), lr=g["lr"],
                    epochs=FINAL_EPOCHS, wd=g["wd"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xm, data.ym, data.Xh, data.yh)
    results["MFNN-Flag3f"] = {"model": m, "time": time.perf_counter() - t0}

    # MFNN-Intermediate3f
    g = h["intermediate3f"]
    m = MFNN_Intermediate3f(x_dim=D, hidden=tuple(g["hidden"]),
                             lr=g["lr"], epochs=FINAL_EPOCHS, wd=g["wd"],
                             w_hf=g["w_hf"], w_mf=g["w_mf"], w_lf=g["w_lf"],
                             lam=g["lam"], verbose=False)
    t0 = time.perf_counter()
    m.fit(data.Xl, data.yl, data.Xm, data.ym, data.Xh, data.yh)
    results["MFNN-Intermediate3f"] = {"model": m, "time": time.perf_counter() - t0}

    return results, Xtest


def run_2f_benchmarks(cfg):
    run_cfg = cfg["run"]["twofid"]
    configs = [
        ("Branin (2D)",)    + branin_funcs(),
        ("Forrester (1D)",) + forrester_funcs(),
        ("Hartmann6 (6D)",) + hartmann6_funcs(),
        ("Booth (2D)",)     + booth_funcs(),
        ("Park91A (4D)",)   + park91a_funcs(),
        ("Borehole (8D)",)  + borehole_funcs(),
    ]

    rows = []
    for i, (name, low, high, bounds) in enumerate(configs):
        print(f"\n  {name}", flush=True)
        is_hartmann = "Hartmann6" in name
        n_low  = run_cfg["n_low_hartmann6"]  if is_hartmann else run_cfg["n_low"]
        n_high = run_cfg["n_high_hartmann6"] if is_hartmann else run_cfg["n_high"]
        seed   = run_cfg["seed_offset"] + i

        data = make_dataset(low, high, bounds, n_low=n_low, n_high=n_high, seed=seed)
        models, Xtest = run_2f_models(data, cfg)
        ytrue = high(Xtest).reshape(-1, 1)

        for model_name, info in models.items():
            ypred = info["model"].predict(Xtest)
            rows.append({
                "Benchmark": name, "Model": model_name, "Fidelity": "2F",
                **_metrics(ytrue, ypred, info["time"])
            })

    return pd.DataFrame(rows)


def run_3f_benchmarks(cfg):
    run_cfg = cfg["run"]["threefid"]
    rows = []

    # Forrester 1D
    print("\n  Forrester-3f (1D)", flush=True)
    low, med, high, bounds = forrester_3f_funcs()
    data = make_dataset_3f(low, med, high, bounds,
                           n_low=run_cfg["n_low"],
                           n_medium=run_cfg["n_medium"],
                           n_high=run_cfg["n_high"], seed=0)
    models, Xtest = run_3f_models(data, cfg)
    ytrue = high(Xtest).reshape(-1, 1)
    for model_name, info in models.items():
        rows.append({"Benchmark": "Forrester-3f (1D)", "Model": model_name,
                     "Fidelity": "3F",
                     **_metrics(ytrue, info["model"].predict(Xtest), info["time"])})

    # Rosenbrock D=2,5,10
    for D, seed in zip([2, 5, 10], [1, 2, 3]):
        name = f"Rosenbrock-3f ({D}D)"
        print(f"\n  {name}", flush=True)
        low, med, high, bounds = rosenbrock_3f_funcs(D)
        data = make_dataset_3f(low, med, high, bounds,
                               n_low=run_cfg["n_low"],
                               n_medium=run_cfg["n_medium"],
                               n_high=run_cfg["n_high"], seed=seed)
        models, Xtest = run_3f_models(data, cfg)
        ytrue = high(Xtest).reshape(-1, 1)
        for model_name, info in models.items():
            rows.append({"Benchmark": name, "Model": model_name, "Fidelity": "3F",
                         **_metrics(ytrue, info["model"].predict(Xtest), info["time"])})

    # Rastrigin D=2,5,10
    for D, seed in zip([2, 5, 10], [4, 5, 6]):
        name = f"Rastrigin-3f ({D}D)"
        print(f"\n  {name}", flush=True)
        low, med, high, bounds = rastrigin_3f_funcs(D)
        data = make_dataset_3f(low, med, high, bounds,
                               n_low=run_cfg["n_low"],
                               n_medium=run_cfg["n_medium"],
                               n_high=run_cfg["n_high"], seed=seed)
        models, Xtest = run_3f_models(data, cfg)
        ytrue = high(Xtest).reshape(-1, 1)
        for model_name, info in models.items():
            rows.append({"Benchmark": name, "Model": model_name, "Fidelity": "3F",
                         **_metrics(ytrue, info["model"].predict(Xtest), info["time"])})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/benchmarks_config.yaml")
    parser.add_argument("--fidelity", default="both", choices=["2f", "3f", "both"])
    parser.add_argument("--out",      default="outputs/results/benchmark_results.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    limit_cpu_threads(cfg.get("cpu_threads", 4))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.fidelity in ("2f", "both"):
        print("\n=== Running 2-fidelity benchmarks ===")
        frames.append(run_2f_benchmarks(cfg))
    if args.fidelity in ("3f", "both"):
        print("\n=== Running 3-fidelity benchmarks ===")
        frames.append(run_3f_benchmarks(cfg))

    df = pd.concat(frames).sort_values(["Fidelity", "Benchmark", "Model"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
