"""
02_run_benchmarks.py
Run all 2f and 3f methods across benchmarks and save results.
Converted from orchestration_2f.ipynb and orchestration_3f.ipynb.

Usage:
    python scripts/02_run_benchmarks.py
    python scripts/02_run_benchmarks.py --fidelity 2f
    python scripts/02_run_benchmarks.py --fidelity 3f
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers_2f import limit_cpu_threads
from orchestration_2f import run_for_function as run_2f
from orchestration_3f import run_for_function as run_3f

from benchmarks.twofid_benchmarks import (
    branin_funcs, forrester_funcs, hartmann6_funcs,
    booth_funcs, park91a_funcs, borehole_funcs,
)
from benchmarks.forrester_3f  import forrester_3f_funcs
from benchmarks.rastrigin_3f  import rastrigin_3f_funcs
from benchmarks.rosenbrock_3f import rosenbrock_3f_funcs


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

    summary = {}
    for i, (name, low, high, bounds) in enumerate(configs):
        is_hartmann = "Hartmann6" in name
        res = run_2f(
            name, low, high, bounds,
            n_low  = run_cfg["n_low_hartmann6"]  if is_hartmann else run_cfg["n_low"],
            n_high = run_cfg["n_high_hartmann6"] if is_hartmann else run_cfg["n_high"],
            seed   = run_cfg["seed_offset"] + i,
        )
        summary[name] = res

    rows = []
    for bench, metrics in summary.items():
        for model, vals in metrics.items():
            rows.append({
                "Benchmark": bench, "Model": model,
                "RMSE": vals["RMSE"], "R2": vals["R2"],
                "Time_sec": vals["Time_sec"], "Fidelity": "2F",
            })
    return pd.DataFrame(rows)


def run_3f_benchmarks(cfg):
    run_cfg = cfg["run"]["threefid"]
    configs = [
        ("Forrester-3f (1D)",) + forrester_3f_funcs(),
        ("Rastrigin-3f (2D)",) + rastrigin_3f_funcs(D=2),
        ("Rosenbrock-3f (2D)",) + rosenbrock_3f_funcs(D=2),
    ]

    summary = {}
    for i, (name, low, med, high, bounds) in enumerate(configs):
        res = run_3f(
            name, low, med, high, bounds,
            n_low    = run_cfg["n_low"],
            n_medium = run_cfg["n_medium"],
            n_high   = run_cfg["n_high"],
            seed     = run_cfg["seed_offset"] + i,
        )
        summary[name] = res

    rows = []
    for bench, metrics in summary.items():
        for model, vals in metrics.items():
            rows.append({
                "Benchmark": bench, "Model": model,
                "RMSE": vals["RMSE"], "R2": vals["R2"],
                "Time_sec": vals["Time_sec"], "Fidelity": "3F",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark experiments")
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

    df = pd.concat(frames).sort_values(
        ["Fidelity", "Benchmark", "Model"]
    ).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
