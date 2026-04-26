"""
01_tune_benchmarks.py
Hyperparameter tuning for all 2-fidelity methods across mf2 benchmarks.
Converted from hyperparameter_tuning_2f.ipynb.

Usage:
    python scripts/01_tune_benchmarks.py
    python scripts/01_tune_benchmarks.py --out outputs/results/tuning_benchmarks.csv
"""
import argparse
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml

# --- path setup ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers_2f import limit_cpu_threads, make_dataset

from orchestration_2f import evaluate_on_test

from benchmarks.twofid_benchmarks import (
    branin_funcs, forrester_funcs, hartmann6_funcs,
    booth_funcs, park91a_funcs, borehole_funcs,
)
from twofid_methods.GPmimic_2f import GPmimic
from twofid_methods.MFGP_2f import MFGP
from twofid_methods.MFNN_delta_2f import MFNN_Delta
from twofid_methods.MFNN_flag_2f import MFNN_Flag
from twofid_methods.MFNN_intermediate_2f import MFNN_Intermediate
from twofid_methods.MFNN_threestep_2f import MFNN_3step
from twofid_methods.MFNN_twostep_2f import MFNN_2step


# -----------------------
# Hyperparameter grid
# -----------------------
NUM_LAYERS  = [2, 3, 4]
NUM_NODES   = [16, 32, 64, 128]
NUM_EPOCHS  = [500]
LRS         = [1e-4, 5e-4, 1e-3]

def make_hidden(n_layers, n_nodes):
    return tuple([n_nodes] * n_layers)


# -----------------------
# Benchmark configs
# -----------------------
TUNING_CONFIGS = [
    ("Branin (2D)",)    + branin_funcs(),
    ("Forrester (1D)",) + forrester_funcs(),
    ("Hartmann6 (6D)",) + hartmann6_funcs(),
    ("Booth (2D)",)     + booth_funcs(),
    ("Park91A (4D)",)   + park91a_funcs(),
    ("Borehole (8D)",)  + borehole_funcs(),
]

def make_tuning_dataset(name, low_f, high_f, bounds, seed):
    n_low  = 200 if "Hartmann6" not in name else 400
    n_high = 40  if "Hartmann6" not in name else 80
    return make_dataset(low_f, high_f, bounds, n_low=n_low, n_high=n_high, seed=seed)


# -----------------------
# Method specs
# -----------------------
METHODS = [
    {
        "name": "GPmimic",
        "build": lambda D, hidden, lr, epochs: GPmimic(
            x_dim=D, hidden=hidden, lr=lr, epochs=epochs,
            alpha=0.1, lam=1e-4, verbose=False),
    },
    {
        "name": "MF-NN-Delta",
        "build": lambda D, hidden, lr, epochs: MFNN_Delta(
            x_dim=D, hidden=hidden, lr=lr, epochs=epochs,
            wd=0.0, verbose=False),
    },
    {
        "name": "MFNN-Intermediate",
        "build": lambda D, hidden, lr, epochs: MFNN_Intermediate(
            x_dim=D, hidden=hidden, hf_hidden=(hidden[0],),
            lr=lr, epochs=epochs, wd=0.0,
            alpha=0.1, lam=1e-4, verbose=False),
    },
    {
        "name": "MFNN-TwoStep",
        "build": lambda D, hidden, lr, epochs: MFNN_2step(
            x_dim=D, hidden=hidden, lr=lr, epochs=epochs,
            wd=0.0, verbose=False),
    },
    {
        "name": "MFNN-ThreeStep",
        "build": lambda D, hidden, lr, epochs: MFNN_3step(
            x_dim=D, hidden_low=hidden, hidden_lin=(hidden[0],),
            hidden_high=(hidden[0],), lr=lr, epochs=epochs,
            wd=0.0, verbose=False),
    },
    {
        "name": "MFNN-Flag",
        "build": lambda D, hidden, lr, epochs: MFNN_Flag(
            x_dim=D, hidden=hidden, lr=lr, epochs=epochs,
            wd=0.0, verbose=False),
    },
]


# -----------------------
# Tuning logic
# -----------------------
def tune_all_methods(methods=METHODS, configs=TUNING_CONFIGS):
    rows = []
    best_cfg_per_method = {}

    for method_spec in methods:
        method_name = method_spec["name"]
        build_model  = method_spec["build"]

        print(f"\n{'='*40}")
        print(f"TUNING METHOD: {method_name}")
        print(f"{'='*40}", flush=True)

        best_cfg      = None
        best_mean_rmse = None

        for L, H, E, lr in product(NUM_LAYERS, NUM_NODES, NUM_EPOCHS, LRS):
            hidden = make_hidden(L, H)
            print(f"\n  {method_name}: layers={L}, nodes={H}, epochs={E}, lr={lr}", flush=True)

            total_rmse = 0.0
            count = 0

            for i, (name, low_f, high_f, bounds) in enumerate(configs):
                dataset = make_tuning_dataset(name, low_f, high_f, bounds, seed=1000 + i)
                D = dataset.Xl.shape[1]

                model = build_model(D, hidden, lr, E)
                model.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)

                rmse, _, _, _ = evaluate_on_test(dataset, method_name, model)
                rmse_val = float(np.asarray(rmse))

                rows.append({
                    "Method": method_name, "Benchmark": name,
                    "layers": L, "nodes": H, "epochs": E, "lr": lr,
                    "RMSE": rmse_val,
                })
                total_rmse += rmse_val
                count += 1

            mean_rmse = total_rmse / count
            print(f"   → mean RMSE={mean_rmse:.4f}", flush=True)

            if best_mean_rmse is None or mean_rmse < best_mean_rmse:
                best_mean_rmse = mean_rmse
                best_cfg = (L, H, E, lr)

        best_cfg_per_method[method_name] = {
            "hidden": list(make_hidden(best_cfg[0], best_cfg[1])),
            "epochs": best_cfg[2],
            "lr": best_cfg[3],
            "best_mean_rmse": best_mean_rmse,
        }
        print(f"\n*** Best {method_name}: layers={best_cfg[0]}, nodes={best_cfg[1]}, "
              f"epochs={best_cfg[2]}, lr={best_cfg[3]} (mean RMSE={best_mean_rmse:.4f})")

    df = pd.DataFrame(rows).sort_values(
        ["Method", "layers", "nodes", "lr", "Benchmark"]
    ).reset_index(drop=True)

    return df, best_cfg_per_method


def tune_alpha_lam(best_cfg_per_method, alpha_grid, lam_grid, configs=TUNING_CONFIGS):
    """Stage-2 tuning of alpha and lam for GPmimic and MFNN-Intermediate."""
    rows = []
    best_alpha_lam = {}

    for method_name in ["GPmimic", "MFNN-Intermediate"]:
        if method_name not in best_cfg_per_method:
            print(f"Skipping {method_name} — not in best_cfg_per_method")
            continue

        hidden = tuple(best_cfg_per_method[method_name]["hidden"])
        lr     = float(best_cfg_per_method[method_name]["lr"])
        epochs = int(best_cfg_per_method[method_name]["epochs"])

        print(f"\n{'='*40}")
        print(f"STAGE-2 TUNING (alpha, lam): {method_name}")
        print(f"Fixed: hidden={hidden}, lr={lr}, epochs={epochs}")
        print(f"{'='*40}", flush=True)

        best_cfg       = None
        best_mean_rmse = None

        for alpha, lam in product(alpha_grid, lam_grid):
            total_rmse = 0.0
            count = 0

            for i, (bname, low_f, high_f, bounds) in enumerate(configs):
                dataset = make_tuning_dataset(bname, low_f, high_f, bounds, seed=2000 + i)
                D = dataset.Xl.shape[1]

                if method_name == "GPmimic":
                    model = GPmimic(x_dim=D, hidden=hidden, lr=lr, epochs=epochs,
                                    alpha=alpha, lam=lam, verbose=False)
                else:
                    model = MFNN_Intermediate(x_dim=D, hidden=hidden, hf_hidden=(hidden[0],),
                                              lr=lr, epochs=epochs, wd=0.0,
                                              alpha=alpha, lam=lam, verbose=False)

                model.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
                rmse, _, _, _ = evaluate_on_test(dataset, method_name, model)
                rmse_val = float(np.asarray(rmse))

                rows.append({
                    "Method": method_name, "Benchmark": bname,
                    "alpha": alpha, "lam": lam,
                    "hidden": str(hidden), "epochs": epochs, "lr": lr,
                    "RMSE": rmse_val,
                })
                total_rmse += rmse_val
                count += 1

            mean_rmse = total_rmse / count
            if best_mean_rmse is None or mean_rmse < best_mean_rmse:
                best_mean_rmse = mean_rmse
                best_cfg = (alpha, lam)

        best_alpha_lam[method_name] = {
            "alpha": best_cfg[0], "lam": best_cfg[1],
            "best_mean_rmse": best_mean_rmse,
        }
        print(f"*** Best {method_name}: alpha={best_cfg[0]}, lam={best_cfg[1]} "
              f"(mean RMSE={best_mean_rmse:.4f})")

    return pd.DataFrame(rows), best_alpha_lam


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune 2f methods on benchmarks")
    parser.add_argument("--config", default="configs/benchmarks_config.yaml")
    parser.add_argument("--out", default="outputs/results/tuning_benchmarks.csv")
    parser.add_argument("--skip-alpha-lam", action="store_true",
                        help="Skip stage-2 alpha/lam tuning")
    args = parser.parse_args()

    # Load config and limit threads
    cfg = yaml.safe_load(open(ROOT / args.config))
    limit_cpu_threads(cfg.get("cpu_threads", 4))

    # Stage 1: tune hidden/lr/epochs
    print("\n=== Stage 1: hidden / lr / epochs ===")
    df_stage1, best_cfg = tune_all_methods()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_stage1.to_csv(out_path, index=False)
    print(f"\nSaved stage-1 results to {out_path}")

    print("\nBest config per method:")
    for m, info in best_cfg.items():
        print(f"  {m}: hidden={info['hidden']}, epochs={info['epochs']}, "
              f"lr={info['lr']}, mean RMSE={info['best_mean_rmse']:.4f}")

    # Stage 2: tune alpha / lam
    best_al = {}
    if not args.skip_alpha_lam:
        print("\n=== Stage 2: alpha / lam (GPmimic + MFNN-Intermediate) ===")
        alpha_grid = [0.01, 0.05, 0.1, 0.2]
        lam_grid   = [1e-5, 1e-4, 1e-3]
        df_stage2, best_al = tune_alpha_lam(best_cfg, alpha_grid, lam_grid)

        al_path = out_path.with_name("tuning_alpha_lam.csv")
        df_stage2.to_csv(al_path, index=False)
        print(f"\nSaved stage-2 results to {al_path}")

        print("\nBest alpha/lam per method:")
        for m, info in best_al.items():
            print(f"  {m}: alpha={info['alpha']}, lam={info['lam']}, "
                  f"mean RMSE={info['best_mean_rmse']:.4f}")

    # -----------------------
    # Write best params back into config so 02_run_benchmarks.py picks them up automatically
    # -----------------------
    method_to_cfg_key = {
        "GPmimic":           "gpmimic",
        "MF-NN-Delta":       "delta",
        "MFNN-Intermediate": "intermediate",
        "MFNN-TwoStep":      "twostep",
        "MFNN-ThreeStep":    "threestep",
        "MFNN-Flag":         "flag",
    }

    for method_name, info in best_cfg.items():
        key = method_to_cfg_key.get(method_name)
        if key and key in cfg["twofid"]:
            cfg["twofid"][key]["hidden"] = list(info["hidden"])
            cfg["twofid"][key]["epochs"] = int(info["epochs"])
            cfg["twofid"][key]["lr"]     = float(info["lr"])

    # Write back alpha/lam for the two methods that have them
    for method_name, info in best_al.items():
        key = method_to_cfg_key.get(method_name)
        if key and key in cfg["twofid"]:
            cfg["twofid"][key]["alpha"] = float(info["alpha"])
            cfg["twofid"][key]["lam"]   = float(info["lam"])

    config_path = ROOT / args.config
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nConfig updated with best hyperparams → {config_path}")
    print("02_run_benchmarks.py will now use these automatically.")
