"""
01_tune_onc.py
Hyperparameter tuning for all 2f/3f methods on ONC real data.
Matches paper Section 2.4.2 exactly:
  - Stage 1: tune hidden/lr on ALL inputs -> time_to_onc only
  - alpha/lam carried forward from benchmark tuning (not re-tuned on ONC)
  - Grid: layers [2,3,4], nodes [16,32,64,128], lr [1e-4,5e-4,1e-3], epochs=500 fixed

Usage:
    python scripts/01_tune_onc.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpers_2f import limit_cpu_threads
from sklearn.metrics import mean_squared_error

from twofid_methods.GPmimic_2f import GPmimic
from twofid_methods.MFNN_delta_2f import MFNN_Delta
from twofid_methods.MFNN_flag_2f import MFNN_Flag
from twofid_methods.MFNN_intermediate_2f import MFNN_Intermediate
from twofid_methods.MFNN_threestep_2f import MFNN_3step
from twofid_methods.MFNN_twostep_2f import MFNN_2step
from threefid_methods.GPmimic_3f import GPmimic3f
from threefid_methods.MFNN_flag_3f import MFNN_Flag3f
from threefid_methods.MFNN_intermediate_3f import MFNN_Intermediate3f
from src.datasets import load_onc_data, make_onc_dataset_2f_lf_hf, make_onc_dataset_3f

# Grid matches Table 3 in paper exactly
NUM_LAYERS = [2, 3, 4]
NUM_NODES  = [16, 32, 64, 128]
NUM_EPOCHS = 500
LRS        = [1e-4, 5e-4, 1e-3]

def make_hidden(n_layers, n_nodes):
    return tuple([n_nodes] * n_layers)

def get_2f_methods(alpha, lam):
    return [
        {"name": "GPmimic",
         "build": lambda D, h, lr, e: GPmimic(x_dim=D, hidden=h, lr=lr, epochs=e, alpha=alpha, lam=lam, verbose=False)},
        {"name": "MF-NN-Delta",
         "build": lambda D, h, lr, e: MFNN_Delta(x_dim=D, hidden=h, lr=lr, epochs=e, wd=0.0, verbose=False)},
        {"name": "MFNN-Intermediate",
         "build": lambda D, h, lr, e: MFNN_Intermediate(x_dim=D, hidden=h, hf_hidden=(h[0],), lr=lr, epochs=e, wd=0.0, alpha=alpha, lam=lam, verbose=False)},
        {"name": "MFNN-Flag",
         "build": lambda D, h, lr, e: MFNN_Flag(x_dim=D, hidden=h, lr=lr, epochs=e, wd=0.0, verbose=False)},
        {"name": "MFNN-TwoStep",
         "build": lambda D, h, lr, e: MFNN_2step(x_dim=D, hidden=h, lr=lr, epochs=e, wd=0.0, verbose=False)},
        {"name": "MFNN-ThreeStep",
         "build": lambda D, h, lr, e: MFNN_3step(x_dim=D, hidden_low=h, hidden_lin=(h[0],), hidden_high=(h[0],), lr=lr, epochs=e, wd=0.0, verbose=False)},
    ]

def get_3f_methods(w_h, w_m, w_l, lam):
    return [
        {"name": "GPmimic3f",
         "build": lambda D, h, lr, e: GPmimic3f(x_dim=D, hidden=h, lr=lr, epochs=e, w_h=w_h, w_m=w_m, w_l=w_l, lam=lam, verbose=False)},
        {"name": "MFNN-Flag3f",
         "build": lambda D, h, lr, e: MFNN_Flag3f(x_dim=D, hidden=h, lr=lr, epochs=e, wd=0.0, verbose=False)},
        {"name": "MFNN-Intermediate3f",
         "build": lambda D, h, lr, e: MFNN_Intermediate3f(x_dim=D, hidden=h, lr=lr, epochs=e, wd=0.0, w_hf=w_h, w_mf=w_m, w_lf=w_l, lam=lam, verbose=False)},
    ]

def tune_methods(methods, data, x_cols, y_col, fidelity, n_lf=200, n_hf=50, seed=42):
    rows = []
    best_cfg_per_method = {}

    for spec in methods:
        name  = spec["name"]
        build = spec["build"]
        print(f"\n{'='*40}\nTUNING: {name}\n{'='*40}", flush=True)

        best_cfg  = None
        best_rmse = None

        for L, H, lr in product(NUM_LAYERS, NUM_NODES, LRS):
            hidden = make_hidden(L, H)
            try:
                if fidelity == "2f":
                    ds, (Xt, yt) = make_onc_dataset_2f_lf_hf(data, x_cols, y_col, n_lf, n_hf, seed)
                    D = ds.Xl.shape[1]
                    m = build(D, hidden, lr, NUM_EPOCHS)
                    m.fit(ds.Xl, ds.yl, ds.Xh, ds.yh)
                else:
                    ds, (Xt, yt) = make_onc_dataset_3f(data, x_cols, y_col, n_lf, n_hf, n_hf, seed)
                    D = ds.Xl.shape[1]
                    m = build(D, hidden, lr, NUM_EPOCHS)
                    m.fit(ds.Xl, ds.yl, ds.Xm, ds.ym, ds.Xh, ds.yh)
                rmse = float(np.sqrt(mean_squared_error(yt, m.predict(Xt))))
            except Exception as e:
                print(f"  ERROR {L},{H},{lr}: {e}", flush=True)
                rmse = float("inf")

            print(f"  layers={L}, nodes={H}, lr={lr} → RMSE={rmse:.4f}", flush=True)
            rows.append({"Method": name, "layers": L, "nodes": H, "lr": lr, "RMSE": rmse})

            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_cfg  = (L, H, lr)

        best_cfg_per_method[name] = {
            "hidden": list(make_hidden(best_cfg[0], best_cfg[1])),
            "epochs": NUM_EPOCHS,
            "lr":     float(best_cfg[2]),
            "best_rmse": best_rmse,
        }
        print(f"*** Best {name}: layers={best_cfg[0]}, nodes={best_cfg[1]}, lr={best_cfg[2]} (RMSE={best_rmse:.4f})")

    return pd.DataFrame(rows), best_cfg_per_method

def write_best_to_config(cfg, best_2f, best_3f, config_path):
    key_map_2f = {"GPmimic": "gpmimic", "MF-NN-Delta": "delta",
                  "MFNN-Intermediate": "intermediate", "MFNN-Flag": "flag",
                  "MFNN-TwoStep": "twostep", "MFNN-ThreeStep": "threestep"}
    key_map_3f = {"GPmimic3f": "gpmimic3f", "MFNN-Flag3f": "flag3f",
                  "MFNN-Intermediate3f": "intermediate3f"}

    for name, info in best_2f.items():
        key = key_map_2f.get(name)
        if key and key in cfg["twofid"]:
            cfg["twofid"][key]["hidden"] = list(info["hidden"])
            cfg["twofid"][key]["epochs"] = int(info["epochs"])
            cfg["twofid"][key]["lr"]     = float(info["lr"])

    for name, info in best_3f.items():
        key = key_map_3f.get(name)
        if key and key in cfg["threefid"]:
            cfg["threefid"][key]["hidden"] = list(info["hidden"])
            cfg["threefid"][key]["epochs"] = int(info["epochs"])
            cfg["threefid"][key]["lr"]     = float(info["lr"])

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nConfig updated → {config_path}")
    print("02_run_onc.py will now use these automatically.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/onc_config.yaml")
    parser.add_argument("--out",    default="outputs/results/tuning_onc.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    limit_cpu_threads(cfg.get("cpu_threads", 4))
    data    = load_onc_data(ROOT, cfg)
    col_cfg = cfg["columns"]

    # Paper Section 2.4.2: ALL inputs -> time_to_onc only
    x_cols = col_cfg["x_all"]
    y_col  = col_cfg["y_onc"]

    # alpha/lam from benchmark tuning Table 6 — carried forward, not re-tuned
    alpha = 0.05
    lam_2f = 1e-5   # GPmimic benchmark best
    lam_inter = 0.1 # Intermediate benchmark best (use per-method in build fns)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print("\n=== Stage 1: Tune 2f methods (All inputs -> Time to ONC) ===")
    df_2f, best_2f = tune_methods(get_2f_methods(alpha=alpha, lam=lam_2f),
                                   data, x_cols, y_col, fidelity="2f")
    df_2f["fidelity"] = "2f"
    all_rows.append(df_2f)

    print("\n=== Stage 1: Tune 3f methods (All inputs -> Time to ONC) ===")
    df_3f, best_3f = tune_methods(get_3f_methods(w_h=0.5, w_m=0.3, w_l=0.2, lam=1e-4),
                                   data, x_cols, y_col, fidelity="3f")
    df_3f["fidelity"] = "3f"
    all_rows.append(df_3f)

    pd.concat(all_rows).reset_index(drop=True).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    write_best_to_config(cfg, best_2f, best_3f, ROOT / args.config)
