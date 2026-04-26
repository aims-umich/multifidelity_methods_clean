"""
01_tune_onc.py
Hyperparameter tuning for all 2f/3f methods on ONC real data.
Converted from hyperparameter_tuning_ONC_fixed.ipynb.

Usage:
    python scripts/01_tune_onc.py
    python scripts/01_tune_onc.py --config configs/onc_config.yaml
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

from helpers_2f import limit_cpu_threads, r2_score
from sklearn.metrics import mean_squared_error

from twofid_methods.GPmimic_2f import GPmimic
from twofid_methods.MFGP_2f import MFGP
from twofid_methods.MFNN_delta_2f import MFNN_Delta
from twofid_methods.MFNN_flag_2f import MFNN_Flag
from twofid_methods.MFNN_intermediate_2f import MFNN_Intermediate
from twofid_methods.MFNN_threestep_2f import MFNN_3step
from twofid_methods.MFNN_twostep_2f import MFNN_2step

from src.datasets import (
    load_onc_data, make_onc_dataset_2f,
    make_onc_dataset_2f_lf_hf, make_onc_dataset_2f_lf_mf, make_onc_dataset_2f_mf_hf,
)


# -----------------------
# Hyperparameter grid
# -----------------------
NUM_LAYERS = [2, 3, 4]
NUM_NODES  = [64, 128]
NUM_EPOCHS = [500]
LRS        = [1e-4, 1e-3]

def make_hidden(n_layers, n_nodes):
    return tuple([n_nodes] * n_layers)


def tune_onc_methods(cfg, data, x_cols_list, y_col):
    """
    Grid search over hidden/lr/epochs for all NN methods on ONC data.
    x_cols_list: list of input column lists to try
    y_col: output column (list with one element)
    """
    rows = []
    best_cfg_per_method = {}

    methods = [
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
            "name": "MFNN-Flag",
            "build": lambda D, hidden, lr, epochs: MFNN_Flag(
                x_dim=D, hidden=hidden, lr=lr, epochs=epochs,
                wd=0.0, verbose=False),
        },
    ]

    for method_spec in methods:
        method_name = method_spec["name"]
        build_model  = method_spec["build"]

        print(f"\n{'='*40}")
        print(f"TUNING: {method_name}  |  output: {y_col}")
        print(f"{'='*40}", flush=True)

        best_cfg       = None
        best_mean_rmse = None

        for L, H, E, lr in product(NUM_LAYERS, NUM_NODES, NUM_EPOCHS, LRS):
            hidden = make_hidden(L, H)
            total_rmse = 0.0
            count = 0

            for x_cols in x_cols_list:
                # use lf+hf 2f dataset for tuning with moderate sizes
                dataset, (Xtest, ytest) = make_onc_dataset_2f_lf_hf(
                    data, x_cols, y_col, n_lf=200, n_hf=50, seed=42)
                D = dataset.Xl.shape[1]

                model = build_model(D, hidden, lr, E)
                model.fit(dataset.Xl, dataset.yl, dataset.Xh, dataset.yh)
                ypred = model.predict(Xtest)
                rmse_val = float(np.sqrt(mean_squared_error(ytest, ypred)))

                rows.append({
                    "Method": method_name, "x_cols": str(x_cols),
                    "y_col": str(y_col), "layers": L, "nodes": H,
                    "epochs": E, "lr": lr, "RMSE": rmse_val,
                })
                total_rmse += rmse_val
                count += 1

            mean_rmse = total_rmse / count
            print(f"  layers={L}, nodes={H}, epochs={E}, lr={lr} → mean RMSE={mean_rmse:.4f}",
                  flush=True)

            if best_mean_rmse is None or mean_rmse < best_mean_rmse:
                best_mean_rmse = mean_rmse
                best_cfg = (L, H, E, lr)

        best_cfg_per_method[method_name] = {
            "hidden": list(make_hidden(best_cfg[0], best_cfg[1])),
            "epochs": best_cfg[2],
            "lr":     best_cfg[3],
            "best_mean_rmse": best_mean_rmse,
        }
        print(f"*** Best {method_name}: layers={best_cfg[0]}, nodes={best_cfg[1]}, "
              f"epochs={best_cfg[2]}, lr={best_cfg[3]} (mean RMSE={best_mean_rmse:.4f})")

    return pd.DataFrame(rows), best_cfg_per_method


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune 2f methods on ONC data")
    parser.add_argument("--config", default="configs/onc_config.yaml")
    parser.add_argument("--out",    default="outputs/results/tuning_onc.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    limit_cpu_threads(cfg.get("cpu_threads", 4))

    # Load data
    data = load_onc_data(ROOT, cfg)

    col_cfg = cfg["columns"]
    x_cols_list = [col_cfg["x_all"], col_cfg["x_temp"], col_cfg["x_temp_htc"]]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    # Accumulate best configs across both outputs — average the RMSE, keep the params
    # from whichever output had the better (lower) mean RMSE per method
    combined_best = {}

    for y_col in [col_cfg["y_onc"], col_cfg["y_tafter"]]:
        print(f"\n\n{'#'*50}")
        print(f"OUTPUT: {y_col}")
        print(f"{'#'*50}")

        df, best = tune_onc_methods(cfg, data, x_cols_list, y_col)
        df["output"] = str(y_col)
        all_rows.append(df)

        print(f"\nBest configs for {y_col}:")
        for m, info in best.items():
            print(f"  {m}: {info}")
            # Keep params from whichever output gives lower RMSE for this method
            if m not in combined_best or info["best_mean_rmse"] < combined_best[m]["best_mean_rmse"]:
                combined_best[m] = info

    pd.concat(all_rows).reset_index(drop=True).to_csv(out_path, index=False)
    print(f"\nSaved tuning results to {out_path}")

    # -----------------------
    # Write best params back into config so 02_run_onc.py picks them up automatically
    # -----------------------
    method_to_cfg_key = {
        "GPmimic":           "gpmimic",
        "MF-NN-Delta":       "delta",
        "MFNN-Intermediate": "intermediate",
        "MFNN-Flag":         "flag",
    }

    for method_name, info in combined_best.items():
        key = method_to_cfg_key.get(method_name)
        if key and key in cfg["twofid"]:
            cfg["twofid"][key]["hidden"] = list(info["hidden"])
            cfg["twofid"][key]["epochs"] = int(info["epochs"])
            cfg["twofid"][key]["lr"]     = float(info["lr"])

    config_path = ROOT / args.config
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nConfig updated with best hyperparams → {config_path}")
    print("02_run_onc.py will now use these automatically.")
