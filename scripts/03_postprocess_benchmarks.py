"""
03_postprocess_benchmarks.py
Generate Tables 7, 8 and Figures 4-7 from paper.

Usage:
    python scripts/03_postprocess_benchmarks.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# -----------------------
# Table helpers
# -----------------------

def export_2f_table(df, out_dir):
    """Table 7: RMSE comparison across 2f benchmarks."""
    subset = df[df["Fidelity"] == "2F"].copy()

    bench_order  = ["Forrester (1D)", "Booth (2D)", "Branin (2D)",
                    "Park91A (4D)", "Hartmann6 (6D)", "Borehole (8D)"]
    model_order  = ["GPmimic", "MF-GP", "MF-NN-Delta", "MFNN-Flag", "MFNN-Intermediate"]

    # pivot: rows=benchmarks, cols=models
    pivot = subset.pivot_table(index="Benchmark", columns="Model",
                               values="RMSE", aggfunc="first")

    lines = []
    lines.append("Benchmark & " + " & ".join(model_order) + " \\\\")
    lines.append("\\hline")

    for bench in bench_order:
        if bench not in pivot.index:
            continue
        row_vals = []
        for model in model_order:
            val = pivot.loc[bench, model] if model in pivot.columns else float("nan")
            row_vals.append(f"{val:.4f}" if not np.isnan(val) else "-")
        lines.append(f"{bench} & " + " & ".join(row_vals) + " \\\\")

    out_path = Path(out_dir) / "table7_2f_benchmarks.tex"
    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")

    # also print as readable table
    print("\nTable 7 — 2f Benchmark RMSE:")
    print(pivot[model_order].to_string(float_format=lambda x: f"{x:.4f}"))


def export_3f_table(df, out_dir):
    """Table 8: RMSE comparison across 3f benchmarks."""
    subset = df[df["Fidelity"] == "3F"].copy()

    bench_order = ["Forrester-3f (1D)", "Rastrigin-3f (2D)", "Rastrigin-3f (5D)",
                   "Rosenbrock-3f (2D)", "Rosenbrock-3f (5D)"]
    model_order = ["GPmimic3f", "MFNN-Intermediate3f", "MFNN-Flag3f"]

    pivot = subset.pivot_table(index="Benchmark", columns="Model",
                               values="RMSE", aggfunc="first")

    lines = []
    lines.append("Benchmark & " + " & ".join(model_order) + " \\\\")
    lines.append("\\hline")

    for bench in bench_order:
        if bench not in pivot.index:
            continue
        row_vals = []
        for model in model_order:
            val = pivot.loc[bench, model] if model in pivot.columns else float("nan")
            row_vals.append(f"{val:.4f}" if not np.isnan(val) else "-")
        lines.append(f"{bench} & " + " & ".join(row_vals) + " \\\\")

    out_path = Path(out_dir) / "table8_3f_benchmarks.tex"
    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")

    print("\nTable 8 — 3f Benchmark RMSE:")
    print(pivot[model_order].to_string(float_format=lambda x: f"{x:.4f}"))


# -----------------------
# Parity + residual plot helper
# -----------------------

def parity_and_residual(preds_dict, title, out_path, model_order=None):
    """
    Top row: parity plots (y_pred vs y_true)
    Bottom row: residual histograms
    Matches Figures 4-7 in paper.
    """
    ytrue = preds_dict["ytrue"].ravel()
    models = [k for k in preds_dict.keys() if k != "ytrue"]
    if model_order:
        models = [m for m in model_order if m in models] + \
                 [m for m in models if m not in model_order]

    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    mn_all = ytrue.min()
    mx_all = ytrue.max()

    for j, model_name in enumerate(models):
        ypred = preds_dict[model_name].ravel()

        # parity plot
        ax = axes[0, j]
        ax.scatter(ytrue, ypred, s=12, alpha=0.7)
        ax.plot([mn_all, mx_all], [mn_all, mx_all], "k--", linewidth=1)
        rmse = float(np.sqrt(mean_squared_error(ytrue, ypred)))
        ax.set_title(f"{model_name}\nRMSE={rmse:.4f}", fontsize=9)
        ax.set_xlabel("y_true (high-fid)", fontsize=8)
        ax.set_ylabel("y_pred", fontsize=8)

        # residual histogram
        ax2 = axes[1, j]
        resid = ypred - ytrue
        ax2.hist(resid, bins=30, alpha=0.9)
        ax2.set_xlabel("Residual (y_pred - y_true)", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.set_title(model_name, fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# -----------------------
# Entry point
# -----------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/results/benchmark_results.csv")
    parser.add_argument("--tex-dir", default="outputs/results")
    parser.add_argument("--fig-dir", default="outputs/figures")
    args = parser.parse_args()

    results_path = ROOT / args.results
    tex_dir      = ROOT / args.tex_dir
    fig_dir      = ROOT / args.fig_dir
    pred_dir     = ROOT / "outputs/results"  # where .npy files are saved

    tex_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")

    print("\n=== LaTeX Tables ===")
    export_2f_table(df, tex_dir)
    export_3f_table(df, tex_dir)

    print("\n=== Parity + Residual Plots ===")

    model_order_2f = ["MF-GP", "GPmimic", "MFNN-Flag", "MF-NN-Delta", "MFNN-Intermediate"]
    model_order_3f = ["GPmimic3f", "MFNN-Intermediate3f", "MFNN-Flag3f"]

    # Figure 4 — Booth 2D
    booth_path = pred_dir / "preds_2f_Booth_2D.npy"
    if booth_path.exists():
        preds = np.load(booth_path, allow_pickle=True).item()
        parity_and_residual(preds,
            "Booth (2D) — Parity and Residuals",
            fig_dir / "fig4_booth_2d.png",
            model_order=model_order_2f)
    else:
        print(f"  Missing {booth_path} — run 02_run_benchmarks.py first")

    # Figure 5 — Borehole 8D
    bore_path = pred_dir / "preds_2f_Borehole_8D.npy"
    if bore_path.exists():
        preds = np.load(bore_path, allow_pickle=True).item()
        parity_and_residual(preds,
            "Borehole (8D) — Parity and Residuals",
            fig_dir / "fig5_borehole_8d.png",
            model_order=model_order_2f)
    else:
        print(f"  Missing {bore_path} — run 02_run_benchmarks.py first")

    # Figure 6 — Rosenbrock 2D (3f)
    rosen_path = pred_dir / "preds_3f_Rosenbrock_2D.npy"
    if rosen_path.exists():
        preds = np.load(rosen_path, allow_pickle=True).item()
        parity_and_residual(preds,
            "Rosenbrock (2D) 3f — Parity and Residuals",
            fig_dir / "fig6_rosenbrock_2d_3f.png",
            model_order=model_order_3f)
    else:
        print(f"  Missing {rosen_path} — run 02_run_benchmarks.py first")

    # Figure 7 — Rastrigin 5D (3f)
    rast_path = pred_dir / "preds_3f_Rastrigin_5D.npy"
    if rast_path.exists():
        preds = np.load(rast_path, allow_pickle=True).item()
        parity_and_residual(preds,
            "Rastrigin (5D) 3f — Parity and Residuals",
            fig_dir / "fig7_rastrigin_5d_3f.png",
            model_order=model_order_3f)
    else:
        print(f"  Missing {rast_path} — run 02_run_benchmarks.py first")

    print("\nDone.")
