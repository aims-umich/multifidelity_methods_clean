"""
03_postprocess_onc.py
Parse ONC results CSV and export LaTeX tables + figures.

Usage:
    python scripts/03_postprocess_onc.py
    python scripts/03_postprocess_onc.py --results outputs/results/onc_results.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# -----------------------
# Input group classification
# matches paper Table 2 exactly
# -----------------------
def classify_input(row):
    """Map raw Inputs column to AllInputs / DominantInputs / NonDominantInputs."""
    if row["Inputs"] == "AllInputs":
        return "AllInputs"
    # Dominant: Temp for ONC, Temp_HTC for Tafter
    if row["Output"] == "ONC"    and row["Inputs"] == "Temp":
        return "DominantInputs"
    if row["Output"] == "Tafter" and row["Inputs"] == "Temp_HTC":
        return "DominantInputs"
    # Everything else is non-dominant
    return "NonDominantInputs"


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    df["InputGroup"] = df.apply(classify_input, axis=1)
    # fill missing MF column with 0
    if "MF" not in df.columns:
        df["MF"] = 0
    df["MF"] = df["MF"].fillna(0)
    df["LF"] = df["LF"].fillna(0)
    df["HF"] = df["HF"].fillna(0)
    print(f"Loaded {len(df)} rows")
    print(f"Combos:      {df['Combo'].unique()}")
    print(f"Inputs:      {df['Inputs'].unique()}")
    print(f"Outputs:     {df['Output'].unique()}")
    print(f"Models:      {df['Model'].unique()}")
    print(f"Budgets:     {sorted(df['TotalBudget'].unique())}")
    return df


# -----------------------
# LaTeX table exporters
# -----------------------

def export_fixed_budget_table(df, combo, budget, out_dir, filename):
    """
    Fixed budget table — matches Tables 10, A13, A14 in paper.
    Shows all input groups x all methods, both outputs side by side.
    """
    subset = df[
        (df["TotalBudget"] == budget) &
        (df["Combo"] == combo)
    ].copy()

    if subset.empty:
        print(f"  No data for combo={combo} budget={budget}, skipping {filename}")
        return

    input_order = ["AllInputs", "DominantInputs", "NonDominantInputs"]
    # model order matches paper
    model_order = ["MF-GP", "GPmimic", "MF-NN-Delta", "MFNN-Flag",
                   "MFNN-Intermediate", "GPmimic3f", "MFNN-Flag3f", "MFNN-Intermediate3f"]

    lines = []
    for group in input_order:
        section = subset[subset["InputGroup"] == group]
        if section.empty:
            continue
        pretty = group.replace("Inputs", " Inputs").replace("NonDominant", "Non-Dominant")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{pretty}}}}} \\\\")
        lines.append("\\hline")

        # get models present, sorted by model_order
        models_present = [m for m in model_order if m in section["Model"].values]
        for model in models_present:
            row_onc  = section[(section["Model"] == model) & (section["Output"] == "ONC")]
            row_temp = section[(section["Model"] == model) & (section["Output"] == "Tafter")]
            if row_onc.empty or row_temp.empty:
                continue
            fid   = row_onc.iloc[0]["Fidelity"]
            label = f"{model} ({fid})"
            onc   = row_onc.iloc[0]
            temp  = row_temp.iloc[0]
            lines.append(
                f" & {label} & "
                f"{onc['RMSE']:.2f} & {onc['R2']:.4f} & {onc['Time_sec']:.2f} & "
                f"{temp['RMSE']:.2f} & {temp['R2']:.4f} & {temp['Time_sec']:.2f} \\\\"
            )
        lines.append("\\hline")

    out_path = Path(out_dir) / filename
    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")


def export_scaling_table(df, input_group, combo, out_dir, filename):
    """
    Scaling table — matches Tables 11, 12 in paper.
    Shows performance across budgets for a fixed input group.
    """
    subset = df[
        (df["Combo"] == combo) &
        (df["Fidelity"] == "2F") &
        (df["InputGroup"] == input_group)
    ].copy()

    if subset.empty:
        print(f"  No data for input_group={input_group} combo={combo}, skipping {filename}")
        return

    model_order = ["MF-GP", "GPmimic", "MF-NN-Delta", "MFNN-Flag", "MFNN-Intermediate"]
    lines = []

    for budget in sorted(subset["TotalBudget"].unique()):
        section = subset[subset["TotalBudget"] == budget]
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Total Budget = {int(budget)}}}}} \\\\")
        lines.append("\\hline")
        models_present = [m for m in model_order if m in section["Model"].values]
        for model in models_present:
            row_onc  = section[(section["Model"] == model) & (section["Output"] == "ONC")]
            row_temp = section[(section["Model"] == model) & (section["Output"] == "Tafter")]
            if row_onc.empty or row_temp.empty:
                continue
            onc  = row_onc.iloc[0]
            temp = row_temp.iloc[0]
            lines.append(
                f" & {model} & "
                f"{onc['RMSE']:.2f} & {onc['R2']:.4f} & {onc['Time_sec']:.2f} & "
                f"{temp['RMSE']:.2f} & {temp['R2']:.4f} & {temp['Time_sec']:.2f} \\\\"
            )
        lines.append("\\hline")

    out_path = Path(out_dir) / filename
    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")


# -----------------------
# Figures
# -----------------------

def plot_timing(df, out_dir):
    """Figure 8: MF-GP runtime vs budget, LF+HF, Time to ONC only."""
    df_mfgp = df[
        (df["Model"] == "MF-GP") &
        (df["Combo"] == "LFHF") &
        (df["Output"] == "ONC") &
        (df["Inputs"] != "Temp_HTC")   # exclude Temp_HTC — dominant for Tafter not ONC
    ].copy()

    if df_mfgp.empty:
        print("  No MF-GP LFHF ONC data for timing plot, skipping")
        return

    group_styles = {
        "AllInputs":        ("o", "blue",  "All Inputs"),
        "DominantInputs":   ("s", "red",   "Dominant Inputs"),
        "NonDominantInputs":("^", "green", "Non-Dominant Inputs"),
    }

    plt.figure()
    for group, (marker, color, label) in group_styles.items():
        subset = df_mfgp[df_mfgp["InputGroup"] == group].sort_values("TotalBudget")
        if subset.empty:
            continue
        plt.scatter(subset["TotalBudget"], subset["Time_sec"],
                    marker=marker, color=color, label=label)

    plt.xlabel("Total Budget")
    plt.ylabel("Time (seconds)")
    plt.title("MF-GP Runtime vs Budget (Time to ONC)")
    plt.legend()
    plt.tight_layout()
    fig_path = Path(out_dir) / "mfgp_timing.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


def plot_parity(df, combo, budget, out_dir, filename):
    """
    Parity plots (pred vs true) for all methods at a fixed budget and combo.
    One subplot per method, two rows (ONC and Tafter).
    """
    subset = df[
        (df["Combo"] == combo) &
        (df["TotalBudget"] == budget) &
        (df["InputGroup"] == "AllInputs")
    ].copy()

    if subset.empty:
        print(f"  No data for parity plot {filename}, skipping")
        return

    models = subset["Model"].unique()
    outputs = ["ONC", "Tafter"]
    n_models = len(models)

    fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for j, model in enumerate(models):
        for i, output in enumerate(outputs):
            ax = axes[i, j]
            row = subset[(subset["Model"] == model) & (subset["Output"] == output)]
            if row.empty:
                ax.axis("off")
                continue
            r = row.iloc[0]
            # we don't have raw predictions in the CSV — just show R2/RMSE as text
            ax.text(0.5, 0.5,
                    f"R²={r['R2']:.4f}\nRMSE={r['RMSE']:.2f}",
                    ha="center", va="center", fontsize=12,
                    transform=ax.transAxes)
            ax.set_title(f"{model}\n{output}", fontsize=9)
            ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")

    fig.suptitle(f"Performance Summary — {combo} Budget={budget}", fontsize=12)
    plt.tight_layout()
    fig_path = Path(out_dir) / filename
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="outputs/results/onc_results.csv")
    parser.add_argument("--tex-dir",  default="outputs/results")
    parser.add_argument("--fig-dir",  default="outputs/figures")
    args = parser.parse_args()

    results_path = ROOT / args.results
    tex_dir      = ROOT / args.tex_dir
    fig_dir      = ROOT / args.fig_dir
    tex_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_path)

    print("\n=== LaTeX Tables ===")
    # Fixed budget tables (Table 10, A13, A14 in paper)
    export_fixed_budget_table(df, "LFHF", 1800, tex_dir, "table_lfhf_1800.tex")
    export_fixed_budget_table(df, "LFMF", 1800, tex_dir, "table_lfmf_1800.tex")
    export_fixed_budget_table(df, "MFHF", 1800, tex_dir, "table_mfhf_1800.tex")

    # Scaling tables (Tables 11, 12 in paper)
    export_scaling_table(df, "DominantInputs", "LFHF", tex_dir, "table_scaling_dominant.tex")
    export_scaling_table(df, "AllInputs",      "LFHF", tex_dir, "table_scaling_allinputs.tex")

    print("\n=== Figures ===")
    plot_timing(df, fig_dir)

    print("\nDone.")
