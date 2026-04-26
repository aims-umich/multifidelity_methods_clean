"""
03_postprocess_onc.py
Parse ONC results CSV and export LaTeX tables + figures.
Converted from MF_ONC_experiments_fixed.ipynb (postprocessing cells).

Usage:
    python scripts/03_postprocess_onc.py
    python scripts/03_postprocess_onc.py --results outputs/results/onc_results.csv
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# -----------------------
# Parsing helpers
# -----------------------

def parse_case(case_string):
    """Extract structured fields from a case name string."""
    match = re.search(r"^(.*?)_to_(.*?)_", case_string)
    inputs = match.group(1) if match else None
    output = match.group(2) if match else None

    lf = int(re.search(r"LF(\d+)", case_string).group(1)) \
        if re.search(r"LF(\d+)", case_string) else 0
    mf = int(re.search(r"MF(\d+)", case_string).group(1)) \
        if re.search(r"MF(\d+)", case_string) else 0
    hf = int(re.search(r"HF(\d+)", case_string).group(1)) \
        if re.search(r"HF(\d+)", case_string) else 0

    count = sum([lf > 0, mf > 0, hf > 0])
    if count == 3:
        num_fidelity, combo = "3F", "LFMFHF"
    elif lf > 0 and hf > 0:
        num_fidelity, combo = "2F", "LFHF"
    elif lf > 0 and mf > 0:
        num_fidelity, combo = "2F", "LFMF"
    elif mf > 0 and hf > 0:
        num_fidelity, combo = "2F", "MFHF"
    else:
        num_fidelity, combo = "1F", "Single"

    total_budget = lf + 2 * mf + 4 * hf

    return pd.Series({
        "Inputs": inputs, "Output": output,
        "NumFidelity": num_fidelity, "ComboType": combo,
        "LF_samples": lf, "MF_samples": mf, "HF_samples": hf,
        "TotalBudget": total_budget,
    })


def classify_input(row):
    if row["Inputs"] == "AllInputs":
        return "AllInputs"
    if row["Output"] == "ONC"    and row["Inputs"] == "Temp":
        return "DominantInputs"
    if row["Output"] == "Tafter" and row["Inputs"] == "Temp_HTC":
        return "DominantInputs"
    return "NonDominantInputs"


def load_and_parse(csv_path):
    df = pd.read_csv(csv_path)

    # Drop old parsed columns if re-running
    df = df.drop(columns=[
        "Inputs", "Output", "NumFidelity", "ComboType",
        "LF_samples", "MF_samples", "HF_samples", "TotalBudget", "InputGroup",
    ], errors="ignore")

    df = pd.concat([df, df["Case"].apply(parse_case)], axis=1)
    df["InputGroup"] = df.apply(classify_input, axis=1)
    return df


# -----------------------
# LaTeX table exporters
# -----------------------

def export_fixed_budget_table(df, combo, budget, out_dir, filename):
    subset = df[
        (df["TotalBudget"] == budget) &
        ((df["ComboType"] == combo) | (combo == "LFHF" and df["ComboType"] == "LFMFHF"))
    ].copy()
    subset["MethodLabel"] = subset["Model"] + " (" + subset["NumFidelity"] + ")"

    input_order = ["AllInputs", "DominantInputs", "NonDominantInputs"]
    lines = []
    for group in input_order:
        section = subset[subset["InputGroup"] == group]
        if section.empty:
            continue
        pretty = group.replace("Inputs", " Inputs")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{pretty}}}}} \\\\")
        lines.append("\\hline")
        for model in sorted(section["MethodLabel"].unique()):
            row_onc  = section[(section["MethodLabel"] == model) & (section["Output"] == "ONC")]
            row_temp = section[(section["MethodLabel"] == model) & (section["Output"] == "Tafter")]
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


def export_scaling_table(df, input_group, out_dir, filename):
    subset = df[
        (df["ComboType"] == "LFHF") &
        (df["NumFidelity"] == "2F") &
        (df["InputGroup"] == input_group)
    ].copy()

    lines = []
    for budget in sorted(subset["TotalBudget"].unique()):
        section = subset[subset["TotalBudget"] == budget]
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Total Budget = {budget}}}}} \\\\")
        lines.append("\\hline")
        for model in sorted(section["Model"].unique()):
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
# Plotting
# -----------------------

def plot_timing(df, out_dir):
    input_order = ["AllInputs", "DominantInputs", "NonDominantInputs"]
    markers = ["o", "s", "^"]
    colors  = ["blue", "red", "green"]

    df_mfgp = df[
        (df["Model"] == "MF-GP") &
        (df["ComboType"] == "LFHF") &
        (df["Output"] == "ONC")
    ]

    plt.figure()
    for group, marker, color in zip(input_order, markers, colors):
        subset = df_mfgp[df_mfgp["InputGroup"] == group].sort_values("TotalBudget")
        label  = group.replace("Inputs", " Inputs")
        plt.scatter(subset["TotalBudget"], subset["Time_sec"],
                    marker=marker, color=color, label=label)

    plt.xlabel("Total Budget")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    fig_path = Path(out_dir) / "mfgp_timing.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess ONC results")
    parser.add_argument("--results",    default="outputs/results/onc_results.csv")
    parser.add_argument("--tex-dir",    default="outputs/results")
    parser.add_argument("--fig-dir",    default="outputs/figures")
    args = parser.parse_args()

    results_path = ROOT / args.results
    tex_dir      = ROOT / args.tex_dir
    fig_dir      = ROOT / args.fig_dir
    tex_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_parse(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")

    # LaTeX tables
    export_fixed_budget_table(df, "LFMF", 1800, tex_dir, "lfmf_1800.tex")
    export_fixed_budget_table(df, "MFHF", 1800, tex_dir, "mfhf_1800.tex")
    export_fixed_budget_table(df, "LFHF", 1800, tex_dir, "lfhf_1800.tex")
    export_scaling_table(df, "DominantInputs", tex_dir, "lfhf_scaling_dominant.tex")
    export_scaling_table(df, "AllInputs",      tex_dir, "lfhf_scaling_allinputs.tex")

    # Figures
    plot_timing(df, fig_dir)

    print("\nDone.")
