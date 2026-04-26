"""
src/datasets.py
Dataset containers and builders for ONC real data.
Extracted from MF_ONC_experiments_fixed.ipynb.

The benchmark datasets (mf2-based) use helpers_2f.make_dataset and helpers_3f.make_dataset
directly — no extra code needed here for those.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------
# Dataset containers
# -----------------------

@dataclass
class Dataset2f:
    """Container for paired low/high-fidelity training data and domain bounds."""
    Xl: np.ndarray  # Low-fidelity inputs  (n_low, D)
    yl: np.ndarray  # Low-fidelity outputs (n_low, 1)
    Xh: np.ndarray  # High-fidelity inputs  (n_high, D)
    yh: np.ndarray  # High-fidelity outputs (n_high, 1)
    bounds: np.ndarray  # (D, 2) box bounds


@dataclass
class Dataset3f:
    """Container for low/medium/high-fidelity training data and domain bounds."""
    Xl: np.ndarray  # Low-fidelity inputs    (n_low, D)
    yl: np.ndarray  # Low-fidelity outputs   (n_low, 1)
    Xm: np.ndarray  # Medium-fidelity inputs  (n_med, D)
    ym: np.ndarray  # Medium-fidelity outputs (n_med, 1)
    Xh: np.ndarray  # High-fidelity inputs    (n_high, D)
    yh: np.ndarray  # High-fidelity outputs   (n_high, 1)
    bounds: np.ndarray  # (D, 2) box bounds


# -----------------------
# Data loading
# -----------------------

def load_onc_data(root: Path, cfg: dict) -> dict:
    """
    Load all ONC CSVs as specified in onc_config.yaml.
    Returns a dict with keys: Xl_full, yl_full, Xm_full, ym_full,
    Xh_full, yh_full, Xh_train, yh_train, Xh_test, yh_test,
    Xm_train, ym_train, Xm_test, ym_test.
    """
    d = cfg["data"]
    p = lambda s: root / s  # shorthand

    data = {
        "Xl_full": pd.read_csv(p(d["low_fidelity"]["inputs"])),
        "yl_full": pd.read_csv(p(d["low_fidelity"]["outputs"])),
        "Xm_full": pd.read_csv(p(d["medium_fidelity"]["inputs"])),
        "ym_full": pd.read_csv(p(d["medium_fidelity"]["outputs"])),
        "Xh_full": pd.read_csv(p(d["high_fidelity"]["inputs"])),
        "yh_full": pd.read_csv(p(d["high_fidelity"]["outputs"])),
        # pre-split HF
        "Xh_train": pd.read_csv(p(d["high_fidelity"]["train_inputs"])),
        "yh_train": pd.read_csv(p(d["high_fidelity"]["train_outputs"])),
        "Xh_test":  pd.read_csv(p(d["high_fidelity"]["test_inputs"])),
        "yh_test":  pd.read_csv(p(d["high_fidelity"]["test_outputs"])),
        # pre-split MF (acts as "high" in LF+MF runs)
        "Xm_train": pd.read_csv(p(d["medium_fidelity"]["train_inputs"])),
        "ym_train": pd.read_csv(p(d["medium_fidelity"]["train_outputs"])),
        "Xm_test":  pd.read_csv(p(d["medium_fidelity"]["test_inputs"])),
        "ym_test":  pd.read_csv(p(d["medium_fidelity"]["test_outputs"])),
    }

    print("Data loaded:")
    print(f"  HF train: {data['Xh_train'].shape}  HF test: {data['Xh_test'].shape}")
    print(f"  MF train: {data['Xm_train'].shape}  MF test: {data['Xm_test'].shape}")
    return data


# -----------------------
# Dataset builders
# -----------------------

def make_onc_dataset_2f(
    Xl_full, yl_full,
    Xh_train_df, yh_train_df,
    Xh_test_df, yh_test_df,
    x_cols, y_col,
    n_lf, n_hf,
    seed=0,
):
    """
    Build a 2-fidelity ONC dataset from pre-split DataFrames.

    Args:
        x_cols: list of input column names
        y_col:  list with one output column name (e.g. ["time_to_onc"])
        n_lf:   number of LF training samples to use
        n_hf:   number of HF training samples to use

    Returns:
        (Dataset2f, (Xh_test, yh_test))
    """
    # LF: first n_lf rows
    Xl = Xl_full[x_cols].values[:n_lf]
    yl = yl_full[y_col].values[:n_lf].reshape(-1, 1)

    # HF training: first n_hf rows of pre-split training set
    assert n_hf <= len(Xh_train_df), \
        f"n_hf={n_hf} exceeds available HF training samples ({len(Xh_train_df)})"
    Xh = Xh_train_df.iloc[:n_hf][x_cols].values
    yh = yh_train_df.iloc[:n_hf][y_col].values.reshape(-1, 1)

    # HF test: full fixed test set
    Xh_test = Xh_test_df[x_cols].values
    yh_test = yh_test_df[y_col].values.reshape(-1, 1)

    bounds = np.column_stack([
        np.min(np.vstack([Xl, Xh]), axis=0),
        np.max(np.vstack([Xl, Xh]), axis=0),
    ])

    return Dataset2f(Xl, yl, Xh, yh, bounds), (Xh_test, yh_test)


def make_onc_dataset_2f_lf_hf(data, x_cols, y_col, n_lf, n_hf, seed=0):
    """Convenience wrapper: LF + HF combination."""
    return make_onc_dataset_2f(
        data["Xl_full"], data["yl_full"],
        data["Xh_train"], data["yh_train"],
        data["Xh_test"],  data["yh_test"],
        x_cols, y_col, n_lf=n_lf, n_hf=n_hf, seed=seed,
    )


def make_onc_dataset_2f_lf_mf(data, x_cols, y_col, n_lf, n_mf, seed=0):
    """Convenience wrapper: LF + MF combination (MF acts as 'high')."""
    return make_onc_dataset_2f(
        data["Xl_full"], data["yl_full"],
        data["Xm_train"], data["ym_train"],
        data["Xm_test"],  data["ym_test"],
        x_cols, y_col, n_lf=n_lf, n_hf=n_mf, seed=seed,
    )


def make_onc_dataset_2f_mf_hf(data, x_cols, y_col, n_mf, n_hf, seed=0):
    """Convenience wrapper: MF + HF combination (MF acts as 'low')."""
    return make_onc_dataset_2f(
        data["Xm_full"], data["ym_full"],
        data["Xh_train"], data["yh_train"],
        data["Xh_test"],  data["yh_test"],
        x_cols, y_col, n_lf=n_mf, n_hf=n_hf, seed=seed,
    )


def make_onc_dataset_3f(
    data, x_cols, y_col,
    n_lf, n_mf, n_hf,
    seed=0,
):
    """
    Build a 3-fidelity ONC dataset (LF + MF + HF).

    Returns:
        (Dataset3f, (Xh_test, yh_test))
    """
    Xl_full = data["Xl_full"];  yl_full = data["yl_full"]
    Xm_full = data["Xm_full"];  ym_full = data["ym_full"]
    Xh_train_df = data["Xh_train"];  yh_train_df = data["yh_train"]
    Xh_test_df  = data["Xh_test"];   yh_test_df  = data["yh_test"]

    assert n_lf <= len(Xl_full), f"n_lf={n_lf} exceeds LF samples"
    assert n_mf <= len(Xm_full), f"n_mf={n_mf} exceeds MF samples"
    assert n_hf <= len(Xh_train_df), f"n_hf={n_hf} exceeds HF training samples"

    Xl = Xl_full[x_cols].values[:n_lf]
    yl = yl_full[y_col].values[:n_lf].reshape(-1, 1)

    Xm = Xm_full[x_cols].values[:n_mf]
    ym = ym_full[y_col].values[:n_mf].reshape(-1, 1)

    Xh = Xh_train_df.iloc[:n_hf][x_cols].values
    yh = yh_train_df.iloc[:n_hf][y_col].values.reshape(-1, 1)

    Xh_test = Xh_test_df[x_cols].values
    yh_test = yh_test_df[y_col].values.reshape(-1, 1)

    bounds = np.column_stack([
        np.min(np.vstack([Xl, Xm, Xh]), axis=0),
        np.max(np.vstack([Xl, Xm, Xh]), axis=0),
    ])

    return Dataset3f(Xl, yl, Xm, ym, Xh, yh, bounds), (Xh_test, yh_test)
