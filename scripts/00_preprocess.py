"""
00_preprocess.py
Split raw ONC CSVs into train/test sets for HF and MF data.
Converted from split_hf.ipynb.

Run this ONCE before any tuning or experiments, or whenever you get new raw data.

Usage:
    python scripts/00_preprocess.py
    python scripts/00_preprocess.py --data-dir ONC_data
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def split_and_save(base_path, n_train, suffix=""):
    """
    Load inputs.csv and outputs.csv from base_path,
    split first n_train rows → train, remainder → test,
    and save to the same folder.

    suffix: appended to filenames, e.g. "500" → hf_training_inputs500.csv
            leave blank for the primary split.
    """
    base_path = Path(base_path)
    inputs_df  = pd.read_csv(base_path / "inputs.csv")
    outputs_df = pd.read_csv(base_path / "outputs.csv")

    assert len(inputs_df) == len(outputs_df), \
        f"Row count mismatch in {base_path}: {len(inputs_df)} inputs vs {len(outputs_df)} outputs"

    fid = base_path.name.split("_")[0]  # "high", "medium", "low"
    prefix = {"high": "hf", "medium": "mf", "low": "lf"}[fid]

    train_inputs  = inputs_df.iloc[:n_train]
    test_inputs   = inputs_df.iloc[n_train:]
    train_outputs = outputs_df.iloc[:n_train]
    test_outputs  = outputs_df.iloc[n_train:]

    train_inputs.to_csv( base_path / f"{prefix}_training_inputs{suffix}.csv",  index=False)
    test_inputs.to_csv(  base_path / f"{prefix}_testing_inputs{suffix}.csv",   index=False)
    train_outputs.to_csv(base_path / f"{prefix}_training_outputs{suffix}.csv", index=False)
    test_outputs.to_csv( base_path / f"{prefix}_testing_outputs{suffix}.csv",  index=False)

    print(f"  {base_path.name}{suffix or ''}: {n_train} train / {len(test_inputs)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ONC raw CSVs into train/test splits")
    parser.add_argument("--data-dir", default="ONC_data",
                        help="Path to ONC_data folder (relative to project root)")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir

    print("=== Preprocessing ONC data ===")
    print(f"Data directory: {data_dir}\n")

    # 1. HF primary split: 200 train / 800 test
    print("High fidelity (primary split: 200/800):")
    split_and_save(data_dir / "high_fidelity", n_train=200)

    # 2. HF alternate split: 500 train / 500 test (kept for reference)
    print("High fidelity (alternate split: 500/500):")
    split_and_save(data_dir / "high_fidelity", n_train=500, suffix="500")

    # 3. MF split: 500 train / 500 test (MF acts as HF in LF+MF runs)
    print("Medium fidelity (500/500):")
    split_and_save(data_dir / "medium_fidelity", n_train=500)

    print("\nDone. All split CSVs saved into their respective fidelity folders.")
    print("You can now run: python scripts/01_tune_onc.py")
