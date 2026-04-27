# Multifidelity Surrogate Modeling — HTGR DLOFC

This repository contains the implementation and experimental pipeline for the paper:

> **Multifidelity Surrogate Modeling of Depressurized Loss of Forced Cooling in High-temperature Gas Reactors**
> Meredith Eaheart, Majdi I. Radaideh — University of Michigan
> [arXiv:2603.14143](https://arxiv.org/abs/2603.14143)

Several multifidelity machine learning methods are evaluated for predicting the time to onset of natural circulation (ONC) and the temperature after ONC for an HTGR DLOFC transient, using CFD simulation data at three mesh fidelity levels.

---

## Repository structure

```
multifidelity_methods_clean/
│
├── twofid_methods/              # 2-fidelity MFNN implementations
│   ├── MFGP_2f.py               # Multifidelity Gaussian Process (co-kriging)
│   ├── GPmimic_2f.py            # GPmimic neural network
│   ├── MFNN_delta_2f.py         # Delta (residual correction) method
│   ├── MFNN_flag_2f.py          # Fidelity-flag method
│   ├── MFNN_intermediate_2f.py  # Intermediate method
│   ├── MFNN_twostep_2f.py       # Two-step method
│   └── MFNN_threestep_2f.py     # Three-step method
│
├── threefid_methods/            # 3-fidelity MFNN implementations
│   ├── GPmimic_3f.py
│   ├── MFNN_flag_3f.py
│   └── MFNN_intermediate_3f.py
│
├── benchmarks/                  # Analytical benchmark functions
│   ├── twofid_benchmarks.py     # mf2 package wrappers (Branin, Forrester, etc.)
│   ├── forrester_3f.py          # 3-fidelity Forrester (1D)
│   ├── rastrigin_3f.py          # 3-fidelity Rastrigin (D=2,5,10)
│   └── rosenbrock_3f.py         # 3-fidelity Rosenbrock (D=2,5,10)
│
├── src/
│   └── datasets.py              # ONC data loading and Dataset2f/3f builders
│
├── configs/
│   ├── benchmarks_config.yaml   # Hyperparameters and run settings for benchmarks
│   └── onc_config.yaml          # Hyperparameters and data paths for ONC experiments
│
├── scripts/                     # Run in numbered order
│   ├── 00_preprocess.py         # Split raw ONC CSVs into train/test sets (run once)
│   ├── 01_tune_benchmarks.py    # Hyperparameter tuning on benchmark functions
│   ├── 01_tune_onc.py           # Hyperparameter tuning on ONC data
│   ├── 02_run_benchmarks.py     # Run 2f + 3f benchmark experiments
│   ├── 02_run_onc.py            # Run 2f + 3f ONC experiments
│   ├── 03_postprocess_benchmarks.py  # Tables 7-8 + Figures 4-7 from paper
│   └── 03_postprocess_onc.py    # Tables 10-12 + Figure 8 from paper
│
├── outputs/
│   ├── results/                 # CSVs and LaTeX .tex files (generated)
│   └── figures/                 # Plots (generated)
│
├── helpers_2f.py                # Shared utilities: MLP, train_torch_regressor,
│                                #   get_device(), limit_cpu_threads()
├── helpers_3f.py                # 3-fidelity dataset container + re-exports
├── orchestration_2f.py          # run_for_function() core logic for 2f benchmarks
└── orchestration_3f.py          # run_for_function() core logic for 3f benchmarks
```

---

## Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm pyyaml mf2 h5py
```

---

## Data

ONC simulation data is not included in this repository. The dataset consists of 1000 CFD simulation samples at each of three mesh fidelity levels (LF: 17,500 elements, MF: 35,000 elements, HF: 70,000 elements), generated using Ansys Fluent. Data is available from the authors upon request.

Place data at the project root before running:

```
ONC_data/
├── high_fidelity/
│   ├── inputs.csv
│   └── outputs.csv
├── medium_fidelity/
│   ├── inputs.csv
│   └── outputs.csv
└── low_fidelity/
    ├── inputs.csv
    └── outputs.csv
```

---

## Reproducing paper results

### Step 1 — Preprocess (run once)
Splits raw CSVs into fixed train/test sets to prevent data leakage.
```bash
python scripts/00_preprocess.py
```

### Step 2 — Benchmark validation (Tables 5-8, Figures 4-7)
```bash
# Optional: re-tune hyperparameters (results already in configs/benchmarks_config.yaml)
python scripts/01_tune_benchmarks.py

# Run experiments
python scripts/02_run_benchmarks.py

# Generate tables and figures
python scripts/03_postprocess_benchmarks.py
```

### Step 3 — ONC experiments (Tables 9-12, Figure 8)
```bash
# Optional: re-tune hyperparameters on ONC data
python scripts/01_tune_onc.py

# Run full cost-matched budget sweep
python scripts/02_run_onc.py

# Generate tables and figures
python scripts/03_postprocess_onc.py
```

For long-running scripts on a shared cluster:
```bash
nohup python scripts/02_run_onc.py > outputs/run_onc.log 2>&1 &
tail -f outputs/run_onc.log
```

---

## Configuration

All hyperparameters and experiment settings are controlled via YAML config files in `configs/`. The scripts should never be edited directly to change hyperparameters.

- `configs/benchmarks_config.yaml` — benchmark hyperparameters (Tables 5-6), sample sizes, and run settings
- `configs/onc_config.yaml` — ONC hyperparameters (Table 9), data paths, and column definitions

Running `01_tune_benchmarks.py` or `01_tune_onc.py` will automatically update the relevant config file with the best hyperparameters found, which are then used by the corresponding run script.

---

## CPU thread limiting

All entry-point scripts set environment variables to limit CPU thread usage before importing any libraries. This is important on shared clusters to avoid consuming all available cores. The default limit is 4 threads, configurable via `cpu_threads:` in each config file.

---

## Methods

| Method | Type | Fidelities |
|--------|------|-----------|
| MF-GP | Gaussian Process (co-kriging) | 2F |
| GPmimic | Neural Network | 2F, 3F |
| MFNN-Delta | Neural Network (residual) | 2F |
| MFNN-Flag | Neural Network (fidelity flag) | 2F, 3F |
| MFNN-Intermediate | Neural Network (shared trunk) | 2F, 3F |
| MFNN-TwoStep | Neural Network | 2F |
| MFNN-ThreeStep | Neural Network | 2F |

---

## Citation

```bibtex
@article{eaheart2026multifidelity,
  title={Multifidelity Surrogate Modeling of Depressurized Loss of Forced Cooling 
         in High-temperature Gas Reactors},
  author={Eaheart, Meredith and Radaideh, Majdi I.},
  journal={arXiv preprint arXiv:2603.14143},
  year={2026}
}
```
