# multifidelity_methods_clean

Clean, script-based version of the MFNN pipeline.
The old `MULTIFIDELITY_METHODS/` folder is untouched — this is a fresh start.

## Folder structure

```
multifidelity_methods_clean/
│
├── twofid_methods/          # 2-fidelity method implementations (device-fixed)
├── threefid_methods/        # 3-fidelity method implementations (device-fixed)
├── benchmarks/              # Benchmark function definitions (mf2 wrappers + custom 3f)
│
├── src/
│   └── datasets.py          # ONC data loading + Dataset2f/3f builders
│
├── configs/
│   ├── benchmarks_config.yaml   # hyperparams + run settings for benchmark experiments
│   └── onc_config.yaml          # hyperparams + data paths for ONC experiments
│
├── scripts/                 # numbered — run in order
│   ├── 00_preprocess.py         # split raw ONC CSVs into train/test (run once)
│   ├── 01_tune_benchmarks.py    # tune 2f methods on mf2 benchmarks
│   ├── 01_tune_onc.py           # tune 2f methods on ONC data
│   ├── 02_run_benchmarks.py     # run 2f + 3f benchmark experiments
│   ├── 02_run_onc.py            # run 2f + 3f ONC experiments
│   └── 03_postprocess_onc.py    # LaTeX tables + figures from ONC results CSV
│
├── outputs/
│   ├── results/             # saved CSVs + .tex files
│   └── figures/             # saved plots
│
├── helpers_2f.py            # shared utilities: MLP, train_torch_regressor,
│                            #   get_device(), limit_cpu_threads()
├── helpers_3f.py            # 3f dataset maker + re-exports from helpers_2f
├── orchestration_2f.py      # run_for_function() for 2f benchmarks
└── orchestration_3f.py      # run_for_function() for 3f benchmarks
```

## Data placement

Your raw data goes here — **not included in the repo, stays local:**
```
multifidelity_methods_clean/
└── ONC_data/
    ├── high_fidelity/
    │   ├── inputs.csv       ← raw (you have this)
    │   └── outputs.csv      ← raw (you have this)
    ├── medium_fidelity/
    │   ├── inputs.csv
    │   └── outputs.csv
    └── low_fidelity/
        ├── inputs.csv
        └── outputs.csv
```
Run `00_preprocess.py` once after placing the data — it creates all the
`hf_training_inputs.csv`, `mf_testing_outputs.csv` etc. that the run scripts need.

## How to run

### Limit CPU threads (important on shared cluster)
All scripts call `limit_cpu_threads()` automatically using the value in the config.
Default is 4. Change `cpu_threads:` in the relevant config file.

### Benchmark pipeline
```bash
# Optional: tune hyperparams first (slow — runs full grid search)
python scripts/01_tune_benchmarks.py

# Run experiments (reads hyperparams from configs/benchmarks_config.yaml)
python scripts/02_run_benchmarks.py                  # both 2f and 3f
python scripts/02_run_benchmarks.py --fidelity 2f    # 2f only
python scripts/02_run_benchmarks.py --fidelity 3f    # 3f only
```

### ONC pipeline
```bash
# Step 0: run ONCE on fresh data to create train/test split CSVs
# Put your ONC_data/ folder at the project root first, then:
python scripts/00_preprocess.py

# Tune on ONC data
python scripts/01_tune_onc.py

# Run full experiment sweep (reads configs/onc_config.yaml)
python scripts/02_run_onc.py
python scripts/02_run_onc.py --fidelity 2f   # 2f only

# Generate LaTeX tables and figures from saved CSV
python scripts/03_postprocess_onc.py
python scripts/03_postprocess_onc.py --results outputs/results/onc_results.csv
```

### Run in background on cluster
```bash
nohup python scripts/02_run_onc.py > outputs/onc_run.log 2>&1 &
tail -f outputs/onc_run.log
```

## Key design decisions

- **One config file per pipeline** — change hyperparams in `configs/`, never in the scripts
- **`get_device()`** — all methods use this single function; GPU if available, CPU otherwise
- **`limit_cpu_threads(n)`** — call once at startup, limits PyTorch + NumPy/sklearn BLAS threads
- **Numbered scripts** — always run 01 → 02 → 03
- **Shared runners** — `orchestration_2f.py` and `orchestration_3f.py` work for both
  benchmarks and ONC; only the data loading differs (via `src/datasets.py`)
