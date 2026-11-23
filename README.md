# 3WNCROD Thesis Research Repository

This repository contains the experimental code, benchmark datasets, and evaluation pipeline used in the thesis work around 3WNCROD and its variants for anomaly detection.

The project is a research workspace rather than a final packaged library. It is organized around reproducible benchmark runs, runtime-optimized implementations, and feature-wise extensions of the original method.

---

## Project goal

The main objective is to study and compare anomaly-detection performance for benchmark datasets using the 3WNCROD family of methods. The repository includes:

- baseline reproduction experiments
- optimized runtime implementations
- feature-wise lambda search variants
- comparison against paper-reported results
- dataset loading and metric evaluation utilities

---

## Current repository layout

```text
.
├── README.md
├── main.py
├── baseline/
│   ├── Code/
│   ├── datasets/
│   └── papers/
├── data_loaders/
│   ├── __init__.py
│   ├── base.py
│   ├── criteria.py
│   └── loader.py
├── evaluators/
│   ├── __init__.py
│   └── metrics.py
├── implementations/
│   ├── 1_reproduction/
│   │   ├── run_baseline.py
│   │   └── generate_paper_tables.py
│   ├── 2_Running Time Optimised/
│   │   ├── Run_optimised.py
│   │   ├── wncrod_optimized.py
│   │   └── ...
│   ├── 3_FeatureWise/
│   │   ├── Run_featurewise.py
│   │   ├── wncrod_featurewise.py
│   │   └── ...
│   ├── 4_FeatureWise2/
│   │   └── Run_multipass_featurewise.py
│   └── 5_FeatureWise3/
│       ├── Run_featurewise3.py
│       └── wncrod_featurewise3.py
├── Idea's/
│   └── 1_featurewise_search_strategies.md
├── result/
│   ├── 1_reproduction/
│   ├── 2_Running Time Optimised/
│   ├── 3_FeatureWise/
│   └── 4_FeatureWise2/
└── .gitignore
```

---

## Main code areas

### baseline/

This folder holds the reference benchmark materials used for comparison and reproduction.

It includes:

- MATLAB/Python reference implementations under `baseline/Code/`
- benchmark datasets under `baseline/datasets/`
- paper-related notes and supporting files under `baseline/papers/`

### data_loaders/

Responsible for dataset loading and subset definitions used in the benchmark pipeline.

Key files:

- `data_loaders/criteria.py` — benchmark subset metadata and selection rules
- `data_loaders/loader.py` — data loading helpers
- `data_loaders/base.py` — common dataset abstractions and defaults

### evaluators/

Contains the evaluation logic used to score predictions.

Typical metrics include:

- ROC-AUC
- Average Precision (AP)
- runtime-aware benchmark summaries

### implementations/

This is the active execution layer of the project. It contains the runnable experimental scripts and the core algorithm variants.

Available folders:

1. `implementations/1_reproduction`
   - baseline reproduction runs and paper table generation

2. `implementations/2_Running Time Optimised`
   - faster version of 3WNCROD with runtime-focused benchmark pipeline

3. `implementations/3_FeatureWise`
   - one lambda per feature, searched with greedy coordinate ascent
   - `Run_custom_lambda.py` runs a user-supplied lambda vector for one subset

4. `implementations/4_FeatureWise2`
   - multi-pass feature-wise coordinate search (up to five passes)

5. `implementations/5_FeatureWise3`
   - multi-pass feature-wise search with convergence and perfect-score stopping

### result/

Contains generated outputs from the experiments, including:

- comparison tables
- time measurements
- charts/plots
- intermediate benchmark artifacts

## Benchmark results: AP and ROC-AUC

The table below reports the recorded Average Precision (AP) and ROC-AUC for all
30 benchmark subsets. Each result cell is formatted as `AP / ROC-AUC`; execution
time, lambda vectors, and paper-reference values are intentionally omitted.
The values are transcribed from the corresponding performance CSVs in `result/`.
The reproduction and runtime-optimized runs have identical recorded scores, but
are shown separately because they are distinct implementations.

`FeatureWise2 P5` and `FeatureWise3 P4` report the corresponding saved pass
outputs; the intermediate pass columns are omitted here. Average rows from the
CSV files are omitted so that every row represents one benchmark subset.

| Dataset | Reproduction | Optimized | FeatureWise | FeatureWise2 P5 | FeatureWise3 P4 |
|---|---:|---:|---:|---:|---:|
| Annthyroid (A1) | 0.437 / 0.732 | 0.437 / 0.732 | 0.694 / 0.892 | 0.694 / 0.892 | 0.694 / 0.892 |
| Annthyroid (A2) | 0.427 / 0.692 | 0.427 / 0.692 | 0.633 / 0.881 | 0.727 / 0.881 | 0.722 / 0.88 |
| Annthyroid (A3) | 0.278 / 0.775 | 0.278 / 0.775 | 0.359 / 0.817 | 0.382 / 0.876 | 0.421 / 0.876 |
| CreditA (C1) | 0.800 / 0.990 | 0.800 / 0.990 | 0.903 / 0.996 | 0.903 / 0.997 | 0.891 / 0.996 |
| CreditA (C2) | 0.993 / 0.999 | 0.993 / 0.999 | 0.993 / 1 | 0.993 / 1 | 0.996 / 1 |
| CreditA (C3) | 0.978 / 0.997 | 0.978 / 0.997 | 0.98 / 0.998 | 0.98 / 0.998 | 0.982 / 0.998 |
| German (G1) | 0.561 / 0.987 | 0.561 / 0.987 | 0.57 / 0.989 | 0.604 / 0.991 | 0.607 / 0.991 |
| German (G2) | 0.543 / 0.986 | 0.543 / 0.986 | 0.619 / 0.99 | 0.619 / 0.99 | 0.582 / 0.991 |
| German (G3) | 0.501 / 0.996 | 0.501 / 0.996 | 0.715 / 0.998 | 0.799 / 0.998 | 0.679 / 0.998 |
| Heart (H1) | 0.931 / 0.990 | 0.931 / 0.990 | 0.986 / 0.999 | 0.986 / 0.999 | 0.975 / 0.997 |
| Heart (H2) | 0.973 / 0.998 | 0.973 / 0.998 | 0.991 / 0.999 | 0.991 / 0.999 | 1 / 1 |
| Heart (H3) | 0.817 / 0.988 | 0.817 / 0.988 | 1 / 1 | 1 / 1 | 0.861 / 0.994 |
| Lymphography (L1) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| Lymphography (L2) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| Lymphography (L3) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| Mammography (MA1) | 0.355 / 0.893 | 0.355 / 0.893 | 0.565 / 0.919 | 0.565 / 0.926 | 0.565 / 0.926 |
| Mammography (MA2) | 0.072 / 0.774 | 0.072 / 0.774 | 0.136 / 0.832 | 0.138 / 0.836 | 0.175 / 0.827 |
| Mammography (MA3) | 0.187 / 0.788 | 0.187 / 0.788 | 0.342 / 0.781 | 0.445 / 0.877 | 0.448 / 0.877 |
| Mushroom (M1) | 0.920 / 0.985 | 0.920 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 |
| Mushroom (M2) | 0.898 / 0.940 | 0.898 / 0.940 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 |
| Mushroom (M3) | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 |
| Musk (MU1) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| Musk (MU2) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| Musk (MU3) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| Thyroid (TH1) | 0.716 / 0.985 | 0.716 / 0.985 | 0.798 / 0.991 | 0.8 / 0.991 | 0.8 / 0.991 |
| Thyroid (TH2) | 0.761 / 0.985 | 0.761 / 0.985 | 0.851 / 0.995 | 0.929 / 0.997 | 0.94 / 0.996 |
| Thyroid (TH3) | 0.852 / 0.992 | 0.852 / 0.992 | 0.882 / 0.996 | 0.897 / 0.996 | 0.884 / 0.996 |
| WDBC (W1) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| WDBC (W2) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 |
| WDBC (W3) | 0.921 / 0.995 | 0.921 / 0.995 | 0.979 / 0.996 | 0.979 / 1 | 1 / 1 |
| **Average** | **0.756 / 0.948** | **0.756 / 0.948** | **0.819 / 0.966** | **0.833 / 0.972** | **0.826 / 0.971** |

The user-supplied lambda runner writes per-run summaries under
`implementations/3_FeatureWise/custom_results/`. Because its inputs are chosen
by the caller and it has no fixed benchmark pass CSV, it is not represented as
a separate score column above.

---

## How to run the experiments

The project is designed for a Python environment with NumPy, SciPy, pandas, scikit-learn, and Matplotlib installed.

### Windows (recommended)

```powershell
py -3 -m pip install numpy pandas scipy scikit-learn matplotlib
py -3 "implementations/1_reproduction/run_baseline.py"
py -3 "implementations/2_Running Time Optimised/Run_optimised.py"
py -3 "implementations/3_FeatureWise/Run_featurewise.py"
```

### Using the project root entry point

The project root also provides `main.py` as a convenience launcher:

```powershell
py -3 main.py --help
```

At present, the implementation-specific runners under `implementations/` are the canonical entry points. They write their generated comparison tables and charts to the corresponding folder under `result/`.

Before a long benchmark run, verify the Python environment and the required packages. Use a virtual environment when the repository is shared across multiple machines:

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

---

## Important notes

- This repository is an evolving research codebase, not a finalized software package.
- The active logic is spread across the `implementations/` directory and the benchmark utilities in `data_loaders/` and `evaluators/`.
- Folder names with spaces require quoted paths when run from a shell.
- Results and tables are stored under `result/` and should be used for comparative analysis and thesis reporting.

---

## Reference context

The project is based on the 3WNCROD anomaly-detection method and its neighborhood-based outlier scoring ideas. The benchmark datasets and result tables are used to compare reproduction quality, optimization gains, and feature-wise search strategies against the original paper baselines.

---

## Recommended starting points

For a new contributor, the best order is:

1. `baseline/` for the original benchmark reference materials
2. `data_loaders/` for loading the dataset subsets
3. `evaluators/` for scoring and metric logic
4. `implementations/1_reproduction/` for the baseline reproduction
5. `implementations/2_Running Time Optimised/` and `implementations/3_FeatureWise/` for the experimental variants

This gives the clearest understanding of the workflow from benchmark setup to result generation.

