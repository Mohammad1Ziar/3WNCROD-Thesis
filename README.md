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
│   │   ├── Run_custom_lambda.py
│   │   ├── Run_featurewise.py
│   │   └── wncrod_featurewise.py
│   ├── 4_FeatureWise2/
│   │   └── Run_multipass_featurewise.py
│   ├── 5_FeatureWise3/
│   │   ├── Run_featurewise3.py
│   │   └── wncrod_featurewise3.py
│   ├── 6_Metaheuristics/
│   │   ├── lambda_search.py
│   │   └── Run_metaheuristic_benchmark.py
│   ├── 6_Optimization1/
│   │   ├── Run_optuna.py
│   │   └── wncrod_optuna.py
│   ├── 7_Optimization2/
│   │   ├── Run_cmaes.py
│   │   └── wncrod_cmaes.py
│   └── 8_Optimization3/
├── Idea's/
│   ├── 1_featurewise_search_strategies.md
│   └── 2_metaheuristic_search_report.md
├── result/
│   ├── 1_reproduction/
│   ├── 2_Running Time Optimised/
│   ├── 3_FeatureWise/
│   ├── 4_FeatureWise2/
│   ├── 5_FeatureWise3/
│   ├── 6_Metaheuristics/
│   ├── 6_Optimization1/
│   │   ├── trials_100/
│   │   ├── trials_200/
│   │   └── trials_400/
│   ├── 7_Optimization2/
│   │   ├── MaxFevals100/
│   │   ├── MaxFevals200/
│   │   └── MaxFevals400/
│   └── 8_Optimization3/
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

6. `implementations/6_Metaheuristics`
   - unified metaheuristic lambda-search benchmark harness (`lambda_search.py`, `Run_metaheuristic_benchmark.py`) comparing 14 search adapters (`GreedyCD`, `RandomSearch`, `NelderMead`, `CMA-ES`, `DE`, `PSO`, `GA`, `BRKGA`, `AntColony`, `SimulatedAnnealing`, `OptunaTPE`, `CMA-ES (no tol)`, `HybridDE_CD`, `HybridCMAES_CD`) under a shared evaluation budget

7. `implementations/6_Optimization1`
   - Bayesian optimization of the feature-wise lambda vector using Optuna's TPE sampler (`Run_optuna.py`, `wncrod_optuna.py`)
   - joint search over active numerical feature lambdas in `[0.1, 3.0]` (step `0.1`, seed `42`), warm-started from all ones, with early stopping at a perfect score (`1.0`); default `400` trials per metric (`AP` and `ROC-AUC`), with saved runs for `100`, `200`, and `400` trials

8. `implementations/7_Optimization2`
   - CMA-ES (Covariance Matrix Adaptation Evolution Strategy via `pycma`) optimization of the feature-wise lambda vector (`Run_cmaes.py`, `wncrod_cmaes.py`)
   - learns the full covariance matrix over active numerical features, initialized at all ones (`sigma0 = 0.3`, bounds `[0.1, 3.0]`, discretization step `0.1`, seed `42`), with early stopping when `AP` or `ROC-AUC` reaches `1.0`; default `400` max function evaluations per metric, with saved runs for `100`, `200`, and `400` evaluations

9. `implementations/8_Optimization3`
   - placeholder directory for the next optimization variant (currently empty)

### result/

Contains generated outputs from the experiments, including:

- comparison tables (`Table1_Performance_Comparison*.csv` and `Table2_RunningTime_Comparison*.csv`)
- time measurements
- charts/plots (under `1_reproduction/charts/` and `2_Running Time Optimised/charts/`)
- budget-grouped subdirectories under `6_Optimization1/` (`trials_100/`, `trials_200/`, `trials_400/`) and `7_Optimization2/` (`MaxFevals100/`, `MaxFevals200/`, `MaxFevals400/`)
- a single-subset AP smoke test on `H2` (budget `120`, seed `1`) under `6_Metaheuristics/`

## Benchmark results: AP and ROC-AUC

The table below reports the recorded Average Precision (AP) and ROC-AUC for all
30 benchmark subsets. Each result cell is formatted as `AP / ROC-AUC`; execution
time, lambda vectors, and paper-reference values are intentionally omitted.
The values are transcribed from the corresponding performance CSVs in `result/`.
The reproduction and runtime-optimized runs have identical recorded scores, but
are shown separately because they are distinct implementations. The three
`Optimization1` columns show the recorded Optuna TPE runs (`100`, `200`, and `400`
trials, seed `42`, lambda step `0.1`), and the three `Optimization2` columns show
the recorded CMA-ES runs (`100`, `200`, and `400` max function evaluations,
`sigma0 = 0.3`, seed `42`, lambda step `0.1`).

`FeatureWise2 P5` and `FeatureWise3 P4` report the corresponding saved pass
outputs; the intermediate pass columns are omitted here. The final row reports
the average scores recorded in each performance CSV.

| Dataset | Reproduction | Optimized | FeatureWise | FeatureWise2 P5 | FeatureWise3 P4 | Optimization1 (100 trials) | Optimization1 (200 trials) | Optimization1 (400 trials) | Optimization2 (100 evals) | Optimization2 (200 evals) | Optimization2 (400 evals) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Annthyroid (A1) | 0.437 / 0.732 | 0.437 / 0.732 | 0.694 / 0.892 | 0.694 / 0.892 | 0.694 / 0.892 | 0.589 / 0.882 | 0.598 / 0.882 | 0.601 / 0.882 | 0.611 / 0.864 | 0.67 / 0.872 | 0.713 / 0.893 |
| Annthyroid (A2) | 0.427 / 0.692 | 0.427 / 0.692 | 0.633 / 0.881 | 0.727 / 0.881 | 0.722 / 0.88 | 0.699 / 0.877 | 0.701 / 0.878 | 0.727 / 0.881 | 0.479 / 0.877 | 0.556 / 0.877 | 0.563 / 0.881 |
| Annthyroid (A3) | 0.278 / 0.775 | 0.278 / 0.775 | 0.359 / 0.817 | 0.382 / 0.876 | 0.421 / 0.876 | 0.381 / 0.789 | 0.383 / 0.802 | 0.383 / 0.812 | 0.383 / 0.803 | 0.383 / 0.833 | 0.383 / 0.847 |
| CreditA (C1) | 0.800 / 0.990 | 0.800 / 0.990 | 0.903 / 0.996 | 0.903 / 0.997 | 0.891 / 0.996 | 0.831 / 0.996 | 0.851 / 0.996 | 0.881 / 0.997 | 0.817 / 0.991 | 0.823 / 0.992 | 0.823 / 0.992 |
| CreditA (C2) | 0.993 / 0.999 | 0.993 / 0.999 | 0.993 / 1 | 0.993 / 1 | 0.996 / 1 | 0.996 / 0.999 | 0.996 / 0.999 | 0.996 / 0.999 | 0.993 / 1 | 0.993 / 1 | 0.993 / 1 |
| CreditA (C3) | 0.978 / 0.997 | 0.978 / 0.997 | 0.98 / 0.998 | 0.98 / 0.998 | 0.982 / 0.998 | 0.981 / 0.998 | 0.982 / 0.998 | 0.982 / 0.998 | 0.981 / 0.998 | 0.981 / 0.998 | 0.981 / 0.998 |
| German (G1) | 0.561 / 0.987 | 0.561 / 0.987 | 0.57 / 0.989 | 0.604 / 0.991 | 0.607 / 0.991 | 0.589 / 0.99 | 0.604 / 0.99 | 0.604 / 0.99 | 0.59 / 0.99 | 0.607 / 0.99 | 0.607 / 0.99 |
| German (G2) | 0.543 / 0.986 | 0.543 / 0.986 | 0.619 / 0.99 | 0.619 / 0.99 | 0.582 / 0.991 | 0.586 / 0.989 | 0.589 / 0.99 | 0.589 / 0.991 | 0.558 / 0.99 | 0.572 / 0.99 | 0.601 / 0.991 |
| German (G3) | 0.501 / 0.996 | 0.501 / 0.996 | 0.715 / 0.998 | 0.799 / 0.998 | 0.679 / 0.998 | 0.761 / 0.998 | 0.768 / 0.998 | 0.768 / 0.998 | 0.585 / 0.997 | 0.585 / 0.997 | 0.61 / 0.997 |
| Heart (H1) | 0.931 / 0.990 | 0.931 / 0.990 | 0.986 / 0.999 | 0.986 / 0.999 | 0.975 / 0.997 | 0.946 / 0.997 | 0.966 / 1 | 0.986 / 1 | 0.931 / 0.991 | 0.931 / 0.994 | 0.943 / 0.994 |
| Heart (H2) | 0.973 / 0.998 | 0.973 / 0.998 | 0.991 / 0.999 | 0.991 / 0.999 | 1 / 1 | 1 / 0.999 | 1 / 1 | 1 / 1 | 0.983 / 0.998 | 0.991 / 0.998 | 0.991 / 0.998 |
| Heart (H3) | 0.817 / 0.988 | 0.817 / 0.988 | 1 / 1 | 1 / 1 | 0.861 / 0.994 | 1 / 0.996 | 1 / 0.998 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Lymphography (L1) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Lymphography (L2) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Lymphography (L3) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Mammography (MA1) | 0.355 / 0.893 | 0.355 / 0.893 | 0.565 / 0.919 | 0.565 / 0.926 | 0.565 / 0.926 | 0.417 / 0.905 | 0.446 / 0.917 | 0.516 / 0.917 | 0.479 / 0.906 | 0.508 / 0.918 | 0.559 / 0.925 |
| Mammography (MA2) | 0.072 / 0.774 | 0.072 / 0.774 | 0.136 / 0.832 | 0.138 / 0.836 | 0.175 / 0.827 | 0.125 / 0.794 | 0.132 / 0.799 | 0.133 / 0.799 | 0.09 / 0.804 | 0.11 / 0.812 | 0.113 / 0.816 |
| Mammography (MA3) | 0.187 / 0.788 | 0.187 / 0.788 | 0.342 / 0.781 | 0.445 / 0.877 | 0.448 / 0.877 | 0.39 / 0.833 | 0.42 / 0.842 | 0.44 / 0.872 | 0.309 / 0.779 | 0.344 / 0.798 | 0.363 / 0.841 |
| Mushroom (M1) | 0.920 / 0.985 | 0.920 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 | 0.92 / 0.985 |
| Mushroom (M2) | 0.898 / 0.940 | 0.898 / 0.940 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 | 0.898 / 0.94 |
| Mushroom (M3) | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 | 0.753 / 0.979 |
| Musk (MU1) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Musk (MU2) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Musk (MU3) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Thyroid (TH1) | 0.716 / 0.985 | 0.716 / 0.985 | 0.798 / 0.991 | 0.8 / 0.991 | 0.8 / 0.991 | 0.799 / 0.991 | 0.799 / 0.991 | 0.8 / 0.991 | 0.777 / 0.99 | 0.795 / 0.99 | 0.8 / 0.991 |
| Thyroid (TH2) | 0.761 / 0.985 | 0.761 / 0.985 | 0.851 / 0.995 | 0.929 / 0.997 | 0.94 / 0.996 | 0.894 / 0.995 | 0.895 / 0.996 | 0.896 / 0.996 | 0.782 / 0.989 | 0.805 / 0.996 | 0.814 / 0.996 |
| Thyroid (TH3) | 0.852 / 0.992 | 0.852 / 0.992 | 0.882 / 0.996 | 0.897 / 0.996 | 0.884 / 0.996 | 0.873 / 0.995 | 0.873 / 0.996 | 0.874 / 0.996 | 0.87 / 0.996 | 0.882 / 0.996 | 0.883 / 0.996 |
| WDBC (W1) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| WDBC (W2) | 1.000 / 1.000 | 1.000 / 1.000 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| WDBC (W3) | 0.921 / 0.995 | 0.921 / 0.995 | 0.979 / 0.996 | 0.979 / 1 | 1 / 1 | 0.979 / 0.999 | 0.979 / 0.999 | 0.983 / 0.999 | 0.929 / 0.983 | 0.929 / 0.991 | 0.994 / 0.995 |
| **Average** | **0.756 / 0.948** | **0.756 / 0.948** | **0.819 / 0.966** | **0.833 / 0.972** | **0.826 / 0.971** | **0.814 / 0.964** | **0.818 / 0.966** | **0.824 / 0.967** | **0.791 / 0.962** | **0.801 / 0.965** | **0.81 / 0.968** |

The user-supplied lambda runner writes per-run summaries under
`implementations/3_FeatureWise/custom_results/`. The `6_Metaheuristics` harness
currently has only a single-subset AP smoke test on `H2` (budget `120`, seed `1`)
in `result/6_Metaheuristics/`. Because neither has a full 30-subset benchmark
pass CSV, they are not represented as columns in the table above.

---

## How to run the experiments

The project is designed for a Python environment with NumPy, SciPy, pandas, scikit-learn, and Matplotlib installed (plus `optuna` for `6_Optimization1`, `cma` for `7_Optimization2`, and `pymoo` for `6_Metaheuristics`).

### Windows (recommended)

```powershell
py -3 -m pip install numpy pandas scipy scikit-learn matplotlib optuna cma
py -3 "implementations/1_reproduction/run_baseline.py"
py -3 "implementations/2_Running Time Optimised/Run_optimised.py"
py -3 "implementations/3_FeatureWise/Run_featurewise.py"
py -3 "implementations/6_Optimization1/Run_optuna.py" ALL --trials 400 --seed 42
py -3 "implementations/7_Optimization2/Run_cmaes.py" ALL --maxfevals 400 --seed 42
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

