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
│   └── 4_FeatureWise2/
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
   - feature-wise lambda search experiments

4. `implementations/4_FeatureWise2`
   - additional feature-wise or second-generation exploratory work

### result/

Contains generated outputs from the experiments, including:

- comparison tables
- time measurements
- charts/plots
- intermediate benchmark artifacts

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

There is a convenience script at `main.py`, which exposes the benchmark targets in the current codebase:

```powershell
py -3 main.py --help
```

This supports the main benchmark choices:

- `baseline`
- `new_idea_1`
- `new_idea_2`
- `paper_tables`
- `all`

> Note: the canonical experimental scripts in active use are the files under `implementations/`. Those are the most reliable entry points for the current thesis workflow.

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

