# 3WNCROD / 3WNCROD+ Thesis Research Repository

> This repository is an active research project created for master’s thesis work. It is continuously evolving, and its folder structure, scripts, and experimental ideas are expected to change as the study progresses.

## 1) Project Overview

This repository contains the implementation, benchmarking, and experimental code related to outlier detection research based on 3WNCROD and its extended variants. The work focuses on detecting anomalous samples in numerical, categorical, and mixed-type datasets using neighborhood-based and decision-theoretic approaches.

The project is intended as a research workspace rather than a final packaged library. It is used to run experiments, test hypotheses, compare variants, and generate evidence for the thesis.

The current objectives include:

- reviewing and implementing the base 3WNCROD method
- developing optimized and hybrid variants
- evaluating performance on standard anomaly detection datasets
- comparing models using ROC-AUC and Average Precision (AP)
- creating a reusable code and data foundation for thesis work

---

## 2) Current Repository Status

The repository is structured as an active research codebase with multiple layers of work:

1. baseline reference code and benchmark data
2. dataset loading and subset definitions
3. evaluator modules for measuring performance
4. experimental scripts for new ideas and optimization attempts
5. results and intermediate outputs for analysis

This means the repository mixes reference materials, exploratory code, active experiments, and current development artifacts in one place.

---

## 3) Repository Structure

```text
.
├── README.md                          # Main project documentation
├── main.py                            # Placeholder / development entry point
├── result-all.png                     # Result-summary visualization
├── verify_subsets.py                  # Checks subset definitions against benchmark expectations
├── test_trash.py                      # Temporary experimental file
├── baseline/                          # Reference implementation and benchmark datasets
│   ├── Code/
│   ├── datasets/
│   ├── papers/
│   └── reproduction/
├── data_loaders/                      # Data loading and subset registration logic
│   ├── base.py
│   ├── criteria.py
│   ├── loader.py
│   └── __init__.py
├── evaluators/                        # Evaluation metrics and scoring logic
│   ├── metrics.py
│   └── __init__.py
├── New_idea/                          # New ideas and experimental benchmark runs
│   ├── 1/
│   └── 2/
├── utils/                             # Diagnostics and helper scripts
│   └── 1_initial_diagnostics/
├── 2. 3WNCROD+/                       # Main experimental area with multiple algorithm variants
│   ├── 1st - FeatureWise/
│   ├── 2nd - Gen 1/
│   ├── 3rd - Running Time Optimised/
│   ├── 4th - Gen 2/
│   ├── 5th - Gen 3/
│   ├── 6th - FeatureWise + Gen 1/
│   ├── 7th - Gen 4/
│   └── Gen's/
└── .venv/                             # Local Python environment
```

> Note: some folders contain spaces in their names, so quoted paths are recommended when running scripts from the shell.

---

## 4) Key Project Areas

### 4.1) baseline

This folder contains the reference implementation and benchmark datasets used as the starting point for comparison and validation.

It includes:

- reference code for the original method
- datasets used in experiments
- paper-related outputs and reproduction scripts

### 4.2) data_loaders

This part of the repository is responsible for loading the benchmark datasets and matching them to the correct predefined subset criteria.

Main files:

- `data_loaders/criteria.py` — defines benchmark subset families and expected counts
- `data_loaders/loader.py` — loads the dataset subset for each experiment

### 4.3) evaluators

This part implements the model evaluation logic, primarily focusing on:

- ROC-AUC
- Average Precision (AP)
- outlier ranking and anomaly identification

### 4.4) New_idea

This directory stores the newer experimental directions, including hybrid and optimized versions of the method.

Examples include:

- `New_idea/1/`
- `New_idea/2/run_hybrid_benchmark.py`
- `New_idea/2/wncrod_hybrid.py`

### 4.5) 2. 3WNCROD+

This is the main research and experimentation area of the project. It contains multiple versions and generations of the method, such as:

- FeatureWise
- Gen 1
- Gen 2
- Gen 3
- Gen 4
- runtime-optimized implementations

This layout reflects the fact that the repository is still in an active thesis-development phase and not yet a finalized stable release.

---

## 5) Common Setup and Execution

### 5.1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.2) Install dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

### 5.3) Validate the benchmark subsets

```bash
python verify_subsets.py
```

### 5.4) Run the experimental benchmark pipeline

```bash
python "New_idea/2/run_hybrid_benchmark.py"
```

### 5.5) Run scripts inside folders with spaces in their names

```bash
python "2. 3WNCROD+/1. Implementation/7th - Gen 4/3WNCROD_IV_Implementation-1.py"
```

---

## 6) Research Workflow

The normal workflow for this project is:

1. define a new hypothesis or variant
2. run experiments on benchmark datasets
3. compare results with the baseline or earlier versions
4. inspect metric values, failures, and runtime behavior
5. refine the implementation and repeat

Because of this workflow, the repository is expected to evolve continuously while the thesis work develops.

---

## 7) Main Reference

The base algorithm used in this project is derived from the following paper:

> Zhang, X., Yuan, Z., and Miao, D.  
> "Outlier Detection Using Three-Way Neighborhood Characteristic Regions and Corresponding Fusion Measurement"  
> IEEE Transactions on Knowledge and Data Engineering, 2024.

The `baseline/` directory and the `result-all.png` file are the primary reference points for benchmark context and comparison outputs.

---

## 8) Important Notes

- This is a thesis research repository, not a final packaged software release.
- Many folders and files are exploratory or experimental by design.
- Before deleting or renaming files, confirm whether they are still used in active experiments.
- Always use quoted paths when executing files under directories with spaces in their names.
- For documentation and comparison, use the versions under `baseline/` and `New_idea/` as a reference.

---

## 9) Summary

This repository is a live research environment for developing and testing outlier-detection methods based on 3WNCROD and its variants. It combines reference code, benchmark datasets, active experimental implementations, and thesis-driven exploration in a single workspace.

If you are beginning work on this project, start with `baseline/`, `data_loaders/`, and `New_idea/2/` to understand the current experimental structure and workflow.

