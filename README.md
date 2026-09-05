# 3WNCROD Thesis Repository

This repository contains the complete experimental and archival material for the 3WNCROD thesis project. It brings together the original implementation, multiple extended variants of the method, runtime-optimized experiments, AI-inspired variants, and the dataset resources used for validation and comparison.

## Project overview

The research workflow evolved through several stages:

- Baseline 3WNCROD reference implementation
- Feature-wise lambda experimentation
- AI-inspired variant exploration
- Runtime optimization and benchmarking
- MA-focused experimental runs and result comparison

The repository is organized to preserve the full history of those iterations while keeping the main experiments grouped by phase.

## Repository structure

- `1. 3WNCROD/`
  - Original 3WNCROD materials and reference implementation
  - `1. Perfect Implementation/`
    - clean implementation and MA-related notebook experiments
  - `3WNCROD-main/`
    - source code, datasets, and project paper copy from the original implementation package

- `2. 3WNCROD+/`
  - Extended research branch covering multiple experimental ideas and implementation variants
  - `1. Implementation/`
    - `1st - FeatureWise Lambda/`
      - initial feature-wise lambda analysis and early notebook variants
    - `2nd - AI Idea's/`
      - AI-assisted exploratory implementations and variant tests
    - `3rd - Running Time Optimised/`
      - optimized runtime implementation and benchmarking-focused notebook/script files
    - `4th AI Idea's/`
      - newer prototype investigation and exploratory MA experiments

- `3.Datasets/`
  - All dataset files used for training, testing, and analysis across the project

- `3WNCROD-2024.pdf`
  - thesis report or final project document

- `main.py`
  - main execution entry script for the repository workflow

- `.venv/`
  - local Python environment used for analysis, notebooks, and experiments

## Research workflow by phase

### 1. Original implementation
The baseline 3WNCROD code and supporting notebooks are kept under `1. 3WNCROD/` to preserve the original formulation and comparison reference.

### 2. Extended variants
The `2. 3WNCROD+/` directory records multiple successive experiments, including:

- feature-wise lambda tuning
- AI-driven variant testing
- MA-specific evaluation pipelines
- runtime optimization trials

### 3. Runtime optimization
The `3rd - Running Time Optimised` section focuses on reducing computational cost while preserving the behavioral logic of the original method. It includes optimized Python implementations and notebook-based benchmarking.

### 4. Final experimental archive
The later folders capture the most recent AI-based and optimized exploration attempts, making the repository useful as both a research archive and a working experimental notebook collection.

## How to use this repository

1. Start with the notebooks in `1. 3WNCROD/` for the original baseline.
2. Compare them with the experiments in `2. 3WNCROD+/1. Implementation/`.
3. Use the dataset files in `3.Datasets/` for reproducible testing.
4. Run the optimized or MA-focused scripts from the relevant implementation folders when benchmarking or reproducing results.

## Notes

- The project is heavily notebook-driven, with experimental analysis stored in Jupyter notebooks.
- The repository reflects an iterative research process rather than a single final static implementation.
- Datasets, notebooks, and source files are all included to support reproducibility and thesis documentation.
