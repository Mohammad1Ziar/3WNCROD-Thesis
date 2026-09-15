# 3WNCROD+ : Advanced Three-Way Neighborhood Characteristic Region Outlier Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Thesis%20Research%20Archive-success.svg)]()
[![Paper Reference](https://img.shields.io/badge/IEEE%20TKDE-2024-informational.svg)](https://doi.org/10.1109/TKDE.2023.3312108)

This repository hosts the comprehensive experimental codebase, mathematical treatises, datasets, and benchmark results for the **3WNCROD+** thesis project. It advances the state of the art in neighborhood rough-set and Three-Way Decision (3WD) anomaly detection, building upon the foundational paper:

> **Zhang, X., Yuan, Z., & Miao, D.** *"Outlier Detection Using Three-Way Neighborhood Characteristic Regions and Corresponding Fusion Measurement,"* **IEEE Transactions on Knowledge and Data Engineering (TKDE)**, vol. 36, no. 5, pp. 2082–2095, May 2024. [DOI: 10.1109/TKDE.2023.3312108](https://doi.org/10.1109/TKDE.2023.3312108).

---

## 🔬 Project Overview & Key Contributions

While the baseline **3WNCROD** algorithm introduced multiple neighborhood outlier factors (MNOF) based on neighborhood characteristic regions (NEB, NPB, POS), it relied on a single global granulation hyperparameter ($\lambda$), computationally intensive pairwise distance calculations ($O(n^2)$), and standard parametric scale measures.

**3WNCROD+** addresses these limitations across three primary dimensions:

1. **Feature-Wise Granulation ($\lambda_j$)**: Replaces the single uniform $\lambda$ with decoupled, attribute-specific neighborhood scales to handle multi-modal and heteroscedastic feature spaces.
2. **High-Performance Vectorization (~89× Speedup)**: Implements Single-Attribute Neighborhood Characterization (SANC) sorting ($O(n \log n)$), BLAS matrix operations, and blocked `uint8` counters to achieve drastic runtime reductions while guaranteeing bit-for-bit mathematical equivalence with the original baseline.
3. **Four-Generation Closed-Form Mathematical Evolution**:
   - **Gen 1 (Robust Granulation & Subspace Augmentation)**: Introduces robust scale estimators (MADN, IQR/1.34), Silverman AMISE bandwidth rules, synthetic subspace attributes via deflationary power iteration (SPE), and discriminative approximation weights $\gamma_{c_j}(X)$.
   - **Gen 2 (Decoupled Multi-Scale & Unsupervised Weighting)**: Resolves distribution skewness via asymmetric half-spread dispersion ($IQR_{eff}$), rate-calibrated cross-attribute voting, and unsupervised coefficient-of-variation feature weighting.
   - **Gen 3 (Copula Subspace Banks & Dual-Dispersion)**: Employs Kendall's $\tau$ and Gaussian copula mutual information ($\Delta_{jk}$) to detect nonlinear dependencies, dual-dispersion statistics, smoothed IDF-style region rarity weights, and dyadic multi-scale granulation.
   - **Gen 4 / 3WNCROD-IV (Topological Stability Principle)**: Solves AMISE neighborhood collapse with the $k^*$-Stable Radius law (Eq. 8†), variance-explained gated subspace augmentation ($\eta_r \ge 0.60$), effective-dimension adaptive NOM thresholds (Eq. 11†), and codomain-restored calibrated discriminative weights (Eq. 10†).

---

## 📊 Benchmark Results Summary

Experiments were conducted on multiple established outlier detection benchmark subsets (e.g., A1–A3 for Annthyroid, MA1–MA3 for Mammography, G1–G3 for German, etc.). Below is a summary comparison between baseline **3WNCROD** and **3WNCROD+** (full matrix in `result-all.png`):

| Dataset Subset | Baseline ROC-AUC | 3WNCROD+ ROC-AUC | Baseline AP | 3WNCROD+ AP |
|:---|:---:|:---:|:---:|:---:|
| **Annthyroid (A1)** | 0.732 | **0.898** | 0.437 | **0.704** |
| **Annthyroid (A2)** | 0.692 | **0.910** | 0.427 | **0.678** |
| **Annthyroid (A3)** | 0.775 | **0.945** | 0.278 | **0.526** |
| **German (G1)** | 0.987 | **0.989** | 0.561 | **0.570** |
| **German (G2)** | 0.986 | **0.990** | 0.543 | **0.620** |
| **German (G3)** | 0.996 | **0.998** | 0.501 | **0.715** |
| **Mammography (MA1)** | 0.893 | **0.920** | 0.355 | **0.565** |
| **Mammography (MA2)** | 0.775 | **0.832** | 0.073 | **0.122** |
| **Mammography (MA3)** | 0.788 | **0.799** | 0.187 | **0.318** |
| **Thyroid (TH1–TH3)** | 0.985–0.992 | **0.998** | 0.716–0.852 | **High/Stable** |
| **Lymphography / Musk / WDBC** | **1.000** | **1.000** | **1.000** | **1.000** |

*Note: Average Precision (AP / PR-AUC) is the primary metric given the extreme class imbalance inherent to anomaly detection.*

---

## 📁 Repository Structure

```text
├── 1. 3WNCROD/                        # Baseline implementation & reference paper
│   ├── 1. Perfect Implementation/      # Clean reference notebooks & MA benchmarks
│   ├── 3WNCROD-main/                  # Original MATLAB/Python package from authors
│   └── 3WNCROD-2024.pdf               # Reference TKDE publication PDF
│
├── 2. 3WNCROD+/                       # Research branch: extensions & algorithmic evolution
│   └── 1. Implementation/
│       ├── 1st - FeatureWise/         # Feature-wise lambda tuning across benchmark datasets
│       ├── 2nd -  Gen 1/              # Generation 1 implementations & exploratory notebooks
│       ├── 3rd - Running Time Optimised/ # ~89x faster vectorized drop-in replacement (SANC)
│       ├── 4th - Gen 2/               # Generation 2 multi-scale & voting experiments
│       ├── 5th - Gen 3/               # Generation 3 copula subspace & dual-dispersion runs
│       ├── 6th - FeatureWise + Gen 1/ # Combined feature-wise & Gen 1 hybrid trials
│       ├── 7th - Gen 4/               # Gen 4 (3WNCROD-IV) reference implementations
│       └── Gen's/                     # Consolidated scripts & notebooks (Gen 1-3 & Gen 4 treatise)
│
├── 3.Datasets/                        # Curated benchmark datasets (.csv, .mat, .xls)
│   ├── annthyroid                     # Annthyroid dataset
│   ├── creditA                        # Credit Approval dataset
│   ├── german                         # German Credit dataset
│   ├── heart270                       # Statlog Heart dataset
│   ├── lymphography                   # Lymphography dataset
│   ├── mammography                    # Mammography dataset (MA1, MA2, MA3 subsets)
│   ├── mushroom                       # Mushroom nominal dataset
│   ├── musk                           # Musk (Clean2) dataset
│   ├── thyroid                        # Standard Thyroid dataset
│   └── wdbc                           # Wisconsin Diagnostic Breast Cancer dataset
│
├── 4. Idea's/                         # Formal mathematical treatises & design specs
│   ├── Gen 1.md                       # Treatise I: Closed-Form Enhancements
│   ├── Gen 2.md                       # Treatise II: Multi-Scale Granulation & Calibrated Voting
│   └── Gen 3.md                       # Treatise III: Copula Subspace Banks & Novel Mechanisms
│
├── result-all.png                     # Benchmark comparison table (3WNCROD vs. 3WNCROD+)
└── README.md                          # Repository documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and activate a Python 3.10+ virtual environment:

```bash
git clone https://github.com/Mohammad1Ziar/3WNCROD-Thesis.git
cd 3WNCROD-Thesis

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core scientific dependencies
pip install numpy scipy pandas scikit-learn matplotlib jupyter
```

### 2. Running Baseline vs. Optimized 3WNCROD

To test the **~89× faster drop-in replacement** against the original implementation:

```python
# Import the optimized function
import sys
sys.path.append("2. 3WNCROD+/1. Implementation/3rd - Running Time Optimised")
from wncrod_optimized import WNCROD

# Execute outlier scoring (bit-for-bit identical to baseline)
# out_scores = WNCROD(data, X_tem, lammda)
```

### 3. Running Fourth-Generation (3WNCROD-IV) Experiments

To run the complete parameter-free, topologically-stabilized implementation:

```bash
python "2. 3WNCROD+/1. Implementation/7th - Gen 4/3WNCROD_IV_Implementation-1.py"
```

You can also explore and interact with the experiments using Jupyter:

```bash
jupyter notebook
```
Navigate to `2. 3WNCROD+/1. Implementation/Gen's/` to inspect `Gen 1.ipynb`, `Gen 2.ipynb`, or `Gen 3.ipynb`.

---

## 📖 Citation & Acknowledgments

If you find this repository, code, or experimental methodology useful in your academic research, please cite the primary literature:

```bibtex
@article{zhang2024outlier,
  title={Outlier Detection Using Three-Way Neighborhood Characteristic Regions and Corresponding Fusion Measurement},
  author={Zhang, Xianyong and Yuan, Zhong and Miao, Duoqian},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={36},
  number={5},
  pages={2082--2095},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TKDE.2023.3312108}
}
```

---
*Maintained as part of the 3WNCROD+ Graduate Thesis Archive by [Mohammad Ziar](https://github.com/Mohammad1Ziar).*
