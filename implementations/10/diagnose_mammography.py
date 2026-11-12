"""
diagnose_mammography.py
Diagnostic tool to analyze Scale Collapse, Density, and Neighborhood Statistics
on Mammography (MA1, MA2, MA3) and Annthyroid (A1).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Setup project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IDEA2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IDEA2_DIR))

from data_loaders import load_subset
from scale_estimators import sigma_mad, sigma_sn, sigma_hybrid

TARGET_SUBSETS = ["MA1", "MA2", "MA3", "A1"]

def analyze_subset(sub_name: str):
    print("=" * 80)
    print(f"📊 DIAGNOSTIC ANALYSIS FOR SUBSET: {sub_name}")
    print("=" * 80)
    
    X, y, X_tem = load_subset(sub_name)
    n_samples, m_features = X.shape
    outlier_ratio = np.mean(y == 1) * 100
    
    print(f"Total Objects (N): {n_samples} | Features: {m_features} | Outliers: {np.sum(y == 1)} ({outlier_ratio:.2f}%)")
    
    records = []
    
    for j in range(m_features):
        col = X[:, j]
        
        # 1. Classical vs Robust Scales
        s_std = float(np.std(col))
        s_mad = sigma_mad(col)
        s_sn = sigma_sn(col)
        s_hyb = sigma_hybrid(col, w=0.5)
        
        ratio_std_mad = s_std / (s_mad + 1e-12)
        ratio_std_sn = s_std / (s_sn + 1e-12)
        ratio_std_hyb = s_std / (s_hyb + 1e-12)
        
        # 2. Distribution Properties
        col_med = float(np.median(col))
        q25, q75 = np.percentile(col, [25, 75])
        iqr = q75 - q25
        
        # Calculate percentage of ties and exact zeros
        _, counts = np.unique(col, return_counts=True)
        max_tie_pct = (np.max(counts) / n_samples) * 100
        zeros_pct = (np.sum(col == 0.0) / n_samples) * 100
        near_median_pct = (np.sum(np.abs(col - col_med) <= 1e-4) / n_samples) * 100
        
        col_skew = float(skew(col))
        col_kurt = float(kurtosis(col))
        
        # 3. Neighborhood Density Impact (at lambda = 1.0)
        # Fraction of points falling in radius for point 0
        eps_paper = s_std / 1.0
        eps_hybrid = s_hyb / 1.0
        
        # Average neighbors per sample on this single feature (sample subset of 1000 for speed)
        sub_idx = np.random.choice(n_samples, min(1000, n_samples), replace=False)
        sub_col = col[sub_idx]
        diffs = np.abs(sub_col[:, None] - sub_col[None, :])
        
        avg_neigh_paper = np.mean(np.sum(diffs <= eps_paper, axis=1))
        avg_neigh_hybrid = np.mean(np.sum(diffs <= eps_hybrid, axis=1))
        
        records.append({
            "Feature": f"F{j+1}",
            "std": round(s_std, 4),
            "MAD": round(s_mad, 4),
            "Sn": round(s_sn, 4),
            "Hybrid": round(s_hyb, 4),
            "Ratio_std/Hyb": round(ratio_std_hyb, 1),
            "Median": round(col_med, 4),
            "IQR": round(iqr, 4),
            "Max_Tie(%)": round(max_tie_pct, 1),
            "Near_Med(%)": round(near_median_pct, 1),
            "Zeros(%)": round(zeros_pct, 1),
            "Skewness": round(col_skew, 2),
            "Kurtosis": round(col_kurt, 2),
            "Neigh_Paper": round(avg_neigh_paper, 1),
            "Neigh_Hybrid": round(avg_neigh_hybrid, 1),
        })
        
    df_res = pd.DataFrame(records)
    print(df_res.to_string(index=False))
    print("\n")
    return df_res

if __name__ == "__main__":
    for sub in TARGET_SUBSETS:
        analyze_subset(sub)