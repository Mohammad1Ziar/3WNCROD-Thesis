"""
data_loaders/loader.py
Primary loader interface. Combines CSV index extraction with MAT feature matrices.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from .criteria import SUBSET_REGISTRY
from .base import load_csv_dataframe, load_mat_matrix

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "baseline" / "datasets"


def load_subset(
    subset_name: str,
    datasets_dir: Optional[Path | str] = None,
    validate: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a benchmark subset according to Table V specifications.

    Parameters
    ----------
    subset_name : str
        Name of the subset (e.g. 'MA1', 'MA2', 'A1', 'G3', etc.)
    datasets_dir : Path or str, optional
        Directory where datasets are stored. Defaults to 3.Datasets.
    validate : bool, optional
        Whether to check sample count and outlier count against Table V.

    Returns
    -------
    X : np.ndarray
        Normalized feature matrix of shape (n_samples, n_features) from .mat.
    y : np.ndarray
        Binary labels array of shape (n_samples,) where 1=outlier, 0=normal.
    X_tem : np.ndarray
        Array of selected row indices extracted from the corresponding .csv.
    """
    name = subset_name.upper()
    if name not in SUBSET_REGISTRY:
        raise ValueError(f"Unknown subset: '{subset_name}'. Choose from: {list(SUBSET_REGISTRY.keys())}")

    spec = SUBSET_REGISTRY[name]
    dir_path = Path(datasets_dir) if datasets_dir else DEFAULT_DATA_DIR

    csv_path = dir_path / spec["csv_file"]
    mat_path = dir_path / spec["mat_file"]

    # 1. Extract row indices from CSV using Table V criteria
    df = load_csv_dataframe(csv_path, has_header=spec["has_csv_header"])
    condition_mask = spec["filter"](df)
    subset_df = df[condition_mask]
    X_tem = subset_df.index.to_numpy()

    # 2. Load feature matrix and labels from MAT file
    X, y = load_mat_matrix(mat_path, key=spec["mat_key"])

    # 3. Sanity check validation against Table V
    if validate:
        n_selected = len(X_tem)
        n_outliers = int(np.sum(y[X_tem]))
        expected_n = spec["expected_objects"]
        expected_out = spec["expected_outliers"]

        assert n_selected == expected_n, (
            f"[{name}] Sample count mismatch: got {n_selected}, expected {expected_n} in Table V."
        )
        assert n_outliers == expected_out, (
            f"[{name}] Outlier count mismatch: got {n_outliers}, expected {expected_out} in Table V."
        )

    return X, y, X_tem