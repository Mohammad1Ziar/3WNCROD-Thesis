"""
data_loaders/base.py
Low-level readers for .csv and .mat files, along with scaling and preprocessing.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


def load_csv_dataframe(filepath: Path | str, has_header: bool = False) -> pd.DataFrame:
    """Reads a CSV file into a DataFrame with integer column indices."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    header_param = 0 if has_header else None
    df = pd.read_csv(path, header=header_param)
    df.columns = list(range(df.shape[1]))
    return df


def load_mat_matrix(filepath: Path | str, key: str = "trandata") -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a MATLAB .mat file.
    Extracts features (all columns except the last) and labels (last column).
    Applies MinMaxScaler on numerical feature columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    mat_dict = loadmat(str(path))
    if key in mat_dict:
        full_data = np.asarray(mat_dict[key], dtype=float)
        X = full_data[:, :-1]
        y_raw = full_data[:, -1]
    elif "X" in mat_dict and "y" in mat_dict:
        X = np.asarray(mat_dict["X"], dtype=float)
        y_raw = np.asarray(mat_dict["y"], dtype=float).flatten()
    else:
        raise KeyError(f"Unable to find data keys in {path}. Available keys: {list(mat_dict.keys())}")

    # Standardize labels: 1 for outlier, 0 for normal
    unique_labels = np.unique(y_raw)
    if set(unique_labels) == {1.0, 2.0}:
        y = np.where(y_raw == 2.0, 1, 0)
    else:
        y = np.where(y_raw == 1.0, 1, 0)

    # Normalize numerical features into [0, 1] range as in reference code
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    return X_norm, y