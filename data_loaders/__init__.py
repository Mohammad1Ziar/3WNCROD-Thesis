"""
data_loaders
Module for loading experimental anomaly detection subsets.
"""

from .loader import load_subset
from .criteria import SUBSET_REGISTRY

__all__ = ["load_subset", "SUBSET_REGISTRY"]