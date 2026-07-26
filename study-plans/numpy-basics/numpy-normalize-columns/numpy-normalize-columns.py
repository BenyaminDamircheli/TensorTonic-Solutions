import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    x = np.array(data, dtype=np.float64)
    col_mean = x.mean(axis=0)
    col_std = x.std(axis=0)

    return (x - col_mean) / col_std
    