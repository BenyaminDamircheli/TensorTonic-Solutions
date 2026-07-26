import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    a = np.array(data, dtype=np.float64)

    indices = np.argsort(a, axis)
    vals = np.sort(a, axis)

    return np.stack([vals, indices])