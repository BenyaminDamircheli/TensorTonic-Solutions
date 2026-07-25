import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    a = np.array(data, dtype=np.float64)

    original = a[row_idx].copy()

    new = np.clip(original,lo, hi)

    return np.stack([original, new])