import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    ca = np.clip(a, lo, hi)
    cb = np.clip(b, lo, hi)

    can = (ca - lo) / (hi - lo)
    cbn = (cb - lo) / (hi - lo)

    return np.abs(can - cbn)

    
    