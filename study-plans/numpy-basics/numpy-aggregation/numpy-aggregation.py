import numpy as np

def summarize(data, axis):
    """Returns: np.ndarray of shape (4, k), rows are mean, std, min, max"""    
    a = np.array(data, dtype=np.float64)

    return np.stack([a.mean(axis), a.std(axis), a.min(axis), a.max(axis)])