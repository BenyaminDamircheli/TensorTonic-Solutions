import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dp = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0.0 or nb == 0.0:
        return 0.0

    return dp/(na * nb)