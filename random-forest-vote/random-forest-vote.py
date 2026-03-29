import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    preds = np.array(predictions)

    res = []

    for i in range(preds.shape[1]):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        index = np.argmax(counts)
        res.append(int(vals[index]))

    return res
