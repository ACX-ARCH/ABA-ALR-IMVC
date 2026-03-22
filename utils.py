"""
Utility functions for data loading, preprocessing, and evaluation.
"""

import numpy as np
import scipy.io as sio


def load_dataset(dataset_name):
    """
    Load dataset from .mat file.
    Returns: X1, X2, y_true
    """
    path = f'./datasets/{dataset_name}.mat'
    data = sio.loadmat(path)

    X1 = data['X1']
    X2 = data['X2']
    y_true = data['y'].flatten() if 'y' in data else data['gt'].flatten()

    # Normalize to [0, 1]
    X1 = (X1 - X1.min(axis=0)) / (X1.max(axis=0) - X1.min(axis=0) + 1e-8)
    X2 = (X2 - X2.min(axis=0)) / (X2.max(axis=0) - X2.min(axis=0) + 1e-8)

    return X1, X2, y_true


def create_missing_mask(n_samples, missing_rate=0.3, seed=42):
    """
    Create random missing mask for two views.
    Returns: mask (N, 2), 1 = observed, 0 = missing
    """
    np.random.seed(seed)
    mask = np.ones((n_samples, 2))
    for v in range(2):
        missing_idx = np.random.choice(n_samples, size=int(n_samples * missing_rate), replace=False)
        mask[missing_idx, v] = 0
    return mask


def evaluate(y_true, y_pred):
    """
    Compute clustering metrics: ACC, NMI, ARI.
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    # ACC with Hungarian matching
    n_clusters = len(np.unique(y_true))
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Build confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)

    acc = cm[row_ind, col_ind].sum() / len(y_true) * 100

    # NMI and ARI
    nmi = normalized_mutual_info_score(y_true, y_pred) * 100
    ari = adjusted_rand_score(y_true, y_pred) * 100

    return acc, nmi, ari