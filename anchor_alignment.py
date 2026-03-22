"""
Anchor alignment module: dynamic baseline selection + Hungarian algorithm.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def anchor_alignment(Z, mask):
    """
    Align anchor graphs across views using dynamic baseline selection.

    Args:
        Z: list of anchor graph matrices, each shape (N, k)
        mask: missing mask, shape (N, 2)

    Returns:
        Z_aligned: list of aligned anchor graphs
        P: list of permutation matrices
    """
    V = len(Z)
    k = Z[0].shape[1]

    # Dynamic baseline: select view with most complete samples
    complete_counts = mask.sum(axis=0)
    baseline_idx = np.argmax(complete_counts)

    Z_aligned = [None] * V
    P = [None] * V

    # Baseline view: no alignment needed
    Z_aligned[baseline_idx] = Z[baseline_idx].copy()
    P[baseline_idx] = np.eye(k)

    # Align other views to baseline
    for v in range(V):
        if v == baseline_idx:
            continue

        # Compute cost matrix: negative similarity between anchor graphs
        # Use only complete samples
        complete_mask = (mask[:, 0] == 1) & (mask[:, 1] == 1)
        Z_v = Z[v][complete_mask]
        Z_base = Z[baseline_idx][complete_mask]

        # Cost: Frobenius norm between columns
        cost = -np.abs(Z_v.T @ Z_base)  # Negative for minimization

        # Solve linear assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build permutation matrix
        P_v = np.zeros((k, k))
        P_v[row_ind, col_ind] = 1

        # Apply alignment
        Z_aligned[v] = Z[v] @ P_v
        P[v] = P_v

    return Z_aligned, P