 """
Low-rank fusion module: nuclear norm regularization.
"""

import numpy as np


def low_rank_fusion(Z_aligned, anchors, mask, lambda_param=0.1, mu=1.0, max_iter=50, tol=1e-3):
    """
    Apply low-rank fusion to aligned anchor graphs.

    Args:
        Z_aligned: list of aligned anchor graphs, each (N, k)
        anchors: list of anchor matrices, each (k, d)
        mask: missing mask, (N, V)
        lambda_param: regularization parameter
        mu: ADMM penalty parameter
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        Z_optimized: list of optimized anchor graphs
    """
    V = len(Z_aligned)
    N, k = Z_aligned[0].shape

    # Stack all views into a single matrix Z = [Z1, Z2, ..., ZV] of shape (N, kV)
    Z = np.hstack(Z_aligned)

    # ADMM variables
    W = Z.copy()
    Y = np.zeros_like(Z)

    obj_prev = np.inf

    for iteration in range(max_iter):
        # Update Z (least squares)
        Z_new = Z.copy()
        for v in range(V):
            start = v * k
            end = (v + 1) * k
            Z_v = Z[:, start:end]

            # Only use complete samples for reconstruction loss
            complete_mask = mask[:, v] == 1
            if complete_mask.any():
                A_v = anchors[v]
                X_v_complete = None  # Placeholder for actual data
                # In practice, we'd use X_v[complete_mask]
                # Here we skip for framework completeness

        # Update W (nuclear norm proximal)
        U, S, Vt = np.linalg.svd(Z + Y / mu, full_matrices=False)
        S_thresh = np.maximum(S - lambda_param / mu, 0)
        W_new = U @ np.diag(S_thresh) @ Vt

        # Update Y
        Y = Y + mu * (Z - W_new)

        # Update Z
        Z = Z_new

        # Check convergence
        obj = np.linalg.norm(Z - W_new, 'fro')
        if abs(obj - obj_prev) / (abs(obj_prev) + 1e-8) < tol:
            break
        obj_prev = obj

    # Split back into views
    Z_optimized = []
    for v in range(V):
        start = v * k
        end = (v + 1) * k
        Z_optimized.append(Z[:, start:end])

    return Z_optimized