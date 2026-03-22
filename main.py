"""
ABA-ALR-IMVC: Anchor-Based Alignment and Adaptive Low-Rank Fusion
for Incomplete Multi-View Clustering

Main entry point for running experiments.
"""

import numpy as np
import argparse
from utils import load_dataset, create_missing_mask, evaluate
from anchor_alignment import anchor_alignment
from low_rank_fusion import low_rank_fusion
from admm_solver import admm_optimize


def main():
    parser = argparse.ArgumentParser(description='ABA-ALR-IMVC')
    parser.add_argument('--dataset', type=str, default='BDGP',
                        help='Dataset name (BDGP, Scene15, HandWritten, 100leaves, MNIST-USPS)')
    parser.add_argument('--missing_rate', type=float, default=0.3,
                        help='Missing rate (0.1, 0.2, 0.3, 0.4, 0.5)')
    parser.add_argument('--k', type=int, default=100,
                        help='Number of anchors')
    parser.add_argument('--lambda_param', type=float, default=0.1,
                        help='Regularization parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Missing Rate: {args.missing_rate * 100}%")
    print(f"Anchors: {args.k}")
    print(f"λ: {args.lambda_param}")
    print("=" * 50)

    # 1. 加载数据
    X1, X2, y_true = load_dataset(args.dataset)
    N = X1.shape[0]

    # 2. 构建缺失掩码
    mask = create_missing_mask(N, missing_rate=args.missing_rate, seed=args.seed)

    # 3. 锚点初始化（K-means on complete samples）
    from sklearn.cluster import KMeans
    complete_idx = np.where((mask[:, 0] == 1) & (mask[:, 1] == 1))[0]
    A1 = KMeans(n_clusters=args.k, random_state=args.seed).fit(X1[complete_idx]).cluster_centers_
    A2 = KMeans(n_clusters=args.k, random_state=args.seed).fit(X2[complete_idx]).cluster_centers_
    anchors = [A1, A2]

    # 4. 锚图构建（高斯核）
    sigma = 0.1
    Z1 = np.exp(-np.sum((X1[:, None, :] - anchors[0][None, :, :]) ** 2, axis=2) / (2 * sigma ** 2))
    Z2 = np.exp(-np.sum((X2[:, None, :] - anchors[1][None, :, :]) ** 2, axis=2) / (2 * sigma ** 2))
    Z = [Z1, Z2]

    # 5. 锚点对齐（动态基准视图）
    Z_aligned, P = anchor_alignment(Z, mask)

    # 6. ADMM优化（低秩融合）
    Z_optimized = admm_optimize(Z_aligned, anchors, mask, lambda_param=args.lambda_param)

    # 7. 自适应权重融合
    omega = mask.sum(axis=0) / mask.sum()
    consensus = np.zeros((N, args.k))
    for v in range(2):
        consensus += omega[v] * Z_optimized[v]

    # 8. 谱聚类
    from sklearn.cluster import SpectralClustering
    n_clusters = len(np.unique(y_true))
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=args.seed)
    y_pred = sc.fit_predict(consensus @ consensus.T)

    # 9. 评估
    acc, nmi, ari = evaluate(y_true, y_pred)
    print("\n" + "=" * 50)
    print("Results:")
    print(f"  ACC = {acc:.2f}%")
    print(f"  NMI = {nmi:.2f}%")
    print(f"  ARI = {ari:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()