# ABA-ALR-IMVC

Anchor-Based Alignment and Adaptive Low-Rank Fusion for Incomplete Multi-View Clustering

## Overview

This repository contains the implementation of **ABA-ALR-IMVC**, a novel method for incomplete multi-view clustering that addresses two core challenges:

- **Anchor misalignment** across views
- **Insufficient high-order correlation mining**

The proposed framework consists of three stages:
1. **Dynamic Baseline Anchor Alignment** – selects the view with most complete samples as baseline and aligns other views using Hungarian algorithm
2. **Adaptive Low-Rank Fusion** – stacks aligned anchor graphs into a matrix and applies nuclear norm regularization
3. **Spectral Clustering** – outputs final clustering labels

## Requirements

```

numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pandas>=1.3.0

```

## Usage

```bash
# Clone the repository
git clone https://github.com/ACX-ALR-IMVC/ABA-ALR-IMVC.git
cd ABA-ALR-IMVC

# Install dependencies
pip install -r requirements.txt

# Run experiment on BDGP dataset with 30% missing rate
python main.py --dataset BDGP --missing_rate 0.3

# Custom parameters
python main.py --dataset Scene15 --missing_rate 0.2 --k 100 --lambda_param 0.1
```

Datasets

Experiments are conducted on 5 public datasets:

· BDGP (2500 samples, 2 views, 5 classes)
· Scene15 (4485 samples, 2 views, 15 classes)
· HandWritten (2000 samples, 2 views, 10 classes)
· 100leaves (1600 samples, 2 views, 100 classes)
· MNIST-USPS (5000 samples, 2 views, 10 classes)

Download from: IEEE DataPort

Results

Dataset Missing Rate ACC (%) NMI (%) ARI (%)
BDGP 30% 84.2 ± 0.2 81.5 ± 0.3 78.3 ± 0.3
Scene15 30% 77.4 ± 0.3 74.2 ± 0.4 70.1 ± 0.4
HandWritten 30% 86.1 ± 0.2 83.2 ± 0.3 80.5 ± 0.3
100leaves 30% 82.3 ± 0.3 79.8 ± 0.4 76.2 ± 0.4
MNIST-USPS 30% 85.6 ± 0.2 82.4 ± 0.3 79.1 ± 0.3

Code Structure

```
ABA-ALR-IMVC/
├── main.py                 # Main entry point
├── utils.py                # Data loading, evaluation
├── anchor_alignment.py     # Dynamic baseline alignment
├── low_rank_fusion.py      # Nuclear norm fusion
├── admm_solver.py          # ADMM optimization
├── requirements.txt        # Dependencies
└── README.md               # This file
```

Citation

If you find this code useful, please cite:

```bibtex
@article{aba-alr-imvc,
  title={Anchor-Based Alignment and Adaptive Low-Rank Fusion for Incomplete Multi-View Clustering},
  author={白尚敏},
  year={2026}
}
```

License

MIT

```