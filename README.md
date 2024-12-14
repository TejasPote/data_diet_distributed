# Data Diet Distributed Training

This repository provides an implementation of the **Data Diet** algorithm combined with Distributed Data Parallel (DDP) training in PyTorch. The project aims to optimize deep learning model training by pruning redundant data points and leveraging distributed GPU setups for faster and more efficient training.

## Features
- Implementation of the **Data Diet** pruning technique.
- Distributed Data Parallel (DDP) training for scalable multi-GPU setups.
- Support for custom neural network models and datasets.
- Training scripts for sparse and dense datasets.
- YAML-based configuration for flexible experiment management.

---

## Repository Structure
```plaintext
data_diet_distributed-master/
│
├── config.yaml              # Configuration file for training
├── data/                    # Placeholder directory for datasets
├── ddp.py                   # Core script for Distributed Data Parallel training
├── get_scores_and_prune.py  # Implements Data Diet's pruning mechanism
├── models/                  # Contains model definitions
├── requirements.txt         # Required Python libraries and versions
├── test.ipynb               # Jupyter Notebook for testing the implementation
├── train.py                 # Script for training on full (dense) datasets
├── train_sparse.py          # Script for training on pruned (sparse) datasets
└── trainer/                 # Module for managing training loops and utilities
