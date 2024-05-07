## Requirements

* python==3.11.5
* pytorch==2.1.0

## Overview

This repository contains the implementation of the AGCN (Adaptive Graph Convolutional Network) model as described in our paper. The code is organized into several scripts, each serving a different purpose in the training and evaluation process of the model.

### Files Description

- `AGCN.py`: Implements the AGCN model architecture.
- `AGCN_domain_loss.py`: Contains the domain-specific loss functions used for training the AGCN model.
- `train.py`: Basic training script for training the AGCN model on a single domain dataset.
- `train_transfer_domain_adaptation.py`: Script for training the AGCN model with domain adaptation techniques.
- `train_transfer_domain_generalization.py`: Script for training the AGCN model with domain generalization techniques.