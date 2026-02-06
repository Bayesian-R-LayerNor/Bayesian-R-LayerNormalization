# Bayesian-R-LayerNormalization
# Bayesian R-LayerNorm: Uncertainty-Aware Robust Normalization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/bayesian-r-layernorm/blob/main/examples/colab_demo.ipynb)

Official implementation of **Bayesian R-LayerNorm**: A theoretically grounded normalization layer with uncertainty quantification for robust deep learning on noisy data.

## ✨ Features

- **Bayesian Uncertainty Quantification**: Extends R-LayerNorm with formal uncertainty estimation
- **Provable Stability**: Mathematical guarantees for numerical and gradient stability
- **Drop-in Replacement**: Compatible with existing PyTorch normalization layers
- **Noise Adaptation**: Dynamically adjusts normalization based on local entropy
- **Minimal Overhead**: Only ∼10% computational increase
- **Cloud-Ready**: Tested on Google Colab with variable GPU allocation

## 📊 Performance

| Noise Type | LayerNorm | R-LayerNorm | Bayesian R-LayerNorm |
|------------|-----------|-------------|----------------------|
| Gaussian   | 25.80%    | 27.80%      | **31.00%**           |
| Shot Noise | 28.80%    | **32.40%**  | 30.60%               |
| Blur       | 31.40%    | 32.00%      | **34.40%**           |
| Contrast   | **32.80%**| 33.60%      | 32.00%               |

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install bayesian-rlayernorm

# Or install from source
git clone https://github.com/Bayesian-R-LayerNor/Bayesian-R-LayerNormalizatione/bayesian-r-layernorm.git
cd bayesian-r-layernorm
pip install -e .
```

```bash
import torch
import torch.nn as nn
from bayesian_rlayernorm import BayesianRLayerNorm

# Create a simple model with Bayesian R-LayerNorm
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    BayesianRLayerNorm(16, lambda_init=0.01),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    BayesianRLayerNorm(32, lambda_init=0.01),
    nn.Tanh()
)

# Forward pass with uncertainty
x = torch.randn(32, 3, 32, 32)
output, uncertainty = model(x, return_uncertainty=True)
```
🎯 Key Features
1. Bayesian Uncertainty
```bash
# Get uncertainty estimates
output, uncertainty = bayesian_norm(x, return_uncertainty=True)
print(f"Prediction uncertainty: {uncertainty.mean():.4f}")
```
