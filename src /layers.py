"""
Bayesian R-LayerNorm implementation with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union


class StandardLayerNorm(nn.Module):
    """
    Standard Layer Normalization (baseline for comparison)
    Implements: LN(x) = γ ⊙ [(x - μ) / σ] + β
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))  # γ parameter
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))   # β parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance over spatial dimensions (H,W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        # Normalization operation
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable affine transformation
        return self.weight * x_normalized + self.bias

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


class RLayerNorm(nn.Module):
    """
    Robust Layer Normalization (R-LayerNorm)
    Implements: R-LN(x) = γ ⊙ [(x - μ) / (σ ⊙ (1 + λ · E(x)))] + β
    """
    def __init__(self, num_features: int, lambda_init: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Standard statistics
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        # Efficient local variance computation for noise estimation
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        local_means = F.avg_pool2d(x_padded, kernel_size=3, stride=1)
        local_vars = F.avg_pool2d(x_padded**2, kernel_size=3, stride=1) - local_means**2
        local_vars = local_vars.clamp(min=0)  # Ensure non-negative variance

        # Noise estimate - mean of local variance
        noise_estimate = local_vars.mean(dim=[2, 3], keepdim=True)

        # Noise-aware scaling factor
        lambda_safe = self.lambda_param.clamp(1e-3, 1.0)
        noise_scale = 1 + lambda_safe * noise_estimate / (var + self.eps)

        # Normalize with noise-aware scaling
        x_normalized = (x - mean) / (std * noise_scale + self.eps)
        
        return self.weight * x_normalized + self.bias

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, lambda_init={self.lambda_param.item():.4f}, eps={self.eps}"


class BayesianRLayerNorm(nn.Module):
    """
    Bayesian R-LayerNorm with uncertainty quantification
    Implements: B-R-LN(x) = γ ⊙ [(x - μ) / σ_eff] + β
    where σ_eff^2 = σ^2 · exp(2α · ψ(λE))
    and ψ(t) = log(1+t) - t/(1+t) (stable function)
    """
    def __init__(self, num_features: int, lambda_init: float = 0.01, 
                 alpha: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.alpha = alpha
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    @staticmethod
    def psi_function(t: torch.Tensor) -> torch.Tensor:
        """
        Stable function ψ(t) = log(1+t) - t/(1+t)
        Properties:
        1. ψ(t) ≈ t²/2 for t→0
        2. ψ(t) ≈ log(t) for t→∞
        3. 0 ≤ ψ'(t) ≤ 1 (bounded derivative)
        """
        return torch.log1p(t) - t / (1 + t)

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, H, W = x.shape

        # Standard statistics
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        # Efficient noise estimation
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        local_means = F.avg_pool2d(x_padded, kernel_size=3, stride=1)
        local_vars = F.avg_pool2d(x_padded**2, kernel_size=3, stride=1) - local_means**2
        local_vars = local_vars.clamp(min=0)
        noise_estimate = local_vars.mean(dim=[2, 3], keepdim=True)

        # Bayesian adjustment using ψ function
        lambda_safe = self.lambda_param.clamp(1e-3, 1.0)
        lambdaE = lambda_safe * noise_estimate / (var + self.eps)
        psi_term = self.psi_function(lambdaE)

        # Effective standard deviation with Bayesian correction
        effective_std = std * torch.exp(self.alpha * psi_term)

        # Normalize with effective standard deviation
        normalized = (x - mean) / (effective_std + self.eps)
        output = self.weight * normalized + self.bias

        # Return uncertainty if requested
        if return_uncertainty:
            uncertainty = 1.0 / (effective_std**2 + self.eps)
            return output, uncertainty

        return output

    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, lambda_init={self.lambda_param.item():.4f}, "
                f"alpha={self.alpha}, eps={self.eps}")


# Factory function for easy layer creation
def create_normalization_layer(norm_type: str, num_features: int, **kwargs) -> nn.Module:
    """
    Factory function to create normalization layers.
    
    Args:
        norm_type: 'layer', 'r_layer', or 'bayesian_r_layer'
        num_features: Number of input features
        **kwargs: Additional arguments for specific layer types
    
    Returns:
        Normalization layer instance
    """
    if norm_type == 'layer':
        return StandardLayerNorm(num_features, **kwargs)
    elif norm_type == 'r_layer':
        return RLayerNorm(num_features, **kwargs)
    elif norm_type == 'bayesian_r_layer':
        return BayesianRLayerNorm(num_features, **kwargs)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")
