"""
Model architectures using Bayesian R-LayerNorm.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .layers import create_normalization_layer


class EfficientCNN(nn.Module):
    """
    Memory-efficient CNN for comparative experiments.
    """
    def __init__(self, norm_type: str = 'layer', num_classes: int = 10, 
                 in_channels: int = 3, use_tanh: bool = True):
        super().__init__()
        self.norm_type = norm_type
        
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.norm1 = create_normalization_layer(norm_type, 16)
        self.act1 = nn.Tanh() if use_tanh else nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.norm2 = create_normalization_layer(norm_type, 32)
        self.act2 = nn.Tanh() if use_tanh else nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm3 = create_normalization_layer(norm_type, 64)
        self.act3 = nn.Tanh() if use_tanh else nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = []
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        features.append(x) if return_features else None
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        features.append(x) if return_features else None
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)
        features.append(x) if return_features else None
        x = self.pool3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if return_features:
            return x, features
        return x


class ResNetBlock(nn.Module):
    """ResNet block with Bayesian R-LayerNorm."""
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, norm_type: str = 'bayesian_r_layer'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.norm1 = create_normalization_layer(norm_type, out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.norm2 = create_normalization_layer(norm_type, out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                create_normalization_layer(norm_type, out_channels)
            )
            
        self.act2 = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act2(out)
        
        return out


class VisionTransformerBlock(nn.Module):
    """Vision Transformer block with Bayesian R-LayerNorm."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 norm_type: str = 'bayesian_r_layer', dropout: float = 0.0):
        super().__init__()
        self.norm1 = create_normalization_layer(norm_type, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = create_normalization_layer(norm_type, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm before attention
        x_norm = self.norm1(x)
        
        # Self-attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # LayerNorm before MLP
        x_norm = self.norm2(x)
        
        # MLP
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        
        return x
