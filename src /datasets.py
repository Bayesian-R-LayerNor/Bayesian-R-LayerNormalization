"""
Dataset utilities for noisy image data.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple, Optional, Callable
import os


class CachedNoisyCIFAR10(Dataset):
    """
    CIFAR-10 dataset with various noise corruptions and in-memory caching.
    """
    CORRUPTION_TYPES = [
        'gaussian', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'frost', 'contrast', 'blur', 'brightness', 'saturate'
    ]
    
    def __init__(self, noise_type: str = 'gaussian', severity: int = 3,
                 num_samples: int = 1000, train: bool = True,
                 root: str = './data', download: bool = True):
        """
        Args:
            noise_type: Type of corruption to apply
            severity: Severity level (1-5)
            num_samples: Number of samples to cache
            train: Whether to use training or test set
            root: Root directory for dataset
            download: Whether to download the dataset
        """
        self.noise_type = noise_type
        self.severity = severity
        self.train = train
        self.num_samples = num_samples
        
        # Download CIFAR-10
        self.clean_dataset = CIFAR10(root=root, train=train, 
                                    download=download, transform=None)
        
        # Cache samples
        self.cached_samples = []
        self.cached_labels = []
        
        indices = list(range(min(num_samples, len(self.clean_dataset))))
        
        for idx in indices:
            img, label = self.clean_dataset[idx]
            img_tensor = self.apply_noise_and_transform(img)
            self.cached_samples.append(img_tensor)
            self.cached_labels.append(label)
    
    def apply_noise_and_transform(self, img_pil: Image.Image) -> torch.Tensor:
        """Apply specific noise corruption to image."""
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        
        # Apply noise based on type
        if self.noise_type == 'gaussian':
            noise = np.random.randn(*img_np.shape) * 0.1 * self.severity
            img_np = img_np + noise
            
        elif self.noise_type == 'shot_noise':
            mask = np.random.random(img_np.shape) < 0.05 * self.severity
            salt = np.random.random(mask.sum()) > 0.5
            img_np_flat = img_np.reshape(-1)
            mask_flat = mask.reshape(-1)
            img_np_flat[mask_flat] = salt.astype(np.float32)
            img_np = img_np_flat.reshape(img_np.shape)
            
        elif self.noise_type == 'blur':
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=self.severity*0.5))
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            
        elif self.noise_type == 'contrast':
            mean = img_np.mean()
            contrast_factor = max(0.5, 1.0 - 0.2 * self.severity)
            img_np = contrast_factor * (img_np - mean) + mean
            
        elif self.noise_type == 'frost':
            # Simulate frost effect
            frost_pattern = np.random.uniform(0.7, 1.0, img_np.shape)
            img_np = img_np * frost_pattern
            
        elif self.noise_type == 'brightness':
            img_np = img_np * (1.0 - 0.1 * self.severity)
            
        # Clip and convert to tensor
        img_np = np.clip(img_np, 0, 1)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        
        # Normalize
        if self.train:
            img_tensor = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            )(img_tensor)
        else:
            img_tensor = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            )(img_tensor)
            
        return img_tensor
    
    def __len__(self) -> int:
        return len(self.cached_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.cached_samples[idx], self.cached_labels[idx]


class MixedCorruptionDataset(Dataset):
    """
    Dataset that mixes multiple corruption types for robust training.
    """
    def __init__(self, corruption_types: List[str], severity: int = 3,
                 samples_per_type: int = 250, train: bool = True):
        """
        Args:
            corruption_types: List of corruption types to mix
            severity: Severity level for all corruptions
            samples_per_type: Number of samples per corruption type
            train: Whether to use training data
        """
        self.datasets = []
        
        for corruption in corruption_types:
            dataset = CachedNoisyCIFAR10(
                noise_type=corruption,
                severity=severity,
                num_samples=samples_per_type,
                train=train
            )
            self.datasets.append(dataset)
            
        self.combined_dataset = ConcatDataset(self.datasets)
        
    def __len__(self) -> int:
        return len(self.combined_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.combined_dataset[idx]
