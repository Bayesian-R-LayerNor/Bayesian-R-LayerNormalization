#!/usr/bin/env python3
"""
Training script for Bayesian R-LayerNorm experiments.
"""

import torch
import argparse
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from bayesian_rlayernorm.models import EfficientCNN
from bayesian_rlayernorm.datasets import MixedCorruptionDataset
from bayesian_rlayernorm.trainers import MemoryManagedTrainer
from bayesian_rlayernorm.utils import setup_experiment, save_results

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Train Bayesian R-LayerNorm')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment run')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.device:
        config['experiment']['device'] = args.device
    
    # Setup experiment
    setup_experiment(config)
    device = torch.device(config['experiment']['device'])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MixedCorruptionDataset(
        corruption_types=config['dataset']['corruption_types'],
        severity=config['dataset']['severity'],
        samples_per_type=config['dataset']['train_samples_per_type'],
        train=True
    )
    
    test_dataset = MixedCorruptionDataset(
        corruption_types=config['dataset']['corruption_types'],
        severity=config['dataset']['severity'],
        samples_per_type=config['dataset']['test_samples_per_type'],
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers']
    )
    
    # Create model
    print(f"Creating model with {config['model']['norm_type']}...")
    model = EfficientCNN(
        norm_type=config['model']['norm_type'],
        num_classes=config['model']['num_classes'],
        use_tanh=config['model']['use_tanh']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=config['training']['patience'],
        min_lr=config['training']['min_lr']
    )
    
    # Trainer
    trainer = MemoryManagedTrainer(
        device=device,
        use_mixed_precision=config['training']['mixed_precision'],
        gradient_accumulation_steps=config['training']['gradient_accumulation']
    )
    
    # Save path
    save_dir = Path(config['logging']['save_dir']) / config['experiment']['name']
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'best_model.pth'
    
    # Train
    print("Starting training...")
    history = trainer.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        scheduler=scheduler,
        save_path=str(save_path)
    )
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'final_metrics': {
            'train_acc': history['train_acc'][-1],
            'val_acc': history['val_acc'][-1]
        }
    }
    
    save_results(results, save_dir / 'results.json')
    print("Training completed!")


if __name__ == '__main__':
    main()
