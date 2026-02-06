"""
Training utilities for Bayesian R-LayerNorm experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gc


class MemoryManagedTrainer:
    """
    Trainer with GPU memory management for cloud environments.
    """
    def __init__(self, device: torch.device, 
                 use_mixed_precision: bool = True,
                 gradient_accumulation_steps: int = 1):
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
    def clear_memory(self):
        """Clear GPU and CPU memory."""
        torch.cuda.empty_cache()
        gc.collect()
        
    def train_epoch(self, model: nn.Module, 
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   num_epochs: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, model: nn.Module,
                test_loader: DataLoader,
                criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Evaluating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        
        results = {'accuracy': accuracy}
        if criterion is not None:
            results['loss'] = total_loss / len(test_loader)
            
        return results
    
    def train_model(self, model: nn.Module,
                   train_loader: DataLoader,
                   test_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   num_epochs: int,
                   scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                   save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Complete training loop.
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch, num_epochs
            )
            
            # Evaluate
            val_results = self.evaluate(model, test_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step(val_results['accuracy'] 
                              if 'accuracy' in val_results else train_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_results.get('loss', 0.0))
            history['val_acc'].append(val_results.get('accuracy', 0.0))
            
            # Save best model
            if val_results['accuracy'] > best_acc and save_path:
                best_acc = val_results['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_acc,
                    'history': history
                }, save_path)
                print(f'Saved best model with accuracy: {best_acc:.2f}%')
            
            # Clear memory every few epochs
            if epoch % 2 == 0:
                self.clear_memory()
        
        return history
