import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import numpy as np



def train_transforms(augment: bool = False):
    """Returns data transforms for CIFAR-100 training.

    Args:
        augment: If True, applies strong augmentation strategy:
                - Random crop with padding
                - Random horizontal flip
                - Random erasing (cutout)
                - Normalization with CIFAR-100 mean/std
                
    Returns:
        torchvision.transforms.Compose with appropriate transforms
    """
    MEAN = [0.485, 0.456, 0.4068]
    STD = [0.229, 0.224, 0.225]
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def test_transforms():
    """Returns deterministic transforms for CIFAR-100 evaluation/test sets.
    
    Includes:
        - ToTensor conversion
        - Normalization with CIFAR-100 specific mean/std values
        
    No augmentations are applied to ensure consistent evaluation.
    """
    MEAN = [0.485, 0.456, 0.4068]
    STD = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),              # Resize the shorter side to 256
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


# Mixup augmentation
def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    GRAD_CLIP = 1.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup augmentation
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, device=device)
        
        # Zero the gradient buffers
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss with mixup
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Optimize
        optimizer.step()
        scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        
        # For accuracy calculation, use original targets
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        # pbar.set_postfix({
        #     'loss': f'{running_loss/(batch_idx+1):.3f}',
        #     'acc': f'{100.*correct/total:.2f}%',
        #     'lr': f'{current_lr:.6f}'
        # })
