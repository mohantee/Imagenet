import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import autoaugment
from tqdm import tqdm
import numpy as np
import logging


def train_transforms(augment: bool = False, use_randaugment: bool = True):
    """Returns advanced data transforms for ImageNet training.

    Args:
        augment: If True, applies strong augmentation strategy
        use_randaugment: If True, uses RandAugment (state-of-the-art augmentation)
                
    Returns:
        torchvision.transforms.Compose with appropriate transforms
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    if augment:
        if use_randaugment:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

def test_transforms(resolution: int = 224):
    """Deterministic eval transforms at arbitrary resolution (FixRes-friendly)."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    resize_side = int(round(resolution / 0.875))
    return transforms.Compose([
        transforms.Resize(resize_side),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def fixres_train_transforms(resolution: int = 288):
    """FixRes fine-tune (classifier-only) transforms at higher resolution.
    Mild aug only; backbone is frozen so we avoid heavy distortions.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def fixres_val_transforms(resolution: int = 288):
    """Validation transforms matching FixRes eval resolution."""
    return test_transforms(resolution)


# Mixup augmentation
def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Mixup augmentation - mixes two samples"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# CutMix augmentation - combines two images by cutting and pasting patches
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center coordinates
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """CutMix augmentation - cuts and pastes patches between images"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss computation for Mixup/CutMix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, 
          use_mixed_precision=False, dtype=torch.float16, scaler=None,
          use_cutmix=True, cutmix_prob=0.5, mixup_alpha=0.2, cutmix_alpha=1.0,
          step_scheduler_on_batch=False, gradient_accumulation_steps=1):
    """
    Advanced training function with Mixup/CutMix, gradient accumulation, and mixed precision.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device
        epoch: Current epoch
        use_mixed_precision: Enable mixed precision training
        dtype: Data type for mixed precision (float16 or bfloat16)
        scaler: Gradient scaler (created externally for proper state management)
        use_cutmix: Enable CutMix augmentation
        cutmix_prob: Probability of applying CutMix (vs Mixup)
        mixup_alpha: Mixup alpha parameter
        cutmix_alpha: CutMix alpha parameter
        step_scheduler_on_batch: Step scheduler per batch (True) or per epoch (False)
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    GRAD_CLIP = 1.0
    
    # Create scaler if not provided and mixed precision is enabled
    if scaler is None and use_mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            logging.info("[Warning] Mixed precision disabled: CUDA not available.")
            use_mixed_precision = False
            scaler = None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # Apply Mixup or CutMix augmentation
        if np.random.rand() < cutmix_prob and use_cutmix:
            # Use CutMix
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=cutmix_alpha, device=device)
        else:
            # Use Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha, device=device)

        # Forward pass under autocast if mixed precision is enabled
        with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=dtype):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass
        if use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every N accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Step LR scheduler per batch if requested
            if step_scheduler_on_batch and scheduler is not None:
                scheduler.step()

        # ---- Accuracy calculation ----
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() +
                    (1 - lam) * predicted.eq(targets_b).sum().item())

        # Loss tracking (unscale for display)
        running_loss += loss.item() * gradient_accumulation_steps
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100.0 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        
        pbar.set_postfix({
            "loss": f"{avg_loss:.3f}",
            "acc": f"{acc:.2f}%",
            "lr": f"{current_lr:.6f}"
        })
    
    return scaler  # Return scaler for state persistence across epochs
