import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import logging


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
    MEAN = [0.485, 0.456, 0.406]
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
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),              # Resize the shorter side to 256
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


# Mixup augmentation
def rand_bbox(size, lam):
    # size is (B, C, H, W)
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    # center coords (x,y) where x in [0, W-1], y in [0, H-1]
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    # return as (x1, y1, x2, y2)
    return int(x1), int(y1), int(x2), int(y2)


def mixup_cutmix(inputs, targets, alpha=1.0, cutmix_prob=0.5):
    # Use actual device of inputs
    device = inputs.device
    rand_index = torch.randperm(inputs.size(0), device=device)
    lam = float(np.random.beta(alpha, alpha))
    if np.random.rand() < cutmix_prob:
        # get bbox in (x1,y1,x2,y2) and slice as [y1:y2, x1:x2]
        x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
        # inputs shape: (B, C, H, W) so slice in height (y) then width (x)
        inputs[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]
        # recompute lambda as area ratio
        H, W = inputs.size(2), inputs.size(3)
        box_area = float((y2 - y1) * (x2 - x1))
        lam = 1.0 - (box_area / float(H * W))
        targets_a, targets_b = targets, targets[rand_index]
        return inputs, targets_a, targets_b, lam
    else:
        inputs = lam * inputs + (1.0 - lam) * inputs[rand_index, :]
        targets_a, targets_b = targets, targets[rand_index]
        return inputs, targets_a, targets_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, optimizer, scheduler, device, epoch,
          use_mixed_precision=False, dtype=torch.float16, step_scheduler_on_batch: bool = False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    # instantiate GradScaler correctly
    scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        if epoch < 5:  # avoid mixup early
            lam = 1.0
            targets_a, targets_b = targets, targets
        else:
            inputs, targets_a, targets_b, lam = mixup_cutmix(inputs, targets)


        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=dtype):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        # step scheduler per-batch only when requested
        if step_scheduler_on_batch:
            scheduler.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
        running_loss += loss.item() * targets.size(0)
        pbar.set_postfix({
            'loss': f'{running_loss / total:.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
