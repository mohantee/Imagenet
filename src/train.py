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
    MEAN = [0.485, 0.456, 0.4068]
    STD = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),              # Resize the shorter side to 256
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


# Mixup augmentation
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(inputs, targets, alpha=1.0, cutmix_prob=0.5, device='cuda'):
    rand_index = torch.randperm(inputs.size(0)).to(device)
    lam = np.random.beta(alpha, alpha)
    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    else:
        inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
        targets_a, targets_b = targets, targets[rand_index]
        return inputs, targets_a, targets_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, optimizer, scheduler, device, epoch,
          use_mixed_precision=False, dtype=torch.float16, step_scheduler_on_batch: bool = False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_cutmix(inputs, targets, device=device)


        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=dtype):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
        running_loss += loss.item() * targets.size(0)
        pbar.set_postfix({
            'loss': f'{running_loss / total:.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
