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
def mixup_data(x, y, alpha=0.4, device='cuda'):
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


def train(model, train_loader, criterion, optimizer, scheduler, device, epoch,
          use_mixed_precision=False, dtype=torch.float16, step_scheduler_on_batch: bool = False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_samples = 0
    GRAD_CLIP = 1.0
    
    # Initialize gradient scaler for mixed precision training if CUDA is available
    scaler = None
    if use_mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            logging.info("[Warning] Mixed precision disabled: CUDA not available.")
            use_mixed_precision = False

    # Auto-detect common per-batch schedulers and warn/adjust step_scheduler_on_batch.
    # OneCycleLR and CyclicLR are intended to be stepped every batch. ReduceLROnPlateau
    # should be stepped externally with validation metrics. For unknown schedulers we
    # keep the user's preference.
    if scheduler is not None:
        sched_name = scheduler.__class__.__name__
        per_step_defaults = ("OneCycleLR", "CyclicLR")
        if sched_name in per_step_defaults and not step_scheduler_on_batch:
            logging.info(f"Detected scheduler '{sched_name}' which is typically stepped per-batch. Enabling step_scheduler_on_batch=True automatically.")
            step_scheduler_on_batch = True
        elif sched_name == "ReduceLROnPlateau" and step_scheduler_on_batch:
            logging.warning(f"Detected scheduler 'ReduceLROnPlateau' which expects metric-based stepping. It's unusual to step it per-batch.")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixup augmentation
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, device=device)

        optimizer.zero_grad()

        # Forward pass under autocast if mixed precision is enabled
        with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=dtype):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward + optimization
        if use_mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Step LR scheduler per iteration only when explicitly requested
        # (e.g., OneCycleLR or user sets step_scheduler_on_batch=True)
        if scheduler is not None and step_scheduler_on_batch:
            try:
                scheduler.step()
            except Exception:
                # Some schedulers expect epoch-level stepping; skip here and let caller handle epoch steps
                pass

        # ---- Accuracy calculation ----
        # For mixup, this is an approximation: count correct if matches either target_a or target_b
        _, predicted = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        total_samples += batch_size
        correct += (lam * predicted.eq(targets_a).sum().item() +
                    (1 - lam) * predicted.eq(targets_b).sum().item())

        # Loss tracking: accumulate per-sample so average is meaningful
        running_loss += loss.item() * batch_size
        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "loss": f"{avg_loss:.3f}",
            "acc": f"{acc:.2f}%",
            "lr": f"{current_lr:.6f}"
        })
