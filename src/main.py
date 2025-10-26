import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import argparse
from tqdm import tqdm
import numpy as np
import boto3
import logging
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config
from botocore.exceptions import ClientError
import tempfile
from pathlib import Path

from model import ResNet50
from train import train, train_transforms, test_transforms, mixup_data, mixup_criterion
from test import evaluate

# --- add to imports at top of main.py ---
from torch.cuda.amp import GradScaler, autocast


def train_mixed_precision(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    scaler=None,
    use_mixed_precision=False,
    precision_type="fp16",
):
    """
    Optimized training function with mixed precision support.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device (cuda/cpu)
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision
        use_mixed_precision: Whether to use mixed precision
        precision_type: fp16 or bf16
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    GRAD_CLIP = 1.0

    # Determine autocast dtype
    if use_mixed_precision:
        autocast_dtype = torch.float16 if precision_type == "fp16" else torch.bfloat16

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # Apply mixup augmentation
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, device=device)

        # Zero gradients
        optimizer.zero_grad()

        if use_mixed_precision and scaler is not None:
            # Mixed precision forward pass
            with autocast(dtype=autocast_dtype, enabled=True):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "loss": f"{running_loss / (batch_idx + 1):.3f}",
                "acc": f"{100.0 * correct / total:.2f}%",
                "lr": f"{current_lr:.6f}",
                "mp": "🚀" if use_mixed_precision else "📊",
            }
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    print(f"Training: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate_mixed_precision(model, val_loader, criterion, device, use_mixed_precision=False, precision_type="fp16"):
    """
    Optimized evaluation function with mixed precision support.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (cuda/cpu)
        use_mixed_precision: Whether to use mixed precision
        precision_type: fp16 or bf16

    Returns:
        accuracy: Validation accuracy
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    # Determine autocast dtype
    if use_mixed_precision:
        autocast_dtype = torch.float16 if precision_type == "fp16" else torch.bfloat16

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if use_mixed_precision:
                # Mixed precision inference
                with autocast(dtype=autocast_dtype, enabled=True):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                # Standard precision inference
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{test_loss / total:.3f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "mp": "🚀" if use_mixed_precision else "📊",
                }
            )

    accuracy = 100.0 * correct / total
    print(f"Validation: Loss: {test_loss / total:.3f}, Accuracy: {accuracy:.2f}%")
    return accuracy


def get_s3_client():
    """Create an S3 client with optimized configuration."""
    config = Config(
        max_pool_connections=50,  # Increase connection pool
        retries=dict(max_attempts=3),  # Retry failed requests
        read_timeout=60,  # Longer timeout for large files
        connect_timeout=60,
    )
    return boto3.client("s3", config=config)


def load_cache_metadata(cache_dir):
    """Load cached file metadata."""
    cache_file = os.path.join(cache_dir, ".cache_metadata.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_cache_metadata(cache_dir, metadata):
    """Save cached file metadata."""
    cache_file = os.path.join(cache_dir, ".cache_metadata.json")
    with open(cache_file, "w") as f:
        json.dump(metadata, f)


def download_file_if_needed(args):
    """Download a single file from S3 if needed using parallel chunk downloads."""
    bucket_name, s3_key, local_path, etag = args

    # Check if file exists and matches ETag
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            content = f.read()
            file_hash = hashlib.md5(content).hexdigest()
            if file_hash == etag.strip('"'):
                return None  # File is up to date

    try:
        s3_client = get_s3_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_path)
        return s3_key
    except Exception as e:
        logging.error(f"Error downloading {s3_key}: {e}")
        return None


def download_from_s3(bucket_name, s3_path, local_path, max_workers=16):
    """
    Download data from S3 bucket to local path with parallel processing and caching.

    Args:
        bucket_name: Name of the S3 bucket
        s3_path: Path within the S3 bucket
        local_path: Local path to save the data
        max_workers: Maximum number of concurrent downloads
    """
    s3_client = get_s3_client()

    try:
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Load cache metadata
        cache_metadata = load_cache_metadata(os.path.dirname(local_path))

        # List all objects and prepare download tasks
        paginator = s3_client.get_paginator("list_objects_v2")
        download_tasks = []

        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_path):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                if obj["Key"].endswith("/"):
                    continue

                local_file_path = os.path.join(local_path, os.path.relpath(obj["Key"], s3_path))

                # Add to download tasks if file needs updating
                etag = obj["ETag"]
                cache_key = f"{bucket_name}:{obj['Key']}"
                if cache_key not in cache_metadata or cache_metadata[cache_key] != etag:
                    download_tasks.append((bucket_name, obj["Key"], local_file_path, etag))
                    cache_metadata[cache_key] = etag

        # Download files in parallel
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(download_file_if_needed, task): task for task in download_tasks}

            for future in tqdm(as_completed(future_to_file), total=len(download_tasks), desc="Downloading files"):
                result = future.result()
                if result:
                    downloaded_files.append(result)

        # Save updated cache metadata
        save_cache_metadata(os.path.dirname(local_path), cache_metadata)

        if downloaded_files:
            logging.info(f"Downloaded {len(downloaded_files)} new or updated files")
        else:
            logging.info("All files are up to date")

    except Exception as e:
        logging.error(f"Error during S3 download: {e}")
        raise


def get_imagenet100_dataset(data_source, train_folder, val_folder, local_dir, train=True, augment=False, use_s3=True):
    """
    Load ImageNet-100 dataset from S3 or local directory.

    Args:
        data_source: S3 bucket name (if use_s3=True) or local data directory path (if use_s3=False)
        train_folder: Full path to training data folder in S3 bucket or relative to local directory
        val_folder: Full path to validation data folder in S3 bucket or relative to local directory
        local_dir: Local directory to cache the dataset (used only if use_s3=True)
        train: If True, load training set; if False, load validation set
        augment: If True, apply data augmentation for training
        use_s3: If True, download from S3; if False, use local directory

    Returns:
        ImageFolder dataset
    """
    if use_s3:
        # Original S3 functionality
        s3_path = train_folder if train else val_folder
        local_path = os.path.join(local_dir, "train" if train else "val")

        # Download data with optimized parallel downloading and caching
        logging.info(f"Checking and downloading {'training' if train else 'validation'} dataset from S3...")
        download_from_s3(
            bucket_name=data_source,  # data_source is bucket_name when use_s3=True
            s3_path=s3_path,
            local_path=local_path,
            max_workers=16,  # Adjust based on available CPU cores and network bandwidth
        )
        dataset_path = local_path
    else:
        # Use local directory directly
        folder_name = train_folder if train else val_folder
        if os.path.isabs(folder_name):
            # If absolute path is provided, use it directly
            dataset_path = folder_name
        else:
            # If relative path, join with data_source directory
            dataset_path = os.path.join(data_source, folder_name)

        logging.info(f"Using {'training' if train else 'validation'} dataset from local path: {dataset_path}")

        # Verify the path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Set up transforms
    transform = train_transforms(augment=augment) if train else test_transforms()

    # Create dataset
    dataset = ImageFolder(dataset_path, transform=transform)
    return dataset


def create_data_loaders(
    data_source, train_folder, val_folder, local_dir, batch_size=32, num_workers=4, augment=True, use_s3=True
):
    """
    Create training and validation data loaders for ImageNet-100 from S3 or local directory.

    Args:
        data_source: S3 bucket name (if use_s3=True) or local data directory path (if use_s3=False)
        train_folder: Full path to training data folder in S3 bucket or relative to local directory
        val_folder: Full path to validation data folder in S3 bucket or relative to local directory
        local_dir: Local directory to cache the dataset (used only if use_s3=True)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation to training set
        use_s3: If True, download from S3; if False, use local directory

    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    # Load datasets
    logging.info("Loading training dataset...")
    train_dataset = get_imagenet100_dataset(
        data_source, train_folder, val_folder, local_dir, train=True, augment=augment, use_s3=use_s3
    )
    logging.info("Loading validation dataset...")
    val_dataset = get_imagenet100_dataset(
        data_source, train_folder, val_folder, local_dir, train=False, augment=False, use_s3=use_s3
    )

    # Get number of classes
    num_classes = len(train_dataset.classes)
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, filepath, bucket_name=None, s3_key=None):
    """
    Save model checkpoint locally and optionally to S3.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The scheduler to save
        epoch: Current epoch number
        accuracy: Current accuracy
        filepath: Local path to save the checkpoint
        bucket_name: Optional S3 bucket name
        s3_key: Optional S3 key for the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save checkpoint locally
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "accuracy": accuracy,
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved locally to {filepath}")

    # # Upload to S3 if specified
    # if bucket_name and s3_key:
    #     try:
    #         s3_client = boto3.client('s3')
    #         s3_client.upload_file(filepath, bucket_name, s3_key)
    #         logging.info(f"Checkpoint uploaded to s3://{bucket_name}/{s3_key}")
    #     except ClientError as e:
    #         logging.error(f"Error uploading checkpoint to S3: {e}")
    #         raise


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, bucket_name=None, s3_key=None):
    """
    Load model checkpoint from local file or S3.

    Args:
        filepath: Local path to save/load the checkpoint
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        bucket_name: Optional S3 bucket name
        s3_key: Optional S3 key for the checkpoint

    Returns:
        Tuple of (epoch number, accuracy)
    """
    # Download from S3 if specified
    if bucket_name and s3_key:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            s3_client = boto3.client("s3")
            s3_client.download_file(bucket_name, s3_key, filepath)
            logging.info(f"Checkpoint downloaded from s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            logging.error(f"Error downloading checkpoint from S3: {e}")
            if not os.path.exists(filepath):
                return 0, 0.0

    if not os.path.exists(filepath):
        logging.info(f"No checkpoint found at {filepath}")
        return 0, 0.0

    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    accuracy = checkpoint["accuracy"]

    logging.info(f"Checkpoint loaded from {filepath}")
    (f"Resuming from epoch {epoch}, accuracy: {accuracy:.2f}%")

    return epoch, accuracy


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    print(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageNet-100")

    # Data source type
    parser.add_argument("--use_s3", action="store_true", help="Use S3 for dataset (default: use local directory)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./imagenet100",
        help="Local dataset directory path (used when --use_s3 is not specified)",
    )

    # S3 Dataset Arguments (required only when using S3)
    parser.add_argument(
        "--bucket_name", type=str, help="Name of the S3 bucket containing the dataset (required when using S3)"
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        default="train",
        help="Path to training data folder (S3 path when using S3, relative to data_dir when local)",
    )
    parser.add_argument(
        "--val_folder",
        type=str,
        default="val",
        help="Path to validation data folder (S3 path when using S3, relative to data_dir when local)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="/tmp/imagenet100",
        help="Local directory to cache the dataset when using S3 (default: /tmp/imagenet100)",
    )

    # S3 Checkpoint Arguments
    parser.add_argument("--checkpoint_bucket", type=str, help="S3 bucket for saving/loading checkpoints")
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="checkpoints",
        help="S3 prefix for checkpoint files (default: checkpoints)",
    )

    # Logging Arguments
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for log files (default: ./logs)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate (default: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers (default: 4)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")

    # Mixed Precision Training Arguments
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training (FP16/BF16)")
    parser.add_argument(
        "--precision_type",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Precision type for mixed precision training (default: fp16)",
    )

    args = parser.parse_args()

    # Validate S3 arguments if using S3
    if args.use_s3 and not args.bucket_name:
        parser.error("--bucket_name is required when using --use_s3")

    # Set device

    device = torch.device("cuda")
    logging.info(f"Automatically selected device: {device}")

    logging.info(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set up logging
    setup_logging(args.log_dir)

    # Create data loaders
    if args.use_s3:
        logging.info("Loading ImageNet-100 dataset from S3...")
        data_source = args.bucket_name
    else:
        logging.info(f"Loading ImageNet-100 dataset from local directory: {args.data_dir}")
        data_source = args.data_dir
        # Verify data directory exists
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {args.data_dir}")

    train_loader, val_loader, num_classes = create_data_loaders(
        data_source,
        args.train_folder,
        args.val_folder,
        args.local_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        use_s3=args.use_s3,
    )

    # Create model
    logging.info("Creating ResNet-50 model...")
    model = ResNet50(num_classes=num_classes).to(device)

    # logging.info model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed Precision Training Setup
    scaler = None
    if args.mixed_precision:
        if args.precision_type == "bf16" and not torch.cuda.is_bf16_supported():
            print("Warning: BF16 not supported on this GPU, falling back to FP16")
            args.precision_type = "fp16"

        scaler = GradScaler()
        print(f"🚀 Mixed precision training enabled with {args.precision_type.upper()}")
        print(f"   Expected benefits: ~2x speedup, ~50% memory reduction")
    else:
        print("📊 Standard FP32 precision training")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0.0

    if args.resume:
        start_epoch, best_accuracy = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            bucket_name=args.checkpoint_bucket,
            s3_key=f"{args.checkpoint_prefix}/{os.path.basename(args.resume)}",
        )
        start_epoch += 1  # Resume from next epoch

    # Training loop
    logging.info(f"\nStarting training for {args.epochs} epochs...")
    logging.info(f"Training from epoch {start_epoch + 1} to {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logging.info("-" * 50)

        # Train with mixed precision support
        train_mixed_precision(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            scaler=scaler,
            use_mixed_precision=args.mixed_precision,
            precision_type=args.precision_type,
        )

        # Evaluate with mixed precision support
        logging.info("\nEvaluating on validation set...")
        accuracy = evaluate_mixed_precision(
            model,
            val_loader,
            criterion,
            device,
            use_mixed_precision=args.mixed_precision,
            precision_type=args.precision_type,
        )

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pth"
            checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                accuracy,
                checkpoint_path,
                bucket_name=args.checkpoint_bucket,
                s3_key=f"{args.checkpoint_prefix}/{checkpoint_name}",
            )

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = "best_model.pth"
            best_model_path = os.path.join(args.save_dir, best_model_name)
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                accuracy,
                best_model_path,
                bucket_name=args.checkpoint_bucket,
                s3_key=f"{args.checkpoint_prefix}/{best_model_name}",
            )
            logging.info(f"New best accuracy: {best_accuracy:.2f}%")

        logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    logging.info("\nTraining completed!")
    logging.info(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
