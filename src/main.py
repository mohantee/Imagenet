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
from botocore.exceptions import ClientError
import tempfile
from pathlib import Path

from model import ResNet50
from train import train, train_transforms, test_transforms, mixup_data, mixup_criterion
from test import evaluate


def download_from_s3(bucket_name, s3_path, local_path):
    """
    Download data from S3 bucket to local path.
    
    Args:
        bucket_name: Name of the S3 bucket
        s3_path: Path within the S3 bucket
        local_path: Local path to save the data
    """
    s3_client = boto3.client('s3')
    try:
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        s3_client.download_file(bucket_name, s3_path, local_path)
        logging.info(f"Successfully downloaded {s3_path} to {local_path}")
    except ClientError as e:
        logging.error(f"Error downloading from S3: {e}")
        raise

def get_imagenet100_dataset(bucket_name, s3_prefix, local_dir, train=True, augment=False):
    """
    Load ImageNet-100 dataset from S3.
    
    Args:
        bucket_name: Name of the S3 bucket containing the dataset
        s3_prefix: Prefix path in the S3 bucket
        local_dir: Local directory to cache the dataset
        train: If True, load training set; if False, load validation set
        augment: If True, apply data augmentation for training
        
    Returns:
        ImageFolder dataset
    """
    # Set up paths
    split = 'train' if train else 'val'
    s3_path = f"{s3_prefix}/{split}"
    local_path = os.path.join(local_dir, split)
    
    # Create local cache directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # List all objects in the S3 path
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Download all files from S3
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_path):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            # Skip directories
            if obj['Key'].endswith('/'):
                continue
                
            # Get relative path and create local directory structure
            rel_path = obj['Key'][len(s3_path):].lstrip('/')
            local_file_path = os.path.join(local_path, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download if file doesn't exist locally
            if not os.path.exists(local_file_path):
                logging.info(f"Downloading {obj['Key']} to {local_file_path}")
                s3_client.download_file(bucket_name, obj['Key'], local_file_path)
    
    # Set up transforms
    transform = train_transforms(augment=augment) if train else test_transforms()
    
    # Create dataset
    dataset = ImageFolder(local_path, transform=transform)
    return dataset


def create_data_loaders(bucket_name, s3_prefix, local_dir, batch_size=32, num_workers=4, augment=True):
    """
    Create training and validation data loaders for ImageNet-100 from S3.
    
    Args:
        bucket_name: Name of the S3 bucket containing the dataset
        s3_prefix: Prefix path in the S3 bucket
        local_dir: Local directory to cache the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation to training set
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    # Load datasets
    logging.info("Loading training dataset from S3...")
    train_dataset = get_imagenet100_dataset(bucket_name, s3_prefix, local_dir, train=True, augment=augment)
    logging.info("Loading validation dataset from S3...")
    val_dataset = get_imagenet100_dataset(bucket_name, s3_prefix, local_dir, train=False, augment=False)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
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
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved locally to {filepath}")
    
    # Upload to S3 if specified
    if bucket_name and s3_key:
        try:
            s3_client = boto3.client('s3')
            s3_client.upload_file(filepath, bucket_name, s3_key)
            logging.info(f"Checkpoint uploaded to s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            logging.error(f"Error uploading checkpoint to S3: {e}")
            raise


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
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket_name, s3_key, filepath)
            logging.info(f"Checkpoint downloaded from s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            logging.error(f"Error downloading checkpoint from S3: {e}")
            if not os.path.exists(filepath):
                return 0, 0.0
    
    if not os.path.exists(filepath):
        logging.info(f"No checkpoint found at {filepath}")
        return 0, 0.0
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    
    logging.info(f"Checkpoint loaded from {filepath}")
    logging.info(f"Resuming from epoch {epoch}, accuracy: {accuracy:.2f}%")
    
    return epoch, accuracy


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet-100')
    # S3 Dataset Arguments
    parser.add_argument('--bucket_name', type=str, required=True,
                        help='Name of the S3 bucket containing the dataset')
    parser.add_argument('--s3_prefix', type=str, default='imagenet100',
                        help='Prefix path in the S3 bucket (default: imagenet100)')
    parser.add_argument('--local_dir', type=str, default='/tmp/imagenet100',
                        help='Local directory to cache the dataset (default: /tmp/imagenet100)')
    
    # S3 Checkpoint Arguments
    parser.add_argument('--checkpoint_bucket', type=str,
                        help='S3 bucket for saving/loading checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='checkpoints',
                        help='S3 prefix for checkpoint files (default: checkpoints)')
    
    # Logging Arguments
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for log files (default: ./logs)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(args.log_dir)
    
    # Create data loaders
    logging.info("Loading ImageNet-100 dataset from S3...")
    train_loader, val_loader, num_classes = create_data_loaders(
        args.bucket_name,
        args.s3_prefix,
        args.local_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment
    )
    
    # Create model
    print("Creating ResNet-50 model...")
    model = ResNet50(num_classes=num_classes).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0.0
    
    if args.resume:
        start_epoch, best_accuracy = load_checkpoint(
            args.resume, model, optimizer, scheduler,
            bucket_name=args.checkpoint_bucket,
            s3_key=f"{args.checkpoint_prefix}/{os.path.basename(args.resume)}"
        )
        start_epoch += 1  # Resume from next epoch
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training from epoch {start_epoch + 1} to {args.epochs}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        accuracy = evaluate(model, val_loader, criterion, device)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pth'
            checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
            save_checkpoint(
                model, optimizer, scheduler, epoch, accuracy, checkpoint_path,
                bucket_name=args.checkpoint_bucket,
                s3_key=f"{args.checkpoint_prefix}/{checkpoint_name}"
            )
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = 'best_model.pth'
            best_model_path = os.path.join(args.save_dir, best_model_name)
            save_checkpoint(
                model, optimizer, scheduler, epoch, accuracy, best_model_path,
                bucket_name=args.checkpoint_bucket,
                s3_key=f"{args.checkpoint_prefix}/{best_model_name}"
            )
            print(f"New best accuracy: {best_accuracy:.2f}%")
        
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()
