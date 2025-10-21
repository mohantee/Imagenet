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

from model import ResNet50
from train import train, train_transforms, test_transforms, mixup_data, mixup_criterion
from test import evaluate


def get_mini_imagenet_dataset(data_dir, train=True, augment=False):
    """
    Load Mini-ImageNet dataset.
    
    Args:
        data_dir: Path to the Mini-ImageNet dataset directory
        train: If True, load training set; if False, load test set
        augment: If True, apply data augmentation for training
        
    Returns:
        ImageFolder dataset
    """
    if train:
        dataset_path = os.path.join(data_dir, 'train')
        transform = train_transforms(augment=augment)
    else:
        dataset_path = os.path.join(data_dir, 'test')
        transform = test_transforms()
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    dataset = ImageFolder(dataset_path, transform=transform)
    return dataset


def create_data_loaders(data_dir, batch_size=32, num_workers=4, augment=True):
    """
    Create training and test data loaders for Mini-ImageNet.
    
    Args:
        data_dir: Path to the Mini-ImageNet dataset directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation to training set
        
    Returns:
        Tuple of (train_loader, test_loader, num_classes)
    """
    # Load datasets
    train_dataset = get_mini_imagenet_dataset(data_dir, train=True, augment=augment)
    test_dataset = get_mini_imagenet_dataset(data_dir, train=False, augment=False)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, 0.0
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch}, accuracy: {accuracy:.2f}%")
    
    return epoch, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-50 on Mini-ImageNet')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Mini-ImageNet dataset directory')
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
    
    # Create data loaders
    print("Loading Mini-ImageNet dataset...")
    train_loader, test_loader, num_classes = create_data_loaders(
        args.data_dir, 
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
            args.resume, model, optimizer, scheduler
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
        print("\nEvaluating on test set...")
        accuracy = evaluate(model, test_loader, criterion, device)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, accuracy, checkpoint_path)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, accuracy, best_model_path)
            print(f"New best accuracy: {best_accuracy:.2f}%")
        
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Best accuracy: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()
