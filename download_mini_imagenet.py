#!/usr/bin/env python3
"""
Script to download Mini-ImageNet dataset from Hugging Face for ResNet-50 training.
"""

import os
import sys
from tqdm import tqdm


def download_mini_imagenet(data_dir):
    """Download Mini-ImageNet from Hugging Face."""
    try:
        from datasets import load_dataset
        print("Loading Mini-ImageNet from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset('timm/mini-imagenet')
        
        # Create directory structure
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Save training data
        print("Saving training data...")
        train_data = dataset['train']
        for i, example in enumerate(tqdm(train_data, desc="Processing training data")):
            image = example['image']
            label = example['label']
            
            # Create class directory
            class_dir = os.path.join(train_dir, f'class_{label}')
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(class_dir, f'image_{i}.jpg')
            image.save(image_path)
        
        # Save test data
        print("Saving test data...")
        test_data = dataset['validation']  # Hugging Face uses 'validation' for test
        for i, example in enumerate(tqdm(test_data, desc="Processing test data")):
            image = example['image']
            label = example['label']
            
            # Create class directory
            class_dir = os.path.join(test_dir, f'class_{label}')
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(class_dir, f'image_{i}.jpg')
            image.save(image_path)
        
        print(f"Mini-ImageNet dataset saved to {data_dir}")
        return True
        
    except ImportError:
        print("Error: datasets library not found. Install it with: pip install datasets")
        return False
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        return False


def main():
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './mini-imagenet'
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading Mini-ImageNet dataset to {data_dir}")
    
    success = download_mini_imagenet(data_dir)
    
    if success:
        print("\n✅ Mini-ImageNet dataset download completed successfully!")
        print(f"Dataset location: {data_dir}")
        print("\nYou can now run training with:")
        print(f"python src/main.py --data_dir {data_dir}")
    else:
        print("\n❌ Failed to download Mini-ImageNet dataset")
        print("Make sure you have installed the required dependencies:")
        print("pip install datasets tqdm")


if __name__ == '__main__':
    main()
