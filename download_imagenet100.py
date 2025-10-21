#!/usr/bin/env python3
"""
Script to download ImageNet-100 dataset from Kaggle for ResNet-50 training.
"""

import os
import sys
import zipfile
from tqdm import tqdm


def download_imagenet100(data_dir):
    """Download ImageNet-100 from Kaggle."""
    try:
        import kaggle
        print("Downloading ImageNet-100 from Kaggle...")
        
        # Download the dataset
        kaggle.api.dataset_download_files('ambityga/imagenet100', path=data_dir, unzip=True)
        
        print(f"ImageNet-100 dataset downloaded to {data_dir}")
        return True
        
    except ImportError:
        print("Error: kaggle library not found. Install it with: pip install kaggle")
        print("You also need to set up Kaggle API credentials.")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create API token and download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nAlternative: Manual download")
        print("1. Go to: https://www.kaggle.com/datasets/ambityga/imagenet100")
        print("2. Download the dataset")
        print("3. Extract to:", data_dir)
        print("4. Ensure structure: data_dir/train/ and data_dir/val/")
        return False


def main():
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './imagenet100'
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading ImageNet-100 dataset to {data_dir}")
    
    success = download_imagenet100(data_dir)
    
    if success:
        print("\n✅ ImageNet-100 dataset download completed successfully!")
        print(f"Dataset location: {data_dir}")
        print("\nYou can now run training with:")
        print(f"python src/main.py --data_dir {data_dir}")
    else:
        print("\n❌ Failed to download ImageNet-100 dataset")
        print("Make sure you have installed the required dependencies:")
        print("pip install kaggle tqdm")
        print("\nAnd set up Kaggle API credentials as shown above.")


if __name__ == '__main__':
    main()
