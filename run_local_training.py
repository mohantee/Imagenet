#!/usr/bin/env python3
"""
Example script to run ResNet-50 training with local ImageNet-100 dataset on EBS volume.
"""

import subprocess
import sys
import os


def run_training_local():
    """Run training with local dataset."""

    # Configuration for local training
    config = {
        "data_dir": "/path/to/your/ebs/imagenet100",  # UPDATE THIS PATH
        "train_folder": "train",  # Relative to data_dir
        "val_folder": "val",  # Relative to data_dir
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.1,
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        "num_workers": 4,
        "save_freq": 10,
    }

    # Build command
    cmd = [
        sys.executable,
        "src/main.py",
        "--data_dir",
        config["data_dir"],
        "--train_folder",
        config["train_folder"],
        "--val_folder",
        config["val_folder"],
        "--epochs",
        str(config["epochs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--save_dir",
        config["save_dir"],
        "--log_dir",
        config["log_dir"],
        "--num_workers",
        str(config["num_workers"]),
        "--save_freq",
        str(config["save_freq"]),
        # Note: --use_s3 is NOT included, so it defaults to local mode
    ]

    print("Starting ResNet-50 training with local ImageNet-100 dataset...")
    print(f"Dataset directory: {config['data_dir']}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    # Verify dataset directory exists
    if not os.path.exists(config["data_dir"]):
        print(f"ERROR: Dataset directory does not exist: {config['data_dir']}")
        print("Please update the 'data_dir' path in this script to point to your EBS volume dataset.")
        return False

    # Verify train and val folders exist
    train_path = os.path.join(config["data_dir"], config["train_folder"])
    val_path = os.path.join(config["data_dir"], config["val_folder"])

    if not os.path.exists(train_path):
        print(f"ERROR: Training folder does not exist: {train_path}")
        return False

    if not os.path.exists(val_path):
        print(f"ERROR: Validation folder does not exist: {val_path}")
        return False

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False


def main():
    """Main function."""
    print("ResNet-50 ImageNet-100 Local Training Script")
    print("=" * 50)

    # Show expected dataset structure
    print("\nExpected dataset structure on your EBS volume:")
    print("/path/to/your/ebs/imagenet100/")
    print("├── train/")
    print("│   ├── class1/")
    print("│   │   ├── image1.jpg")
    print("│   │   ├── image2.jpg")
    print("│   │   └── ...")
    print("│   ├── class2/")
    print("│   └── ...")
    print("└── val/")
    print("    ├── class1/")
    print("    ├── class2/")
    print("    └── ...")
    print()

    # Check if we should proceed
    response = input("Have you updated the data_dir path in this script? (y/n): ")
    if response.lower() != "y":
        print("Please edit this script and update the 'data_dir' path to point to your EBS volume dataset.")
        return

    # Run training
    success = run_training_local()

    if success:
        print("\n🎉 Training completed successfully!")
        print("Check the './checkpoints' directory for saved models.")
        print("Check the './logs' directory for training logs.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
