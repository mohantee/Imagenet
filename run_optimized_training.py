#!/usr/bin/env python3
"""
Optimized Mixed Precision Training Script for ResNet-50 on ImageNet-100
Provides maximum performance with FP16/BF16 mixed precision training.
"""

import subprocess
import sys
import os
import torch


def check_gpu_support():
    """Check GPU capabilities for mixed precision training."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Mixed precision requires GPU.")
        return False, False

    gpu_name = torch.cuda.get_device_name(0)
    fp16_support = True  # Most modern GPUs support FP16
    bf16_support = torch.cuda.is_bf16_supported()

    print(f"🔍 GPU Detected: {gpu_name}")
    print(f"   FP16 Support: {'✅' if fp16_support else '❌'}")
    print(f"   BF16 Support: {'✅' if bf16_support else '❌'}")

    return fp16_support, bf16_support


def run_optimized_training():
    """Run optimized mixed precision training."""

    fp16_support, bf16_support = check_gpu_support()

    if not fp16_support:
        print("❌ This GPU doesn't support mixed precision training.")
        return False

    # Optimized configurations for different scenarios
    configs = {
        "high_performance_fp16": {
            "name": "🚀 High Performance FP16 (Recommended)",
            "data_dir": "/mnt/ebs-volume/imagenet100",  # UPDATE THIS
            "mixed_precision": True,
            "precision_type": "fp16",
            "batch_size": 64,  # 2x larger than FP32
            "epochs": 100,
            "lr": 0.1,
            "num_workers": 8,
            "save_freq": 10,
        },
        "ultra_stable_bf16": {
            "name": "🎯 Ultra Stable BF16 (Ampere+ GPUs)",
            "data_dir": "/mnt/ebs-volume/imagenet100",  # UPDATE THIS
            "mixed_precision": True,
            "precision_type": "bf16",
            "batch_size": 64,
            "epochs": 100,
            "lr": 0.1,
            "num_workers": 8,
            "save_freq": 10,
            "available": bf16_support,
        },
        "memory_optimized": {
            "name": "💾 Memory Optimized (Large Batch)",
            "data_dir": "/mnt/ebs-volume/imagenet100",  # UPDATE THIS
            "mixed_precision": True,
            "precision_type": "fp16",
            "batch_size": 128,  # 4x larger than FP32
            "epochs": 100,
            "lr": 0.15,  # Slightly higher LR for larger batch
            "num_workers": 8,
            "save_freq": 10,
        },
        "fp32_baseline": {
            "name": "📊 FP32 Baseline (Comparison)",
            "data_dir": "/mnt/ebs-volume/imagenet100",  # UPDATE THIS
            "mixed_precision": False,
            "precision_type": "fp16",  # Ignored
            "batch_size": 32,  # Standard batch size
            "epochs": 100,
            "lr": 0.1,
            "num_workers": 8,
            "save_freq": 10,
        },
    }

    # Filter available configurations
    available_configs = {}
    for key, config in configs.items():
        if config.get("available", True):  # Include if not specified or True
            available_configs[key] = config

    # Show available configurations
    print("\\nAvailable Training Configurations:")
    print("=" * 50)
    for i, (key, config) in enumerate(available_configs.items(), 1):
        print(f"{i}. {config['name']}")
        if config["mixed_precision"]:
            print(f"   Precision: {config['precision_type'].upper()}")
            print(f"   Batch Size: {config['batch_size']} (vs 32 for FP32)")
            print(f"   Expected Speedup: ~2x")
            print(f"   Memory Savings: ~50%")
        else:
            print(f"   Standard FP32 training for comparison")
        print()

    # Get user choice
    while True:
        try:
            choice = input(f"Select configuration (1-{len(available_configs)}): ")
            choice_idx = int(choice) - 1
            config_key = list(available_configs.keys())[choice_idx]
            config = available_configs[config_key]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")

    print(f"\\n🚀 Selected: {config['name']}")

    # Build optimized command
    cmd = [
        sys.executable,
        "src/main.py",
        "--data_dir",
        config["data_dir"],
        "--batch_size",
        str(config["batch_size"]),
        "--epochs",
        str(config["epochs"]),
        "--lr",
        str(config["lr"]),
        "--num_workers",
        str(config["num_workers"]),
        "--save_freq",
        str(config["save_freq"]),
        "--save_dir",
        f"./checkpoints_{config['precision_type'] if config['mixed_precision'] else 'fp32'}",
        "--log_dir",
        f"./logs_{config['precision_type'] if config['mixed_precision'] else 'fp32'}",
    ]

    # Add mixed precision flags
    if config["mixed_precision"]:
        cmd.extend(["--mixed_precision"])
        cmd.extend(["--precision_type", config["precision_type"]])

    # Show training summary
    print("\\n" + "=" * 60)
    print("🎯 TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {config['data_dir']}")
    print(f"Mixed Precision: {'✅ ENABLED' if config['mixed_precision'] else '❌ DISABLED'}")
    if config["mixed_precision"]:
        print(f"Precision Type: {config['precision_type'].upper()}")
        print(f"Expected Benefits:")
        print(f"  • Training Speed: ~2x faster")
        print(f"  • Memory Usage: ~50% reduction")
        print(f"  • Batch Size: {config['batch_size']} (vs 32 standard)")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Workers: {config['num_workers']}")
    print("=" * 60)

    # Verify dataset exists
    if not os.path.exists(config["data_dir"]):
        print(f"\\n❌ ERROR: Dataset directory not found: {config['data_dir']}")
        print("Please update the data_dir path in this script.")
        return False

    print(f"\\nCommand: {' '.join(cmd)}")
    print("\\n🚀 Starting optimized training...")

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\\n🎉 Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
        return False


def show_performance_tips():
    """Show performance optimization tips."""
    print("\\n🔧 PERFORMANCE OPTIMIZATION TIPS")
    print("=" * 50)
    print("1. 🚀 Use FP16 for broad compatibility")
    print("2. 🎯 Use BF16 on Ampere+ GPUs for stability")
    print("3. 💾 Increase batch size with mixed precision")
    print("4. ⚡ Use more workers with fast storage (EBS)")
    print("5. 📊 Monitor GPU utilization with nvidia-smi")
    print("6. 🔍 Compare with FP32 baseline for accuracy")
    print()
    print("Expected Performance Gains:")
    print("• RTX 3090: ~1.8x speedup with FP16")
    print("• A100: ~2.0x speedup with BF16")
    print("• Memory: ~50% reduction")
    print("• Batch size: Can typically double")


def main():
    """Main function."""
    print("🚀 OPTIMIZED MIXED PRECISION TRAINING")
    print("ResNet-50 on ImageNet-100")
    print("=" * 50)

    show_performance_tips()

    response = input("\\nProceed with optimized training? (y/n): ")
    if response.lower() != "y":
        print("Training cancelled.")
        return

    success = run_optimized_training()

    if success:
        print("\\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("\\n📊 Next Steps:")
        print("1. Compare training time with FP32 baseline")
        print("2. Verify accuracy is within expected range")
        print("3. Check saved checkpoints in output directory")
        print("4. Monitor GPU memory usage during training")
    else:
        print("\\n❌ TRAINING FAILED")
        print("\\n🛠️ Troubleshooting:")
        print("1. Check GPU memory with: nvidia-smi")
        print("2. Reduce batch size if out of memory")
        print("3. Verify dataset path is correct")
        print("4. Ensure CUDA and PyTorch are compatible")


if __name__ == "__main__":
    main()
