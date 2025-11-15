# ResNet-50 ImageNet Training 🚀

A comprehensive implementation of ResNet-50 for training on ImageNet dataset, with testing full ImageNet-1K. Features advanced training techniques including mixed precision, advanced augmentations, and optimal learning rate scheduling.

## Model Architecture 🏗️

ResNet-50 implementation with:
- Input: 224×224×3 images
- 4 feature extraction stages with bottleneck blocks
- Output features: 2048
- Final classifier: 2048 -> num_classes
- BatchNorm and ReLU activations

## Dataset Strategy 📊

### Phase 1: ImageNet-100 Subset Testing
- 100 carefully selected classes
- ~130K training images
- ~5K validation images
- Used for rapid prototyping and validation

### Phase 2: Full ImageNet (1K classes)
- Complete ImageNet dataset
- ~1.2M training images
- ~50K validation images
- Full-scale training

## Training Pipeline 🛠️

### Advanced Augmentations
- RandAugment for robust feature learning (--use_randaugment)
- Mixup augmentation (α = 0.2, --mixup_alpha=0.2)
- CutMix (probability = 0.5, α = 1.0, --cutmix_prob=0.5 --cutmix_alpha=1.0)
- Random resized crops and flips
- Normalized with ImageNet stats

### Optimization Settings
- Optimizer: SGD (--optimizer=sgd)
  - Base learning rate: 0.06 (--lr=0.06)
  - Momentum: 0.9 (--momentum=0.9)
  - Weight decay: 1e-4 (--weight_decay=1e-4)
- Learning Rate Schedule
  - Warmup epochs: 10 (--warmup_epochs=10)
  - EMA enabled (--use_ema)
- Mixed Precision: BF16 (--mixed_precision --precision_type=bf16)

### Training Configuration
- Batch size: 256 (--batch_size=256)
- Epochs: 120 (--epochs=120)
- Label smoothing: 0.1 (--label_smoothing=0.1)
- Data loading workers: 4 (--num_workers=4)
- Checkpoint frequency: Every 5 epochs (--save_freq=5)
- Checkpoint directory: /mnt/imagenet/data/checkpoints (--checkpoint_prefix)
- Resume training: Supported (--resume=/mnt/imagenet/data/checkpoints/best_model.pth)

## Current Results 📈

### ImageNet-100 Subset Performance
Based on test_subset2.ipynb:
- Peak validation accuracy: ~82%
- Convergence: ~100 epochs
- Training time: ~8 hours (V100 GPU)
- Early stopping patience: 10 epochs
- Best checkpoint saved at model_best.pth

### Training Curves
- Steady learning rate warmup (first 15% epochs)
- Consistent accuracy improvement until plateau
- Effective early stopping prevents overfitting

## Usage Guide 🚀

### For Subset Testing
```bash
python run_local_training.py \
    --data_dir ./imagenet100 \
    --mixed_precision \
    --batch_size 128 \
    --epochs 100 \
    --label_smoothing 0.15
```

### For Full Dataset
```bash
python run_local_training.py \
    --data_dir=/mnt/imagenet/data \
    --batch_size=256 \
    --epochs=120 \
    --lr=0.06 \
    --momentum=0.9 \
    --warmup_epochs=10 \
    --weight_decay=1e-4 \
    --label_smoothing=0.1 \
    --use_ema \
    --use_cutmix \
    --cutmix_prob=0.5 \
    --mixup_alpha=0.2 \
    --cutmix_alpha=1.0 \
    --use_randaugment \
    --mixed_precision \
    --precision_type=bf16 \
    --optimizer=sgd \
    --num_workers=4 \
    --save_freq=5 \
    --checkpoint_prefix=/mnt/imagenet/data/checkpoints
```

## Dependencies 📦
```
torch>=2.0.0
torchvision
numpy
tqdm
kaggle  # for dataset download
```

## Monitoring 📊

Progress monitoring available through:
- Real-time loss/accuracy tracking
- Learning rate scheduling visualization
- Resource utilization metrics
- Automatic checkpointing

## Model Export 📤
- Regular checkpoints saved every epoch
- Best model saved based on validation accuracy
- Support for FP32 and quantized models
- ONNX export capability
```
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `/mnt/imagenet/data` | Path to ImageNet dataset |
| `--batch_size` | `256` | Training batch size |
| `--epochs` | `120` | Number of training epochs |
| `--lr` | `0.06` | Initial learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--optimizer` | `sgd` | Optimizer type (sgd/adamw) |
| `--warmup_epochs` | `10` | Number of warmup epochs |
| `--weight_decay` | `1e-4` | Weight decay (L2 regularization) |
| `--label_smoothing` | `0.1` | Label smoothing factor |
| `--mixed_precision` | `True` | Enable mixed precision training |
| `--precision_type` | `bf16` | Precision type (bf16/fp16) |
| `--use_ema` | `True` | Use Exponential Moving Average |
| `--use_cutmix` | `True` | Enable CutMix augmentation |
| `--cutmix_prob` | `0.5` | Probability of applying CutMix |
| `--cutmix_alpha` | `1.0` | Alpha parameter for CutMix |
| `--mixup_alpha` | `0.2` | Alpha parameter for Mixup |
| `--use_randaugment` | `True` | Enable RandAugment |
| `--num_workers` | `4` | Number of data loading workers |
| `--save_freq` | `5` | Save checkpoint frequency (epochs) |
| `--checkpoint_prefix` | `/checkpoints` | Checkpoint directory path |
| `--resume` | `None` | Path to checkpoint for resuming training |



## 🔧 Model Architecture

### ResNet-50 Components
- **Initial Conv Layer**: 7x7 conv, stride 2, 64 filters
- **Max Pooling**: 3x3 kernel, stride 2
- **Stage 1**: 3 bottleneck blocks (64 → 256 channels)
- **Stage 2**: 4 bottleneck blocks (256 → 512 channels)
- **Stage 3**: 6 bottleneck blocks (512 → 1024 channels)
- **Stage 4**: 3 bottleneck blocks (1024 → 2048 channels)
- **Global Average Pooling**: 1x1 output
- **Final FC Layer**: 2048 → 100 classes

### Bottleneck Block
Each bottleneck block consists of:
1. 1x1 convolution (dimension reduction)
2. 3x3 convolution (main processing)
3. 1x1 convolution (dimension expansion)
4. Shortcut connection with residual learning

## 📈 Data Augmentation

### Training Augmentations
- **Random Resized Crop**: 224x224
- **Random Horizontal Flip**: 50% probability
- **Color Jittering**: Brightness, contrast, saturation (0.2)
- **Random Affine**: Rotation ±15°, translation (0.1, 0.1)
- **Mixup**: α=0.2 for label smoothing
- **Normalization**: ImageNet mean/std

### Test Augmentations
- **Resize**: 256x256
- **Center Crop**: 224x224
- **Normalization**: ImageNet mean/std

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Monitor training progress
python monitor_training.py
```

### Checkpoint Management
- **Best Model**: `checkpoints/best_model.pth`
- **Periodic Checkpoints**: `checkpoints/checkpoint_epoch_N.pth`
- **Resume Training**: `--resume path/to/checkpoint.pth`

### Training Output
```
Evaluating: 100%|██████████| 196/196 [01:18<00:00,  2.48it/s, loss=1.942, acc=75.00%, skipped=0]
INFO:root:Test set: Average loss: 1.942, Accuracy: 75.00%
INFO:root:💾 Saved checkpoint (best_model.pth) with acc=75.00%, time=51.84 mins
INFO:root:Epoch 120/125 completed in 0:51:50, Accuracy: 75.00%, Best: 75.00% 
Epoch 120: 100%|██████████| 5005/5005 [50:31<00:00,  1.65it/s, loss=2.887, acc=59.87%, lr=0.000279]

Evaluating: 100%|██████████| 196/196 [01:36<00:00,  2.03it/s, loss=1.944, acc=75.08%, skipped=0]
INFO:root:Test set: Average loss: 1.944, Accuracy: 75.08%
INFO:root:✅ FixRes done @ 288 — acc=75.08%
```

## 🎯 Expected Results

### Training Performance
- **Initial Accuracy**: ~1% (random)
- **Convergence**: 50-80% after 50-100 epochs
- **Best Accuracy**: 70-85% (depending on hyperparameters)

### Training Time and Infrastructure
- **Total Training Time**: 85-87 hours (full 120 epochs)
- **Infrastructure**:
  - AWS G5.2xlarge EC2 instance (single GPU)
  - Spot instance configuration
  - EBS storage for dataset

### Cost Breakdown 💰
- **Compute Cost**: ~$44 (85-87 hours × $0.51/hour spot price)
- **Storage Cost**: ~$11 (EBS volume for dataset)
- **Total Cost**: ~$55 for complete training

### Resource Requirements
- **GPU Memory**: NVIDIA A10G GPU (24GB VRAM)
- **Storage**: EBS volume for dataset and checkpoints
- **Instance Type**: g5.2xlarge (4 vCPU, 16GB RAM)


### Further enhancements with OneCycleLR

The training to be optimized with OneCycleLR scheduler to reduce training time and improve convergence:
- Faster convergence (reduces needed epochs by ~30-40%)
- Better generalization through learning rate annealing
- Automatic warmup and cooldown phases
- Momentum cycling for improved stability
- Typically reaches 75% accuracy in ~60 epochs vs 120+ with step LR


## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python src/main.py --batch_size 64
```

**2. Slow Training**
```bash
# Increase number of workers
python src/main.py --num_workers 8
```

**3. Dataset Not Found**
```bash
# Re-download dataset
python download_imagenet100.py
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## 📚 Dependencies

### Core Dependencies
- `torch>=2.1.0` - PyTorch framework
- `torchvision>=0.16.0` - Computer vision utilities
- `numpy>=1.24.0` - Numerical computing
- `pillow>=10.0.0` - Image processing
- `tqdm>=4.65.0` - Progress bars

### Dataset Dependencies
- `datasets>=2.14.0` - Hugging Face datasets
- `requests>=2.31.0` - HTTP requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ResNet Paper**: Deep Residual Learning for Image Recognition (He et al., 2016)
- **ImageNet-100**: Few-Shot Learning with 100-ImageNet (Vinyals et al., 2016)
- **Mixup**: Beyond Empirical Risk Minimization (Zhang et al., 2017)
- **Hugging Face**: For providing the 100-ImageNet dataset


**Happy Training! 🚀**
