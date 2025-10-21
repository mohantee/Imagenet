# ResNet-50 Mini-ImageNet Training

A complete implementation of ResNet-50 for training on the Mini-ImageNet dataset with advanced data augmentation techniques including Mixup.

## 🚀 Features

- **Custom ResNet-50 Implementation**: Built from scratch with proper bottleneck blocks
- **Mini-ImageNet Dataset**: Automated download and setup from Hugging Face
- **Advanced Data Augmentation**: Mixup, random crops, color jittering, and more
- **Training Pipeline**: Complete training loop with checkpointing and evaluation
- **Progress Monitoring**: Real-time training progress with tqdm
- **Model Checkpointing**: Automatic saving of best models and periodic checkpoints

## 📁 Project Structure

```
Imagenet/
├── src/
│   ├── model.py          # ResNet-50 implementation
│   ├── train.py          # Training functions with mixup
│   ├── test.py           # Evaluation functions
│   └── main.py           # Main training script
├── download_mini_imagenet.py  # Dataset download script
├── monitor_training.py        # Training monitoring tool
├── setup.py                   # Quick setup script
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Setup
```bash
# Clone or download the project
cd Imagenet

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pillow tqdm datasets requests

# Download Mini-ImageNet dataset
python download_mini_imagenet.py
```

## 📊 Dataset

### Mini-ImageNet
- **Classes**: 100
- **Training Images**: 50,000 (500 per class)
- **Test Images**: 10,000 (100 per class)
- **Image Size**: 84x84 pixels (resized to 224x224 for training)
- **Source**: Hugging Face Datasets Hub

### Download
```bash
# Download to default location (./mini-imagenet)
python download_mini_imagenet.py

# Download to custom location
python download_mini_imagenet.py /path/to/custom/directory
```

## 🏃‍♂️ Training

### Basic Training
```bash
# Activate virtual environment
source venv/bin/activate

# Start training with default parameters
python src/main.py --data_dir ./mini-imagenet
```

### Advanced Training Options
```bash
python src/main.py \
    --data_dir ./mini-imagenet \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --num_workers 4 \
    --save_freq 10
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./mini-imagenet` | Path to Mini-ImageNet dataset |
| `--batch_size` | `32` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `0.1` | Initial learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--weight_decay` | `1e-4` | Weight decay (L2 regularization) |
| `--num_workers` | `4` | Data loading workers |
| `--device` | `auto` | Device (cuda/cpu/auto) |
| `--save_freq` | `10` | Save checkpoint every N epochs |
| `--no_augment` | `False` | Disable data augmentation |

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
Epoch 1/100
--------------------------------------------------
Epoch 1: 100%|██████████| 1563/1563 [05:23<00:00, 4.85it/s]

Evaluating on test set...
Evaluating: 100%|██████████| 313/313 [00:45<00:00, 6.89it/s]

Test set: Average loss: 4.605, Accuracy: 1.00%
New best accuracy: 1.00%
Current learning rate: 0.100000
```

## 🎯 Expected Results

### Training Performance
- **Initial Accuracy**: ~1% (random)
- **Convergence**: 50-80% after 50-100 epochs
- **Best Accuracy**: 70-85% (depending on hyperparameters)

### Training Time
- **CPU**: ~2-3 hours for 50 epochs
- **GPU**: ~30-60 minutes for 50 epochs
- **Memory**: ~2-4 GB RAM

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python src/main.py --batch_size 16
```

**2. Slow Training**
```bash
# Increase number of workers
python src/main.py --num_workers 8
```

**3. Dataset Not Found**
```bash
# Re-download dataset
python download_mini_imagenet.py
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
- **Mini-ImageNet**: Few-Shot Learning with Mini-ImageNet (Vinyals et al., 2016)
- **Mixup**: Beyond Empirical Risk Minimization (Zhang et al., 2017)
- **Hugging Face**: For providing the Mini-ImageNet dataset

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](../../issues)
3. Create a new [Issue](../../issues/new) with detailed information

---

**Happy Training! 🚀**