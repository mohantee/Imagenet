# Using Local Dataset on EBS Volume

The code has been modified to support both S3 and local dataset sources. Since you've copied the ImageNet-100 dataset to your EBS volume, you can now use it directly without downloading from S3.

## Changes Made

### 1. Modified `src/main.py`

- Added `--use_s3` flag to enable S3 mode (defaults to local mode)
- Added `--data_dir` argument for local dataset directory
- Made S3 arguments optional when using local mode
- Updated dataset loading functions to support both S3 and local paths

### 2. New Arguments

**Local Mode (Default):**
- `--data_dir`: Path to your local dataset directory (e.g., `/mnt/ebs/imagenet100`)
- `--train_folder`: Subfolder for training data (default: `train`)
- `--val_folder`: Subfolder for validation data (default: `val`)

**S3 Mode (when using `--use_s3`):**
- `--bucket_name`: S3 bucket name (required when using S3)
- `--train_folder`: S3 path to training data
- `--val_folder`: S3 path to validation data
- `--local_dir`: Local cache directory for S3 downloads

## Usage Examples

### Running with Local Dataset (Recommended for EBS)

```bash
# Basic usage - update the data_dir path to your EBS volume
python src/main.py --data_dir /mnt/ebs/imagenet100

# With custom parameters
python src/main.py \
    --data_dir /mnt/ebs/imagenet100 \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.1 \
    --save_dir ./checkpoints \
    --num_workers 8
```

### Running with S3 Dataset (Original functionality)

```bash
python src/main.py \
    --use_s3 \
    --bucket_name your-s3-bucket \
    --train_folder path/to/train \
    --val_folder path/to/val \
    --local_dir /tmp/imagenet100
```

## Expected Dataset Structure

Your EBS volume dataset should be organized as follows:

```
/path/to/your/ebs/imagenet100/
├── train/
│   ├── n01440764/  # class folder
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/  # another class folder
│   │   ├── n01443537_2368.JPEG
│   │   └── ...
│   └── ... (100 class folders total)
└── val/
    ├── n01440764/
    │   ├── n01440764_18.JPEG
    │   └── ...
    ├── n01443537/
    │   └── ...
    └── ... (100 class folders total)
```

## Quick Start Script

Use the provided `run_local_training.py` script for easy setup:

1. Edit `run_local_training.py` and update the `data_dir` path to point to your EBS volume
2. Run: `python run_local_training.py`

## Advantages of Local Dataset

1. **Faster data loading**: No network download time
2. **Reliable**: No dependency on S3 connectivity
3. **Cost-effective**: No S3 transfer costs
4. **Consistent performance**: No network variability

## Performance Tips for EBS

1. **Use GP3 or io2 volumes** for better IOPS
2. **Increase num_workers** if you have sufficient CPU cores
3. **Use larger batch_size** if you have sufficient GPU memory
4. **Consider using pin_memory=True** (already enabled) for faster GPU transfers

## Troubleshooting

### Common Issues:

1. **"Dataset directory does not exist"**: Update the `--data_dir` path
2. **"No such file or directory"**: Verify the dataset structure matches the expected format
3. **Slow data loading**: Increase `--num_workers` or upgrade EBS volume type
4. **Out of memory**: Reduce `--batch_size`

### Verifying Dataset:

```bash
# Check if dataset structure is correct
ls /path/to/your/ebs/imagenet100/
ls /path/to/your/ebs/imagenet100/train/ | wc -l    # Should show 100 classes
ls /path/to/your/ebs/imagenet100/val/ | wc -l      # Should show 100 classes
```

## Original S3 Functionality

The original S3 functionality is preserved. Use `--use_s3` flag to enable S3 mode when needed.