# main_refactored.py
import os
import time
import json
import numpy as np
import torch
import logging
import argparse
import tempfile
from tqdm import tqdm
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import ResNet50
from train import train_transforms, test_transforms, fixres_train_transforms, fixres_val_transforms
from torch.amp import GradScaler

# Optional: EMA for better model stability
try:
    from torch_ema import ExponentialMovingAverage
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    logging.warning("torch_ema not available. Install with: pip install torch-ema")

# -------------------------------------------------------------------
# Storage Handler: single class for S3 / Local storage operations
# -------------------------------------------------------------------
class StorageHandler:
    def __init__(self, storage_type="local", bucket=None, prefix="checkpoints", cache_dir="/tmp"):
        self.storage_type = storage_type
        self.bucket = bucket
        self.prefix = prefix
        self.cache_dir = cache_dir
        if storage_type == "s3":
            self.client = boto3.client("s3", config=Config(max_pool_connections=50, retries={'max_attempts': 3}))

    def _local_path(self, key):
        return os.path.join(self.prefix if self.storage_type == "local" else self.cache_dir, key)

    # --- Dataset Handling ---
    def sync_dataset(self, s3_path, local_path):
        if self.storage_type == "local":
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Dataset not found: {local_path}")
            return local_path

        paginator = self.client.get_paginator("list_objects_v2")
        os.makedirs(local_path, exist_ok=True)
        tasks = []

        cache_file = os.path.join(local_path, ".cache_metadata.json")
        cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}

        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_path):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                rel_path = os.path.relpath(obj["Key"], s3_path)
                local_file = os.path.join(local_path, rel_path)
                etag = obj["ETag"]
                cache_key = f"{self.bucket}:{obj['Key']}"

                if cache.get(cache_key) != etag or not os.path.exists(local_file):
                    tasks.append((self.bucket, obj["Key"], local_file, etag))
                    cache[cache_key] = etag

        with ThreadPoolExecutor(max_workers=16) as executor:
            for fut in tqdm(as_completed([executor.submit(self._download_file, t) for t in tasks]),
                            total=len(tasks), desc="Downloading"):
                fut.result()

        json.dump(cache, open(cache_file, "w"))
        return local_path

    def _download_file(self, task):
        bucket, key, path, etag = task
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.client.download_file(bucket, key, path)

    # --- Checkpoint Handling ---
    def save_checkpoint(self, checkpoint_dict, filename):
        if self.storage_type == "local":
            os.makedirs(self.prefix, exist_ok=True)
            torch.save(checkpoint_dict, os.path.join(self.prefix, filename))
            return

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(checkpoint_dict, tmp.name)
            self.client.upload_file(tmp.name, self.bucket, f"{self.prefix}/{filename}")
            os.unlink(tmp.name)

    def load_checkpoint(self, filename):
        if self.storage_type == "local":
            path = os.path.join(self.prefix, filename)
            return torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                self.client.download_file(self.bucket, f"{self.prefix}/{filename}", tmp.name)
                ckpt = torch.load(tmp.name, map_location="cpu")
                os.unlink(tmp.name)
                return ckpt
            except ClientError:
                return None


# -------------------------------------------------------------------
# Training and evaluation (using train.py and test.py modules)
# -------------------------------------------------------------------
from train import train as train_epoch
from test import evaluate as evaluate_epoch


# -------------------------------------------------------------------
# Data loader creation
# -------------------------------------------------------------------
def build_dataloaders(storage: StorageHandler, train_path, val_path, batch_size, workers,
                      use_randaugment=True, eval_resolution: int = 224):
    if storage.storage_type == "s3":
        train_dir = storage.sync_dataset(train_path, os.path.join(storage.cache_dir, "train"))
        val_dir = storage.sync_dataset(val_path, os.path.join(storage.cache_dir, "val"))
    else:
        train_dir, val_dir = train_path, val_path

    # Use advanced augmentation for training
    train_set = ImageFolder(train_dir, transform=train_transforms(augment=True, use_randaugment=use_randaugment))
    val_set = ImageFolder(val_dir, transform=test_transforms(eval_resolution))

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, 
                   persistent_workers=True if workers > 0 else False),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                   persistent_workers=True if workers > 0 else False),
        len(train_set.classes),
    )






# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use_s3", action="store_true")
    p.add_argument("--bucket", type=str)
    p.add_argument("--data_dir", default="./imagenet100")
    p.add_argument("--train_folder", default="train")
    p.add_argument("--val_folder", default="val")
    p.add_argument("--checkpoint_prefix", default="checkpoints")
    p.add_argument("--checkpoint_bucket", type=str)
    p.add_argument("--save_dir", choices=["local", "s3"], default="local")
    p.add_argument("--epochs", type=int, default=120)  # More epochs for better convergence
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)  # More workers for faster data loading
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--precision_type", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--use_cutmix", action="store_true", default=True, help="Use CutMix augmentation")
    p.add_argument("--cutmix_prob", type=float, default=0.5, help="Probability of CutMix vs Mixup")
    p.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter")
    p.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix alpha parameter")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--use_randaugment", action="store_true", default=True, help="Use RandAugment")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for learning rate")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd", help="Optimizer type")
    p.add_argument("--fixres_enable", action="store_true", help="Enable FixRes fine-tuning at higher resolution")
    p.add_argument("--fixres_resolution", type=int, default=288, help="FixRes fine-tune/eval resolution")
    p.add_argument("--fixres_epochs", type=int, default=5, help="FixRes fine-tune epochs (classifier only)")
    p.add_argument("--fixres_lr", type=float, default=0.01, help="FixRes LR for classifier")
    p.add_argument("--fixres_batch_size", type=int, default=None, help="Optional batch size for FixRes (defaults to --batch_size)")
    args = p.parse_args()

    os.makedirs(args.checkpoint_prefix, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storage = StorageHandler(
        storage_type="s3" if args.use_s3 else "local",
        bucket=args.bucket if args.use_s3 else args.checkpoint_bucket,
        prefix=args.checkpoint_prefix,
        cache_dir="/tmp/imagenet100"
    )

    train_loader, val_loader, num_classes = build_dataloaders(
        storage,
        os.path.join(args.data_dir, args.train_folder) if not args.use_s3 else args.train_folder,
        os.path.join(args.data_dir, args.val_folder) if not args.use_s3 else args.val_folder,
        args.batch_size,
        args.num_workers,
        use_randaugment=args.use_randaugment,
        eval_resolution=224,
    )

    model = ResNet50(num_classes=num_classes).to(device)
    
    if not args.fixres_enable:
        # Use SGD for ImageNet (typically better than AdamW)
        if args.optimizer == "sgd":
            # Scale learning rate by batch size (linear scaling rule)
            base_lr = args.lr * (args.batch_size / 256.0)
            optimizer = optim.SGD(
                model.parameters(), 
                lr=base_lr, 
                momentum=args.momentum, 
                weight_decay=args.weight_decay,
                nesterov=True  # Nesterov momentum for better convergence
            )
            
            # Cosine annealing with warmup (best for ImageNet)
            warmup_steps = args.warmup_epochs * len(train_loader)
            total_steps = args.epochs * len(train_loader)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            step_scheduler_on_batch = True
        else:
            # AdamW with OneCycleLR (alternative)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                steps_per_epoch=len(train_loader),
                epochs=args.epochs,
                pct_start=args.warmup_epochs / args.epochs,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
            step_scheduler_on_batch = True
        
        # Loss with label smoothing (improves generalization)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        dtype = torch.float16 if args.precision_type == "fp16" else torch.bfloat16
        
        # Mixed precision scaler
        scaler = GradScaler("cuda") if args.mixed_precision else None
        
        # EMA for model stability (optional but recommended)
        ema = None
        if args.use_ema and EMA_AVAILABLE:
            ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
            logging.info(f"✅ EMA enabled with decay={args.ema_decay}")
        elif args.use_ema and not EMA_AVAILABLE:
            logging.warning("EMA requested but torch_ema not installed. Install with: pip install torch-ema")

        start_epoch, best_acc = 0, 0
        if args.resume:
            ckpt = storage.load_checkpoint(args.resume)
            if ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_acc = ckpt.get("accuracy", 0)
                if ema and "ema_state_dict" in ckpt:
                    ema.load_state_dict(ckpt["ema_state_dict"])
                logging.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")

        logging.info("=" * 80)
        logging.info("Advanced Training Configuration:")
        logging.info(f"  Optimizer: {args.optimizer.upper()}")
        logging.info(f"  Learning Rate: {args.lr}")
        logging.info(f"  Batch Size: {args.batch_size}")
        logging.info(f"  Label Smoothing: {args.label_smoothing}")
        logging.info(f"  CutMix: {args.use_cutmix} (prob={args.cutmix_prob})")
        logging.info(f"  Mixup Alpha: {args.mixup_alpha}, CutMix Alpha: {args.cutmix_alpha}")
        logging.info(f"  EMA: {'Enabled' if ema else 'Disabled'}")
        logging.info(f"  Mixed Precision: {args.mixed_precision} ({args.precision_type})")
        logging.info(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
        logging.info("=" * 80)

        for epoch in range(start_epoch, args.epochs):
            start_ts = time.perf_counter()
            
            # Train for one epoch
            scaler = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch,
                use_mixed_precision=args.mixed_precision,
                dtype=dtype,
                scaler=scaler,
                use_cutmix=args.use_cutmix,
                cutmix_prob=args.cutmix_prob,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                step_scheduler_on_batch=step_scheduler_on_batch,
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
            
            # Update EMA after training epoch
            if ema:
                ema.update()
            
            # Evaluate with EMA weights if available
            # CRITICAL: Always use FP32 for evaluation to prevent numerical instability
            if ema:
                ema.store()
                ema.copy_to()
                # Validate EMA weights before evaluation (check for NaN/Inf)
                has_nan = False
                for param in model.parameters():
                    if param is not None and (torch.isnan(param).any() or torch.isinf(param).any()):
                        has_nan = True
                        logging.warning("⚠️  EMA weights contain NaN/Inf - restoring original weights")
                        break
                if has_nan:
                    ema.restore()
                    logging.warning("⚠️  Skipping EMA evaluation, using original weights")
            
            # CRITICAL: Always disable mixed precision for evaluation (prevents loss explosion)
            # Even if mixed_precision is enabled for training, evaluation must use FP32
            acc = evaluate_epoch(
                model, val_loader, criterion, device,
                use_mixed_precision=False, dtype=torch.float32  # Force FP32 evaluation
            )
            
            # Restore original weights after evaluation
            if ema:
                ema.restore()
            
            # Step scheduler per epoch if not stepping per batch
            if not step_scheduler_on_batch and scheduler is not None:
                scheduler.step()
            
            if (epoch + 1) % args.save_freq == 0 or acc > best_acc:
                # Save EMA state if available
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": acc,
                }
                
                if scheduler is not None:
                    ckpt["scheduler_state_dict"] = scheduler.state_dict()
                
                if ema:
                    ckpt["ema_state_dict"] = ema.state_dict()
                
                name = f"checkpoint_epoch_{epoch+1}.pth" if acc <= best_acc else "best_model.pth"
                storage.save_checkpoint(ckpt, name)
                best_acc = max(best_acc, acc)
                logging.info(f"💾 Saved checkpoint ({name}) with acc={acc:.2f}%, time={(time.perf_counter() - start_ts)/60:.2f} mins")

            logging.info(f"Epoch {epoch+1}/{args.epochs} completed in {timedelta(seconds=int(time.perf_counter() - start_ts))}, Accuracy: {acc:.2f}%, Best: {best_acc:.2f}% \n")

        logging.info(f"🎉 Training completed. Best accuracy: {best_acc:.2f}%")

    elif args.fixres_enable:
        logging.info("=" * 80)
        logging.info(f"🔧 FixRes finetune ONLY @ {args.fixres_resolution}")

        fr_bs = args.fixres_batch_size or args.batch_size

        train_dir = os.path.join(args.data_dir, args.train_folder)
        val_dir   = os.path.join(args.data_dir, args.val_folder)

        fr_train = ImageFolder(train_dir, transform=fixres_train_transforms(args.fixres_resolution))
        fr_val   = ImageFolder(val_dir,   transform=fixres_val_transforms(args.fixres_resolution))

        fr_train_loader = DataLoader(fr_train, batch_size=fr_bs, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
        fr_val_loader   = DataLoader(fr_val, batch_size=fr_bs, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

        optimizer = optim.SGD(model.fc.parameters(), lr=args.fixres_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        model.train()
        model.to(device)

        start_epoch, best_acc = 0, 0
        if args.resume:
            ckpt = storage.load_checkpoint(args.resume)
            if ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except ValueError as e:
                    logging.warning(
                        "Optimizer state not loaded due to mismatch with current optimizer: %s. \n"
                    )
                start_epoch = ckpt["epoch"] + 1
                best_acc = ckpt.get("accuracy", 0)
                logging.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")


        for e in range(args.fixres_epochs):
            _ = train_epoch(
                model, fr_train_loader, criterion, optimizer, None, device, e,
                use_mixed_precision=False, dtype=torch.float32,
                mixup_alpha=0.0,
                use_cutmix=False,
                gradient_accumulation_steps=1
            )

        acc_fixres = evaluate_epoch(
            model, fr_val_loader, criterion, device,
            use_mixed_precision=False, dtype=torch.float32
        )
        logging.info(f"✅ FixRes done @ {args.fixres_resolution} — acc={acc_fixres:.2f}%")

        ckpt = {
            "epoch": args.fixres_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": acc_fixres,
        }
    # Save fixres checkpoint (filename only — storage handles prefix)
    storage.save_checkpoint(ckpt, "best_model_fixres.pth")

if __name__ == "__main__":
    main()
