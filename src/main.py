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
from train import train_transforms, test_transforms
from torch.amp import GradScaler

# EMA
try:
    from torch_ema import ExponentialMovingAverage
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    logging.warning("torch_ema not available. Install with: pip install torch-ema")


# ============================================================
# Storage Handler (S3 + local)
# ============================================================
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


# ============================================================
# train / eval import
# ============================================================
from train import train as train_epoch
from test import evaluate as evaluate_epoch


# ============================================================
# Build Dataloaders
# ============================================================
def build_dataloaders(storage, train_path, val_path, batch_size, workers, use_randaugment=True):
    if storage.storage_type == "s3":
        train_dir = storage.sync_dataset(train_path, os.path.join(storage.cache_dir, "train"))
        val_dir = storage.sync_dataset(val_path, os.path.join(storage.cache_dir, "val"))
    else:
        train_dir, val_dir = train_path, val_path

    train_set = ImageFolder(train_dir, transform=train_transforms(augment=True, use_randaugment=use_randaugment))
    val_set = ImageFolder(val_dir, transform=test_transforms())

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                   pin_memory=True, persistent_workers=True if workers > 0 else False),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                   pin_memory=True, persistent_workers=True if workers > 0 else False),
        len(train_set.classes),
    )


# ============================================================
# MAIN
# ============================================================
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
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--precision_type", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--use_cutmix", action="store_true", default=True)
    p.add_argument("--cutmix_prob", type=float, default=0.5)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--use_randaugment", action="store_true", default=True)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.checkpoint_prefix, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storage = StorageHandler(
        storage_type="s3" if args.use_s3 else "local",
        bucket=args.bucket if args.use_s3 else args.checkpoint_bucket,
        prefix=args.checkpoint_prefix,
        cache_dir="/tmp/imagenet100"
    )

    train_loader, val_loader, num_classes = build_dataloaders(
        storage,
        os.path.join(args.data_dir, args.train_folder),
        os.path.join(args.data_dir, args.val_folder),
        args.batch_size,
        args.num_workers,
        use_randaugment=args.use_randaugment
    )

    model = ResNet50(num_classes=num_classes).to(device)

    # ============================================================
    # Optimizer + scheduler
    # ============================================================
    if args.optimizer == "sgd":
        base_lr = args.lr * (args.batch_size / 256.0)
        optimizer = optim.SGD(
            model.parameters(), lr=base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )

        warmup_steps = args.warmup_epochs * len(train_loader)
        total_steps = args.epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        step_scheduler_on_batch = True

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=args.warmup_epochs / args.epochs
        )
        step_scheduler_on_batch = True

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    dtype = torch.float16 if args.precision_type == "fp16" else torch.bfloat16
    scaler = GradScaler("cuda") if args.mixed_precision else None

    ema = None
    if args.use_ema and EMA_AVAILABLE:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    # ============================================================
    # Resume checkpoint
    # ============================================================
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

            logging.info(f"Resumed training from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ============================================================
    # FC AUTO-FIX (Option C)
    # ============================================================
    def fc_stats(m):
        w = m.fc.weight.data.float().cpu()
        return float(w.mean()), float(w.std())

    fc_mean, fc_std = fc_stats(model)
    logging.info(f"[FC CHECK] mean={fc_mean:.6f}, std={fc_std:.6f}")

    if fc_std < 0.015:
        repair_epochs = 3
        logging.warning("⚠️ FC severely untrained → performing 3-epoch FC repair")
    elif fc_std < 0.03:
        repair_epochs = 1
        logging.warning("⚠️ FC partially untrained → performing 1-epoch FC repair")
    else:
        repair_epochs = 0
        logging.info("✅ FC appears trained — no repair needed")

    # ------------------------------------------------------------
    # Run FC repair
    # ------------------------------------------------------------
    if repair_epochs > 0:
        # Freeze all but FC
        for name, p in model.named_parameters():
            p.requires_grad = ("fc" in name)

        fc_optimizer = optim.SGD(
            model.fc.parameters(),
            lr=0.02,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        fc_loss_fn = nn.CrossEntropyLoss()

        for ep in range(repair_epochs):
            model.train()
            losses = []
            for i, (images, labels) in enumerate(train_loader):
                if i >= 300:
                    break
                images, labels = images.to(device), labels.to(device)
                fc_optimizer.zero_grad()
                out = model(images)
                loss = fc_loss_fn(out, labels)
                loss.backward()
                fc_optimizer.step()
                losses.append(loss.item())

            logging.info(f"FC repair {ep+1}/{repair_epochs} → loss={np.mean(losses):.4f}")

        # Unfreeze all parameters
        for p in model.parameters():
            p.requires_grad = True

        # Rebuild optimizer/scheduler
        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=base_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True
            )
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        fc_mean, fc_std = fc_stats(model)
        logging.info(f"🔧 FC repair complete — new std={fc_std:.6f}")

    # ============================================================
    # Training Loop
    # ============================================================
    logging.info("=" * 80)
    logging.info("Training Configuration Loaded")
    logging.info(f"Optimizer={args.optimizer}, LR={args.lr}, Batch={args.batch_size}")
    logging.info("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.perf_counter()

        # Train one epoch
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

        if ema:
            ema.update()

        # FP32 evaluation
        if ema:
            ema.store()
            ema.copy_to()

            # Safety check
            if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                logging.warning("⚠️ EMA contains NaNs → restoring original weights")
                ema.restore()

        acc = evaluate_epoch(
            model, val_loader, criterion, device,
            use_mixed_precision=False,
            dtype=torch.float32
        )

        if ema:
            ema.restore()

        if not step_scheduler_on_batch:
            scheduler.step()

        # Save checkpoint
        if acc > best_acc or (epoch + 1) % args.save_freq == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": acc
            }
            if scheduler:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if ema:
                ckpt["ema_state_dict"] = ema.state_dict()

            fname = "best_model.pth" if acc > best_acc else f"checkpoint_epoch_{epoch+1}.pth"
            storage.save_checkpoint(ckpt, fname)

            best_acc = max(best_acc, acc)
            logging.info(f"💾 Saved checkpoint: {fname} (acc={acc:.2f}%)")

        elapsed = time.perf_counter() - start_time
        logging.info(
            f"Epoch {epoch+1}/{args.epochs} — Acc={acc:.2f}% — Best={best_acc:.2f}% — Time={timedelta(seconds=int(elapsed))}"
        )

    logging.info(f"🎉 Training Complete — Best Accuracy: {best_acc:.2f}%")


# ============================================================
if __name__ == "__main__":
    main()
