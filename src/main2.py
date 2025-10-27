# main_refactored.py
import os
import time
import json
import torch
import logging
import argparse
import tempfile
import hashlib
from tqdm import tqdm
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import ResNet50
from train import train_transforms, test_transforms, mixup_data, mixup_criterion
from torch.amp import GradScaler, autocast

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
            return torch.load(path, map_location="cpu") if os.path.exists(path) else None

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                self.client.download_file(self.bucket, f"{self.prefix}/{filename}", tmp.name)
                ckpt = torch.load(tmp.name, map_location="cpu")
                os.unlink(tmp.name)
                return ckpt
            except ClientError:
                return None


# -------------------------------------------------------------------
# Training and evaluation (mixed precision capable)
# -------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, mp, dtype, epoch):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        x, ya, yb, lam = mixup_data(x, y, device=device)

        optimizer.zero_grad()
        ctx = autocast("cuda", dtype=dtype, enabled=mp)

        with ctx:
            out = model(x)
            loss = mixup_criterion(criterion, out, ya, yb, lam)

        if mp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

        pbar.set_postfix(loss=running_loss/(total/loader.batch_size), acc=f"{100*correct/total:.2f}%")

    return running_loss/len(loader), 100*correct/total


def evaluate_epoch(model, loader, criterion, device, mp, dtype):
    model.eval()
    total, correct, loss_total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ctx = autocast("cuda", dtype=dtype, enabled=mp)
            with ctx:
                out = model(x)
                loss = criterion(out, y)
            loss_total += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100 * correct / total


# -------------------------------------------------------------------
# Data loader creation
# -------------------------------------------------------------------
def build_dataloaders(storage: StorageHandler, train_path, val_path, batch_size, workers):
    if storage.storage_type == "s3":
        train_dir = storage.sync_dataset(train_path, os.path.join(storage.cache_dir, "train"))
        val_dir = storage.sync_dataset(val_path, os.path.join(storage.cache_dir, "val"))
    else:
        train_dir, val_dir = train_path, val_path

    train_set = ImageFolder(train_dir, transform=train_transforms())
    val_set = ImageFolder(val_dir, transform=test_transforms())

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--precision_type", choices=["fp16", "bf16"], default="fp16")
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
    )

    model = ResNet50(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    dtype = torch.float16 if args.precision_type == "fp16" else torch.bfloat16
    scaler = GradScaler("cuda") if args.mixed_precision else None

    start_epoch, best_acc = 0, 0
    if args.resume:
        ckpt = storage.load_checkpoint(args.resume)
        if ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt["accuracy"]

    for epoch in range(start_epoch, args.epochs):
        start_ts = time.perf_counter()
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, args.mixed_precision, dtype, epoch)
        acc = evaluate_epoch(model, val_loader, criterion, device, args.mixed_precision, dtype)

        if (epoch + 1) % args.save_freq == 0 or acc > best_acc:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "accuracy": acc,
            }
            name = f"checkpoint_epoch_{epoch+1}.pth" if acc <= best_acc else "best_model.pth"
            storage.save_checkpoint(ckpt, name)
            best_acc = max(best_acc, acc)
            logging.info(f"Saved checkpoint ({name}) with acc={acc:.2f}, time={(time.perf_counter() - start_ts)/60} mins")

    logging.info(f"Training completed. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
