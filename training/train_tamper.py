"""
DocFraudDetector — Tamper Detection Model Training Script
Trains an EfficientNet-B0 binary classifier for document tamper detection.

Features:
- Uses timm (PyTorch Image Models) for EfficientNet-B0
- Augmentation via Albumentations
- Learning rate scheduling, early stopping, checkpointing
- Training metrics: accuracy, precision, recall, F1, AUC-ROC
- Mixed precision training support

Author: Pranav Kashyap | IIIT Dharwad
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import cv2


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class TamperDataset(Dataset):
    """Dataset for genuine vs tampered document images."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load genuine images (label = 0)
        genuine_dir = os.path.join(data_dir, "genuine")
        if os.path.exists(genuine_dir):
            for fname in sorted(os.listdir(genuine_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(genuine_dir, fname))
                    self.labels.append(0)
        
        # Load tampered images (label = 1)
        tampered_dir = os.path.join(data_dir, "tampered")
        if os.path.exists(tampered_dir):
            for fname in sorted(os.listdir(tampered_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(tampered_dir, fname))
                    self.labels.append(1)
        
        print(f"[Dataset] Loaded {len(self.samples)} images "
              f"({self.labels.count(0)} genuine, {self.labels.count(1)} tampered)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, config.TAMPER_INPUT_SIZE)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(img)
        
        return img, label


# ═══════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════

def get_train_transforms():
    """Training augmentations."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

def create_model(pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 model for binary classification."""
    model = timm.create_model(
        config.TAMPER_MODEL_NAME,
        pretrained=pretrained,
        num_classes=config.TAMPER_NUM_CLASSES,
    )
    return model


# ═══════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    
    # AUC-ROC (only if both classes present)
    if len(set(all_labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc_roc"] = 0.0
    
    return metrics


def train(args):
    """Main training function."""
    print("=" * 60)
    print("  TAMPER DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"  Device: {config.DEVICE}")
    print(f"  Model: {config.TAMPER_MODEL_NAME}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)
    
    device = torch.device(config.DEVICE)
    
    # ── Dataset ──
    data_dir = args.data_dir or config.SYNTHETIC_DIR
    
    full_dataset = TamperDataset(data_dir, transform=None)
    
    if len(full_dataset) == 0:
        print("\n⚠️  No training data found!")
        print("Run `python data/synthetic_generator.py` first to generate training data.")
        return
    
    # Split into train/val
    val_size = int(len(full_dataset) * config.TRAIN_VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=config.TRAIN_NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=config.TRAIN_NUM_WORKERS,
    )
    
    print(f"\nTrain: {train_size} | Val: {val_size}")
    
    # ── Model ──
    model = create_model(pretrained=True)
    model = model.to(device)
    
    # ── Training setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.TRAIN_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.TRAIN_SCHEDULER_PATIENCE,
        factor=0.5, verbose=True,
    )
    
    # ── Training loop ──
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])
        
        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # History
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc_roc"],
            "lr": current_lr,
        })
        
        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            
            torch.save(model.state_dict(), config.TAMPER_CHECKPOINT)
            print(f"  ✅ Best model saved! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\n⏹️  Early stopping triggered (no improvement for {args.early_stopping} epochs)")
            break
    
    # ── Final report ──
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Model saved: {config.TAMPER_CHECKPOINT}")
    
    # Save training history
    history_path = os.path.join(config.MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")
    
    # Final validation with best model
    model.load_state_dict(torch.load(config.TAMPER_CHECKPOINT, map_location=device, weights_only=True))
    final_metrics = validate(model, val_loader, criterion, device)
    
    print(f"\n  Final Validation Metrics:")
    for k, v in final_metrics.items():
        print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tamper Detection Model")
    parser.add_argument("--data-dir", type=str, default=None, help="Training data directory")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.TRAIN_LR)
    parser.add_argument("--early-stopping", type=int, default=7)
    
    args = parser.parse_args()
    train(args)
