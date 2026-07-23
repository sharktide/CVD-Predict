#!/usr/bin/env python3
"""
Train cardiac arrest detection model with k-fold cross-validation.
Uses synthetic PPG data from wristppg simulator.

Output:
- Best model checkpoint (PyTorch)
- ONNX export for deployment
- Per-fold evaluation metrics
- Confusion matrices and ROC curves
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.dann.model_v2 import CardiacArrestDetector, build_model


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticPPGDataset(Dataset):
    """Dataset for synthetic PPG segments with features."""

    def __init__(self, data_dir, split=None, augment=False):
        """
        Args:
            data_dir: Path to synthetic_v2 directory
            split: Optional list of segment indices to use
            augment: Whether to apply data augmentation
        """
        data_dir = Path(data_dir)
        
        # Load features
        self.features_df = pd.read_csv(data_dir / "features.csv")
        
        # Load PPG segments (memory-mapped for large arrays)
        self.ppg_data = np.load(data_dir / "ppg_segments.npy", mmap_mode='r')
        
        # Feature columns (exclude metadata)
        exclude = {"segment_id", "profile", "label"}
        self.feat_cols = [c for c in self.features_df.columns if c not in exclude]
        self.n_features = len(self.feat_cols)
        
        # Labels
        self.labels = self.features_df["label"].values
        
        # Filter to split indices if provided
        if split is not None:
            self.indices = split
        else:
            self.indices = np.arange(len(self.features_df))
        
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Load PPG
        ppg = self.ppg_data[real_idx].copy().astype(np.float32)
        
        # Augmentation
        if self.augment:
            ppg = self._augment_ppg(ppg)
        
        # Normalize PPG
        ppg = ppg - np.mean(ppg)
        std = np.std(ppg)
        if std > 1e-8:
            ppg = ppg / std
        ppg = ppg.astype(np.float32)
        
        # Features
        features = self.features_df.iloc[real_idx][self.feat_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -1e6, 1e6).astype(np.float32)
        
        # Label (binary: 1 = cardiac arrest, 0 = normal)
        label = 1 if self.labels[real_idx] == 2 else 0
        
        return {
            "ppg": torch.tensor(ppg).unsqueeze(0),  # (1, 1500)
            "features": torch.tensor(features),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def _augment_ppg(self, ppg):
        """Apply random augmentation to PPG signal."""
        rng = np.random.RandomState()
        
        # Time shift
        if rng.random() < 0.3:
            shift = rng.randint(-50, 50)
            ppg = np.roll(ppg, shift)
        
        # Amplitude scaling
        if rng.random() < 0.3:
            scale = rng.uniform(0.8, 1.2)
            ppg = ppg * scale
        
        # Gaussian noise
        if rng.random() < 0.3:
            noise = rng.normal(0, 0.05, len(ppg)).astype(np.float32)
            ppg = ppg + noise
        
        # Random dropout (simulates signal loss)
        if rng.random() < 0.1:
            start = rng.randint(0, len(ppg) - 100)
            ppg[start:start + rng.randint(10, 100)] = 0
        
        return ppg.astype(np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        ppg = batch["ppg"].to(device)
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(ppg, features)
        loss = criterion(outputs["logit"], labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = (outputs["probability"] > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            ppg = batch["ppg"].to(device)
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(ppg, features)
            loss = criterion(outputs["logit"], labels)
            
            total_loss += loss.item() * len(labels)
            all_preds.extend((outputs["probability"] > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs["probability"].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": float(np.mean(all_preds == all_labels)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
    }
    
    try:
        metrics["auroc"] = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        metrics["auroc"] = 0.0
    
    try:
        metrics["auprc"] = float(average_precision_score(all_labels, all_probs))
    except Exception:
        metrics["auprc"] = 0.0
    
    return metrics, all_preds, all_labels, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "processed" / "synthetic_v2"
    output_dir = project_root / "models" / "cardiac_arrest_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARDIAC ARREST DETECTION MODEL TRAINING")
    print("=" * 70)

    # Config
    config = {
        "n_features": 40,  # Will be detected from data
        "latent_dim": 128,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "n_epochs": 50,
        "patience": 10,
        "n_folds": 5,
        "seed": 42,
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load full dataset to detect feature count
    full_dataset = SyntheticPPGDataset(data_dir)
    config["n_features"] = full_dataset.n_features
    print(f"Dataset: {len(full_dataset)} segments, {config['n_features']} features")
    print(f"Label distribution: CA={np.sum(full_dataset.labels == 2)}, "
          f"GenDet={np.sum(full_dataset.labels == 1)}, "
          f"Healthy={np.sum(full_dataset.labels == 0)}")

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["seed"])
    binary_labels = (full_dataset.labels == 2).astype(int)  # Binary: CA vs non-CA
    
    fold_results = []
    best_fold_metrics = {"f1": 0}
    
    print(f"\nTraining with {config['n_folds']}-fold cross-validation...")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(binary_labels)), binary_labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{config['n_folds']}")
        print(f"{'='*70}")
        print(f"  Train: {len(train_idx)} segments")
        print(f"  Val:   {len(val_idx)} segments")

        # Create data loaders
        train_dataset = SyntheticPPGDataset(data_dir, split=train_idx, augment=True)
        val_dataset = SyntheticPPGDataset(data_dir, split=val_idx, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                                  shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                                shuffle=False, num_workers=0, pin_memory=False)

        # Build model
        model = build_model(config).to(device)
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss (weighted for class imbalance)
        n_ca = np.sum(binary_labels[train_idx] == 1)
        n_normal = np.sum(binary_labels[train_idx] == 0)
        pos_weight = torch.tensor([n_normal / (n_ca + 1e-6)], dtype=torch.float32).to(device)
        pos_weight = torch.clamp(pos_weight, max=10.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        fold_start = time.time()

        for epoch in range(config["n_epochs"]):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics, val_preds, val_labels, val_probs = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                
                # Save best fold model
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                }, output_dir / f"best_fold_{fold}.pt")
                
                print(f"  Epoch {epoch+1:3d}: loss={train_metrics['loss']:.4f} "
                      f"train_acc={train_metrics['accuracy']:.3f} | "
                      f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                      f"val_f1={val_metrics['f1']:.3f} val_auroc={val_metrics['auroc']:.3f} "
                      f"★")
            else:
                patience_counter += 1
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}: loss={train_metrics['loss']:.4f} "
                          f"train_acc={train_metrics['accuracy']:.3f} | "
                          f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                          f"val_f1={val_metrics['f1']:.3f}")
                
                if patience_counter >= config["patience"]:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        fold_time = time.time() - fold_start
        
        # Load best model for this fold and evaluate
        checkpoint = torch.load(output_dir / f"best_fold_{fold}.pt", 
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics, final_preds, final_labels, final_probs = evaluate(
            model, val_loader, criterion, device
        )
        
        fold_results.append({
            "fold": fold,
            "metrics": final_metrics,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "time_s": fold_time,
        })
        
        print(f"\n  Fold {fold+1} Results:")
        print(f"    Accuracy:  {final_metrics['accuracy']:.3f}")
        print(f"    Precision: {final_metrics['precision']:.3f}")
        print(f"    Recall:    {final_metrics['recall']:.3f}")
        print(f"    F1:        {final_metrics['f1']:.3f}")
        print(f"    AUROC:     {final_metrics['auroc']:.3f}")
        print(f"    AUPRC:     {final_metrics['auprc']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(final_labels, final_preds)
        print(f"    Confusion Matrix:")
        print(f"      TN={cm[0,0]:4d} FP={cm[0,1]:4d}")
        print(f"      FN={cm[1,0]:4d} TP={cm[1,1]:4d}")
        
        # Track best fold
        if final_metrics["f1"] > best_fold_metrics["f1"]:
            best_fold_metrics = final_metrics
            best_fold_idx = fold

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    
    avg_metrics = {}
    for key in fold_results[0]["metrics"]:
        if isinstance(fold_results[0]["metrics"][key], (int, float)):
            values = [fr["metrics"][key] for fr in fold_results]
            avg_metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    
    for key, vals in avg_metrics.items():
        print(f"  {key:12s}: {vals['mean']:.3f} ± {vals['std']:.3f} "
              f"(range: {vals['min']:.3f} - {vals['max']:.3f})")
    
    print(f"\n  Best fold: {best_fold_idx + 1} (F1={best_fold_metrics['f1']:.3f})")

    # ── Train final model on all data ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)
    
    # Create full dataset with augmentation
    full_train = SyntheticPPGDataset(data_dir, augment=True)
    full_loader = DataLoader(full_train, batch_size=config["batch_size"], 
                             shuffle=True, num_workers=0)
    
    final_model = build_model(config).to(device)
    
    # Weighted loss for full data
    n_ca = np.sum(full_dataset.labels == 2)
    n_normal = np.sum(full_dataset.labels != 2)
    pos_weight = torch.tensor([n_normal / (n_ca + 1e-6)], dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config["n_epochs"]):
        final_model.train()
        total_loss = 0
        total = 0
        
        for batch in full_loader:
            ppg = batch["ppg"].to(device)
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = final_model(ppg, features)
            loss = criterion(outputs["logit"], labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            total += len(labels)
        
        avg_loss = total_loss / total
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": final_model.state_dict(),
                "config": config,
                "cv_metrics": avg_metrics,
            }, output_dir / "best_model.pt")
            
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} ★ saved")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}")
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Export to ONNX ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPORTING TO ONNX")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    final_model.eval()
    
    onnx_path = output_dir / "cardiac_arrest_detector.onnx"
    final_model.export_onnx(onnx_path, n_features=config["n_features"])
    
    # Save config and results
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump({
            "fold_results": fold_results,
            "avg_metrics": avg_metrics,
            "best_fold": best_fold_idx,
            "best_fold_metrics": best_fold_metrics,
            "final_model_epoch": checkpoint["epoch"],
        }, f, indent=2, default=str)
    
    print(f"\nAll results saved to {output_dir}")
    print(f"  - best_model.pt (PyTorch checkpoint)")
    print(f"  - cardiac_arrest_detector.onnx (deployment)")
    print(f"  - config.json")
    print(f"  - cv_results.json")


if __name__ == "__main__":
    main()
