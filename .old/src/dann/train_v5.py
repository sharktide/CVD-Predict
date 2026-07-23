#!/usr/bin/env python3
"""
V5 Training: Multi-Branch PPG+ACC+Biodata Cardiac Arrest Detection.

Pipeline:
1. Train core neural network (PPG+ACC+Biodata → P(ischemia))
2. Extract P(ischemia) on train set
3. Train Edge Decision Gate (Random Forest) on P(ischemia) + biodata
4. Export ONNX for deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, average_precision_score, f1_score, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys

import sys
# Find project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir
for _ in range(5):
    if (project_root / "data").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
from src.dann.model_v5 import CardiacArrestDetectorV5


# ── Dataset ──────────────────────────────────────────────────────────────────

class WristPPGDataset(Dataset):
    """Dataset for PPG + ACC + Biodata."""

    def __init__(self, data_dir, indices=None, augment=False):
        data_dir = Path(data_dir)
        
        self.ppg = np.load(data_dir / "ppg.npy", mmap_mode='r')
        self.accel = np.load(data_dir / "accel.npy", mmap_mode='r')
        self.labels = np.load(data_dir / "labels.npy")
        self.biodata = pd.read_csv(data_dir / "biodata.csv").values.astype(np.float32)
        
        self.indices = indices if indices is not None else np.arange(len(self.labels))
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        ppg = self.ppg[real_idx].copy().astype(np.float32)
        accel = self.accel[real_idx].copy().astype(np.float32)
        biodata = self.biodata[real_idx].copy()
        label = float(self.labels[real_idx])
        
        if self.augment:
            ppg, accel = self._augment(ppg, accel)
        
        # Ensure float32
        ppg = ppg.astype(np.float32)
        accel = accel.astype(np.float32)
        biodata = biodata.astype(np.float32)
        
        return {
            "ppg": torch.tensor(ppg).unsqueeze(0),       # (1, 1500)
            "accel": torch.tensor(accel).permute(1, 0),   # (3, 1500)
            "biodata": torch.tensor(biodata),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def _augment(self, ppg, accel):
        rng = np.random.RandomState()
        
        # PPG augmentation
        if rng.random() < 0.3:
            shift = rng.randint(-50, 50)
            ppg = np.roll(ppg, shift)
        if rng.random() < 0.3:
            ppg = ppg * rng.uniform(0.8, 1.2)
        if rng.random() < 0.3:
            noise = rng.normal(0, 0.05, len(ppg)).astype(np.float32)
            ppg = ppg + noise
        
        # ACC augmentation (more aggressive - real wrist ACC is messy)
        if rng.random() < 0.4:
            # Add random motion burst
            start = rng.randint(0, len(accel) - 200)
            burst = rng.normal(0, 0.5, (200, 3)).astype(np.float32)
            accel[start:start+200] += burst
        if rng.random() < 0.3:
            accel = accel * rng.uniform(0.7, 1.3)
        
        return ppg.astype(np.float32), accel.astype(np.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        ppg = batch["ppg"].to(device)
        accel = batch["accel"].to(device)
        biodata = batch["biodata"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(ppg, accel, biodata)
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
    all_logits = []
    
    with torch.no_grad():
        for batch in loader:
            ppg = batch["ppg"].to(device)
            accel = batch["accel"].to(device)
            biodata = batch["biodata"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(ppg, accel, biodata)
            loss = criterion(outputs["logit"], labels)
            
            total_loss += loss.item() * len(labels)
            all_preds.extend((outputs["probability"] > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs["probability"].cpu().numpy())
            all_logits.extend(outputs["logit"].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    
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
    
    return metrics, all_preds, all_labels, all_probs, all_logits


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = project_root / "data" / "processed" / "synthetic_v5"
    output_dir = project_root / "models" / "cardiac_arrest_v5"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V5 CARDIAC ARREST DETECTION (PPG + ACC + Biodata)")
    print("=" * 70)

    config = {
        "n_biodata": 16,
        "ppg_dim": 128,
        "acc_dim": 64,
        "latent_dim": 128,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "n_epochs": 50,
        "patience": 10,
        "n_folds": 5,
        "seed": 42,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    labels = np.load(data_dir / "labels.npy")
    print(f"Dataset: {len(labels)} segments")
    print(f"  CA: {(labels == 1).sum()}, Normal: {(labels == 0).sum()}")

    # K-Fold CV
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["seed"])
    fold_results = []

    print(f"\nTraining with {config['n_folds']}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{config['n_folds']}")
        print(f"{'='*70}")

        # Data loaders
        train_dataset = WristPPGDataset(data_dir, train_idx, augment=True)
        val_dataset = WristPPGDataset(data_dir, val_idx, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

        # Model
        model = CardiacArrestDetectorV5(
            n_biodata=config["n_biodata"],
            ppg_dim=config["ppg_dim"],
            acc_dim=config["acc_dim"],
        ).to(device)
        print(f"  Parameters: {model.n_params:,}")

        # Loss
        n_ca = (labels[train_idx] == 1).sum()
        n_normal = (labels[train_idx] == 0).sum()
        pos_weight = torch.tensor([min(n_normal / (n_ca + 1e-6), 10.0)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        # Train
        best_f1 = 0
        patience_counter = 0
        fold_start = time.time()

        for epoch in range(config["n_epochs"]):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                patience_counter = 0
                torch.save({
                    "fold": fold, "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics, "config": config,
                }, output_dir / f"best_fold_{fold}.pt")
                print(f"  Epoch {epoch+1:3d}: loss={train_metrics['loss']:.4f} "
                      f"val_f1={val_metrics['f1']:.3f} val_auroc={val_metrics['auroc']:.3f} ★")
            else:
                patience_counter += 1
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}: loss={train_metrics['loss']:.4f} "
                          f"val_f1={val_metrics['f1']:.3f}")
                if patience_counter >= config["patience"]:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final eval
        checkpoint = torch.load(output_dir / f"best_fold_{fold}.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics, _, val_labels, val_probs, val_logits = evaluate(model, val_loader, criterion, device)
        
        fold_time = time.time() - fold_start
        fold_results.append({"fold": fold, "metrics": final_metrics, "time_s": fold_time})
        
        print(f"\n  Fold {fold+1} final: acc={final_metrics['accuracy']:.3f} "
              f"f1={final_metrics['f1']:.3f} auroc={final_metrics['auroc']:.3f}")

    # ── CV Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CV SUMMARY")
    print("=" * 70)
    for key in ["accuracy", "f1", "auroc", "recall", "precision"]:
        values = [fr["metrics"][key] for fr in fold_results]
        print(f"  {key:12s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    # ── Train final model on all data ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    full_dataset = WristPPGDataset(data_dir, augment=True)
    full_loader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    final_model = CardiacArrestDetectorV5(
        n_biodata=config["n_biodata"],
        ppg_dim=config["ppg_dim"],
        acc_dim=config["acc_dim"],
    ).to(device)

    n_ca = (labels == 1).sum()
    n_normal = (labels == 0).sum()
    pos_weight = torch.tensor([min(n_normal / (n_ca + 1e-6), 10.0)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(final_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["n_epochs"]):
        final_model.train()
        total_loss = 0
        total = 0
        for batch in full_loader:
            ppg = batch["ppg"].to(device)
            accel = batch["accel"].to(device)
            biodata = batch["biodata"].to(device)
            labels_t = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = final_model(ppg, accel, biodata)
            loss = criterion(outputs["logit"], labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(labels_t)
            total += len(labels_t)
        
        avg_loss = total_loss / total
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": final_model.state_dict(), "config": config,
            }, output_dir / "best_model.pt")
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} ★")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Train Edge Decision Gate ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING EDGE DECISION GATE (Random Forest)")
    print("=" * 70)

    final_model.eval()
    all_probs = []
    all_labels_rf = []
    all_biodata = []

    full_eval = WristPPGDataset(data_dir, augment=False)
    full_eval_loader = DataLoader(full_eval, batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in full_eval_loader:
            ppg = batch["ppg"].to(device)
            accel = batch["accel"].to(device)
            biodata = batch["biodata"].to(device)
            labels_rf = batch["label"]
            
            outputs = final_model(ppg, accel, biodata)
            all_probs.extend(outputs["probability"].cpu().numpy())
            all_labels_rf.extend(labels_rf.numpy())
            all_biodata.extend(biodata.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels_rf = np.array(all_labels_rf)
    all_biodata = np.array(all_biodata)

    # Edge Decision Gate: Random Forest on [P(ischemia), biodata]
    X_gate = np.column_stack([all_probs, all_biodata])
    y_gate = all_labels_rf

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_gate, y_gate)
    
    rf_pred = rf.predict(X_gate)
    rf_prob = rf.predict_proba(X_gate)[:, 1]
    rf_auc = roc_auc_score(y_gate, rf_prob)
    rf_f1 = f1_score(y_gate, rf_pred)
    print(f"  RF on full data: AUROC={rf_auc:.4f}, F1={rf_f1:.4f}")

    # Save RF
    with open(output_dir / "edge_gate.pkl", "wb") as f:
        pickle.dump(rf, f)

    # Feature importance
    feature_names = ["P(ischemia)"] + [f"biodata_{i}" for i in range(all_biodata.shape[1])]
    importances = rf.feature_importances_
    print(f"\n  Edge Gate feature importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
        print(f"    {name:25s}: {imp:.4f}")

    # ── Export ONNX ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPORTING TO ONNX")
    print("=" * 70)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    final_model.eval()

    onnx_path = output_dir / "cardiac_arrest_detector_v5.onnx"
    final_model.export_onnx(onnx_path, n_biodata=config["n_biodata"])

    # Verify
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX verified! Size: {onnx_path.stat().st_size / 1024:.1f} KB")

    # Save config + results
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "cv_results.json", "w") as f:
        json.dump({
            "fold_results": fold_results,
            "avg_metrics": {k: float(np.mean([fr["metrics"][k] for fr in fold_results]))
                           for k in ["accuracy", "f1", "auroc", "recall", "precision"]},
            "edge_gate_auroc": float(rf_auc),
            "edge_gate_f1": float(rf_f1),
        }, f, indent=2)

    print(f"\nDone! Files saved to {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - cardiac_arrest_detector_v5.onnx")
    print(f"  - edge_gate.pkl")
    print(f"  - config.json")
    print(f"  - cv_results.json")


if __name__ == "__main__":
    main()
