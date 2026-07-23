"""
Phase 4: Dataset and Training Pipeline for DANN Cardiac Arrest Prediction

Handles:
- Data loading and preprocessing
- Class balancing (oversample cardiac arrest, undersample healthy)
- Window stratification for balanced batches
- Training loop with Focal Loss and GRL
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ── Dataset ────────────────────────────────────────────────────────────────────

class CardiacArrestDataset(Dataset):
    """
    Dataset for cardiac arrest prediction.

    Loads PPG segments and pre-extracted features from cohort_v1.
    """

    # Event type mapping
    EVENT_MAP = {
        None: 0,           # Healthy
        0.0: 0,            # Healthy (numeric)
        "sepsis": 1,       # General deterioration
        "aki": 1,          # General deterioration
        "heart_failure": 1, # General deterioration
        "respiratory_failure": 1, # General deterioration
        "cardiac_arrest": 2, # Cardiac arrest (target)
    }

    # Domain mapping
    DOMAIN_CLINICAL = 0
    DOMAIN_WEARABLE = 1

    def __init__(self, cohort_dir, features_csv=None, mode="train",
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 target_fs=25, segment_length_s=60, seed=42):
        """
        Args:
            cohort_dir: Path to cohort_v1 directory
            features_csv: Path to features CSV (optional, for feature-based training)
            mode: "train", "val", or "test"
            train_ratio, val_ratio, test_ratio: Dataset splits
            target_fs: Target sampling frequency (Hz)
            segment_length_s: Segment length in seconds
            seed: Random seed for reproducibility
        """
        self.cohort_dir = Path(cohort_dir)
        self.target_fs = target_fs
        self.segment_length_s = segment_length_s
        self.segment_length = int(target_fs * segment_length_s)  # 1500 samples

        # Load windows metadata - prefer combined windows if available
        combined_windows = self.cohort_dir / "augmented" / "combined_windows.csv"
        if combined_windows.exists():
            self.windows_df = pd.read_csv(combined_windows)
            self.aug_ppg_dir = self.cohort_dir / "augmented" / "ppg_segments"
        else:
            self.windows_df = pd.read_csv(self.cohort_dir / "windows.csv")
            self.aug_ppg_dir = None

        # Load features if available - features is the PRIMARY dataset
        self.features_df = None
        if features_csv and Path(features_csv).exists():
            self.features_df = pd.read_csv(features_csv)
            # Merge with windows metadata where available (left join from features)
            self.windows_df = self.features_df.merge(
                self.windows_df, on="window_id", how="left", suffixes=("", "_win")
            )
        else:
            # No features - use windows only
            pass

        # Stratified split by subject_id AND primary_event
        np.random.seed(seed)

        # Separate synthetic (subject_id <= 0) from real data
        real_mask = self.windows_df["subject_id"] > 0
        synthetic_df = self.windows_df[~real_mask]
        real_df = self.windows_df[real_mask]

        # Group real subjects
        subject_events = real_df.groupby("subject_id")["primary_event"].first().reset_index()

        # Map rare events for stratification
        stratify_labels = subject_events["primary_event"].fillna("none").replace("sepsis", "aki").values

        from sklearn.model_selection import train_test_split
        train_subjects, temp_subjects = train_test_split(
            subject_events["subject_id"].values,
            test_size=(val_ratio + test_ratio),
            stratify=stratify_labels,
            random_state=seed,
        )

        temp_events = subject_events[subject_events["subject_id"].isin(temp_subjects)]
        stratify_labels_temp = temp_events["primary_event"].fillna("none").replace("sepsis", "aki").values
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=stratify_labels_temp,
            random_state=seed,
        )

        # Combine: synthetic goes to train, real goes to train/val/test
        if mode == "train":
            train_real = real_df[real_df["subject_id"].isin(train_subjects)]
            self.windows_df = pd.concat([train_real, synthetic_df], ignore_index=True)
        elif mode == "val":
            self.windows_df = real_df[real_df["subject_id"].isin(val_subjects)].copy()
        elif mode == "test":
            self.windows_df = real_df[real_df["subject_id"].isin(test_subjects)].copy()

        # Compute class weights for balancing
        self._compute_class_weights()

        print(f"  {mode}: {len(self.windows_df)} windows from {self.windows_df['subject_id'].nunique()} subjects")
        print(f"    Event distribution: {self.windows_df['primary_event'].value_counts().to_dict()}")

    def _compute_class_weights(self):
        """Compute class weights for balanced sampling (inverse frequency, capped)."""
        event_counts = self.windows_df["primary_event"].value_counts()
        total = len(self.windows_df)

        self.class_weights = {}
        for event_type, count in event_counts.items():
            # Inverse frequency, capped at 5x
            weight = min(5.0, total / (len(event_counts) * count))
            self.class_weights[event_type] = weight

        # Mild boost for cardiac arrest (not 2x)
        if "cardiac_arrest" in self.class_weights:
            self.class_weights["cardiac_arrest"] = min(5.0, self.class_weights["cardiac_arrest"] * 1.5)

    def __len__(self):
        return len(self.windows_df)

    def __getitem__(self, idx):
        """
        Returns:
            ppg: PPG segment (1, segment_length) - randomly sampled from window
            features: Feature vector (n_features,) if available, else zeros
            event_label: Event type label (0=Healthy, 1=General, 2=Arrest)
            domain_label: Domain label (0=Clinical, 1=Wearable)
            time_to_event: Time-to-event in hours (0-24), inf for healthy
        """
        row = self.windows_df.iloc[idx]
        window_id = row["window_id"]
        n_segments = int(row["n_segments"])

        # Randomly select a segment from this window
        seg_idx = np.random.randint(0, n_segments)
        seg_path = self.cohort_dir / "ppg_segments" / f"{window_id}_s{seg_idx:02d}.npy"
        
        # Check augmented directory for augmented windows
        if not seg_path.exists() and self.aug_ppg_dir is not None:
            seg_path = self.aug_ppg_dir / f"{window_id}.npy"

        # Load PPG segment
        if seg_path.exists():
            ppg = np.load(seg_path).astype(np.float32)
            # Handle NaN/inf
            ppg = np.nan_to_num(ppg, nan=0.0, posinf=0.0, neginf=0.0)
            # Pad or truncate to fixed length
            if len(ppg) < self.segment_length:
                ppg = np.pad(ppg, (0, self.segment_length - len(ppg)), mode='constant')
            elif len(ppg) > self.segment_length:
                ppg = ppg[:self.segment_length]
        else:
            # Synthetic/augmented row without PPG - use zero signal
            ppg = np.zeros(self.segment_length, dtype=np.float32)

        # Normalize PPG
        ppg_std = ppg.std()
        if ppg_std > 1e-8:
            ppg = (ppg - ppg.mean()) / ppg_std
        else:
            ppg = np.zeros(self.segment_length, dtype=np.float32)

        # Features - numeric only, exclude ALL metadata
        numeric_cols = self.windows_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["subject_id", "n_segments", "subject_id_win", "study_id",
                       "time_to_event_hours", "time_to_event_hours_win",
                       "n_segments_win", "total_hours_available"]
        feat_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if feat_cols:
            features = self.windows_df.iloc[idx][feat_cols].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = np.clip(features, -1e6, 1e6)
        else:
            features = np.zeros(60, dtype=np.float32)

        # Labels
        event_label = self.EVENT_MAP.get(row["primary_event"], 0)
        domain_label = self.DOMAIN_CLINICAL  # All MIMIC data is clinical domain

        # Time-to-event
        tte = row["time_to_event_hours"]
        if tte == float('inf') or pd.isna(tte) or np.isinf(tte):
            time_to_event = 24.0  # Cap healthy at 24h
        else:
            time_to_event = min(float(tte), 24.0)

        return {
            "ppg": torch.tensor(ppg).unsqueeze(0),  # (1, segment_length)
            "features": torch.tensor(features),
            "event_label": torch.tensor(event_label, dtype=torch.long),
            "domain_label": torch.tensor(domain_label, dtype=torch.long),
            "time_to_event": torch.tensor(time_to_event / 24.0, dtype=torch.float32),  # Normalize to 0-1
            "is_healthy": torch.tensor(row["is_healthy"], dtype=torch.bool),
        }


# ── Balanced Batch Sampler ────────────────────────────────────────────────────

class BalancedBatchSampler:
    """
    Sampler that ensures each batch has balanced event types.
    Specifically ensures cardiac arrest samples are well-represented.
    """

    def __init__(self, dataset, batch_size=32, n_arrest_per_batch=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_arrest_per_batch = n_arrest_per_batch

        # Group indices by event type
        self.event_indices = {}
        for idx in range(len(dataset)):
            row = dataset.windows_df.iloc[idx]
            event = row["primary_event"]
            if event not in self.event_indices:
                self.event_indices[event] = []
            self.event_indices[event].append(idx)

        # Compute oversampling weights
        self.weights = {}
        for event, indices in self.event_indices.items():
            self.weights[event] = len(indices)

        print(f"  Balanced sampler: {len(self.event_indices)} event types")
        for event, indices in self.event_indices.items():
            print(f"    {event}: {len(indices)} samples")

    def __iter__(self):
        """Generate balanced batches."""
        # Determine how many samples per event type
        n_other_per_batch = (self.batch_size - self.n_arrest_per_batch) // max(1, len(self.event_indices) - 1)

        while True:
            # Shuffle indices within each event type
            for event in self.event_indices:
                np.random.shuffle(self.event_indices[event])

            # Generate batches
            batches = []
            arrest_indices = self.event_indices.get("cardiac_arrest", [])
            other_indices = []
            for event, indices in self.event_indices.items():
                if event != "cardiac_arrest":
                    other_indices.extend(indices)

            # Oversample cardiac arrest
            if arrest_indices:
                arrest_oversampled = np.random.choice(
                    arrest_indices,
                    size=min(len(arrest_indices) * 3, len(other_indices)),
                    replace=True
                )
            else:
                arrest_oversampled = np.array([])

            # Combine and shuffle
            all_indices = np.concatenate([arrest_oversampled, other_indices])
            np.random.shuffle(all_indices)

            # Create batches
            for i in range(0, len(all_indices), self.batch_size):
                batch_indices = all_indices[i:i + self.batch_size]
                if len(batch_indices) > 0:
                    yield batch_indices.tolist()

    def __len__(self):
        return len(self.dataset) // self.batch_size


# ── Data Loaders ──────────────────────────────────────────────────────────────

def create_data_loaders(cohort_dir, features_csv=None, batch_size=32, num_workers=4):
    """Create train/val/test data loaders."""
    print("Creating data loaders...")

    train_dataset = CardiacArrestDataset(cohort_dir, features_csv, mode="train")
    val_dataset = CardiacArrestDataset(cohort_dir, features_csv, mode="val")
    test_dataset = CardiacArrestDataset(cohort_dir, features_csv, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_weights


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    event_correct = 0
    domain_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        ppg = batch["ppg"].to(device)
        features = batch["features"].to(device)
        event_labels = batch["event_label"].to(device)
        domain_labels = batch["domain_label"].to(device)
        time_to_event = batch["time_to_event"].to(device)
        is_healthy = batch["is_healthy"].to(device)

        # GRL alpha schedule (linear warmup)
        alpha = min(1.0, (epoch * len(train_loader) + batch_idx) / (total_epochs * len(train_loader) * 0.5))

        # Forward pass
        outputs = model(ppg, features, alpha=alpha)

        # Compute loss
        loss_dict = criterion(
            outputs, event_labels, domain_labels,
            time_to_event.unsqueeze(1), is_healthy
        )

        # Backward pass
        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss_dict["total_loss"].item()
        event_pred = outputs["event_logits"].argmax(dim=-1)
        event_correct += (event_pred == event_labels).sum().item()
        domain_pred = outputs["domain_logits"].argmax(dim=-1)
        domain_correct += (domain_pred == domain_labels).sum().item()
        total_samples += event_labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"loss={loss_dict['total_loss'].item():.4f} "
                  f"event_acc={event_correct / total_samples:.3f} "
                  f"domain_acc={domain_correct / total_samples:.3f}")

    return {
        "loss": total_loss / len(train_loader),
        "event_accuracy": event_correct / total_samples,
        "domain_accuracy": domain_correct / total_samples,
    }


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    event_correct = 0
    total_samples = 0
    all_event_preds = []
    all_event_labels = []

    with torch.no_grad():
        for batch in val_loader:
            ppg = batch["ppg"].to(device)
            features = batch["features"].to(device)
            event_labels = batch["event_label"].to(device)
            domain_labels = batch["domain_label"].to(device)
            time_to_event = batch["time_to_event"].to(device)
            is_healthy = batch["is_healthy"].to(device)

            outputs = model(ppg, features, alpha=0.0)  # No GRL during validation

            loss_dict = criterion(
                outputs, event_labels, domain_labels,
                time_to_event.unsqueeze(1), is_healthy
            )

            total_loss += loss_dict["total_loss"].item()
            event_pred = outputs["event_logits"].argmax(dim=-1)
            event_correct += (event_pred == event_labels).sum().item()
            total_samples += event_labels.size(0)

            all_event_preds.extend(event_pred.cpu().numpy())
            all_event_labels.extend(event_labels.cpu().numpy())

    # Compute per-class accuracy
    all_event_preds = np.array(all_event_preds)
    all_event_labels = np.array(all_event_labels)

    per_class_acc = {}
    for cls in [0, 1, 2]:
        mask = all_event_labels == cls
        if mask.sum() > 0:
            per_class_acc[cls] = (all_event_preds[mask] == cls).mean()

    return {
        "loss": total_loss / len(val_loader),
        "event_accuracy": event_correct / total_samples,
        "per_class_accuracy": per_class_acc,
    }


# ── Main Training Script ──────────────────────────────────────────────────────

def main():
    """Main training function."""
    project_root = Path(__file__).parent.parent.parent
    cohort_dir = project_root / "data" / "processed" / "cohort_v1"
    features_csv = project_root / "data" / "processed" / "cohort_v1" / "augmented" / "combined_features.csv"
    output_dir = project_root / "models" / "dann_cardiac_arrest_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        "input_dim": 1500,
        "latent_dim": 128,
        "n_features": 60,
        "n_event_types": 3,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "n_epochs": 100,
        "patience": 15,
        "lambda_domain": 0.5,
        "gamma_time": 1.0,
    }

    print("=" * 70)
    print("PHASE 4: DANN TRAINING")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        cohort_dir, features_csv, batch_size=config["batch_size"]
    )

    # Detect feature count from actual data
    sample_batch = next(iter(train_loader))
    n_features = sample_batch["features"].shape[1]
    config["n_features"] = n_features
    print(f"Detected {n_features} features from data")

    # Build model
    import sys
    sys.path.insert(0, str(project_root))
    from src.dann.model import build_dann_model, DANNLoss
    model = build_dann_model(config)
    model = model.to(device)

    # Loss function
    event_weights = [class_weights.get(None, 1.0),
                     class_weights.get("aki", 1.0),
                     class_weights.get("cardiac_arrest", 1.0)]
    criterion = DANNLoss(
        event_weights=event_weights,
        lambda_domain=config["lambda_domain"],
        gamma_time=config["gamma_time"],
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for {config['n_epochs']} epochs...")
    for epoch in range(config["n_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['n_epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, config["n_epochs"]
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"event_acc={train_metrics['event_accuracy']:.3f}, "
              f"domain_acc={train_metrics['domain_accuracy']:.3f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, "
              f"event_acc={val_metrics['event_accuracy']:.3f}")
        print(f"  Per-class val acc: {val_metrics['per_class_accuracy']}")

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": config,
            }, output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (val_loss={val_metrics['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_metrics["loss"],
        "config": config,
    }, output_dir / "final_model.pt")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
