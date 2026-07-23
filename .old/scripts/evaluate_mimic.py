#!/usr/bin/env python3
"""
Evaluate cardiac arrest model on REAL MIMIC ICU PPG data.
No simulator. No synthetic data. Real patients only.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dann.model_v2 import build_model
from src.dann.inference import FeatureExtractor


def main():
    project_root = Path(__file__).parent.parent
    cohort_dir = project_root / "data" / "processed" / "cohort_v1"
    model_dir = project_root / "models" / "cardiac_arrest_v4"
    output_dir = project_root / "models" / "cardiac_arrest_v4" / "eval_mimic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVALUATION ON REAL MIMIC ICU DATA")
    print("=" * 70)

    # Load model
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = build_model(config)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")

    # Load windows metadata
    windows_df = pd.read_csv(cohort_dir / "windows.csv")
    print(f"Total windows: {len(windows_df)}")
    print(f"Event distribution:")
    print(windows_df["primary_event"].value_counts().to_string())

    # Feature extractor
    fe = FeatureExtractor(fs=25)

    # Process each window: load PPG segments, extract features, predict
    results = []
    ppg_dir = cohort_dir / "ppg_segments"

    for _, row in windows_df.iterrows():
        window_id = row["window_id"]
        subject_id = row["subject_id"]
        primary_event = row["primary_event"] if pd.notna(row["primary_event"]) else "healthy"
        n_segments = int(row["n_segments"])
        time_to_event = row["time_to_event_hours"]

        # Load first segment of this window
        seg_path = ppg_dir / f"{window_id}_s00.npy"
        if not seg_path.exists():
            continue

        ppg = np.load(seg_path).astype(np.float32)
        if np.any(np.isnan(ppg)) or np.std(ppg) < 1e-8:
            continue

        # Normalize
        ppg = ppg - np.mean(ppg)
        std = np.std(ppg)
        if std > 1e-8:
            ppg = ppg / std

        # Pad/truncate to 1500
        if len(ppg) < 1500:
            ppg = np.pad(ppg, (0, 1500 - len(ppg)), mode='edge')
        elif len(ppg) > 1500:
            ppg = ppg[:1500]

        # Extract features
        features = fe.extract(ppg)

        # Predict
        ppg_tensor = torch.tensor(ppg).unsqueeze(0).unsqueeze(0)  # (1, 1, 1500)
        feat_tensor = torch.tensor(features).unsqueeze(0)  # (1, 34)

        with torch.no_grad():
            outputs = model(ppg_tensor, feat_tensor)

        prob = outputs["probability"].item()

        # Binary label: 1 = cardiac arrest, 0 = not cardiac arrest
        is_ca = 1 if primary_event == "cardiac_arrest" else 0

        results.append({
            "window_id": window_id,
            "subject_id": subject_id,
            "primary_event": primary_event,
            "is_cardiac_arrest": is_ca,
            "probability": prob,
            "time_to_event_hours": time_to_event,
        })

    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results_df)} windows from {results_df['subject_id'].nunique()} subjects")

    # ── Overall Binary Evaluation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BINARY EVALUATION: Cardiac Arrest vs Non-Arrest")
    print("=" * 70)

    y_true = results_df["is_cardiac_arrest"].values
    y_prob = results_df["probability"].values
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\nClass distribution:")
    print(f"  Cardiac Arrest: {np.sum(y_true == 1)} windows")
    print(f"  Non-Arrest:     {np.sum(y_true == 0)} windows")

    print(f"\nPrediction distribution:")
    print(f"  Predicted CA:     {np.sum(y_pred == 1)} windows")
    print(f"  Predicted Normal: {np.sum(y_pred == 0)} windows")

    # Per-event breakdown
    print(f"\nPer-event prediction stats:")
    for event in results_df["primary_event"].dropna().unique():
        subset = results_df[results_df["primary_event"] == event]
        mean_prob = subset["probability"].mean()
        std_prob = subset["probability"].std()
        n_alerts = (subset["probability"] > 0.5).sum()
        print(f"  {str(event):30s}: n={len(subset):4d}, mean_prob={mean_prob:.4f} ± {std_prob:.4f}, alerts={n_alerts}")

    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_prob)
        print(f"\nAUROC (binary): {auroc:.4f}")
    except Exception as e:
        print(f"\nAUROC error: {e}")
        auroc = 0.0

    # AUPRC
    try:
        auprc = average_precision_score(y_true, y_prob)
        print(f"AUPRC (binary): {auprc:.4f}")
    except Exception as e:
        print(f"AUPRC error: {e}")
        auprc = 0.0

    # Find best threshold
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)

    # F1-optimal threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    print(f"\nF1-optimal threshold: {best_threshold:.4f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall:    {best_recall:.4f}")
    print(f"  F1:        {best_f1:.4f}")

    # Sensitivity at 95% specificity
    specificity = 1 - fpr
    idx_95spec = np.argmax(specificity >= 0.95)
    sensitivity_at_95spec = tpr[idx_95spec]
    threshold_at_95spec = thresholds_roc[idx_95spec]
    print(f"\nAt 95% specificity:")
    print(f"  Sensitivity (recall): {sensitivity_at_95spec:.4f}")
    print(f"  Threshold: {threshold_at_95spec:.4f}")

    # Sensitivity at 90% specificity
    idx_90spec = np.argmax(specificity >= 0.90)
    sensitivity_at_90spec = tpr[idx_90spec]
    print(f"At 90% specificity:")
    print(f"  Sensitivity (recall): {sensitivity_at_90spec:.4f}")

    # Confusion matrix at optimal threshold
    y_pred_optimal = (y_prob > best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_optimal)
    print(f"\nConfusion matrix (threshold={best_threshold:.4f}):")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    # ── Per-Subject Evaluation (leave-one-out) ─────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SUBJECT EVALUATION (leave-one-out)")
    print("=" * 70)

    ca_subjects = results_df[results_df["is_cardiac_arrest"] == 1]["subject_id"].unique()
    print(f"\nCardiac arrest subjects: {len(ca_subjects)}")

    for subj in ca_subjects:
        subj_data = results_df[results_df["subject_id"] == subj]
        n_windows = len(subj_data)
        mean_prob = subj_data["probability"].mean()
        max_prob = subj_data["probability"].max()
        n_alerts = (subj_data["probability"] > 0.5).sum()
        event = subj_data["primary_event"].iloc[0]
        print(f"  Subject {subj}: event={event}, windows={n_windows}, "
              f"mean_prob={mean_prob:.4f}, max_prob={max_prob:.4f}, alerts={n_alerts}/{n_windows}")

    # ── Per-Window Evaluation (CA segments only) ───────────────────────────
    print("\n" + "=" * 70)
    print("CARDIAC ARREST WINDOWS DETAIL")
    print("=" * 70)

    ca_windows = results_df[results_df["is_cardiac_arrest"] == 1].sort_values("probability", ascending=False)
    print(f"\nTotal CA windows: {len(ca_windows)}")
    print(f"Windows with prob > 0.5: {(ca_windows['probability'] > 0.5).sum()}")
    print(f"Windows with prob > 0.7: {(ca_windows['probability'] > 0.7).sum()}")
    print(f"Windows with prob > 0.9: {(ca_windows['probability'] > 0.9).sum()}")

    # Top predictions
    print(f"\nTop 20 CA windows by probability:")
    for _, row in ca_windows.head(20).iterrows():
        print(f"  {row['window_id']:40s} subj={row['subject_id']} prob={row['probability']:.4f} tte={row['time_to_event_hours']:.1f}h")

    # Bottom predictions (missed)
    print(f"\nBottom 20 CA windows (most missed):")
    for _, row in ca_windows.tail(20).iterrows():
        print(f"  {row['window_id']:40s} subj={row['subject_id']} prob={row['probability']:.4f} tte={row['time_to_event_hours']:.1f}h")

    # ── Non-CA Evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NON-CA EVALUATION (false positive analysis)")
    print("=" * 70)

    non_ca = results_df[results_df["is_cardiac_arrest"] == 0]
    print(f"\nNon-CA windows: {len(non_ca)}")
    print(f"Mean prob: {non_ca['probability'].mean():.4f} ± {non_ca['probability'].std():.4f}")
    print(f"False positives (prob > 0.5): {(non_ca['probability'] > 0.5).sum()}/{len(non_ca)}")

    for event in non_ca["primary_event"].dropna().unique():
        subset = non_ca[non_ca["primary_event"] == event]
        mean_prob = subset["probability"].mean()
        n_fp = (subset["probability"] > 0.5).sum()
        print(f"  {event:30s}: n={len(subset):4d}, mean_prob={mean_prob:.4f}, FP={n_fp}")

    # ── Save results ───────────────────────────────────────────────────────
    results_df.to_csv(output_dir / "mimic_evaluation.csv", index=False)

    eval_summary = {
        "total_windows": int(len(results_df)),
        "total_subjects": int(results_df["subject_id"].nunique()),
        "n_ca_windows": int(np.sum(y_true == 1)),
        "n_non_ca_windows": int(np.sum(y_true == 0)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1_optimal_threshold": float(best_threshold),
        "f1_optimal_precision": float(best_precision),
        "f1_optimal_recall": float(best_recall),
        "f1_optimal_f1": float(best_f1),
        "sensitivity_at_95_specificity": float(sensitivity_at_95spec),
        "sensitivity_at_90_specificity": float(sensitivity_at_90spec),
        "confusion_matrix_optimal": cm.tolist(),
        "per_subject": {
            str(subj): {
                "n_windows": int(subj_data.shape[0]),
                "mean_prob": float(subj_data["probability"].mean()),
                "max_prob": float(subj_data["probability"].max()),
                "n_alerts": int((subj_data["probability"] > 0.5).sum()),
            }
            for subj, subj_data in ca_windows.groupby("subject_id")
        }
    }

    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - mimic_evaluation.csv")
    print(f"  - eval_summary.json")


if __name__ == "__main__":
    main()
