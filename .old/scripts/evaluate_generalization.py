#!/usr/bin/env python3
"""
Evaluate cardiac arrest model on non-simulator data:
1. MIMIC Finger-to-Wrist PPG (real ICU physiology, wrist-like degradation)
2. MMASH wearable pseudo-PPG (real wearable data)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dann.model_v2 import build_model
from src.dann.inference import FeatureExtractor


def evaluate_dataset(model, fe, ppg_files, labels, dataset_name, output_dir):
    """Evaluate model on a set of PPG files."""
    results = []
    
    for ppg_file, label in zip(ppg_files, labels):
        ppg = np.load(str(ppg_file)).astype(np.float32)
        
        # Skip bad segments
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
        ppg_tensor = torch.tensor(ppg).unsqueeze(0).unsqueeze(0)
        feat_tensor = torch.tensor(features).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(ppg_tensor, feat_tensor)
        
        prob = outputs["probability"].item()
        
        results.append({
            "file": ppg_file.name,
            "label": label,
            "probability": prob,
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION: {dataset_name}")
    print(f"{'='*70}")
    print(f"Segments: {len(results_df)}")
    print(f"Mean probability: {results_df['probability'].mean():.4f} ± {results_df['probability'].std():.4f}")
    print(f"Median probability: {results_df['probability'].median():.4f}")
    
    # Distribution
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    probs = results_df["probability"].values
    print(f"\nProbability distribution:")
    for p in percentiles:
        print(f"  {p:3d}th percentile: {np.percentile(probs, p):.4f}")
    
    # Alert rates at different thresholds
    print(f"\nAlert rates:")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_alerts = (probs > thresh).sum()
        pct = n_alerts / len(probs) * 100
        print(f"  threshold={thresh:.1f}: {n_alerts}/{len(probs)} ({pct:.1f}%)")
    
    # Compare to training distribution
    print(f"\nComparison to training:")
    print(f"  Training cardiac_arrest mean: ~0.85 (from synthetic data)")
    print(f"  Training healthy mean: ~0.15 (from synthetic data)")
    print(f"  This dataset mean: {results_df['probability'].mean():.4f}")
    
    results_df.to_csv(output_dir / f"{dataset_name.lower().replace(' ', '_')}.csv", index=False)
    return results_df


def main():
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "cardiac_arrest_v4"
    output_dir = project_root / "models" / "cardiac_arrest_v4" / "eval_generalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERALIZATION EVALUATION (Non-Simulator Data)")
    print("=" * 70)

    # Load model
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = build_model(config)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fe = FeatureExtractor(fs=25)

    # ── 1. MIMIC Finger-to-Wrist ───────────────────────────────────────────
    wristppg_dir = project_root / "data" / "processed" / "mimic_wristppg" / "ppg"
    metadata = pd.read_csv(project_root / "data" / "processed" / "mimic_wristppg" / "metadata.csv")
    
    # Sample 2000 segments (full set is 21K, too slow)
    sample_metadata = metadata.sample(n=min(2000, len(metadata)), random_state=42)
    
    ppg_files = [wristppg_dir / f"{row['segment_id']}.npy" for _, row in sample_metadata.iterrows()]
    ppg_files = [f for f in ppg_files if f.exists()]
    
    # All are "healthy" (no cardiac arrest labels in this dataset)
    labels = ["healthy"] * len(ppg_files)
    
    mimic_wrist_results = evaluate_dataset(
        model, fe, ppg_files, labels,
        "MIMIC Finger-to-Wrist", output_dir
    )

    # ── 2. MMASH Wearable ──────────────────────────────────────────────────
    signals_dir = project_root / "data" / "processed" / "signals"
    mmash_files = sorted(signals_dir.glob("mmash_*_wear.npy"))
    
    ppg_files_mmash = [f for f in mmash_files if f.exists()]
    labels_mmash = ["healthy"] * len(ppg_files_mmash)
    
    mmash_results = evaluate_dataset(
        model, fe, ppg_files_mmash, labels_mmash,
        "MMASH Wearable", output_dir
    )

    # ── 3. Summary comparison ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-DATASET COMPARISON")
    print("=" * 70)
    
    datasets = {
        "MIMIC Finger-to-Wrist (n={})".format(len(mimic_wrist_results)): mimic_wrist_results,
        "MMASH Wearable (n={})".format(len(mmash_results)): mmash_results,
    }
    
    print(f"\n{'Dataset':<45} {'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8} {'>0.5':>8} {'>0.7':>8}")
    print("-" * 100)
    for name, df in datasets.items():
        probs = df["probability"].values
        mean_p = probs.mean()
        med_p = np.median(probs)
        p90 = np.percentile(probs, 90)
        p95 = np.percentile(probs, 95)
        n50 = (probs > 0.5).sum()
        n70 = (probs > 0.7).sum()
        print(f"  {name:<43} {mean_p:>8.4f} {med_p:>8.4f} {p90:>8.4f} {p95:>8.4f} {n50:>7d} {n70:>7d}")
    
    # Key question: are these healthy segments being classified as healthy?
    all_probs = np.concatenate([df["probability"].values for df in datasets.values()])
    print(f"\nAll non-simulator segments (n={len(all_probs)}):")
    print(f"  Classified as normal (prob < 0.5): {(all_probs < 0.5).sum()} ({(all_probs < 0.5).mean()*100:.1f}%)")
    print(f"  Classified as CA (prob > 0.5): {(all_probs > 0.5).sum()} ({(all_probs > 0.5).mean()*100:.1f}%)")
    print(f"  Classified as CA (prob > 0.7): {(all_probs > 0.7).sum()} ({(all_probs > 0.7).mean()*100:.1f}%)")
    
    # Save summary
    summary = {
        "mimic_wrist": {
            "n_segments": len(mimic_wrist_results),
            "mean_prob": float(mimic_wrist_results["probability"].mean()),
            "std_prob": float(mimic_wrist_results["probability"].std()),
            "median_prob": float(mimic_wrist_results["probability"].median()),
            "p90": float(np.percentile(mimic_wrist_results["probability"].values, 90)),
            "p95": float(np.percentile(mimic_wrist_results["probability"].values, 95)),
            "n_alerts_05": int((mimic_wrist_results["probability"] > 0.5).sum()),
            "n_alerts_07": int((mimic_wrist_results["probability"] > 0.7).sum()),
        },
        "mmash": {
            "n_segments": len(mmash_results),
            "mean_prob": float(mmash_results["probability"].mean()),
            "std_prob": float(mmash_results["probability"].std()),
            "median_prob": float(mmash_results["probability"].median()),
            "p90": float(np.percentile(mmash_results["probability"].values, 90)),
            "p95": float(np.percentile(mmash_results["probability"].values, 95)),
            "n_alerts_05": int((mmash_results["probability"] > 0.5).sum()),
            "n_alerts_07": int((mmash_results["probability"] > 0.7).sum()),
        },
    }
    
    with open(output_dir / "generalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
