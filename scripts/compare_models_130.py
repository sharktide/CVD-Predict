#!/usr/bin/env python3
"""
Compare v4, v5-watch, and v6-watch on the same 130 synthetic Apple Watch test signals.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from scripts.test_apple_watch_approx import AppleWatchPPGGenerator, extract_features_for_apple_watch, load_v4_model, load_feature_columns, run_inference
from src.model_watch import build_watch_model

def load_v6_watch():
    model_path = Path("production/cvd_risk_v6_watch/best_model.keras")
    config_path = Path("production/cvd_risk_v6_watch/config.yaml")

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load using legacy H5 format
    model = tf.keras.models.load_model(str(model_path))
    with open("production/cvd_risk_v6_watch/feature_columns.json") as f:
        feature_cols = json.load(f)
    with open("production/cvd_risk_v6_watch/optimal_threshold.json") as f:
        threshold = json.load(f)["threshold"]

    return model, feature_cols, threshold

def predict_v6(model, feature_cols, threshold, ppg, features_dict):
    feature_array = np.array([[features_dict.get(col, 0) for col in feature_cols]])

    # Pad/truncate PPG to 7500 samples (model expects 7500)
    ppg_padded = np.zeros(7500, dtype=np.float32)
    L = min(len(ppg), 7500)
    ppg_padded[:L] = ppg[:L]
    ppg_input = ppg_padded.reshape(1, -1, 1).astype(np.float32)

    prob = model.predict({"ppg_input": ppg_input, "feature_input": feature_array}, verbose=0)[0][0]
    return prob

def main():
    np.random.seed(42)

    print("Generating 130 synthetic Apple Watch test signals...")
    gen = AppleWatchPPGGenerator(fs=25, seed=42)
    signals = []
    labels = []

    # 50 healthy
    for i in range(50):
        ppg, _ = gen.generate_healthy_profile(duration_s=120.0)
        signals.append(ppg)
        labels.append(0)

    # 50 at-risk
    for i in range(50):
        ppg, _ = gen.generate_at_risk_profile(duration_s=120.0)
        signals.append(ppg)
        labels.append(1)

    # 30 borderline
    for i in range(30):
        ppg, _ = gen.generate_borderline_profile(duration_s=120.0)
        signals.append(ppg)
        labels.append(1)

    labels = np.array(labels)

    print("Loading models...")
    v4_model = load_v4_model()
    v4_feature_columns = load_feature_columns()
    v6_model, v6_feature_cols, v6_threshold = load_v6_watch()

    with open("production/cvd_risk_v5_watch/feature_columns.json") as f:
        v5_feature_cols = json.load(f)
    with open("production/cvd_risk_v5_watch/optimal_threshold.json") as f:
        v5_threshold = json.load(f)["threshold"]

    v4_probs, v5_probs, v6_probs = [], [], []

    print("Running inference on 130 signals...")
    for i, ppg in enumerate(signals):
        feat = extract_features_for_apple_watch(ppg, fs=25, feature_columns=v4_feature_columns)

        # v4
        inference = run_inference(v4_model, ppg, feat, v4_feature_columns)
        v4_prob = inference["event_probability"]
        v4_probs.append(v4_prob)

        # v5 (inverted v4)
        v5_probs.append(1 - v4_prob)

        # v6
        v6_prob = predict_v6(v6_model, v6_feature_cols, v6_threshold, ppg, feat)
        v6_probs.append(v6_prob)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/130 done")

    v4_probs = np.array(v4_probs)
    v5_probs = np.array(v5_probs)
    v6_probs = np.array(v6_probs)

    results = []
    for name, probs, threshold in [
        ("v4 (raw)", v4_probs, 0.05),
        ("v5-watch (inverted)", v5_probs, 0.55),
        ("v6-watch (native)", v6_probs, v6_threshold),
    ]:
        preds = (probs >= threshold).astype(int)
        try:
            auroc = roc_auc_score(labels, probs)
        except ValueError:
            auroc = float('nan')

        results.append({
            "Model": name,
            "AUROC": f"{auroc:.3f}",
            "Accuracy": f"{accuracy_score(labels, preds)*100:.1f}%",
            "Precision": f"{precision_score(labels, preds, zero_division=0)*100:.1f}%",
            "Recall": f"{recall_score(labels, preds, zero_division=0)*100:.1f}%",
            "F1": f"{f1_score(labels, preds, zero_division=0):.3f}",
            "Threshold": threshold,
        })

    df = pd.DataFrame(results)
    print("\n=== Model Comparison on 130 Synthetic Signals ===")
    print(df.to_string(index=False))

    output_path = "evaluation/model_comparison_130.csv"
    os.makedirs("evaluation", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
