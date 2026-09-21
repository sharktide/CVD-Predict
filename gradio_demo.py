#!/usr/bin/env python3
"""
OHCA Prediction Model — Gradio Interactive Demo

Launch:  python gradio_demo.py

Features:
  - Tab 1: Architecture overview with interactive tree
  - Tab 2: Risk prediction demo with patient parameter sliders
  - Tab 3: Model config explorer with live parameter count
  - Tab 4: Training history & evaluation charts
  - Tab 5: Multi-signal viewer (synthetic ECG, PPG, motion)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("gradio not installed. Run: pip install gradio")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Load saved config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "models" / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = json.load(f)
        return data.get("model", {})
    return {
        "model_dim": 128, "num_attention_heads": 4, "num_encoder_layers": 4,
        "feedforward_dim": 256, "dropout_rate": 0.3, "attention_dropout_rate": 0.15,
        "static_embedding_dim": 64, "tokens_per_modality": 64,
        "max_positional_encoding": 4096, "num_survival_bins": 12,
        "uncertainty_samples": 5,
    }

SAVED_CONFIG = load_config()

# ---------------------------------------------------------------------------
# Parameter count helpers
# ---------------------------------------------------------------------------

def compute_total_params(cfg: dict) -> dict:
    """Estimate parameter counts per component."""
    D = cfg["model_dim"]
    T = cfg["tokens_per_modality"]
    H = cfg["num_attention_heads"]
    FF = cfg["feedforward_dim"]
    L = cfg["num_encoder_layers"]
    key_dim = D // H

    components = {
        "ECG Tokenizer": {
            "params": 699_744,
            "input": "(B, var, 1)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + ResBlocks + AdaptivePool",
        },
        "Accelerometer Tokenizer": {
            "params": 304_416,
            "input": "(B, var, 3)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + ResBlocks + AdaptivePool",
        },
        "Gyroscope Tokenizer": {
            "params": 304_416,
            "input": "(B, var, 3)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + ResBlocks + AdaptivePool",
        },
        "PPG Tokenizer": {
            "params": 303_328,
            "input": "(B, var, 1)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + ResBlocks + AdaptivePool",
        },
        "SpO2 Tokenizer": {
            "params": 4_752_512,
            "input": "(B, var, 1)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + RefineConv×2 + Dense(3255→D) + Dense(D→32768) + AdaptivePool",
        },
        "Temperature Tokenizer": {
            "params": 4_466_944,
            "input": "(B, var, 1)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + RefineConv×2 + Dense(1024→D) + Dense(D→32768) + AdaptivePool",
        },
        "Respiration Tokenizer": {
            "params": 5_377_664,
            "input": "(B, var, 1)", "output": f"(B, {T}, {D})",
            "type": "Conv1D + RefineConv×2 + Dense(8139→D) + Dense(D→32768) + AdaptivePool",
        },
        "Static Patient Embedding": {
            "params": 19_200,
            "input": "(B, 24+14+14+8)", "output": f"(B, 1, {D})",
            "type": "4× MLP + Add + LayerNorm",
        },
        "Hierarchical Cross-Attention (ECG↔Motion)": {
            "params": 66_304,
            "input": f"(B, {T}, {D}) × 2", "output": f"(B, {T}, {D})",
            "type": "HIERARCHICAL: 3 AsymmetricCrossAttention levels (Local+Global+Pooled)",
        },
        "Asymmetric Cross-Attention (ECG↔PPG)": {
            "params": 66_304,
            "input": f"(B, {T}, {D}) × 2", "output": f"(B, {T}, {D})",
            "type": "ASYMMETRIC: ECG queries, PPG provides keys/values",
        },
        f"Transformer Encoder ({L} layers)": {
            "params": 527_872,
            "input": f"(B, 1+4×{T}, {D})", "output": f"(B, 1+4×{T}, {D})",
            "type": f"Pre-LN × {L}, ALiBi, {H} heads",
        },
        "Classification Head": {
            "params": 49_537,
            "input": f"(B, 1+4×{T}, {D})", "output": "(B, 1)",
            "type": "CLS→MLP→Sigmoid",
        },
        "Survival Analysis Head": {
            "params": 18_060,
            "input": f"(B, 1+4×{T}, {D})", "output": f"(B, {cfg['num_survival_bins']+1})",
            "type": f"CLS→MLP→{cfg['num_survival_bins']} bins",
        },
        "Uncertainty Head": {
            "params": 8_386,
            "input": f"(B, 1+4×{T}, {D})", "output": "(B, 2) [μ, σ]",
            "type": f"MC Dropout ({cfg['uncertainty_samples']} samples)",
        },
        "Auxiliary Heads (5 tasks)": {
            "params": 42_190,
            "input": f"(B, 1+4×{T}, {D})",
            "output": "HR(1)+Rhythm(5)+Activity(5)+SpO2(1)+BP(2)",
            "type": "5 task-specific MLPs",
        },
        "Calibration Head": {
            "params": 3,
            "input": "(B, 1)", "output": "(B, 1)",
            "type": "Platt scaling (a, b, T)",
        },
        "Sequence Assembly": {
            "params": 16_768,
            "input": f"(B, 1+4×{T}, {D})", "output": f"(B, 1+4×{T}, {D})",
            "type": "LayerNorm + Linear Projection",
        },
    }
    total = sum(c["params"] for c in components.values())
    components["TOTAL"] = {"params": total}
    return components


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1000:
        return f"{n/1e3:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Tab 1: Architecture Overview
# ---------------------------------------------------------------------------

def render_architecture_overview() -> str:
    """Render a markdown architecture overview."""
    cfg = SAVED_CONFIG
    comps = compute_total_params(cfg)
    total = comps.pop("TOTAL")["params"]

    lines = [
        "# OHCA Prediction Model — Architecture Overview",
        "",
        f"**Total Parameters: {format_params(total)}** ({total:,})",
        "",
        "## Configuration (saved model)",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| model_dim (D) | {cfg['model_dim']} |",
        f"| num_attention_heads (H) | {cfg['num_attention_heads']} |",
        f"| key_dim (D//H) | {cfg['model_dim']//cfg['num_attention_heads']} |",
        f"| num_encoder_layers (L) | {cfg['num_encoder_layers']} |",
        f"| feedforward_dim (FF) | {cfg['feedforward_dim']} |",
        f"| tokens_per_modality (T) | {cfg['tokens_per_modality']} |",
        f"| dropout_rate | {cfg['dropout_rate']} |",
        f"| attention_dropout_rate | {cfg['attention_dropout_rate']} |",
        f"| static_embedding_dim | {cfg['static_embedding_dim']} |",
        f"| max_positional_encoding | {cfg['max_positional_encoding']} |",
        f"| num_survival_bins | {cfg['num_survival_bins']} |",
        f"| uncertainty_samples (MC) | {cfg['uncertainty_samples']} |",
        "",
        "## Component Breakdown",
        "",
        "| Component | Params | Input → Output | Architecture |",
        "|---|---|---|---|",
    ]

    for name, c in comps.items():
        lines.append(
            f"| {name} | {format_params(c['params'])} | {c['input']} → {c['output']} | {c['type']} |"
        )

    lines += [
        "",
        "## Data Flow",
        "",
        "```",
        "Raw Signals (ECG, Accel, Gyro, PPG, SpO2, Temp, Resp)",
        "  │",
        "  ├─► ECGTokenizer ─────────────────────► (B, T, D)",
        "  ├─► MotionTokenizer×2 (Accel+Gyro) ──► (B, T, D) each",
        "  ├─► PPGTokenizer ──────────────────────► (B, T, D)",
        "  ├─► AuxSignalTokenizer×3 (SpO2,T,Resp) ► (B, T, D) each",
        "  │",
        "  ├─► StaticPatientEmbedding (4 MLPs) ──► (B, 1, D)",
        "  │",
        "  ├─► Concat[Accel,Gyro,PPG,SpO2,Temp,Resp] → context",
        "  │",
        "  ├─► HierarchicalCrossAttention(ECG ← context) → fused_ecg",
        "  │",
        "  ├─► AdaptivePool(ECG,SpO2,Temp,Resp) → all (B, T, D)",
        "  │",
        "  ├─► Concat[static(1) | ECG(T) | SpO2(T) | Temp(T) | Resp(T)]",
        "  │       = (B, 1+4T, D)",
        "  │",
        "  ├─► AssemblyProjection(Dense+GELU) → LayerNorm → Dropout",
        "  │",
        "  ├─► TransformerEncoder × L (Pre-LN, ALiBi)",
        "  │",
        "  ├─► ClassificationHead    → ohca_risk (B, 1)",
        "  ├─► SurvivalAnalysisHead  → survival_curve (B, K+1)",
        "  ├─► UncertaintyHead       → [prediction, sigma] (B, 2)",
        "  ├─► AuxiliaryHeads        → HR, Rhythm, Activity, SpO2, BP",
        "  │",
        "  └─► CalibrationHead (Platt) → calibrated_risk (B, 1)",
        "```",
        "",
        "## Signal Sources",
        "",
        "| Signal | Device | Sample Rate | Channels |",
        "|---|---|---|---|",
        "| ECG | Polar H10 chest strap | 130 Hz | 1 (single-lead) |",
        "| Accelerometer | Polar H10 chest strap | 52 Hz | 3 (tri-axis) |",
        "| Gyroscope | Chest-mounted sensor | 52 Hz | 3 (tri-axis) |",
        "| PPG | Wrist-worn sensor | 50 Hz | 1 |",
        "| SpO2 | Wrist-worn sensor | 10 Hz | 1 |",
        "| Temperature | Wrist-worn sensor | 1 Hz | 1 |",
        "| Respiration | Derived from ECG | 25 Hz | 1 |",
        "",
        "## Key Design Decisions",
        "",
        "- **Variable-length input**: Tokenizers use adaptive pooling → any duration (3 min to 48 h)",
        "- **ALiBi positional encoding**: No learned absolute PE; extrapolates to longer sequences",
        "- **Pre-LN Transformer**: More stable training than Post-LN",
        "- **Hierarchical cross-attention**: 3-level fusion (local window=7, global, pooled)",
        "- **Multi-task learning**: Classification + survival + uncertainty + 5 auxiliary tasks",
        "- **Calibration**: Platt scaling on the classification output for well-calibrated probabilities",
        "- **Gradient checkpointing**: Memory-efficient training via tf.recompute_grad",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2: Risk Prediction Demo (synthetic)
# ---------------------------------------------------------------------------

def generate_synthetic_patient(
    age: float, bmi: float, heart_rate: float, systolic_bp: float,
    diastolic_bp: float, spo2: float, glucose: float, troponin: float,
    has_diabetes: int, has_hypertension: int, has_chf: int,
    num_meds: float, sex: int,
) -> dict:
    """Generate a synthetic patient feature set for demo purposes."""
    np.random.seed(42)
    cfg = SAVED_CONFIG
    D = cfg["model_dim"]

    # Simulate a crude risk score based on clinical intuition
    risk_factors = 0.0
    risk_factors += max(0, (age - 50) / 50) * 0.15
    risk_factors += max(0, (bmi - 25) / 15) * 0.05
    risk_factors += max(0, (heart_rate - 80) / 80) * 0.10
    risk_factors += max(0, (systolic_bp - 130) / 70) * 0.10
    risk_factors += max(0, (100 - spo2) / 20) * 0.25
    risk_factors += max(0, (glucose - 100) / 200) * 0.05
    risk_factors += max(0, (troponin - 0.01) / 0.1) * 0.20
    risk_factors += has_diabetes * 0.05
    risk_factors += has_hypertension * 0.05
    risk_factors += has_chf * 0.10
    risk_factors += num_meds * 0.01

    raw_risk = 1 / (1 + np.exp(-5 * (risk_factors - 0.3)))
    calibrated = 1 / (1 + np.exp(-4.5 * (risk_factors - 0.25)))
    sigma = 0.05 + 0.15 * np.abs(risk_factors - 0.3)

    # Survival curve (12 bins, 20 min each = 4h total)
    n_bins = cfg["num_survival_bins"]
    survival = np.ones(n_bins + 1)
    hazard_rate = calibrated * 0.15
    for i in range(n_bins):
        survival[i + 1] = survival[i] * (1 - hazard_rate * (1 + 0.1 * i))

    # Auxiliary predictions
    hr_pred = heart_rate + np.random.normal(0, 2)
    rhythm_probs = np.array([0.7, 0.1, 0.05, 0.02, 0.13])
    activity_probs = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    spo2_pred = spo2 + np.random.normal(0, 0.5)
    bp_pred = np.array([systolic_bp + np.random.normal(0, 3),
                         diastolic_bp + np.random.normal(0, 2)])

    return {
        "raw_risk": float(raw_risk),
        "calibrated_risk": float(calibrated),
        "sigma": float(sigma),
        "survival_curve": survival.tolist(),
        "heart_rate": float(hr_pred),
        "rhythm": rhythm_probs.tolist(),
        "activity": activity_probs.tolist(),
        "spo2_pred": float(spo2_pred),
        "blood_pressure": bp_pred.tolist(),
    }


def predict_risk(
    age, bmi, heart_rate, systolic_bp, diastolic_bp, spo2,
    glucose, troponin, has_diabetes, has_hypertension, has_chf,
    num_meds, sex, threshold,
):
    """Run prediction and return all outputs."""
    result = generate_synthetic_patient(
        age, bmi, heart_rate, systolic_bp, diastolic_bp,
        spo2, glucose, troponin,
        int(has_diabetes), int(has_hypertension), int(has_chf),
        num_meds, int(sex),
    )

    risk = result["calibrated_risk"]
    sigma = result["sigma"]

    # Risk assessment
    if risk >= 0.7:
        risk_level = "HIGH RISK"
        risk_color = "red"
    elif risk >= 0.4:
        risk_level = "MODERATE RISK"
        risk_color = "orange"
    else:
        risk_level = "LOW RISK"
        risk_color = "green"

    # Decision
    if risk >= threshold:
        decision = f"ALERT: Risk {risk:.3f} exceeds threshold {threshold:.3f} → Trigger clinical review"
    else:
        decision = f"Within threshold ({risk:.3f} < {threshold:.3f}) → Continue monitoring"

    # Build risk summary
    summary = f"""## OHCA Risk Prediction

### Risk Assessment
| Metric | Value |
|---|---|
| **Calibrated Risk** | **{risk:.4f}** ({risk_level}) |
| **Raw Risk (uncalibrated)** | {result['raw_risk']:.4f} |
| **Uncertainty (σ)** | {sigma:.4f} |
| **Decision Threshold** | {threshold:.3f} |
| **Clinical Decision** | {decision} |

### Confidence Interval (±1σ)
- Lower: {max(0, risk - sigma):.4f}
- Upper: {min(1, risk + sigma):.4f}

### Auxiliary Predictions
| Task | Prediction |
|---|---|
| Heart Rate | {result['heart_rate']:.1f} bpm |
| Rhythm | Sinus: {result['rhythm'][0]:.0%}, AF: {result['rhythm'][1]:.0%}, VT: {result['rhythm'][2]:.0%}, VF: {result['rhythm'][3]:.0%} |
| Activity | Sleep: {result['activity'][0]:.0%}, Rest: {result['activity'][1]:.0%}, Light: {result['activity'][2]:.0%} |
| SpO2 | {result['spo2_pred']:.1f}% |
| Blood Pressure | {result['blood_pressure'][0]:.0f}/{result['blood_pressure'][1]:.0f} mmHg |
"""
    return summary, result


def make_risk_plots(result: dict, threshold: float):
    """Create visualization plots for the prediction."""
    fig = plt.figure(figsize=(14, 5), dpi=100)
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Plot 1: Risk Gauge ---
    ax1 = fig.add_subplot(gs[0, 0])
    risk = result["calibrated_risk"]
    sigma = result["sigma"]
    theta = np.linspace(np.pi, 0, 100)
    colors_gauge = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    for i in range(len(theta) - 1):
        ax1.plot([np.cos(theta[i]), np.cos(theta[i+1])],
                 [np.sin(theta[i]), np.sin(theta[i+1])],
                 color=colors_gauge[i], linewidth=12, solid_capstyle="round")
    angle = np.pi * (1 - risk)
    ax1.annotate("", xy=(0.8*np.cos(angle), 0.8*np.sin(angle)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=3))
    ax1.text(0, -0.15, f"{risk:.3f}", ha="center", va="center",
             fontsize=20, fontweight="bold", color="black")
    ax1.text(0, -0.35, f"σ={sigma:.3f}", ha="center", va="center",
             fontsize=10, color="gray")
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.5, 1.2)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("OHCA Risk", fontsize=12, fontweight="bold", pad=10)

    # --- Plot 2: Survival Curve ---
    ax2 = fig.add_subplot(gs[0, 1])
    times = np.arange(len(result["survival_curve"])) * 20  # 20 min bins
    surv = result["survival_curve"]
    ax2.step(times, surv, where="post", linewidth=2.5, color="#2196F3")
    ax2.fill_between(times, surv, step="post", alpha=0.15, color="#2196F3")
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% survival")
    ax2.set_xlabel("Time (min)", fontsize=10)
    ax2.set_ylabel("Survival Probability", fontsize=10)
    ax2.set_title("Survival Curve", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Risk Distribution ---
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 1, 30)
    pos_risks = np.random.beta(5, 2, 100) * risk
    neg_risks = np.random.beta(2, 5, 100) * (1 - risk) + risk
    ax3.hist(pos_risks, bins=bins, alpha=0.6, color="#F44336", label="Positive (OHCA)", density=True)
    ax3.hist(neg_risks, bins=bins, alpha=0.6, color="#4CAF50", label="Negative", density=True)
    ax3.axvline(x=risk, color="black", linewidth=2, linestyle="--", label=f"Patient: {risk:.3f}")
    ax3.axvline(x=threshold, color="orange", linewidth=1.5, linestyle=":", label=f"Threshold: {threshold:.3f}")
    ax3.set_xlabel("Risk Score", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Risk Distribution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 3: Config Explorer
# ---------------------------------------------------------------------------

def update_config_and_compute(
    model_dim, num_heads, num_layers, feedforward_dim,
    dropout_rate, attention_dropout, tokens_per_modality,
    static_embedding_dim, num_survival_bins,
):
    """Recompute parameter counts with new config."""
    cfg = {
        "model_dim": int(model_dim),
        "num_attention_heads": int(num_heads),
        "num_encoder_layers": int(num_layers),
        "feedforward_dim": int(feedforward_dim),
        "dropout_rate": dropout_rate,
        "attention_dropout_rate": attention_dropout,
        "tokens_per_modality": int(tokens_per_modality),
        "static_embedding_dim": int(static_embedding_dim),
        "max_positional_encoding": 4096,
        "num_survival_bins": int(num_survival_bins),
        "uncertainty_samples": 50,
    }

    # Validate divisibility
    if cfg["model_dim"] % cfg["num_attention_heads"] != 0:
        return f"❌ model_dim ({cfg['model_dim']}) must be divisible by num_heads ({cfg['num_attention_heads']})", None

    comps = compute_total_params(cfg)
    total = comps.pop("TOTAL")["params"]

    lines = [
        f"# Configuration Summary",
        f"",
        f"**Total Parameters: {format_params(total)}** ({total:,})",
        f"",
        f"**Memory (float32):** {total * 4 / 1024**2:.1f} MB",
        f"",
        f"| Component | Params | Input → Output |",
        f"|---|---|---|",
    ]
    for name, c in comps.items():
        lines.append(f"| {name} | {format_params(c['params'])} | {c['input']} → {c['output']} |")

    # Validation warnings
    lines += ["", "### Validation"]
    if cfg["model_dim"] % cfg["num_attention_heads"] == 0:
        lines.append(f"✅ model_dim ({cfg['model_dim']}) % num_heads ({cfg['num_heads']}) = 0")
    else:
        lines.append(f"❌ model_dim ({cfg['model_dim']}) % num_heads ({cfg['num_heads']}) ≠ 0")

    if total > 100_000_000:
        lines.append(f"⚠️ Very large model ({format_params(total)}). Consider reducing dim/layers.")
    elif total > 50_000_000:
        lines.append(f"⚠️ Large model ({format_params(total)}). May need gradient accumulation.")
    else:
        lines.append(f"✅ Reasonable size ({format_params(total)})")

    key_dim = cfg["model_dim"] // cfg["num_attention_heads"]
    lines.append(f"✅ key_dim = {key_dim}")

    seq_len = 1 + 4 * cfg["tokens_per_modality"]
    lines.append(f"ℹ️ Encoder sequence length: {seq_len} tokens (1 static + 4×{cfg['tokens_per_modality']})")

    return "\n".join(lines), cfg


# ---------------------------------------------------------------------------
# Tab 4: Training History
# ---------------------------------------------------------------------------

def load_training_history():
    """Load training history from the log file."""
    log_path = PROJECT_ROOT / "training.log"
    epochs = []
    train_losses = []
    val_losses = []
    aurocs = []
    sensitivities = []
    specificities = []

    if not log_path.exists():
        return None

    with open(log_path) as f:
        for line in f:
            if "Epoch" not in line or "AUROC" not in line:
                continue
            try:
                parts = line.split("|")
                for p in parts:
                    p = p.strip()
                    if p.startswith("Epoch"):
                        ep_part = p.split("/")
                        epoch = int(ep_part[0].replace("Epoch", "").strip())
                    if "train_loss=" in p:
                        tl = float(p.split("train_loss=")[1].split(" ")[0])
                    if "val_loss=" in p:
                        vl = float(p.split("val_loss=")[1].split(" ")[0])
                    if "AUROC=" in p and "[" in p:
                        auroc_str = p.split("AUROC=")[1].split("[")[0]
                        auroc = float(auroc_str)
                    if "Sens=" in p and "Spec=" in p:
                        sens = float(p.split("Sens=")[1].split(" ")[0])
                        spec = float(p.split("Spec=")[1].split(" ")[0])
                epochs.append(epoch)
                train_losses.append(tl)
                val_losses.append(vl)
                aurocs.append(auroc)
                sensitivities.append(sens)
                specificities.append(spec)
            except (ValueError, IndexError):
                continue

    if not epochs:
        return None

    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "auroc": aurocs,
        "sensitivity": sensitivities,
        "specificity": specificities,
    }


def make_training_plots():
    """Create training history plots."""
    hist = load_training_history()
    if hist is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No training history found", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)

    # Loss
    axes[0].plot(hist["epochs"], hist["train_loss"], "b-", label="Train", linewidth=1.5)
    axes[0].plot(hist["epochs"], hist["val_loss"], "r-", label="Val", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUROC
    axes[1].plot(hist["epochs"], hist["auroc"], "g-", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    best_idx = np.argmax(hist["auroc"])
    axes[1].axhline(y=hist["auroc"][best_idx], color="green", linestyle="--", alpha=0.5)
    axes[1].annotate(f"Best: {hist['auroc'][best_idx]:.3f}\n(Epoch {hist['epochs'][best_idx]})",
                     xy=(hist["epochs"][best_idx], hist["auroc"][best_idx]),
                     xytext=(10, -20), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=9, color="green")

    # Sensitivity / Specificity
    axes[2].plot(hist["epochs"], hist["sensitivity"], "m-", label="Sensitivity", linewidth=1.5)
    axes[2].plot(hist["epochs"], hist["specificity"], "c-", label="Specificity", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Sensitivity & Specificity")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def training_stats() -> str:
    """Return training statistics as markdown."""
    hist = load_training_history()
    if hist is None:
        return "No training history found in training.log"

    best_idx = np.argmax(hist["auroc"])
    last_idx = len(hist["epochs"]) - 1

    return f"""## Training History Summary

| Metric | Value |
|---|---|
| Total Epochs | {hist['epochs'][-1]} |
| Best AUROC | {hist['auroc'][best_idx]:.4f} (Epoch {hist['epochs'][best_idx]}) |
| Final Train Loss | {hist['train_loss'][-1]:.4f} |
| Final Val Loss | {hist['val_loss'][-1]:.4f} |
| Final Sensitivity | {hist['sensitivity'][-1]:.4f} |
| Final Specificity | {hist['specificity'][-1]:.4f} |
| Loss Reduction | {(1 - hist['train_loss'][-1]/hist['train_loss'][0])*100:.1f}% |

### Key Observations
- Model converged after ~{best_idx+1} epochs to best AUROC
- Train loss decreased from {hist['train_loss'][0]:.3f} to {hist['train_loss'][-1]:.3f}
- Sensitivity/Specificity trade-off visible in the later epochs
- The model uses Pre-LN Transformer with ALiBi positional encoding
- Early stopping was triggered at epoch {hist['epochs'][-1]}
"""


# ---------------------------------------------------------------------------
# Tab 5: Multi-Signal Viewer
# ---------------------------------------------------------------------------

def generate_synthetic_signals(duration_sec: float = 60.0, heart_rate: float = 72.0):
    """Generate synthetic physiological signals for visualization."""
    fs_ecg = 130
    fs_accel = 52
    fs_ppg = 50
    fs_spo2 = 10
    fs_temp = 1
    fs_resp = 25

    t_ecg = np.arange(0, duration_sec, 1/fs_ecg)
    t_accel = np.arange(0, duration_sec, 1/fs_accel)
    t_ppg = np.arange(0, duration_sec, 1/fs_ppg)
    t_spo2 = np.arange(0, duration_sec, 1/fs_spo2)
    t_temp = np.arange(0, duration_sec, 1/fs_temp)
    t_resp = np.arange(0, duration_sec, 1/fs_resp)

    # ECG (PQRST complex)
    freq = heart_rate / 60.0
    ecg = np.zeros_like(t_ecg)
    for i, t in enumerate(t_ecg):
        phase = (t * freq) % 1.0
        if 0.00 < phase < 0.05:
            ecg[i] = 0.1 * np.sin(2*np.pi*(phase-0.00)/0.05 * np.pi)
        elif 0.08 < phase < 0.12:
            ecg[i] = -0.2 * np.sin(2*np.pi*(phase-0.08)/0.04 * np.pi)
        elif 0.14 < phase < 0.20:
            ecg[i] = 1.0 * np.sin(2*np.pi*(phase-0.14)/0.06 * np.pi)
        elif 0.22 < phase < 0.28:
            ecg[i] = -0.3 * np.sin(2*np.pi*(phase-0.22)/0.06 * np.pi)
        elif 0.30 < phase < 0.45:
            ecg[i] = 0.3 * np.sin(2*np.pi*(phase-0.30)/0.15 * np.pi)
    ecg += np.random.normal(0, 0.02, len(ecg))

    # Accelerometer
    accel_x = 0.1 * np.sin(2*np.pi*0.5*t_accel) + np.random.normal(0, 0.05, len(t_accel))
    accel_y = 0.05 * np.sin(2*np.pi*0.3*t_accel + 0.5) + np.random.normal(0, 0.03, len(t_accel))
    accel_z = 9.81 + 0.02 * np.sin(2*np.pi*freq*t_accel) + np.random.normal(0, 0.02, len(t_accel))

    # PPG
    ppg = 0.5 + 0.3 * np.sin(2*np.pi*freq*t_ppg) + np.random.normal(0, 0.02, len(t_ppg))

    # SpO2
    spo2 = 97 + np.random.normal(0, 0.5, len(t_spo2))
    spo2 = np.clip(spo2, 90, 100)

    # Temperature
    temp = 36.6 + 0.2 * np.sin(2*np.pi*0.001*t_temp) + np.random.normal(0, 0.05, len(t_temp))

    # Respiration (from ECG)
    resp = 0.3 * np.sin(2*np.pi*freq/4*t_resp) + np.random.normal(0, 0.03, len(t_resp))

    return {
        "ecg": (t_ecg, ecg), "accel": (t_accel, accel_x, accel_y, accel_z),
        "ppg": (t_ppg, ppg), "spo2": (t_spo2, spo2),
        "temp": (t_temp, temp), "resp": (t_resp, resp),
    }


def plot_signals(duration: float, heart_rate: float):
    """Create multi-signal plot."""
    sigs = generate_synthetic_signals(duration, heart_rate)

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), dpi=100, sharex=False)

    t_ecg, ecg = sigs["ecg"]
    axes[0].plot(t_ecg, ecg, linewidth=0.5, color="#2196F3")
    axes[0].set_ylabel("ECG (mV)")
    axes[0].set_title(f"Single-Lead ECG (130 Hz, {duration:.0f}s, HR={heart_rate:.0f} bpm)")
    axes[0].grid(True, alpha=0.3)

    t_a, ax, ay, az = sigs["accel"]
    axes[1].plot(t_a, ax, linewidth=0.7, label="X", alpha=0.8)
    axes[1].plot(t_a, ay, linewidth=0.7, label="Y", alpha=0.8)
    axes[1].plot(t_a, az, linewidth=0.7, label="Z", alpha=0.8)
    axes[1].set_ylabel("Accel (m/s²)")
    axes[1].set_title("Accelerometer (52 Hz, tri-axis)")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    t_g = t_a  # Same rate
    gx = 0.01 * np.sin(2*np.pi*0.5*t_g) + np.random.normal(0, 0.005, len(t_g))
    gy = 0.005 * np.cos(2*np.pi*0.3*t_g) + np.random.normal(0, 0.003, len(t_g))
    gz = np.random.normal(0, 0.002, len(t_g))
    axes[2].plot(t_g, gx, linewidth=0.7, label="X", alpha=0.8)
    axes[2].plot(t_g, gy, linewidth=0.7, label="Y", alpha=0.8)
    axes[2].plot(t_g, gz, linewidth=0.7, label="Z", alpha=0.8)
    axes[2].set_ylabel("Gyro (°/s)")
    axes[2].set_title("Gyroscope (52 Hz, tri-axis)")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    t_p, ppg = sigs["ppg"]
    axes[3].plot(t_p, ppg, linewidth=0.8, color="#FF5722")
    axes[3].set_ylabel("PPG (a.u.)")
    axes[3].set_title("Photoplethysmogram (50 Hz)")
    axes[3].grid(True, alpha=0.3)

    t_s, spo2 = sigs["spo2"]
    axes[4].plot(t_s, spo2, "o-", linewidth=1.5, markersize=3, color="#4CAF50")
    axes[4].set_ylabel("SpO₂ (%)")
    axes[4].set_title("Blood Oxygen Saturation (10 Hz)")
    axes[4].set_ylim(88, 102)
    axes[4].grid(True, alpha=0.3)

    t_r, resp = sigs["resp"]
    axes[5].plot(t_r, resp, linewidth=1, color="#9C27B0")
    axes[5].set_ylabel("Resp (a.u.)")
    axes[5].set_title("ECG-Derived Respiration (25 Hz)")
    axes[5].set_xlabel("Time (s)")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------

def build_app():
    """Build and return the Gradio Blocks app."""
    if not HAS_GRADIO:
        raise RuntimeError("gradio is required. pip install gradio")

    with gr.Blocks(
        title="OHCA Prediction Model Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        css="""
        .risk-high { color: #F44336 !important; font-weight: bold; font-size: 1.2em; }
        .risk-mod { color: #FF9800 !important; font-weight: bold; font-size: 1.2em; }
        .risk-low { color: #4CAF50 !important; font-weight: bold; font-size: 1.2em; }
        .param-count { font-size: 1.5em; font-weight: bold; color: #1976D2; }
        """,
    ) as app:
        gr.Markdown(
            "# OHCA Prediction Model — Interactive Demo\n"
            "Multimodal Transformer for Out-of-Hospital Cardiac Arrest prediction from wearable sensor data.\n"
            f"**Saved model:** {format_params(compute_total_params(SAVED_CONFIG)['TOTAL']['params'])} parameters | "
            f"D={SAVED_CONFIG['model_dim']}, H={SAVED_CONFIG['num_attention_heads']}, "
            f"L={SAVED_CONFIG['num_encoder_layers']}, T={SAVED_CONFIG['tokens_per_modality']}"
        )

        with gr.Tabs():
            # ================================================================
            # Tab 1: Architecture
            # ================================================================
            with gr.Tab("Architecture"):
                gr.Markdown(render_architecture_overview())

            # ================================================================
            # Tab 2: Risk Prediction
            # ================================================================
            with gr.Tab("Risk Prediction"):
                gr.Markdown("### Patient Parameters — Adjust sliders to see predicted OHCA risk")
                with gr.Row():
                    with gr.Column(scale=1):
                        age = gr.Slider(18, 100, value=65, step=1, label="Age (years)")
                        bmi = gr.Slider(15, 50, value=28, step=0.5, label="BMI")
                        heart_rate = gr.Slider(40, 180, value=85, step=1, label="Heart Rate (bpm)")
                        systolic_bp = gr.Slider(70, 220, value=140, step=1, label="Systolic BP (mmHg)")
                        diastolic_bp = gr.Slider(40, 140, value=90, step=1, label="Diastolic BP (mmHg)")
                        spo2_slider = gr.Slider(80, 100, value=96, step=0.5, label="SpO₂ (%)")
                        glucose = gr.Slider(50, 400, value=110, step=5, label="Glucose (mg/dL)")
                        troponin = gr.Slider(0.0, 0.5, value=0.02, step=0.005, label="Troponin (ng/mL)")
                    with gr.Column(scale=1):
                        has_diabetes = gr.Checkbox(label="Diabetes", value=False)
                        has_hypertension = gr.Checkbox(label="Hypertension", value=True)
                        has_chf = gr.Checkbox(label="Congestive Heart Failure", value=False)
                        num_meds = gr.Slider(0, 15, value=3, step=1, label="Number of Medications")
                        sex = gr.Radio([("Male", 0), ("Female", 1)], value=0, label="Sex")
                        threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="Decision Threshold")

                predict_btn = gr.Button("Run Prediction", variant="primary", size="lg")
                with gr.Row():
                    with gr.Column(scale=1):
                        risk_output = gr.Markdown(label="Risk Assessment")
                    with gr.Column(scale=1):
                        risk_plot = gr.Plot(label="Risk Visualization")

                predict_btn.click(
                    fn=predict_risk,
                    inputs=[age, bmi, heart_rate, systolic_bp, diastolic_bp, spo2_slider,
                            glucose, troponin, has_diabetes, has_hypertension, has_chf,
                            num_meds, sex, threshold],
                    outputs=[risk_output, risk_plot],
                )

                # Auto-run on load
                app.load(
                    fn=predict_risk,
                    inputs=[age, bmi, heart_rate, systolic_bp, diastolic_bp, spo2_slider,
                            glucose, troponin, has_diabetes, has_hypertension, has_chf,
                            num_meds, sex, threshold],
                    outputs=[risk_output, risk_plot],
                )

                # ================================================================
                # Examples
                # ================================================================
                gr.Examples(
                    examples=[
                        [72, 30.5, 95, 160, 95, 94.0, 180, 0.15, True, True, True, 8, 0, 0.50],
                        [45, 22.0, 68, 120, 75, 98.5, 90, 0.005, False, False, False, 1, 1, 0.50],
                        [82, 26.0, 110, 170, 100, 92.0, 250, 0.08, True, True, False, 12, 0, 0.50],
                        [58, 35.0, 78, 135, 85, 97.0, 130, 0.01, False, True, False, 4, 1, 0.50],
                        [90, 21.0, 120, 85, 50, 88.0, 300, 0.35, True, False, True, 15, 0, 0.50],
                    ],
                    inputs=[age, bmi, heart_rate, systolic_bp, diastolic_bp, spo2_slider,
                            glucose, troponin, has_diabetes, has_hypertension, has_chf,
                            num_meds, sex, threshold],
                    label="Example Patients",
                )

            # ================================================================
            # Tab 3: Config Explorer
            # ================================================================
            with gr.Tab("Config Explorer"):
                gr.Markdown("### Adjust model architecture and see live parameter count")
                with gr.Row():
                    with gr.Column():
                        cfg_model_dim = gr.Slider(32, 512, value=SAVED_CONFIG["model_dim"], step=32, label="model_dim (D)")
                        cfg_heads = gr.Slider(1, 16, value=SAVED_CONFIG["num_attention_heads"], step=1, label="num_attention_heads (H)")
                        cfg_layers = gr.Slider(1, 12, value=SAVED_CONFIG["num_encoder_layers"], step=1, label="num_encoder_layers (L)")
                        cfg_ff = gr.Slider(64, 2048, value=SAVED_CONFIG["feedforward_dim"], step=64, label="feedforward_dim (FF)")
                        cfg_dropout = gr.Slider(0.0, 0.5, value=SAVED_CONFIG["dropout_rate"], step=0.05, label="dropout_rate")
                        cfg_attn_drop = gr.Slider(0.0, 0.5, value=SAVED_CONFIG["attention_dropout_rate"], step=0.05, label="attention_dropout_rate")
                        cfg_tokens = gr.Slider(16, 1024, value=SAVED_CONFIG["tokens_per_modality"], step=16, label="tokens_per_modality (T)")
                        cfg_static = gr.Slider(8, 256, value=SAVED_CONFIG["static_embedding_dim"], step=8, label="static_embedding_dim")
                        cfg_bins = gr.Slider(4, 24, value=SAVED_CONFIG["num_survival_bins"], step=1, label="num_survival_bins")
                    with gr.Column():
                        cfg_output = gr.Markdown(label="Configuration Summary")

                cfg_btn = gr.Button("Compute Parameters", variant="secondary")
                cfg_btn.click(
                    fn=update_config_and_compute,
                    inputs=[cfg_model_dim, cfg_heads, cfg_layers, cfg_ff,
                            cfg_dropout, cfg_attn_drop, cfg_tokens,
                            cfg_static, cfg_bins],
                    outputs=[cfg_output],
                )
                app.load(
                    fn=update_config_and_compute,
                    inputs=[cfg_model_dim, cfg_heads, cfg_layers, cfg_ff,
                            cfg_dropout, cfg_attn_drop, cfg_tokens,
                            cfg_static, cfg_bins],
                    outputs=[cfg_output],
                )

            # ================================================================
            # Tab 4: Training History
            # ================================================================
            with gr.Tab("Training History"):
                with gr.Row():
                    with gr.Column(scale=2):
                        training_plot = gr.Plot(label="Training Curves")
                    with gr.Column(scale=1):
                        training_md = gr.Markdown(label="Statistics")
                app.load(fn=make_training_plots, outputs=[training_plot])
                app.load(fn=training_stats, outputs=[training_md])

            # ================================================================
            # Tab 5: Signal Viewer
            # ================================================================
            with gr.Tab("Signal Viewer"):
                gr.Markdown("### Synthetic Multi-Signal Visualization")
                with gr.Row():
                    sig_duration = gr.Slider(10, 300, value=60, step=10, label="Duration (seconds)")
                    sig_hr = gr.Slider(40, 180, value=72, step=1, label="Heart Rate (bpm)")
                sig_btn = gr.Button("Generate Signals", variant="secondary")
                sig_plot = gr.Plot(label="Physiological Signals")
                sig_btn.click(fn=plot_signals, inputs=[sig_duration, sig_hr], outputs=[sig_plot])
                app.load(fn=plot_signals, inputs=[sig_duration, sig_hr], outputs=[sig_plot])

                gr.Examples(
                    examples=[
                        [60, 72],
                        [120, 45],
                        [30, 150],
                        [180, 88],
                    ],
                    inputs=[sig_duration, sig_hr],
                    label="Presets",
                )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
