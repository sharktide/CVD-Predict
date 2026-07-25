"""
Visualization utilities for the OHCA Predictor project.

Provides OHCAVisualizer with methods for ECG/accelerometer/PPG signal
visualization, training curves, ROC/PR/calibration curves, decision
curve analysis, confusion matrix, attention heatmaps, and survival curves.
All plots are saved to a specified output directory.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper")


class OHCAVisualizer:
    """
    Centralized visualization class for OHCA prediction outputs.

    All plot methods save figures to the configured output directory and
    return the matplotlib Figure object for further customization.
    """

    def __init__(self, output_dir: str = "outputs/figures") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig: plt.Figure, name: str) -> plt.Figure:
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return fig

    def plot_signals(
        self,
        ecg: Optional[np.ndarray] = None,
        accelerometer: Optional[np.ndarray] = None,
        ppg: Optional[np.ndarray] = None,
        sampling_rates: Optional[Dict[str, float]] = None,
        title: str = "Sensor Signals",
        max_duration_seconds: float = 10.0,
        filename: str = "sensor_signals",
    ) -> plt.Figure:
        """
        Plot ECG, accelerometer, and PPG signals side by side.

        Args:
            ecg: 1-D ECG signal array.
            accelerometer: (N, 3) accelerometer array (x, y, z).
            ppg: 1-D PPG signal array.
            sampling_rates: Dict with keys 'ecg_hz', 'accelerometer_hz', 'ppg_hz'.
            title: Figure super-title.
            max_duration_seconds: Maximum time in seconds to display.
            filename: Output filename (without extension).

        Returns:
            matplotlib Figure object.
        """
        if sampling_rates is None:
            sampling_rates = {"ecg_hz": 130.0, "accelerometer_hz": 52.0, "ppg_hz": 50.0}

        panels = []
        labels = []

        if ecg is not None:
            ecg = ecg.squeeze()
            hz = sampling_rates.get("ecg_hz", 130.0)
            n = min(int(hz * max_duration_seconds), len(ecg))
            panels.append((np.arange(n) / hz, ecg[:n]))
            labels.append("ECG")

        if accelerometer is not None:
            hz = sampling_rates.get("accelerometer_hz", 52.0)
            n = min(int(hz * max_duration_seconds), len(accelerometer))
            time_ax = np.arange(n) / hz
            for ch, axis_label in enumerate(["X", "Y", "Z"]):
                panels.append((time_ax, accelerometer[:n, ch]))
                labels.append(f"Accel {axis_label}")

        if ppg is not None:
            ppg = ppg.squeeze()
            hz = sampling_rates.get("ppg_hz", 50.0)
            n = min(int(hz * max_duration_seconds), len(ppg))
            panels.append((np.arange(n) / hz, ppg[:n]))
            labels.append("PPG")

        n_panels = len(panels)
        if n_panels == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No signal data provided", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return self._save(fig, filename)

        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels), sharex=False)
        if n_panels == 1:
            axes = [axes]

        for ax, (t, sig), label in zip(axes, panels, labels):
            ax.plot(t, sig, linewidth=0.6, color="#2c3e50")
            ax.set_ylabel(label, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Time (seconds)", fontsize=10)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        title: str = "Training Curves",
        filename: str = "training_curves",
    ) -> plt.Figure:
        """
        Plot loss and metric curves from training history.

        Args:
            history: Dict mapping metric names to lists of per-epoch values.
                     Expected keys include 'loss', 'val_loss', and any metric names.
            metrics: List of metric names to plot (excluding loss). If None,
                     auto-detects all non-loss keys.
            title: Figure title.
            filename: Output filename.

        Returns:
            matplotlib Figure object.
        """
        if metrics is None:
            metrics = [k for k in history if k not in ("loss", "val_loss")]

        n_plots = 1 + len(metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        ax = axes[0]
        epochs = list(range(1, len(history.get("loss", [])) + 1))
        if "loss" in history:
            ax.plot(epochs, history["loss"], label="Train Loss", linewidth=1.5)
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5,
                    linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()

        for idx, metric_name in enumerate(metrics):
            ax = axes[idx + 1]
            train_key = metric_name
            val_key = f"val_{metric_name}"
            if train_key in history:
                ax.plot(epochs, history[train_key], label=f"Train {metric_name}",
                        linewidth=1.5)
            if val_key in history:
                ax.plot(epochs, history[val_key], label=f"Val {metric_name}",
                        linewidth=1.5, linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        filename: str = "roc_curve",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> plt.Figure:
        """
        Plot ROC curve with bootstrap confidence band.

        Args:
            y_true: Ground-truth binary labels.
            y_scores: Predicted probabilities.
            title: Plot title.
            filename: Output filename.
            n_bootstrap: Number of bootstrap resamples for CI.
            confidence_level: Confidence level for the band.

        Returns:
            matplotlib Figure object.
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)

        rng = np.random.RandomState(42)
        n_samples = len(y_true)
        boot_aucs = []
        fpr_grid = np.linspace(0.0, 1.0, 200)

        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_t = y_true[indices]
            y_s = y_scores[indices]
            if len(np.unique(y_t)) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_t, y_s)
            interp_tpr = np.interp(fpr_grid, fpr_b, tpr_b)
            interp_tpr[0] = 0.0
            boot_aucs.append(interp_tpr)

        boot_aucs = np.array(boot_aucs)
        lower_idx = int((1 - confidence_level) / 2 * len(boot_aucs))
        upper_idx = int((1 + confidence_level) / 2 * len(boot_aucs) - 1)
        lower_idx = max(0, min(lower_idx, len(boot_aucs) - 1))
        upper_idx = max(0, min(upper_idx, len(boot_aucs) - 1))

        sorted_aucs = np.sort(boot_aucs, axis=0)
        lower_band = sorted_aucs[lower_idx]
        upper_band = sorted_aucs[upper_idx]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.fill_between(fpr_grid, lower_band, upper_band, alpha=0.2, color="#3498db",
                         label=f"{int(confidence_level * 100)}% CI")
        ax.plot(fpr, tpr, color="#2c3e50", linewidth=2,
                label=f"ROC (AUC = {roc_auc_val:.3f})")
        ax.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", linewidth=1,
                label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve",
        filename: str = "pr_curve",
    ) -> plt.Figure:
        """
        Plot precision-recall curve.

        Args:
            y_true: Ground-truth binary labels.
            y_scores: Predicted probabilities.
            title: Plot title.
            filename: Output filename.

        Returns:
            matplotlib Figure object.
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        prevalence = np.mean(y_true) if len(y_true) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color="#2c3e50", linewidth=2,
                label=f"PR (AP = {ap:.3f})")
        ax.axhline(y=prevalence, color="#95a5a6", linestyle="--", linewidth=1,
                   label=f"Baseline (prevalence = {prevalence:.3f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=10)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        filename: str = "calibration_curve",
    ) -> plt.Figure:
        """
        Plot calibration (reliability) curve.

        Args:
            y_true: Ground-truth binary labels.
            y_scores: Predicted probabilities.
            n_bins: Number of bins for calibration.
            title: Plot title.
            filename: Output filename.

        Returns:
            matplotlib Figure object.
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = []
        bin_means = []
        bin_counts = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_scores >= lo) & (y_scores < hi)
            if i == n_bins - 1:
                mask = (y_scores >= lo) & (y_scores <= hi)
            count = np.sum(mask)
            if count > 0:
                bin_centers.append((lo + hi) / 2.0)
                bin_means.append(np.mean(y_true[mask]))
                bin_counts.append(count)

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_counts = np.array(bin_counts, dtype=float)

        fraction_of_positives = bin_means
        mean_predicted = bin_centers

        brier = np.mean((y_scores - y_true) ** 2)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(mean_predicted, fraction_of_positives, "s-", color="#2c3e50",
                linewidth=2, markersize=8, label=f"Model (Brier = {brier:.4f})")
        ax.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", linewidth=1,
                label="Perfectly calibrated")
        sc = ax.scatter(mean_predicted, fraction_of_positives, c=bin_counts,
                        cmap="YlOrRd", s=60, zorder=5, edgecolors="black", linewidths=0.5)
        cbar = fig.colorbar(sc, ax=ax, label="Samples per bin")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_decision_curve_analysis(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        title: str = "Decision Curve Analysis",
        filename: str = "decision_curve",
    ) -> plt.Figure:
        """
        Plot decision curve analysis (net benefit vs threshold probability).

        Args:
            y_true: Ground-truth binary labels.
            y_scores: Predicted probabilities.
            thresholds: Array of threshold probabilities. If None, uses
                        100 linearly spaced thresholds in (0.01, 0.99).
            title: Plot title.
            filename: Output filename.

        Returns:
            matplotlib Figure object.
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)

        prevalence = np.mean(y_true) if len(y_true) > 0 else 0.0
        n = len(y_true)

        net_benefits_model = []
        for t in thresholds:
            tp = np.sum((y_scores >= t) & (y_true == 1))
            fp = np.sum((y_scores >= t) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (t / (1 - t)) if t < 1 else 0.0
            net_benefits_model.append(nb)

        net_benefits_all = []
        for t in thresholds:
            nb = prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1 else 0.0
            net_benefits_all.append(nb)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, net_benefits_model, color="#2c3e50", linewidth=2,
                label="Prediction Model")
        ax.plot(thresholds, net_benefits_all, color="#e74c3c", linewidth=1.5,
                linestyle="--", label="Treat All")
        ax.axhline(y=0, color="#95a5a6", linewidth=1, linestyle=":", label="Treat None")
        ax.set_xlabel("Threshold Probability", fontsize=12)
        ax.set_ylabel("Net Benefit", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim([0, 1])
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix",
        normalize: bool = True,
    ) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Class label strings. Defaults to ['No OHCA', 'OHCA'].
            title: Plot title.
            filename: Output filename.
            normalize: If True, normalize rows to show proportions.

        Returns:
            matplotlib Figure object.
        """
        from sklearn.metrics import confusion_matrix

        if labels is None:
            labels = ["No OHCA", "OHCA"]

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            cm_display = cm.astype(float)
            fmt = "d"

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                total = cm.sum(axis=1)[i]
                pct = (count / total * 100) if total > 0 else 0
                ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, color="#555555")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        modality_names: Optional[List[str]] = None,
        title: str = "Cross-Modal Attention",
        filename: str = "attention_heatmap",
    ) -> plt.Figure:
        """
        Plot attention weight heatmap across modalities.

        Args:
            attention_weights: (num_heads, num_modalities, num_modalities) or
                               (num_modalities, num_modalities) array of
                               attention weights.
            modality_names: Names for each modality axis.
            title: Plot title.
            filename: Output filename.

        Returns:
            matplotlib Figure object.
        """
        if modality_names is None:
            modality_names = ["ECG", "Accel", "Gyro", "PPG", "SpO2", "Temp", "Resp",
                              "Demo", "Meds", "Comor", "Labs"]

        if attention_weights.ndim == 3:
            avg_weights = np.mean(attention_weights, axis=0)
        else:
            avg_weights = attention_weights

        n = avg_weights.shape[0]
        modality_names = modality_names[:n]

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            avg_weights,
            annot=True,
            fmt=".3f",
            cmap="magma",
            xticklabels=modality_names,
            yticklabels=modality_names,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_xlabel("Key Modality", fontsize=12)
        ax.set_ylabel("Query Modality", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_survival_curves(
        self,
        survival_probs: np.ndarray,
        time_bins: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Survival Curves",
        filename: str = "survival_curves",
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> plt.Figure:
        """
        Plot Kaplan-Meier-style survival curves.

        Args:
            survival_probs: (num_curves, num_time_bins) array of survival
                            probabilities.
            time_bins: (num_time_bins,) array of time bin edges (hours).
            labels: Label for each curve.
            title: Plot title.
            filename: Output filename.
            confidence_intervals: Tuple of (lower, upper) arrays with same
                                  shape as survival_probs for CI bands.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette("Set2", n_colors=survival_probs.shape[0])

        for i in range(survival_probs.shape[0]):
            ax.step(time_bins, survival_probs[i], where="post",
                    linewidth=2, label=labels[i] if labels else f"Group {i + 1}",
                    color=colors[i])
            if confidence_intervals is not None:
                lower, upper = confidence_intervals
                ax.fill_between(
                    time_bins, lower[i], upper[i], step="post",
                    alpha=0.2, color=colors[i],
                )

        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._save(fig, filename)
