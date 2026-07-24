#!/usr/bin/env python3
"""
Simplified ISEF Evaluation - Runs quickly without training baselines
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix,
                             average_precision_score, log_loss, brier_score_loss, matthews_corrcoef)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = "isef_evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

def comprehensive_metrics(y_true, y_prob, threshold=0.45):
    y_pred = (y_prob >= threshold).astype(int)
    
    n_bootstrap = 1000
    metrics = {}
    
    # Point estimates
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    metrics['auprc'] = average_precision_score(y_true, y_prob)
    metrics['fpr'] = 1 - metrics['specificity']
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['prevalence'] = np.mean(y_true)
    
    # Bootstrap CIs
    boot_metrics = {k: [] for k in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc_roc', 'fpr']}
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]
        y_pred_boot = (y_prob_boot >= threshold).astype(int)
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        boot_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        boot_metrics['sensitivity'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        boot_metrics['specificity'].append(recall_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
        boot_metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        boot_metrics['auc_roc'].append(roc_auc_score(y_true_boot, y_prob_boot))
        boot_metrics['fpr'].append(1 - recall_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
    
    for key in boot_metrics:
        if boot_metrics[key]:
            metrics[f'{key}_ci_lower'] = np.percentile(boot_metrics[key], 2.5)
            metrics[f'{key}_ci_upper'] = np.percentile(boot_metrics[key], 97.5)
    
    metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    metrics['log_loss'] = log_loss(y_true, y_prob)
    metrics['youden_j'] = metrics['sensitivity'] + metrics['specificity'] - 1
    metrics['markmanship'] = metrics['sensitivity'] * metrics['specificity']
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn + 1e-10)
    metrics['positive_lr'] = metrics['sensitivity'] / (1 - metrics['specificity'] + 1e-10)
    metrics['negative_lr'] = (1 - metrics['sensitivity']) / (metrics['specificity'] + 1e-10)
    metrics['nnt'] = 1 / (metrics['sensitivity'] - metrics['fpr'] + 1e-10)
    
    return metrics

def threshold_analysis(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        fpr = 1 - specificity
        f1 = f1_score(y_true, y_pred, zero_division=0)
        youden = sensitivity + specificity - 1
        cost = fn * 10 + fp
        
        results.append({
            'threshold': thr,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'f1': f1,
            'youden_j': youden,
            'cost': cost,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        })
    
    return results

def modality_ablation(model, X_test, y_test):
    results = {}
    
    prob_full = model.predict(X_test, verbose=0).flatten()
    results['full'] = {
        'auc': roc_auc_score(y_test, prob_full),
        'accuracy': accuracy_score(y_test, (prob_full >= 0.45).astype(int)),
        'sensitivity': recall_score(y_test, (prob_full >= 0.45).astype(int)),
        'fpr': 1 - recall_score(y_test, (prob_full >= 0.45).astype(int), pos_label=0)
    }
    
    for mod in ['ecg', 'acc', 'bio']:
        X_ablated = {k: v.copy() for k, v in X_test.items()}
        X_ablated[f'{mod}_input'] = np.zeros_like(X_test[f'{mod}_input'])
        
        prob_ablated = model.predict(X_ablated, verbose=0).flatten()
        
        results[f'no_{mod}'] = {
            'auc': roc_auc_score(y_test, prob_ablated),
            'accuracy': accuracy_score(y_test, (prob_ablated >= 0.45).astype(int)),
            'sensitivity': recall_score(y_test, (prob_ablated >= 0.45).astype(int)),
            'fpr': 1 - recall_score(y_test, (prob_ablated >= 0.45).astype(int), pos_label=0)
        }
    
    return results

def robustness_test(model, X_test, y_test):
    noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = {mod: [] for mod in ['ecg', 'acc', 'bio']}
    
    for noise in noise_levels:
        for mod in ['ecg', 'acc', 'bio']:
            X_noisy = {k: v.copy() for k, v in X_test.items()}
            X_noisy[f'{mod}_input'] = X_noisy[f'{mod}_input'] + np.random.randn(*X_noisy[f'{mod}_input'].shape) * noise
            prob = model.predict(X_noisy, verbose=0).flatten()
            results[mod].append({
                'noise': noise,
                'auc': roc_auc_score(y_test, prob),
                'accuracy': accuracy_score(y_test, (prob >= 0.45).astype(int))
            })
    
    return results

def subgroup_analysis(y_true, y_prob, X_test, threshold=0.45):
    bio = X_test['bio_input']
    y_pred = (y_prob >= threshold).astype(int)
    
    subgroups = {
        'age_<65': bio[:, 0] < 65,
        'age_65-75': (bio[:, 0] >= 65) & (bio[:, 0] < 75),
        'age_>75': bio[:, 0] >= 75,
        'male': bio[:, 1] > 0.5,
        'female': bio[:, 1] <= 0.5,
        'low_spo2': bio[:, 5] < 95,
        'normal_spo2': bio[:, 5] >= 95,
        'high_bp': bio[:, 3] > 140,
        'normal_bp': bio[:, 3] <= 140
    }
    
    results = {}
    for name, mask in subgroups.items():
        if np.sum(mask) < 10:
            continue
            
        y_true_sub = y_true[mask]
        y_prob_sub = y_prob[mask]
        y_pred_sub = y_pred[mask]
        
        if len(np.unique(y_true_sub)) < 2:
            continue
        
        results[name] = {
            'n': int(np.sum(mask)),
            'accuracy': accuracy_score(y_true_sub, y_pred_sub),
            'sensitivity': recall_score(y_true_sub, y_pred_sub, zero_division=0),
            'specificity': recall_score(y_true_sub, y_pred_sub, pos_label=0, zero_division=0),
            'fpr': 1 - recall_score(y_true_sub, y_pred_sub, pos_label=0, zero_division=0),
            'auc': roc_auc_score(y_true_sub, y_prob_sub)
        }
    
    return results

def calibration_analysis(y_true, y_prob, n_bins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_correct = np.histogram(y_prob[y_true == 1], bins=n_bins, range=(0, 1))[0]
    bin_acc = bin_correct / (bin_totals + 1e-10)
    bin_conf = np.histogram(y_prob, bins=n_bins, range=(0, 1))[1][:-1] + 0.05
    ece = np.mean(np.abs(bin_acc - bin_conf) * bin_totals / len(y_true))
    
    return {
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist(),
        'ece': float(ece),
        'brier_score': brier_score_loss(y_true, y_prob)
    }

def create_figures(results, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Performance Overview
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_prob'])
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {results["metrics"]["auc_roc"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # PR Curve
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(results['y_true'], results['y_prob'])
    ax2.plot(recall, precision, 'r-', linewidth=2, label=f'AP = {results["metrics"]["auprc"]:.3f}')
    ax2.fill_between(recall, precision, alpha=0.1, color='red')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision (PPV)', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=11)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0, 1.05])
    
    # Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    y_pred = (results['y_prob'] >= 0.45).astype(int)
    cm = confusion_matrix(results['y_true'], y_pred)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    classes = ['No Arrest', 'Arrest']
    tick_marks = np.arange(len(classes))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(classes, fontsize=11)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(classes, fontsize=11)
    
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max()/2 else "black"
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color=color, fontsize=14)
    
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    
    # Calibration Plot
    ax4 = fig.add_subplot(gs[1, 0])
    fraction_pos, mean_pred = calibration_curve(results['y_true'], results['y_prob'], n_bins=10)
    ax4.plot(mean_pred, fraction_pos, 's-', linewidth=2, markersize=8, label='Model')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
    ax4.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax4.set_ylabel('Fraction of Positives', fontsize=12)
    ax4.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=11)
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.02])
    
    # Threshold Sweep
    ax5 = fig.add_subplot(gs[1, 1])
    thresholds = np.arange(0.1, 0.9, 0.01)
    sens_list, spec_list, fpr_list = [], [], []
    for thr in thresholds:
        y_pred_thr = (results['y_prob'] >= thr).astype(int)
        sens_list.append(recall_score(results['y_true'], y_pred_thr, zero_division=0))
        spec_list.append(recall_score(results['y_true'], y_pred_thr, pos_label=0, zero_division=0))
        fpr_list.append(1 - spec_list[-1])
    
    ax5.plot(thresholds, sens_list, 'g-', linewidth=2, label='Sensitivity')
    ax5.plot(thresholds, fpr_list, 'r-', linewidth=2, label='FPR')
    ax5.axvline(x=0.45, color='gray', linestyle='--', linewidth=1.5, label='Threshold = 0.45')
    ax5.set_xlabel('Decision Threshold', fontsize=12)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Threshold Analysis', fontsize=14, fontweight='bold')
    ax5.legend(loc='center right', fontsize=11)
    ax5.set_xlim([0.08, 0.92])
    ax5.set_ylim([-0.02, 1.05])
    
    # Metrics Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS (95% CI)
    ════════════════════════════════
    Accuracy:     {results['metrics']['accuracy']:.1%} [{results['metrics']['accuracy_ci_lower']:.1%}, {results['metrics']['accuracy_ci_upper']:.1%}]
    Sensitivity:  {results['metrics']['sensitivity']:.1%} [{results['metrics']['sensitivity_ci_lower']:.1%}, {results['metrics']['sensitivity_ci_upper']:.1%}]
    Specificity:  {results['metrics']['specificity']:.1%} [{results['metrics']['specificity_ci_lower']:.1%}, {results['metrics']['specificity_ci_upper']:.1%}]
    FPR:          {results['metrics']['fpr']:.1%} [{results['metrics']['fpr_ci_lower']:.1%}, {results['metrics']['fpr_ci_upper']:.1%}]
    AUC-ROC:      {results['metrics']['auc_roc']:.3f} [{results['metrics']['auc_roc_ci_lower']:.3f}, {results['metrics']['auc_roc_ci_upper']:.3f}]
    AUPRC:        {results['metrics']['auprc']:.3f}
    F1 Score:     {results['metrics']['f1']:.3f}
    Youden's J:   {results['metrics']['youden_j']:.3f}
    Markmanship:  {results['metrics']['markmanship']:.3f}
    DOR:          {results['metrics']['diagnostic_odds_ratio']:.1f}
    +LR:          {results['metrics']['positive_lr']:.2f}
    -LR:          {results['metrics']['negative_lr']:.2f}
    Brier Score:  {results['metrics']['brier_score']:.4f}
    ECE:          {results['calibration']['ece']:.4f}
    """
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig(f'{save_path}/performance_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Ablation and Robustness
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ablation = results['ablation']
    models_list = list(ablation.keys())
    aucs = [ablation[m]['auc'] for m in models_list]
    colors = ['green' if 'full' in m else 'steelblue' for m in models_list]
    bars = ax.bar(models_list, aucs, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Modality Ablation Study', fontsize=14, fontweight='bold')
    ax.set_ylim([0.85, 1.0])
    ax.tick_params(axis='x', rotation=45)
    
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[1]
    robustness = results['robustness']
    for mod in ['ecg', 'acc', 'bio']:
        noises = [r['noise'] for r in robustness[mod]]
        aucs = [r['auc'] for r in robustness[mod]]
        ax.plot(noises, aucs, 'o-', linewidth=2, markersize=6, label=mod.upper())
    
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, label='95% AUC')
    ax.set_xlabel('Noise Level (σ)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Robustness to Signal Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim([-0.02, 0.52])
    ax.set_ylim([0.85, 1.02])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/ablation_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Subgroup Analysis
    if results.get('subgroups'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        subgroups = list(results['subgroups'].keys())
        sensitivities = [results['subgroups'][s]['sensitivity'] for s in subgroups]
        fprs = [results['subgroups'][s]['fpr'] for s in subgroups]
        
        x = np.arange(len(subgroups))
        width = 0.35
        ax.bar(x - width/2, sensitivities, width, label='Sensitivity', color='green', edgecolor='black')
        ax.bar(x + width/2, fprs, width, label='FPR', color='red', edgecolor='black')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance by Patient Subgroup', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subgroups, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='99% Sensitivity')
        ax.axhline(y=0.03, color='red', linestyle='--', alpha=0.5, label='3% FPR')
        
        ax = axes[1]
        aucs = [results['subgroups'][s]['auc'] for s in subgroups]
        ax.barh(subgroups, aucs, color='steelblue', edgecolor='black')
        ax.set_xlabel('AUC-ROC', fontsize=12)
        ax.set_title('AUC by Patient Subgroup', fontsize=14, fontweight='bold')
        ax.set_xlim([0.85, 1.0])
        
        for i, v in enumerate(aucs):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/subgroup_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Figure 4: Error Analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    cm = confusion_matrix(results['y_true'], (results['y_prob'] >= 0.45).astype(int))
    tn, fp, fn, tp = cm.ravel()
    categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [tn, fp, fn, tp]
    colors = ['green', 'orange', 'red', 'blue']
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Outcomes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontsize=11)
    
    ax = axes[1]
    ax.hist(results['y_prob'][results['y_true'] == 0], bins=30, alpha=0.6, color='blue', label='No Arrest', density=True)
    ax.hist(results['y_prob'][results['y_true'] == 1], bins=30, alpha=0.6, color='red', label='Arrest', density=True)
    ax.axvline(x=0.45, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 4 publication-quality figures to {save_path}/")

def run_evaluation():
    print("=" * 70)
    print("ISEF COMPREHENSIVE EVALUATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n[1/8] Loading data...")
    X_test = {
        'ecg_input': np.load('data/X_ecg_test.npy'),
        'acc_input': np.load('data/X_acc_test.npy'),
        'bio_input': np.load('data/X_bio_test.npy')
    }
    y_test = np.load('data/y_test.npy')
    
    print("[2/8] Loading model...")
    model = tf.keras.models.load_model('model_v2.keras', compile=False)
    
    print("[3/8] Generating predictions...")
    y_prob = model.predict(X_test, verbose=0).flatten()
    
    results = {'y_true': y_test, 'y_prob': y_prob, 'timestamp': datetime.now().isoformat()}
    
    print("[4/8] Computing comprehensive metrics...")
    metrics = comprehensive_metrics(y_test, y_prob, threshold=0.45)
    results['metrics'] = metrics
    
    print(f"\n{'='*50}")
    print("KEY PERFORMANCE METRICS (95% CI)")
    print(f"{'='*50}")
    print(f"Accuracy:     {metrics['accuracy']:.1%} [{metrics['accuracy_ci_lower']:.1%}, {metrics['accuracy_ci_upper']:.1%}]")
    print(f"Sensitivity:  {metrics['sensitivity']:.1%} [{metrics['sensitivity_ci_lower']:.1%}, {metrics['sensitivity_ci_upper']:.1%}]")
    print(f"Specificity:  {metrics['specificity']:.1%} [{metrics['specificity_ci_lower']:.1%}, {metrics['specificity_ci_upper']:.1%}]")
    print(f"FPR:          {metrics['fpr']:.1%} [{metrics['fpr_ci_lower']:.1%}, {metrics['fpr_ci_upper']:.1%}]")
    print(f"AUC-ROC:      {metrics['auc_roc']:.3f} [{metrics['auc_roc_ci_lower']:.3f}, {metrics['auc_roc_ci_upper']:.3f}]")
    print(f"{'='*50}")
    
    print("\n[5/8] Optimizing thresholds...")
    threshold_results = threshold_analysis(y_test, y_prob)
    results['thresholds'] = threshold_results
    
    optimal = {
        'best_youden': max(threshold_results, key=lambda x: x['youden_j']),
        'high_sensitivity': max([t for t in threshold_results if t['sensitivity'] >= 0.99], 
                               key=lambda x: x['fpr'], default=None),
        'balanced': max(threshold_results, key=lambda x: x['f1'])
    }
    results['optimal_thresholds'] = optimal
    
    print(f"  Best Youden's J: {optimal['best_youden']['threshold']:.2f} (Sens: {optimal['best_youden']['sensitivity']:.3f}, FPR: {optimal['best_youden']['fpr']:.3f})")
    if optimal['high_sensitivity']:
        print(f"  High Sensitivity: {optimal['high_sensitivity']['threshold']:.2f} (Sens: {optimal['high_sensitivity']['sensitivity']:.3f}, FPR: {optimal['high_sensitivity']['fpr']:.3f})")
    print(f"  Balanced F1: {optimal['balanced']['threshold']:.2f} (F1: {optimal['balanced']['f1']:.3f})")
    
    print("\n[6/8] Running ablation and robustness tests...")
    ablation = modality_ablation(model, X_test, y_test)
    results['ablation'] = ablation
    
    print("\n  Modality Ablation Results:")
    for name, m in ablation.items():
        print(f"    {name:15s} - AUC: {m['auc']:.3f}, Acc: {m['accuracy']:.3f}, Sens: {m['sensitivity']:.3f}")
    
    robustness = robustness_test(model, X_test, y_test)
    results['robustness'] = robustness
    
    print("\n  Robustness Results (AUC at different noise levels):")
    for mod in ['ecg', 'acc', 'bio']:
        print(f"    {mod.upper():4s}: ", end="")
        for r in robustness[mod]:
            print(f"σ={r['noise']:.2f}:{r['auc']:.3f} ", end="")
        print()
    
    print("\n[7/8] Analyzing subgroups and calibration...")
    subgroups = subgroup_analysis(y_test, y_prob, X_test, threshold=0.45)
    results['subgroups'] = subgroups
    
    print("\n  Subgroup Performance:")
    for name, m in subgroups.items():
        print(f"    {name:15s} - Sens: {m['sensitivity']:.3f}, FPR: {m['fpr']:.3f}, AUC: {m['auc']:.3f}")
    
    calibration = calibration_analysis(y_test, y_prob)
    results['calibration'] = calibration
    
    print(f"\n  Calibration: ECE = {calibration['ece']:.4f}, Brier = {calibration['brier_score']:.4f}")
    
    print("\n[8/8] Generating figures and saving results...")
    create_figures(results, f"{RESULTS_DIR}/figures")
    
    with open(f"{RESULTS_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    results = run_evaluation()
