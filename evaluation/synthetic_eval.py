"""
Synthetic patient evaluation — tests model accuracy across 500 simulated patients
covering healthy, at-risk, and acute cardiac scenarios.
"""
import sys
sys.path.insert(0, 'src')
import os, json, numpy as np, pandas as pd, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, brier_score_loss, confusion_matrix)
from utils import load_parquet, load_numpy
from losses import GradientReversalLayer

# ── Load real data for feature distributions ──────────────────────────────────
_DROP_COLS = {
    'HRV_DFA_alpha2','HRV_MFDFA_alpha2_Width','HRV_MFDFA_alpha2_Peak',
    'HRV_MFDFA_alpha2_Mean','HRV_MFDFA_alpha2_Max','HRV_MFDFA_alpha2_Delta',
    'HRV_MFDFA_alpha2_Asymmetry','HRV_MFDFA_alpha2_Fluctuation','HRV_MFDFA_alpha2_Increment',
}

features_df = load_parquet('data/processed/features.parquet')
meta_df = load_parquet('data/processed/cohort_meta.parquet')
features_df['patient_id'] = features_df['patient_id'].astype(str)
meta_df['patient_id'] = meta_df['patient_id'].astype(str)
features_df = features_df.merge(
    meta_df[['patient_id','acuity_score','community_likeness','importance_weight','icu_type']],
    on='patient_id', how='left')
features_df['icu_domain'] = (
    features_df.get('icu_type', pd.Series('unknown', index=features_df.index))
    .isin(['SICU','MICU','CCU','CSRU','TSICU']).astype(np.int32))

num_cols = features_df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    med = features_df[col].median()
    features_df[col] = features_df[col].fillna(med if np.isfinite(med) else 0.0)

features_df['is_event'] = features_df['event_type'].isin(['MI','ARREST'])
pos = features_df[features_df['is_event']]
neg = features_df[~features_df['is_event']]

with open('models/cvd_risk_v2/feature_columns.json') as f:
    feat_cols_v2 = json.load(f)
with open('models/cvd_risk_v3/feature_columns.json') as f:
    feat_cols_v3 = json.load(f)

# Use V2 columns (superset) for generating features
feat_cols = feat_cols_v2  # includes acuity_score etc.

# ── Compute realistic distributions from real data ───────────────────────────
pos_stats = {}
neg_stats = {}
for col in feat_cols:
    if col in pos.columns and pd.api.types.is_numeric_dtype(pos[col]):
        pos_stats[col] = (pos[col].mean(), pos[col].std())
        neg_stats[col] = (neg[col].mean(), neg[col].std())
    else:
        # Fallback: use 0 for missing columns
        pos_stats[col] = (0.0, 1.0)
        neg_stats[col] = (0.0, 1.0)


# ── Synthetic PPG generator ──────────────────────────────────────────────────
def generate_ppg(length=7500, heart_rate=72, noise_std=0.1, is_cardiac=False, rng=None):
    """Generate a realistic synthetic PPG signal."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(length) / 62.5  # 62.5 Hz
    beat_period = 60.0 / heart_rate
    phase = (t % beat_period) / beat_period
    ppg = np.zeros(length, dtype=np.float32)

    # Systolic peak
    systolic_mask = phase < 0.3
    ppg[systolic_mask] = np.sin(np.pi * phase[systolic_mask] / 0.3)

    # Dicrotic notch
    notch_mask = (phase >= 0.3) & (phase < 0.5)
    ppg[notch_mask] = 0.3 * np.sin(np.pi * (phase[notch_mask] - 0.3) / 0.2)

    if is_cardiac:
        ppg *= rng.uniform(0.3, 0.8)
        arrhythmia = rng.normal(0, 0.3, length).astype(np.float32)
        ppg += np.convolve(arrhythmia, np.ones(50)/50, mode='same')
        ppg += 0.2 * np.sin(2 * np.pi * t / 10)
    else:
        ppg *= rng.uniform(0.8, 1.2)

    ppg += rng.normal(0, noise_std, length).astype(np.float32)
    ppg = (ppg - ppg.mean()) / max(ppg.std(), 1e-6)
    return ppg.astype(np.float32)


# ── Patient profile generators ───────────────────────────────────────────────
def _sample_features(feat_cols, pos_stats, neg_stats, blend, rng,
                     override=None, override_range=None):
    """Sample features given a blend factor (0=neg, 1=pos)."""
    features = {}
    for col in feat_cols:
        mu_p, sigma_p = pos_stats[col]
        mu_n, sigma_n = neg_stats[col]
        mu = mu_n * (1 - blend) + mu_p * blend
        sigma = max(sigma_n, sigma_p) * 0.9
        if sigma > 0 and np.isfinite(sigma):
            features[col] = rng.normal(mu, sigma)
        else:
            features[col] = mu
    if override:
        for k, v in override.items():
            if k in features:
                features[k] = v
    if override_range:
        for k, (lo, hi) in override_range.items():
            if k in features:
                features[k] = rng.uniform(lo, hi)
    return features

def gen_healthy_young(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, 0.0, rng,
        override={'horizon_hours': 0.0, 'signal_length': 7500},
        override_range={'sqi': (0.6, 0.95)}), 0.0, 'healthy_young'

def gen_healthy_elderly(rng):
    f = _sample_features(feat_cols, pos_stats, neg_stats, 0.0, rng,
        override={'horizon_hours': 0.0, 'signal_length': 7500},
        override_range={'sqi': (0.5, 0.85)})
    # Slight HRV shift for elderly
    for k in f:
        if 'MeanNN' in k or 'SDNN' in k:
            f[k] *= rng.uniform(0.7, 0.9)
    return f, 0.0, 'healthy_elderly'

def gen_risk_factor(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, rng.uniform(0.2, 0.5), rng,
        override={'signal_length': 7500},
        override_range={'horizon_hours': (0, 5), 'sqi': (0.4, 0.7)}), rng.uniform(0.0, 0.3), 'risk_factor'

def gen_pre_mi(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, rng.uniform(0.5, 0.8), rng,
        override={'signal_length': 7500},
        override_range={'horizon_hours': (2, 12), 'sqi': (0.3, 0.6)}), rng.uniform(0.5, 0.9), 'pre_mi'

def gen_acute_mi(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, 1.0, rng,
        override={'signal_length': 7500},
        override_range={'horizon_hours': (0, 6), 'sqi': (0.2, 0.5)}), 1.0, 'acute_mi'

def gen_cardiac_arrest(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, 1.0, rng,
        override={'signal_length': 7500},
        override_range={'horizon_hours': (0, 3), 'sqi': (0.1, 0.4)}), 1.0, 'cardiac_arrest'

def gen_icu_stable(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, rng.uniform(0.1, 0.3), rng,
        override={'signal_length': 7500},
        override_range={'horizon_hours': (0, 8), 'sqi': (0.35, 0.7)}), 0.0, 'icu_stable'

def gen_wearable_healthy(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, 0.0, rng,
        override={'horizon_hours': 0.0, 'signal_length': 1500},
        override_range={'sqi': (0.5, 0.9)}), 0.0, 'wearable_healthy'

def gen_wearable_atrisk(rng):
    return _sample_features(feat_cols, pos_stats, neg_stats, rng.uniform(0.15, 0.45), rng,
        override={'signal_length': 1500},
        override_range={'horizon_hours': (0, 6), 'sqi': (0.4, 0.75)}), rng.uniform(0.2, 0.7), 'wearable_atrisk'


# ── Generate synthetic cohort ────────────────────────────────────────────────
def generate_cohort(n_patients=500, seed=42):
    """Generate a diverse synthetic patient cohort."""
    rng = np.random.default_rng(seed)

    profiles = [
        (gen_healthy_young,     0.15),
        (gen_healthy_elderly,   0.10),
        (gen_risk_factor,       0.15),
        (gen_pre_mi,            0.10),
        (gen_acute_mi,          0.15),
        (gen_cardiac_arrest,    0.05),
        (gen_icu_stable,        0.10),
        (gen_wearable_healthy,  0.10),
        (gen_wearable_atrisk,   0.10),
    ]

    all_features = []
    all_labels = []
    all_ppg = []
    all_profiles = []
    all_patient_ids = []

    for patient_idx in range(n_patients):
        r = rng.random()
        cumsum = 0
        selected_fn = profiles[-1][0]
        for fn, frac in profiles:
            cumsum += frac
            if r < cumsum:
                selected_fn = fn
                break

        (feat_dict, label, profile_name) = selected_fn(rng)
        all_features.append(feat_dict)
        all_labels.append(label)
        all_profiles.append(profile_name)
        all_patient_ids.append(f'SYN_{patient_idx:04d}')

        is_cardiac = label > 0.5
        hr = rng.uniform(50, 60) if not is_cardiac else rng.uniform(90, 140)
        noise = rng.uniform(0.02, 0.08) if not is_cardiac else rng.uniform(0.05, 0.2)
        ppg = generate_ppg(
            length=int(feat_dict.get('signal_length', 7500)),
            heart_rate=hr, noise_std=noise, is_cardiac=is_cardiac, rng=rng)
        all_ppg.append(ppg)

    return all_features, all_labels, all_ppg, all_profiles, all_patient_ids


# ── Build model input arrays ─────────────────────────────────────────────────
def build_model_inputs(features_list, ppg_list, model_feat_cols, target_length=7500):
    """Convert feature dicts + PPG arrays into model-ready numpy arrays."""
    n = len(features_list)
    X_feat = np.zeros((n, len(model_feat_cols)), dtype=np.float32)
    for i, feat_dict in enumerate(features_list):
        for j, col in enumerate(model_feat_cols):
            X_feat[i, j] = feat_dict.get(col, 0.0)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    X_ppg = np.zeros((n, target_length), dtype=np.float32)
    for i, ppg in enumerate(ppg_list):
        length = min(len(ppg), target_length)
        X_ppg[i, :length] = ppg[:length]
    X_ppg = X_ppg[..., np.newaxis]
    X_ppg = np.nan_to_num(X_ppg, nan=0.0, posinf=0.0, neginf=0.0)

    return X_ppg, X_feat


# ── Model evaluation ─────────────────────────────────────────────────────────
def evaluate_model_on_cohort(model_path, model_feat_cols, X_ppg, X_feat, y_true,
                              profiles, threshold=0.5, model_name="model"):
    """Evaluate a model on a synthetic cohort."""
    model = tf.keras.models.load_model(model_path, compile=False,
        custom_objects={'GradientReversalLayer': GradientReversalLayer})

    preds = model({'ppg_input': X_ppg, 'feature_input': X_feat}, training=False)
    y_prob = preds[0].numpy().ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # Binarize labels for classification metrics (some profiles have continuous labels)
    y_true_binary = (y_true >= 0.5).astype(int)
    auroc = roc_auc_score(y_true_binary, y_prob) if len(np.unique(y_true_binary)) > 1 else float('nan')
    y_true = y_true_binary  # use binary for all classification metrics
    results = {
        'model': model_name,
        'threshold': threshold,
        'n_patients': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int(len(y_true) - y_true.sum()),
        'auroc': auroc,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'mean_prob_pos': float(y_prob[y_true == 1].mean()),
        'mean_prob_neg': float(y_prob[y_true == 0].mean()),
    }

    profile_results = {}
    for pname in sorted(set(profiles)):
        mask = np.array([p == pname for p in profiles])
        if mask.sum() < 3:
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        ypred = y_pred[mask]
        try:
            auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float('nan')
        except:
            auc = float('nan')
        profile_results[pname] = {
            'n': int(mask.sum()),
            'true_label': float(yt.mean()),
            'pred_prob_mean': float(yp.mean()),
            'pred_prob_std': float(yp.std()),
            'auroc': auc,
            'accuracy': float((yt == ypred).mean()),
            'precision': float(precision_score(yt, ypred, zero_division=0)),
            'recall': float(recall_score(yt, ypred, zero_division=0)),
            'f1': float(f1_score(yt, ypred, zero_division=0)),
        }
    results['profiles'] = profile_results
    return results, y_prob


# ── Noise robustness test ────────────────────────────────────────────────────
def test_noise_robustness(model_path, X_ppg_base, X_feat, y_true, threshold=0.5):
    model = tf.keras.models.load_model(model_path, compile=False,
        custom_objects={'GradientReversalLayer': GradientReversalLayer})
    y_true_binary = (y_true >= 0.5).astype(int)
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results = []
    rng = np.random.default_rng(42)

    for noise_std in noise_levels:
        X_noisy = X_ppg_base + rng.normal(0, noise_std, X_ppg_base.shape).astype(np.float32)
        X_noisy = np.nan_to_num(X_noisy, nan=0.0, posinf=0.0, neginf=0.0)
        preds = model({'ppg_input': X_noisy, 'feature_input': X_feat}, training=False)
        y_prob = preds[0].numpy().ravel()
        y_pred = (y_prob >= threshold).astype(int)
        try:
            auroc = roc_auc_score(y_true_binary, y_prob) if len(np.unique(y_true_binary)) > 1 else float('nan')
        except:
            auroc = float('nan')
        results.append({
            'noise_std': noise_std,
            'auroc': auroc,
            'accuracy': float((y_true_binary == y_pred).mean()),
            'precision': float(precision_score(y_true_binary, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true_binary, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true_binary, y_pred, zero_division=0)),
        })
    return results


# ── Feature importance test ──────────────────────────────────────────────────
def test_feature_perturbation(model_path, X_ppg, X_feat, y_true, model_feat_cols, threshold=0.5):
    model = tf.keras.models.load_model(model_path, compile=False,
        custom_objects={'GradientReversalLayer': GradientReversalLayer})
    y_true_binary = (y_true >= 0.5).astype(int)

    preds = model({'ppg_input': X_ppg, 'feature_input': X_feat}, training=False)
    y_prob_base = preds[0].numpy().ravel()
    base_acc = float((y_true_binary == (y_prob_base >= threshold).astype(int)).mean())

    feature_importance = []
    rng = np.random.default_rng(42)

    for j, col in enumerate(model_feat_cols):
        X_corrupted = X_feat.copy()
        X_corrupted[:, j] = rng.uniform(
            X_feat[:, j].min(), X_feat[:, j].max(), size=len(X_feat)).astype(np.float32)
        preds = model({'ppg_input': X_ppg, 'feature_input': X_corrupted}, training=False)
        y_prob = preds[0].numpy().ravel()
        acc = float((y_true_binary == (y_prob >= threshold).astype(int)).mean())
        feature_importance.append({
            'feature': col,
            'accuracy_drop': base_acc - acc,
            'accuracy_after': acc,
        })
    feature_importance.sort(key=lambda x: x['accuracy_drop'], reverse=True)
    return feature_importance[:15], base_acc


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    N_PATIENTS = 500
    logger.info('Generating %d synthetic patients...', N_PATIENTS)
    features_list, labels, ppg_list, profiles, patient_ids = generate_cohort(N_PATIENTS)

    y_true = np.array(labels, dtype=np.float32)
    logger.info('Cohort: %d patients, %d positive (%.1f%%)',
                N_PATIENTS, int(y_true.sum()), y_true.mean() * 100)

    from collections import Counter
    for p, c in sorted(Counter(profiles).items()):
        logger.info('  %s: %d (%.1f%%)', p, c, c / N_PATIENTS * 100)

    all_results = {}

    # ── V2 evaluation ────────────────────────────────────────────────────────
    logger.info('='*60)
    logger.info('EVALUATING V2 MODEL')
    logger.info('='*60)
    X_ppg_v2, X_feat_v2 = build_model_inputs(features_list, ppg_list, feat_cols_v2)

    for t_name, t_val in [('t05', 0.05), ('t10', 0.10), ('t50', 0.50)]:
        key = f'v2_{t_name}'
        res, prob = evaluate_model_on_cohort(
            'models/cvd_risk_v2/best_model.keras', feat_cols_v2,
            X_ppg_v2, X_feat_v2, y_true, profiles, threshold=t_val, model_name=key)
        all_results[key] = res
        logger.info('%s: AUROC=%.4f Acc=%.1f%% Prec=%.1f%% Rec=%.1f%% F1=%.3f',
                     key, res['auroc'], res['accuracy']*100, res['precision']*100,
                     res['recall']*100, res['f1'])

    # ── V3 evaluation ────────────────────────────────────────────────────────
    logger.info('='*60)
    logger.info('EVALUATING V3 MODEL')
    logger.info('='*60)
    X_ppg_v3, X_feat_v3 = build_model_inputs(features_list, ppg_list, feat_cols_v3)

    for t_name, t_val in [('t05', 0.05), ('t10', 0.10)]:
        key = f'v3_{t_name}'
        res, prob = evaluate_model_on_cohort(
            'models/cvd_risk_v3/best_model.keras', feat_cols_v3,
            X_ppg_v3, X_feat_v3, y_true, profiles, threshold=t_val, model_name=key)
        all_results[key] = res
        logger.info('%s: AUROC=%.4f Acc=%.1f%% Prec=%.1f%% Rec=%.1f%% F1=%.3f',
                     key, res['auroc'], res['accuracy']*100, res['precision']*100,
                     res['recall']*100, res['f1'])

    # ── Noise robustness ─────────────────────────────────────────────────────
    logger.info('='*60)
    logger.info('NOISE ROBUSTNESS TEST')
    logger.info('='*60)
    all_results['noise_v2'] = test_noise_robustness(
        'models/cvd_risk_v2/best_model.keras', X_ppg_v2, X_feat_v2, y_true, threshold=0.05)
    all_results['noise_v3'] = test_noise_robustness(
        'models/cvd_risk_v3/best_model.keras', X_ppg_v3, X_feat_v3, y_true, threshold=0.10)

    for nr in all_results['noise_v2']:
        logger.info('  V2 noise=%.2f: AUROC=%.4f Acc=%.1f%% F1=%.3f',
                     nr['noise_std'], nr['auroc'], nr['accuracy']*100, nr['f1'])

    # ── Feature importance ───────────────────────────────────────────────────
    logger.info('='*60)
    logger.info('FEATURE IMPORTANCE (V2)')
    logger.info('='*60)
    feat_imp, base_acc = test_feature_perturbation(
        'models/cvd_risk_v2/best_model.keras', X_ppg_v2, X_feat_v2, y_true, feat_cols_v2, threshold=0.05)
    all_results['feature_importance_v2'] = feat_imp
    all_results['base_accuracy_v2'] = base_acc
    for fi in feat_imp[:10]:
        logger.info('  %s: drop=%.4f', fi['feature'], fi['accuracy_drop'])

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs('models/cvd_risk_v4', exist_ok=True)
    with open('models/cvd_risk_v4/synthetic_eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info('Results saved to models/cvd_risk_v4/synthetic_eval_results.json')
