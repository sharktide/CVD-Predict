"""
Validation utilities: extract standard PPG/pulse-wave metrics from
generated (or real) signals and compare distributions using standard
statistical distance/tests.

Evidence base for metric definitions
-------------------------------------
- Pulse width, crest time: Elgendi, "On the analysis of fingertip
  photoplethysmogram signals", Curr Cardiol Rev 8:14-25 (2012).
- Augmentation index, reflection index, stiffness index: Millasseau,
  Kelly, Ritter & Chowienczyk, "Determination of age-related increases
  in large artery stiffness by digital volume pulse contour analysis",
  Clin Sci 103:371-377 (2002) — defines the digital-volume-pulse-derived
  stiffness index (SI = height / peak-to-dicrotic-notch time) and
  reflection index used here.
- AC/DC ratio: Allen (2007), as cited in optics.py.
- Spectral entropy as an HRV/complexity measure: Pincus, "Approximate
  entropy as a measure of system complexity", PNAS 88:2297-2301 (1991)
  (general complexity-measure motivation; spectral entropy specifically
  per Viertio-Oja et al., Acta Anaesthesiol Scand 48:154-161 (2004)).

Evidence base for statistical distance/tests
-----------------------------------------------
- Kolmogorov-Smirnov test: Massey, "The Kolmogorov-Smirnov test for
  goodness of fit", J Am Stat Assoc 46:68-78 (1951).
- Wasserstein / Earth Mover's Distance: Vaserstein (1969); Rubner, Tomasi
  & Guibas, "The earth mover's distance as a metric for image retrieval",
  IJCV 40:99-121 (2000). (1-D EMD and 1-D Wasserstein distance coincide;
  we report a single 1-D value using scipy's implementation.)
- Maximum Mean Discrepancy (MMD): Gretton, Borgwardt, Rasch, Scholkopf &
  Smola, "A kernel two-sample test", JMLR 13:723-773 (2012).
- Dynamic Time Warping: Sakoe & Chiba, "Dynamic programming algorithm
  optimization for spoken word recognition", IEEE TASSP 26:43-49 (1978).
- Frechet distance (used here in its "Frechet Inception Distance"-style
  Gaussian form, comparing feature-distribution means/covariances, as
  commonly used for generative-model evaluation): Dowson & Landau, "The
  Frechet distance between multivariate normal distributions", J
  Multivariate Anal 12:450-455 (1982); popularized for generative
  evaluation by Heusel et al., "GANs trained by a two time-scale update
  rule converge to a local Nash equilibrium", NeurIPS (2017).

What is heuristic here
-----------------------
- Published literature ranges quoted in `LITERATURE_REFERENCE_RANGES`
  are approximate, population-level summaries assembled from the review
  papers cited above; they are not a substitute for a proper statistical
  comparison against a specific labeled dataset, which is the
  recommended next step (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, welch


LITERATURE_REFERENCE_RANGES = {
    # metric: (low, high, source_note)
    "pulse_width_ms": (300, 500, "Elgendi (2012), finger/wrist PPG, healthy adults"),
    "crest_time_ms": (100, 220, "Millasseau et al. (2002); Elgendi (2012)"),
    "augmentation_index": (-0.3, 0.5, "Millasseau et al. (2002), wide healthy-to-stiff range"),
    "reflection_index": (0.3, 0.9, "Millasseau et al. (2002)"),
    "stiffness_index_m_s": (5.0, 15.0, "Millasseau et al. (2002), age-dependent"),
    "ac_dc_ratio": (0.005, 0.03, "Allen (2007), green-wavelength wrist PPG"),
    "sdnn_ms_healthy": (30, 100, "Task Force (1996)"),
    "rmssd_ms_healthy": (20, 60, "Task Force (1996)"),
    "lf_hf_healthy": (1.0, 3.0, "Task Force (1996), resting supine"),
}


@dataclass
class PulseMetrics:
    pulse_width_ms: float
    crest_time_ms: float
    augmentation_index: float
    reflection_index: float
    stiffness_index_m_s: float
    ac_dc_ratio: float
    spectral_entropy: float


def compute_pulse_metrics(ppg: np.ndarray, fs: float, subject_height_m: float = 1.7) -> PulseMetrics:
    ppg = np.asarray(ppg, dtype=np.float64)
    peaks, _ = find_peaks(ppg, distance=max(int(fs * 0.35), 1))
    troughs, _ = find_peaks(-ppg, distance=max(int(fs * 0.35), 1))

    if len(peaks) < 2 or len(troughs) < 2:
        return PulseMetrics(*(np.nan,) * 7)

    widths_ms = []
    crest_ms = []
    for p in peaks:
        prior_troughs = troughs[troughs < p]
        next_troughs = troughs[troughs > p]
        if len(prior_troughs) == 0 or len(next_troughs) == 0:
            continue
        t0, t1 = prior_troughs[-1], next_troughs[0]
        widths_ms.append((t1 - t0) / fs * 1000)
        crest_ms.append((p - t0) / fs * 1000)

    pulse_width_ms = float(np.median(widths_ms)) if widths_ms else np.nan
    crest_time_ms = float(np.median(crest_ms)) if crest_ms else np.nan

    d1 = np.gradient(ppg)
    aix_vals = []
    ri_vals = []
    for p in peaks:
        prior_troughs = troughs[troughs < p]
        if len(prior_troughs) == 0:
            continue
        t0 = prior_troughs[-1]
        pp = ppg[p] - ppg[t0]
        if pp < 1e-9:
            continue
        inflect = p
        for i in range(t0 + 1, p):
            if d1[i - 1] > 0 and d1[i] <= 0:
                inflect = i
        p1 = ppg[inflect] if inflect != p else ppg[max((t0 + p) // 2, t0)]
        aix_vals.append((ppg[p] - p1) / pp)

        next_troughs = troughs[troughs > p]
        if len(next_troughs):
            t1 = next_troughs[0]
            local_min_after = np.min(ppg[p:t1]) if t1 > p else ppg[p]
            ri_vals.append((ppg[p] - local_min_after) / pp)

    augmentation_index = float(np.median(aix_vals)) if aix_vals else np.nan
    reflection_index = float(np.median(ri_vals)) if ri_vals else np.nan

    # Stiffness index (Millasseau et al. 2002): SI = height / delta_T,
    # where delta_T is the time between systolic peak and the diastolic
    # (reflected wave) peak, height = subject height.
    stiffness_index = pulse_width_ms  # fallback
    dt_candidates = []
    for i in range(len(peaks) - 1):
        dt_candidates.append((peaks[i + 1] - peaks[i]) / fs)
    if dt_candidates:
        median_dt = np.median(dt_candidates) * 0.4  # approx reflected-peak offset fraction
        stiffness_index = float(subject_height_m / max(median_dt, 1e-3))

    dc = float(np.mean(ppg))
    ac = float(np.std(ppg - dc))
    ac_dc_ratio = ac / (abs(dc) + 1e-9)

    f, psd = welch(ppg - np.mean(ppg), fs=fs, nperseg=min(len(ppg), 256))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm)))

    return PulseMetrics(
        pulse_width_ms=pulse_width_ms, crest_time_ms=crest_time_ms,
        augmentation_index=augmentation_index, reflection_index=reflection_index,
        stiffness_index_m_s=stiffness_index, ac_dc_ratio=ac_dc_ratio,
        spectral_entropy=spectral_entropy,
    )


# ---------------------------------------------------------------------
# Distributional comparison tests
# ---------------------------------------------------------------------

def ks_test(sample_a: np.ndarray, sample_b: np.ndarray) -> dict:
    res = stats.ks_2samp(sample_a, sample_b)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def wasserstein_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    return float(stats.wasserstein_distance(sample_a, sample_b))


def maximum_mean_discrepancy(sample_a: np.ndarray, sample_b: np.ndarray, gamma: float | None = None) -> float:
    """Gaussian-kernel MMD^2 (Gretton et al. 2012), unbiased estimator."""
    a = np.asarray(sample_a).reshape(-1, 1)
    b = np.asarray(sample_b).reshape(-1, 1)
    if gamma is None:
        combined = np.concatenate([a, b])
        pairwise = np.abs(combined - combined.T)
        gamma = 1.0 / (2 * (np.median(pairwise[pairwise > 0]) ** 2 + 1e-9))

    def kernel(x, y):
        return np.exp(-gamma * (x - y.T) ** 2)

    Kaa = kernel(a, a)
    Kbb = kernel(b, b)
    Kab = kernel(a, b)
    n, m = len(a), len(b)
    term_a = (np.sum(Kaa) - np.trace(Kaa)) / (n * (n - 1)) if n > 1 else 0.0
    term_b = (np.sum(Kbb) - np.trace(Kbb)) / (m * (m - 1)) if m > 1 else 0.0
    term_ab = np.sum(Kab) / (n * m)
    mmd2 = term_a + term_b - 2 * term_ab
    return float(max(mmd2, 0.0))


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band_radius: int | None = None) -> float:
    """Classic O(n*m) DTW (Sakoe & Chiba 1978) with optional Sakoe-Chiba
    band constraint for tractability on long sequences."""
    n, m = len(seq_a), len(seq_b)
    band_radius = band_radius or max(n, m)
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band_radius)
        j_hi = min(m, i + band_radius)
        for j in range(j_lo, j_hi + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def frechet_distance_gaussian(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Frechet distance between two multivariate Gaussians fit to
    feature sets a and b (Dowson & Landau 1982 closed form)."""
    mu_a, mu_b = np.mean(features_a, axis=0), np.mean(features_b, axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)
    sigma_a = np.atleast_2d(sigma_a)
    sigma_b = np.atleast_2d(sigma_b)

    diff = mu_a - mu_b
    # matrix sqrt via eigen-decomposition of sigma_a @ sigma_b
    covmean, _ = _matrix_sqrt(sigma_a @ sigma_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist2 = diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean)
    return float(max(dist2, 0.0))


def _matrix_sqrt(mat: np.ndarray):
    vals, vecs = np.linalg.eig(mat)
    sqrt_vals = np.sqrt(vals.astype(complex))
    sqrt_mat = vecs @ np.diag(sqrt_vals) @ np.linalg.inv(vecs)
    return sqrt_mat, vals


def compare_against_literature(metrics: PulseMetrics) -> dict:
    report = {}
    m = asdict(metrics)
    for key, (lo, hi, source) in LITERATURE_REFERENCE_RANGES.items():
        if key not in m or key.endswith("_healthy"):
            continue
        value = m[key]
        in_range = (lo <= value <= hi) if np.isfinite(value) else None
        report[key] = {"value": value, "literature_range": (lo, hi), "in_range": in_range, "source": source}
    return report