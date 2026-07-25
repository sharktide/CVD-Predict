"""
Clinical Alert Orchestrator for OHCA Predictor.

Compiles patient reports from model output and raw signals, computes
signal-level features, and sends clinical alerts via LLM API.

CRITICAL SAFETY CONSTRAINT:
The LLM provides EXPLANATORY SUPPORT ONLY. It does NOT:
- Make autonomous treatment decisions
- Replace clinical judgment
- Override physician assessment
- Generate legally binding medical advice

All LLM outputs are advisory and must be reviewed by qualified
medical professionals before any clinical action is taken.
"""

import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.signal import find_peaks

from ohca_predictor.config import LLMConfig
from ohca_predictor.llm.prompts import (
    CLINICAL_SYSTEM_PROMPT,
    PatientReportBuilder,
)

logger = logging.getLogger(__name__)


class ClinicalAlertOrchestrator:
    """Orchestrates clinical alert generation and LLM-based report compilation.

    This class bridges the ML model's numerical predictions with LLM-generated
    clinical explanations. It computes signal-level features, builds structured
    patient reports, and sends them to an LLM endpoint for clinical interpretation.

    CRITICAL: All outputs are EXPLANATORY SUPPORT ONLY.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the orchestrator with LLM configuration.

        Args:
            config: LLMConfig instance containing endpoint URL, model ID,
                token limits, temperature, timeout, and retry settings.
        """
        self.config = config
        self.report_builder = PatientReportBuilder()
        self._api_call_log: List[Dict[str, Any]] = []

    def compile_patient_report(
        self,
        model_output: Dict[str, Any],
        raw_signals: Dict[str, np.ndarray],
        patient_features: Dict[str, Any],
        signal_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile a structured patient report from model output and raw data.

        Combines ML model predictions, computed signal features, and patient
        demographics into a structured JSON report suitable for clinical review
        and LLM consumption.

        Args:
            model_output: Dictionary containing model prediction results with
                keys: ohca_risk, survival_curve, uncertainty, auxiliary.
            raw_signals: Dictionary mapping signal names to numpy arrays.
                Expected keys: ecg, accelerometer, gyroscope, ppg, spo2,
                temperature, respiration.
            patient_features: Dictionary with patient demographics and clinical
                data: age, sex, bmi, comorbidities, medications, etc.
            signal_features: Optional pre-computed signal features. If None,
                features are computed from raw_signals.

        Returns:
            Structured dictionary containing the compiled report with sections
            for patient summary, model assessment, signal analysis, risk level,
            uncertainty, and raw data references.
        """
        if signal_features is None:
            signal_features = self._compute_signal_features(raw_signals)

        patient_summary = self.report_builder.build_patient_summary(patient_features)
        model_assessment = self.report_builder.build_model_assessment(model_output)
        signal_analysis = self.report_builder.build_signal_analysis(signal_features)

        ohca_risk = model_output.get("ohca_risk", 0.0)
        risk_level = self.report_builder.determine_risk_level(ohca_risk)

        uncertainty = model_output.get("uncertainty", {})
        if isinstance(uncertainty, dict):
            total_uncertainty = uncertainty.get(
                "total", uncertainty.get("combined", 0.0)
            )
        else:
            total_uncertainty = float(uncertainty) if uncertainty else 0.0
        uncertainty_level = self.report_builder.determine_uncertainty_level(
            total_uncertainty
        )

        system_prompt = CLINICAL_SYSTEM_PROMPT.format(
            patient_summary=patient_summary,
            model_assessment=model_assessment,
            signal_analysis=signal_analysis,
        )

        user_prompt = (
            "Analyze the following patient data and provide a clinical "
            "interpretation of the OHCA risk assessment.\n\n"
            f"Patient Summary:\n{patient_summary}\n\n"
            f"Model Assessment:\n{model_assessment}\n\n"
            f"Signal Analysis:\n{signal_analysis}"
        )

        report: Dict[str, Any] = {
            "patient_summary": patient_summary,
            "model_assessment": model_assessment,
            "signal_analysis": signal_analysis,
            "risk_level": risk_level,
            "uncertainty_level": uncertainty_level,
            "total_uncertainty": total_uncertainty,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_output": {
                "ohca_risk": ohca_risk,
                "risk_level": risk_level,
                "uncertainty_level": uncertainty_level,
            },
            "patient_features": patient_features,
            "signal_features": signal_features,
        }

        survival_curve = model_output.get("survival_curve")
        if survival_curve is not None:
            if isinstance(survival_curve, np.ndarray):
                report["model_output"]["survival_curve"] = survival_curve.tolist()
            elif isinstance(survival_curve, (list, tuple)):
                report["model_output"]["survival_curve"] = list(survival_curve)

        auxiliary = model_output.get("auxiliary", {})
        if auxiliary:
            serializable_auxiliary = {}
            for key, value in auxiliary.items():
                if isinstance(value, np.ndarray):
                    serializable_auxiliary[key] = value.tolist()
                elif isinstance(value, (int, float, str, bool)):
                    serializable_auxiliary[key] = value
                elif isinstance(value, (list, tuple)):
                    serializable_auxiliary[key] = list(value)
                else:
                    serializable_auxiliary[key] = str(value)
            report["model_output"]["auxiliary"] = serializable_auxiliary

        return report

    def send_clinical_alert(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Send a compiled patient report to the LLM endpoint for clinical interpretation.

        Posts the report to the configured LLM endpoint with retry logic
        and exponential backoff. Returns the LLM response or an error dict.

        Args:
            report: Structured patient report dictionary as returned by
                compile_patient_report. Must contain 'system_prompt' and
                'user_prompt' keys.

        Returns:
            Dictionary containing either:
            - 'response': The LLM's clinical interpretation text
            - 'success': True
            - 'latency_ms': Response time in milliseconds
            Or on error:
            - 'error': Error description string
            - 'success': False
            - 'attempts': Number of attempts made
            - 'last_status_code': HTTP status code if available
        """
        system_prompt = report.get("system_prompt", "")
        user_prompt = report.get("user_prompt", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        last_error: Optional[str] = None
        last_status_code: Optional[int] = None

        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.monotonic()
            try:
                logger.info(
                    "LLM API call attempt %d/%d to %s",
                    attempt,
                    self.config.max_retries,
                    self.config.endpoint_url,
                )

                response = requests.post(
                    self.config.endpoint_url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                    headers={"Content-Type": "application/json"},
                )

                latency_ms = (time.monotonic() - start_time) * 1000

                if response.status_code == 200:
                    response_json = response.json()
                    choices = response_json.get("choices", [])
                    if choices and len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                    else:
                        content = json.dumps(response_json)

                    self._log_api_call(
                        endpoint=self.config.endpoint_url,
                        request=payload,
                        response=response_json,
                        latency=latency_ms,
                    )

                    logger.info(
                        "LLM API call succeeded on attempt %d (%.0f ms)",
                        attempt,
                        latency_ms,
                    )

                    return {
                        "response": content,
                        "success": True,
                        "latency_ms": latency_ms,
                        "model": self.config.model_id,
                        "usage": response_json.get("usage", {}),
                    }

                last_status_code = response.status_code
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"

                self._log_api_call(
                    endpoint=self.config.endpoint_url,
                    request=payload,
                    response={"error": last_error, "status_code": response.status_code},
                    latency=latency_ms,
                )

                logger.warning(
                    "LLM API call attempt %d failed with status %d: %s",
                    attempt,
                    response.status_code,
                    last_error,
                )

            except requests.exceptions.Timeout:
                latency_ms = (time.monotonic() - start_time) * 1000
                last_error = f"Request timed out after {self.config.timeout_seconds}s"
                logger.warning(
                    "LLM API call attempt %d timed out (%.0f ms)",
                    attempt,
                    latency_ms,
                )
                self._log_api_call(
                    endpoint=self.config.endpoint_url,
                    request=payload,
                    response={"error": last_error},
                    latency=latency_ms,
                )

            except requests.exceptions.ConnectionError as exc:
                latency_ms = (time.monotonic() - start_time) * 1000
                last_error = f"Connection error: {exc}"
                logger.warning(
                    "LLM API call attempt %d connection error: %s",
                    attempt,
                    last_error,
                )
                self._log_api_call(
                    endpoint=self.config.endpoint_url,
                    request=payload,
                    response={"error": last_error},
                    latency=latency_ms,
                )

            except requests.exceptions.RequestException as exc:
                latency_ms = (time.monotonic() - start_time) * 1000
                last_error = f"Request exception: {exc}"
                logger.warning(
                    "LLM API call attempt %d failed: %s",
                    attempt,
                    last_error,
                )
                self._log_api_call(
                    endpoint=self.config.endpoint_url,
                    request=payload,
                    response={"error": last_error},
                    latency=latency_ms,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                latency_ms = (time.monotonic() - start_time) * 1000
                last_error = f"Response parsing error: {exc}"
                logger.warning(
                    "LLM API call attempt %d parse error: %s",
                    attempt,
                    last_error,
                )
                self._log_api_call(
                    endpoint=self.config.endpoint_url,
                    request=payload,
                    response={"error": last_error},
                    latency=latency_ms,
                )

            if attempt < self.config.max_retries:
                backoff_seconds = min(2 ** attempt, 30)
                logger.info(
                    "Retrying in %d seconds (attempt %d/%d)...",
                    backoff_seconds,
                    attempt + 1,
                    self.config.max_retries,
                )
                time.sleep(backoff_seconds)

        return {
            "error": last_error or "Unknown error after all retries",
            "success": False,
            "attempts": self.config.max_retries,
            "last_status_code": last_status_code,
        }

    def _compute_signal_features(self, raw_signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute comprehensive signal features from raw sensor data.

        Extracts heart rate, HRV metrics, rhythm analysis, motion features,
        signal quality estimates, ST segment analysis, QT interval estimation,
        respiratory rate, and SpO2 features from raw signal arrays.

        Args:
            raw_signals: Dictionary mapping signal names to numpy arrays.
                Expected keys: ecg, accelerometer, gyroscope, ppg, spo2,
                temperature, respiration. Sampling rates are assumed from
                the project's SimulationConfig defaults.

        Returns:
            Dictionary with computed signal features organized by modality.
        """
        features: Dict[str, Any] = {}

        ecg = raw_signals.get("ecg")
        if ecg is not None and len(ecg) > 0:
            ecg_signal = np.asarray(ecg, dtype=np.float64).flatten()
            ecg_hz = 130.0

            r_peaks = self._detect_r_peaks(ecg_signal, ecg_hz)
            features["hr"] = self._compute_heart_rate(r_peaks, ecg_hz)

            if len(r_peaks) >= 3:
                rr_intervals = np.diff(r_peaks) / ecg_hz * 1000.0
                features["hrv"] = self._compute_hrv_metrics(rr_intervals)
            else:
                features["hrv"] = {"rmssd": 0.0, "sdnn": 0.0, "lf_hf_ratio": 0.0}

            features["rhythm"] = self._analyze_rhythm(ecg_signal, r_peaks, ecg_hz)
            features["st_segment"] = self._compute_st_segment(ecg_signal, r_peaks, ecg_hz)
            features["qt_interval"] = self._compute_qt_interval(ecg_signal, r_peaks, ecg_hz)

        accel = raw_signals.get("accelerometer")
        if accel is not None and len(accel) > 0:
            accel_signal = np.asarray(accel, dtype=np.float64)
            if accel_signal.ndim > 1:
                accel_magnitude = np.sqrt(np.sum(accel_signal ** 2, axis=-1))
            else:
                accel_magnitude = accel_signal.flatten()
            features["motion"] = self._analyze_motion(accel_magnitude)

        ppg = raw_signals.get("ppg")
        spo2 = raw_signals.get("spo2")
        features["spo2"] = self._compute_spo2_features(ppg, spo2)

        resp = raw_signals.get("respiration")
        if resp is not None and len(resp) > 0:
            resp_signal = np.asarray(resp, dtype=np.float64).flatten()
            features["respiratory_rate"] = self._compute_respiratory_rate(resp_signal, 25.0)

        features["signal_quality"] = self._estimate_signal_quality(raw_signals)

        return features

    def _detect_r_peaks(self, ecg_signal: np.ndarray, ecg_hz: float) -> np.ndarray:
        """Detect R-peaks in an ECG signal using bandpass filtering and peak detection.

        Applies a simple bandpass filter (5-15 Hz) to enhance R-peaks,
        then uses scipy.signal.find_peaks with adaptive thresholding.

        Args:
            ecg_signal: Raw ECG signal as 1D numpy array.
            ecg_hz: Sampling frequency in Hz.

        Returns:
            1D numpy array of R-peak sample indices.
        """
        if len(ecg_signal) < int(ecg_hz):
            return np.array([], dtype=int)

        filtered = self._bandpass_filter(ecg_signal, lowcut=5.0, highcut=15.0, fs=ecg_hz, order=2)

        squared = filtered ** 2

        window_size = int(0.15 * ecg_hz)
        if window_size < 1:
            window_size = 1
        kernel = np.ones(window_size) / window_size
        envelope = np.convolve(squared, kernel, mode="same")

        threshold = np.mean(envelope) + 0.3 * np.std(envelope)

        min_distance = int(0.3 * ecg_hz)
        if min_distance < 1:
            min_distance = 1

        peaks, properties = find_peaks(
            envelope,
            height=threshold,
            distance=min_distance,
        )

        if len(peaks) == 0:
            peaks, properties = find_peaks(
                envelope,
                height=np.mean(envelope) * 0.5,
                distance=min_distance,
            )

        return peaks.astype(int)

    def _bandpass_filter(
        self,
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 2,
    ) -> np.ndarray:
        """Apply a Butterworth bandpass filter to a signal.

        Uses scipy.signal.butter and scipy.signal.sosfilt for numerically
        stable second-order-section filtering.

        Args:
            signal: Input 1D signal array.
            lowcut: Lower cutoff frequency in Hz.
            highcut: Upper cutoff frequency in Hz.
            fs: Sampling frequency in Hz.
            order: Filter order (default 2).

        Returns:
            Filtered signal as 1D numpy array.
        """
        from scipy.signal import butter, sosfilt

        nyquist = fs / 2.0
        low = max(lowcut / nyquist, 0.001)
        high = min(highcut / nyquist, 0.999)

        if low >= high:
            return signal.copy()

        sos = butter(order, [low, high], btype="band", output="sos")
        filtered = sosfilt(sos, signal)
        return filtered

    def _compute_heart_rate(self, r_peaks: np.ndarray, ecg_hz: float) -> Dict[str, Any]:
        """Compute heart rate statistics from R-peak locations.

        Args:
            r_peaks: Array of R-peak sample indices.
            ecg_hz: ECG sampling frequency in Hz.

        Returns:
            Dictionary with 'mean', 'min', 'max', 'range', and 'std' heart rate values.
            Returns zeroed dict if fewer than 2 R-peaks are detected.
        """
        if len(r_peaks) < 2:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "range": (0.0, 0.0), "std": 0.0}

        rr_samples = np.diff(r_peaks)
        rr_seconds = rr_samples / ecg_hz
        hr_bpm = 60.0 / rr_seconds

        hr_bpm = hr_bpm[(hr_bpm > 20) & (hr_bpm < 300)]

        if len(hr_bpm) == 0:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "range": (0.0, 0.0), "std": 0.0}

        return {
            "mean": float(np.mean(hr_bpm)),
            "min": float(np.min(hr_bpm)),
            "max": float(np.max(hr_bpm)),
            "range": (float(np.min(hr_bpm)), float(np.max(hr_bpm))),
            "std": float(np.std(hr_bpm)),
        }

    def _compute_hrv_metrics(self, rr_intervals: np.ndarray) -> Dict[str, Any]:
        """Compute time-domain and frequency-domain HRV metrics.

        Time-domain: RMSSD (root mean square of successive differences),
        SDNN (standard deviation of NN intervals).

        Frequency-domain: LF power (0.04-0.15 Hz), HF power (0.15-0.40 Hz),
        and LF/HF ratio computed via periodogram on interpolated RR series.

        Args:
            rr_intervals: RR intervals in milliseconds.

        Returns:
            Dictionary with keys: rmssd, sdnn, lf_power, hf_power, lf_hf_ratio.
        """
        if len(rr_intervals) < 2:
            return {
                "rmssd": 0.0,
                "sdnn": 0.0,
                "lf_power": 0.0,
                "hf_power": 0.0,
                "lf_hf_ratio": 0.0,
            }

        diffs = np.diff(rr_intervals)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        sdnn = float(np.std(rr_intervals, ddof=1))

        lf_power = 0.0
        hf_power = 0.0
        lf_hf_ratio = 0.0

        if len(rr_intervals) >= 10:
            try:
                mean_rr = np.mean(rr_intervals)
                cumulative_time = np.cumsum(rr_intervals) / 1000.0
                cumulative_time = np.insert(cumulative_time, 0, 0.0)

                interp_fs = 4.0
                t_interp = np.arange(
                    cumulative_time[0], cumulative_time[-1], 1.0 / interp_fs
                )
                if len(t_interp) > 2:
                    rr_interp = np.interp(t_interp, cumulative_time[1:], rr_intervals)
                    rr_interp = rr_interp - np.mean(rr_interp)

                    nperseg = min(len(rr_interp), 256)
                    if nperseg >= 8:
                        freqs, psd = self._periodogram(rr_interp, interp_fs)

                        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                        hf_mask = (freqs > 0.15) & (freqs <= 0.40)

                        if np.any(lf_mask):
                            lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
                        if np.any(hf_mask):
                            hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask]))

                        if hf_power > 0:
                            lf_hf_ratio = lf_power / hf_power
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                pass

        return {
            "rmssd": rmssd,
            "sdnn": sdnn,
            "lf_power": lf_power,
            "hf_power": hf_power,
            "lf_hf_ratio": lf_hf_ratio,
        }

    def _periodogram(
        self, signal: np.ndarray, fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a simple periodogram power spectral density estimate.

        Uses a Hanning window and FFT to estimate the one-sided power
        spectral density.

        Args:
            signal: Input 1D signal array.
            fs: Sampling frequency in Hz.

        Returns:
            Tuple of (frequencies, psd) arrays for positive frequencies.
        """
        n = len(signal)
        windowed = signal * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        psd = (np.abs(fft_vals) ** 2) / (fs * n)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs, psd

    def _analyze_rhythm(
        self, ecg_signal: np.ndarray, r_peaks: np.ndarray, ecg_hz: float
    ) -> Dict[str, Any]:
        """Analyze cardiac rhythm regularity and ectopy burden.

        Classifies rhythm as 'regular', 'slightly irregular', 'irregular',
        or 'highly irregular' based on RR interval coefficient of variation.
        Estimates ectopy burden from anomalous RR intervals.

        Args:
            ecg_signal: Raw ECG signal.
            r_peaks: R-peak sample indices.
            ecg_hz: ECG sampling frequency.

        Returns:
            Dictionary with 'classification', 'regularity_index',
            and 'ectopy_burden' keys.
        """
        if len(r_peaks) < 3:
            return {
                "classification": "indeterminate",
                "regularity_index": 0.0,
                "ectopy_burden": 0.0,
            }

        rr_samples = np.diff(r_peaks)
        rr_seconds = rr_samples / ecg_hz
        mean_rr = np.mean(rr_seconds)

        if mean_rr <= 0:
            return {
                "classification": "indeterminate",
                "regularity_index": 0.0,
                "ectopy_burden": 0.0,
            }

        cv = np.std(rr_seconds) / mean_rr

        if cv < 0.05:
            classification = "regular"
        elif cv < 0.10:
            classification = "slightly irregular"
        elif cv < 0.20:
            classification = "irregular"
        else:
            classification = "highly irregular"

        regularity_index = max(0.0, 1.0 - cv)

        rr_median = np.median(rr_seconds)
        deviation = np.abs(rr_seconds - rr_median) / rr_median
        ectopic_beats = np.sum(deviation > 0.20)
        ectopy_burden = float(ectopic_beats / len(rr_seconds)) if len(rr_seconds) > 0 else 0.0

        return {
            "classification": classification,
            "regularity_index": float(regularity_index),
            "ectopy_burden": ectopy_burden,
        }

    def _compute_st_segment(
        self, ecg_signal: np.ndarray, r_peaks: np.ndarray, ecg_hz: float
    ) -> Dict[str, Any]:
        """Compute ST segment deviation from isoelectric line.

        Estimates ST deviation at J+60ms and J+80ms relative to the
        R-peak location, then averages across detected beats.

        Args:
            ecg_signal: Raw ECG signal.
            r_peaks: R-peak sample indices.
            ecg_hz: ECG sampling frequency.

        Returns:
            Dictionary with 'deviation_j60', 'deviation_j80',
            'mean_deviation', and 'lead_deviations' keys.
        """
        if len(r_peaks) < 1:
            return {
                "deviation_j60": 0.0,
                "deviation_j80": 0.0,
                "mean_deviation": 0.0,
                "lead_deviations": {},
            }

        j_offset_samples_60 = int(0.060 * ecg_hz)
        j_offset_samples_80 = int(0.080 * ecg_hz)
        qrs_width_samples = int(0.080 * ecg_hz)

        st_values_60: List[float] = []
        st_values_80: List[float] = []

        for r_peak in r_peaks:
            j_point = r_peak + qrs_width_samples
            st_60_idx = j_point + j_offset_samples_60
            st_80_idx = j_point + j_offset_samples_80

            if st_80_idx < len(ecg_signal):
                st_values_60.append(float(ecg_signal[st_60_idx]))
                st_values_80.append(float(ecg_signal[st_80_idx]))

        if not st_values_60:
            return {
                "deviation_j60": 0.0,
                "deviation_j80": 0.0,
                "mean_deviation": 0.0,
                "lead_deviations": {},
            }

        deviation_j60 = float(np.mean(st_values_60))
        deviation_j80 = float(np.mean(st_values_80))
        mean_deviation = float(np.mean([deviation_j60, deviation_j80]))

        return {
            "deviation_j60": deviation_j60,
            "deviation_j80": deviation_j80,
            "mean_deviation": mean_deviation,
            "lead_deviations": {
                "Lead II (J+60ms)": deviation_j60,
                "Lead II (J+80ms)": deviation_j80,
            },
        }

    def _compute_qt_interval(
        self, ecg_signal: np.ndarray, r_peaks: np.ndarray, ecg_hz: float
    ) -> Dict[str, Any]:
        """Estimate QT interval and corrected QT (QTc) using Bazett's formula.

        The QT interval is estimated from the R-peak to the end of the T-wave,
        approximated as the point where the signal returns to near-baseline
        after the R-peak.

        QTc = QT / sqrt(RR) where RR is in seconds.

        Args:
            ecg_signal: Raw ECG signal.
            r_peaks: R-peak sample indices.
            ecg_hz: ECG sampling frequency.

        Returns:
            Dictionary with 'qt' (ms), 'qtc' (ms), and 'rr' (ms) keys.
        """
        if len(r_peaks) < 2:
            return {"qt": 0.0, "qtc": 0.0, "rr": 0.0}

        rr_samples = np.diff(r_peaks)
        rr_seconds = rr_samples / ecg_hz
        mean_rr = np.mean(rr_seconds)

        qt_values: List[float] = []

        for i in range(len(r_peaks) - 1):
            r_peak = r_peaks[i]
            search_end = r_peaks[i + 1]

            if search_end >= len(ecg_signal):
                continue

            segment = ecg_signal[r_peak:search_end]

            if len(segment) < int(0.20 * ecg_hz):
                continue

            r_peak_val = ecg_signal[r_peak]
            baseline = np.mean(ecg_signal[max(0, r_peak - int(0.05 * ecg_hz)):r_peak])

            if abs(r_peak_val - baseline) < 1e-10:
                continue

            t_wave_search_start = int(0.10 * ecg_hz)
            t_wave_region = segment[t_wave_search_start:]

            if len(t_wave_region) < 2:
                continue

            abs_signal = np.abs(t_wave_region)
            threshold = 0.2 * np.max(abs_signal) if np.max(abs_signal) > 0 else 0.0

            crossing_indices = np.where(
                (abs_signal[:-1] >= threshold) & (abs_signal[1:] < threshold)
            )[0]

            if len(crossing_indices) > 0:
                t_end_relative = t_wave_search_start + crossing_indices[-1] + 1
                qt_samples = t_end_relative
            else:
                qt_samples = int(0.35 * ecg_hz)

            qt_ms = (qt_samples / ecg_hz) * 1000.0
            qt_values.append(qt_ms)

        if not qt_values:
            return {"qt": 0.0, "qtc": 0.0, "rr": 0.0}

        median_qt = float(np.median(qt_values))
        qt_sec = median_qt / 1000.0
        rr_sec = mean_rr

        if rr_sec > 0:
            qtc = qt_sec / math.sqrt(rr_sec)
            qtc_ms = qtc * 1000.0
        else:
            qtc_ms = 0.0

        return {
            "qt": median_qt,
            "qtc": qtc_ms,
            "rr": mean_rr * 1000.0,
        }

    def _analyze_motion(self, accel_magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze motion from accelerometer magnitude signal.

        Computes peak acceleration, mean activity level, and classifies
        activity as 'stationary', 'walking', 'active', or 'high_intensity'.

        Args:
            accel_magnitude: 1D accelerometer magnitude array.

        Returns:
            Dictionary with 'peak_acceleration', 'mean_acceleration',
            'std_acceleration', and 'classification' keys.
        """
        if len(accel_magnitude) == 0:
            return {
                "peak_acceleration": 0.0,
                "mean_acceleration": 0.0,
                "std_acceleration": 0.0,
                "classification": "unknown",
            }

        peak_accel = float(np.max(np.abs(accel_magnitude)))
        mean_accel = float(np.mean(accel_magnitude))
        std_accel = float(np.std(accel_magnitude))

        gravity_component = 9.81
        dynamic_accel = std_accel

        if dynamic_accel < 0.05:
            classification = "stationary"
        elif dynamic_accel < 0.5:
            classification = "walking"
        elif dynamic_accel < 2.0:
            classification = "active"
        else:
            classification = "high_intensity"

        return {
            "peak_acceleration": peak_accel,
            "mean_acceleration": mean_accel,
            "std_acceleration": std_accel,
            "classification": classification,
        }

    def _compute_spo2_features(
        self,
        ppg: Optional[np.ndarray],
        spo2: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute SpO2 features from PPG and/or SpO2 signals.

        If a dedicated SpO2 signal is available, computes mean and
        variability. Otherwise, estimates from PPG signal characteristics.

        Args:
            ppg: PPG signal array, or None if unavailable.
            spo2: Dedicated SpO2 signal array, or None if unavailable.

        Returns:
            Dictionary with 'mean', 'variability', and 'min' SpO2 values.
        """
        if spo2 is not None and len(spo2) > 0:
            spo2_signal = np.asarray(spo2, dtype=np.float64).flatten()
            valid_spo2 = spo2_signal[(spo2_signal > 50) & (spo2_signal <= 100)]
            if len(valid_spo2) > 0:
                return {
                    "mean": float(np.mean(valid_spo2)),
                    "variability": float(np.std(valid_spo2)),
                    "min": float(np.min(valid_spo2)),
                }

        if ppg is not None and len(ppg) > 0:
            ppg_signal = np.asarray(ppg, dtype=np.float64).flatten()
            ppg_mean = np.mean(ppg_signal)
            ppg_std = np.std(ppg_signal)

            if ppg_mean > 0:
                pulsatility = ppg_std / ppg_mean
                estimated_spo2 = 95.0 - 10.0 * max(0.0, 0.5 - pulsatility)
                estimated_spo2 = max(70.0, min(100.0, estimated_spo2))
            else:
                estimated_spo2 = 95.0

            return {
                "mean": estimated_spo2,
                "variability": float(ppg_std),
                "min": max(70.0, estimated_spo2 - 3.0 * ppg_std) if ppg_std > 0 else estimated_spo2,
            }

        return {"mean": 0.0, "variability": 0.0, "min": 0.0}

    def _compute_respiratory_rate(
        self, resp_signal: np.ndarray, resp_hz: float
    ) -> Dict[str, Any]:
        """Estimate respiratory rate from respiration signal.

        Uses bandpass filtering (0.1-0.5 Hz) and peak detection to count
        respiratory cycles and compute breaths per minute.

        Args:
            resp_signal: Respiration signal array.
            resp_hz: Sampling frequency of respiration signal.

        Returns:
            Dictionary with 'mean', 'min', 'max', and 'std' respiratory rate.
        """
        if len(resp_signal) < resp_hz:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        filtered = self._bandpass_filter(
            resp_signal, lowcut=0.1, highcut=0.5, fs=resp_hz, order=2
        )

        squared = filtered ** 2
        window_size = int(1.0 * resp_hz)
        if window_size < 1:
            window_size = 1
        kernel = np.ones(window_size) / window_size
        envelope = np.convolve(squared, kernel, mode="same")

        min_distance = int(2.0 * resp_hz)
        if min_distance < 1:
            min_distance = 1

        peaks, _ = find_peaks(
            envelope,
            height=np.mean(envelope) * 0.3,
            distance=min_distance,
        )

        if len(peaks) < 2:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        duration_seconds = len(resp_signal) / resp_hz
        rr_bpm = (len(peaks) / duration_seconds) * 60.0

        rr_intervals_bpm: List[float] = []
        peak_times = peaks / resp_hz
        for i in range(1, len(peak_times)):
            interval = peak_times[i] - peak_times[i - 1]
            if interval > 0:
                rr_intervals_bpm.append(60.0 / interval)

        if rr_intervals_bpm:
            return {
                "mean": float(np.mean(rr_intervals_bpm)),
                "min": float(np.min(rr_intervals_bpm)),
                "max": float(np.max(rr_intervals_bpm)),
                "std": float(np.std(rr_intervals_bpm)),
            }

        return {"mean": rr_bpm, "min": rr_bpm, "max": rr_bpm, "std": 0.0}

    def _estimate_signal_quality(
        self, raw_signals: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Estimate signal quality for each modality using SNR heuristics.

        Computes a rough signal-to-noise ratio for each available signal
        and maps it to a quality score in [0, 1].

        Args:
            raw_signals: Dictionary of raw signal arrays.

        Returns:
            Dictionary mapping modality names to quality scores in [0, 1].
        """
        quality: Dict[str, float] = {}

        for modality, signal_data in raw_signals.items():
            if signal_data is None or len(signal_data) == 0:
                quality[modality] = 0.0
                continue

            sig = np.asarray(signal_data, dtype=np.float64).flatten()

            sig_mean = np.mean(sig)
            sig_std = np.std(sig)

            if abs(sig_mean) < 1e-10:
                quality[modality] = 0.0
                continue

            snr = abs(sig_mean) / max(sig_std, 1e-10)

            min_val = np.min(sig)
            max_val = np.max(sig)
            dynamic_range = max_val - min_val

            if dynamic_range < 1e-10:
                range_score = 0.0
            else:
                range_score = min(1.0, dynamic_range / (abs(sig_mean) + 1e-10))

            snr_score = min(1.0, snr / 10.0)

            quality_score = 0.6 * snr_score + 0.4 * range_score
            quality[modality] = max(0.0, min(1.0, quality_score))

        return quality

    def _log_api_call(
        self,
        endpoint: str,
        request: Dict[str, Any],
        response: Any,
        latency: float,
    ) -> None:
        """Log an LLM API call for audit and debugging purposes.

        Logs the endpoint, request payload (truncated), response summary,
        and latency. Appends to the internal call log list.

        Args:
            endpoint: The LLM API endpoint URL.
            request: The request payload dictionary.
            response: The response object or error dictionary.
            latency: Response latency in milliseconds.
        """
        request_summary = {
            "model": request.get("model"),
            "message_count": len(request.get("messages", [])),
            "max_tokens": request.get("max_tokens"),
        }

        if isinstance(response, dict):
            if "error" in response:
                response_summary = {"error": response["error"], "success": False}
            else:
                response_summary = {
                    "has_choices": "choices" in response,
                    "success": True,
                    "usage": response.get("usage", {}),
                }
        else:
            response_summary = {"raw_type": type(response).__name__}

        log_entry = {
            "endpoint": endpoint,
            "request": request_summary,
            "response": response_summary,
            "latency_ms": latency,
            "timestamp": time.time(),
        }

        self._api_call_log.append(log_entry)

        logger.info(
            "LLM API call: endpoint=%s, latency=%.0f ms, success=%s",
            endpoint,
            latency,
            response_summary.get("success", False),
        )
