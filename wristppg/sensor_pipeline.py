"""
Approximate wearable acquisition chain: converts a "clean" optical PPG
waveform into what a photodiode + AGC + ADC + firmware filtering chain
would plausibly output.

IMPORTANT: Apple does not publish the internal analog front-end, AGC
algorithm, or firmware filter design of Apple Watch. Everything in this
module is a generic, textbook approximation of a reflectance-mode
pulse-oximetry-style acquisition chain (as used broadly in consumer
wearables and described in the PPG instrumentation literature below),
NOT a reverse-engineered or validated model of actual Apple Watch
hardware/firmware. Treat all component-level behavior (AGC time
constants, ADC bit depth, filter cutoffs) as illustrative defaults that
should be replaced with device-specific values if/when available.

Evidence base (generic PPG/wearable instrumentation, not device-specific)
--------------------------------------------------------------------------
- Photodiode shot-noise-limited detection & AGC to keep the DC operating
  point within the ADC's dynamic range: Tamura, Maeda, Sekine & Yoshida,
  "Wearable Photoplethysmographic Sensors - Past and Present",
  Electronics 3:282-302 (2014).
- Adaptive LED drive current / automatic gain control as standard
  practice in reflectance pulse oximetry front-ends (e.g., Maxim
  MAX86141/MAX30101-class AFE application notes; generic architecture,
  not Apple-specific).
- Anti-aliasing filter ahead of ADC sampling: standard DSP practice
  (Oppenheim & Schafer, "Discrete-Time Signal Processing").
- Sampling jitter and clock drift as generic embedded-ADC nonidealities:
  Kester, "Data Conversion Handbook" (Analog Devices), Ch. 2.
- Motion-based sample rejection / accelerometer fusion for wearable PPG:
  well documented generically (e.g., Zhang, Pi & Liu, "TROIKA: A general
  framework for heart rate monitoring using wrist-type PPG signals during
  intensive physical exercise", IEEE TBME 62:522-31 (2015)) though not
  describing Apple's specific implementation.

What is heuristic / unverifiable
---------------------------------
- Apple Watch's actual sample rate, bit depth, AGC control law, and
  firmware filter coefficients are proprietary and not publicly
  documented; the values below (e.g., 12-bit ADC, ~64-256 Hz internal
  sampling before decimation, simple PI-controller AGC) are reasonable
  engineering defaults for this class of device, not confirmed facts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class SensorPipelineParams:
    adc_bits: int = 12
    fs_internal_hz: float = 128.0
    fs_output_hz: float = 25.0
    agc_time_constant_s: float = 2.0
    clock_drift_ppm: float = 20.0
    jitter_std_s: float = 150e-6
    saturation_margin: float = 0.98
    antialias_cutoff_hz: float = 8.0


class SensorPipeline:
    def __init__(self, rng: np.random.Generator, params: SensorPipelineParams | None = None):
        self.rng = rng
        self.params = params or SensorPipelineParams()

    def apply_agc(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Simple automatic-gain-control loop: adjusts a scalar LED-drive
        /photodiode gain so the DC level tracks toward the ADC's
        mid-range, with a first-order (RC-like) response time constant.
        """
        target = 0.5
        n = len(signal)
        gain = np.empty(n)
        g = 1.0
        alpha = 1.0 / (self.params.agc_time_constant_s * fs)
        dc_est = signal[0]
        for i in range(n):
            dc_est = (1 - alpha) * dc_est + alpha * signal[i]
            error = target - dc_est * g
            g = np.clip(g + 0.05 * error, 0.05, 50.0)
            gain[i] = g
        return signal * gain, gain

    def apply_clock_drift_and_jitter(self, t: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Resample onto a slightly drifting/jittered time base and
        interpolate back onto the nominal grid, emulating a real crystal
        oscillator's slow drift plus sample-to-sample timing jitter.
        """
        drift = self.params.clock_drift_ppm * 1e-6
        t_drifted = t * (1.0 + drift)
        jitter = self.rng.normal(0, self.params.jitter_std_s, size=len(t))
        t_actual = t_drifted + jitter
        t_actual = np.sort(t_actual)
        return np.interp(t, t_actual, signal)

    def apply_antialias(self, signal: np.ndarray, fs: float) -> np.ndarray:
        nyq = fs / 2.0
        cutoff = min(self.params.antialias_cutoff_hz, nyq * 0.9)
        b, a = butter(4, cutoff / nyq, btype="low")
        return filtfilt(b, a, signal)

    def quantize(self, signal: np.ndarray) -> np.ndarray:
        levels = 2 ** self.params.adc_bits
        lo, hi = np.min(signal), np.max(signal)
        if hi - lo < 1e-12:
            return signal
        step = (hi - lo) / levels
        q = np.round((signal - lo) / step) * step + lo
        return q

    def apply_saturation(self, signal: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(signal, [0.5, 99.5])
        span = hi - lo
        clip_lo = lo - 0.05 * span
        clip_hi = hi + 0.05 * span
        return np.clip(signal, clip_lo, clip_hi)

    def decimate(self, signal: np.ndarray, fs_in: float) -> np.ndarray:
        factor = max(int(round(fs_in / self.params.fs_output_hz)), 1)
        return signal[::factor]

    def run(self, clean_ppg: np.ndarray, fs_in: float) -> dict:
        n = len(clean_ppg)
        t = np.arange(n) / fs_in

        drifted = self.apply_clock_drift_and_jitter(t, clean_ppg)
        gained, gain_trace = self.apply_agc(drifted, fs_in)
        saturated = self.apply_saturation(gained)
        filtered = self.apply_antialias(saturated, fs_in)
        quantized = self.quantize(filtered)
        output = self.decimate(quantized, fs_in)

        return {
            "raw_sensor_output": output.astype(np.float32),
            "agc_gain_trace": gain_trace.astype(np.float32),
            "fs_output_hz": self.params.fs_output_hz,
        }