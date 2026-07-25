"""
Wearable sensor physics simulation for the OHCA Predictor.

Implements physics-based models for ECG, accelerometer, gyroscope, PPG, SpO2,
temperature, respiration, and comprehensive artifact injection.  Each sensor
class wraps a deterministic generation pipeline seeded by an independent RNG
so that the entire multimodal suite can be reproduced from a single patient
state and random seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, signal

from ohca_predictor.config import OHCAConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PI2 = 2.0 * math.pi


def _wrap_phase(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap phase angle into [0, 2*pi)."""
    return np.mod(theta, _PI2)


def _bandpass_filter(
    signal_arr: NDArray[np.float64],
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.9999)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, signal_arr).astype(signal_arr.dtype)


def _lowpass_filter(
    signal_arr: NDArray[np.float64],
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth lowpass filter."""
    nyq = fs / 2.0
    cutoff = min(cutoff_hz / nyq, 0.9999)
    b, a = signal.butter(order, cutoff, btype="low")
    return signal.filtfilt(b, a, signal_arr).astype(signal_arr.dtype)


def _highpass_filter(
    signal_arr: NDArray[np.float64],
    cutoff_hz: float,
    fs: float,
    order: int = 2,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth highpass filter."""
    nyq = fs / 2.0
    cutoff = max(cutoff_hz / nyq, 1e-5)
    b, a = signal.butter(order, cutoff, btype="high")
    return signal.filtfilt(b, a, signal_arr).astype(signal_arr.dtype)


def _resample_linear(
    signal_arr: NDArray[np.float64],
    src_hz: float,
    dst_hz: float,
) -> NDArray[np.float64]:
    """Linearly resample a 1-D signal from src_hz to dst_hz."""
    ratio = dst_hz / src_hz
    n_src = len(signal_arr)
    n_dst = int(n_src * ratio)
    if n_dst == 0:
        return np.array([], dtype=signal_arr.dtype)
    src_t = np.arange(n_src) / src_hz
    dst_t = np.arange(n_dst) / dst_hz
    return np.interp(dst_t, src_t, signal_arr).astype(signal_arr.dtype)


# ---------------------------------------------------------------------------
# 1. ECG Sensor – McSharry limit-cycle model
# ---------------------------------------------------------------------------

@dataclass
class _WaveParams:
    """Single Gaussian attractor parameter set."""
    a: float
    b: float
    theta0: float


class ECGSensor:
    """Physics-based ECG generator using the McSharry limit-cycle ODE model.

    The waveform is produced by driving a phase oscillator with Gaussian
    attractors placed on the phase circle.  Each heartbeat corresponds to
    one full revolution.  Heart-rate variability is injected through
    respiratory sinus arrhythmia (RSA), LF, HF, VLF, and ULF modulation
    bands applied to the instantaneous angular velocity omega.
    """

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["ecg_hz"]

    # -- Gaussian attractor definitions -----------------------------------

    @staticmethod
    def _default_waves() -> Dict[str, _WaveParams]:
        return {
            "P": _WaveParams(a=0.15, b=0.08, theta0=0.15),
            "Q": _WaveParams(a=0.10, b=0.03, theta0=0.75),
            "R": _WaveParams(a=1.10, b=0.02, theta0=0.85),
            "S": _WaveParams(a=0.15, b=0.03, theta0=0.95),
            "T": _WaveParams(a=0.28, b=0.18, theta0=1.55),
            "U": _WaveParams(a=0.05, b=0.06, theta0=1.95),
        }

    # -- Demographic / disease modifications -----------------------------

    def _modify_waves(
        self,
        waves: Dict[str, _WaveParams],
        patient_state: Dict[str, Any],
    ) -> Dict[str, _WaveParams]:
        """Apply demographic and disease-specific modifications to wave params."""
        modified = {k: _WaveParams(v.a, v.b, v.theta0) for k, v in waves.items()}
        age = patient_state.get("age", 40)
        sex = patient_state.get("sex", "male")
        bmi = patient_state.get("bmi", 25.0)
        diseases = patient_state.get("diseases", [])

        # Age effects
        if age > 65:
            modified["P"].theta0 += 0.05
            modified["P"].b *= 1.15
            modified["Q"].theta0 -= 0.02
            modified["S"].theta0 += 0.02
            modified["R"].b *= 1.20
            modified["T"].b *= 1.10
            modified["T"].a *= 0.90

        # Sex effects
        if sex == "female":
            modified["T"].theta0 -= 0.03
            modified["T"].a *= 0.95

        # BMI effects
        if bmi > 30:
            scale = 1.0 / (1.0 + 0.02 * (bmi - 22))
            for key in modified:
                modified[key].a *= scale
                if key in ("P", "T"):
                    modified[key].b *= 1.15

        # Disease effects
        for disease in diseases:
            # --- Ischemia / ACS ---
            if disease in ("acs_unstable_angina", "acs_nstemi", "acs_stemi", "ischemia"):
                modified["ST"] = _WaveParams(a=0.08, b=0.05, theta0=1.20)
                modified["T"].a *= -0.7
                modified["T"].b *= 0.8
                modified["Q"].a *= 1.8
                # STEMI: additional ST elevation in affected leads
                if disease == "acs_stemi":
                    modified["ST"].a = 0.20
                    modified["ST"].b = 0.06
                # Pathological Q-waves in old infarction
                if disease in ("acs_nstemi", "acs_stemi"):
                    modified["Q"].a *= 2.5
                    modified["Q"].b *= 1.5

            # --- Cardiomyopathies / LVH ---
            if disease in ("dcm", "hcm", "arvc", "lvh"):
                modified["R"].a *= 1.6
                modified["R"].b *= 1.3
                modified["S"].a *= 1.3
                modified["ST"] = _WaveParams(a=0.05, b=0.04, theta0=1.20)
                # HCM: asymmetric septal hypertrophy → deeper S in lateral leads
                if disease == "hcm":
                    modified["S"].a *= 1.5
                    modified["T"].a *= -0.5
                # DCM: global wall motion → diffuse ST-T changes
                if disease == "dcm":
                    modified["T"].a *= 0.6
                    modified["T"].b *= 1.2
                # ARVC: epsilon waves, T-wave inversion in V1-V3
                if disease == "arvc":
                    modified["P"].a *= 0.7

            # --- Bundle branch blocks ---
            if disease in ("lbbb", "rbbb"):
                modified["Q"].b *= 2.0
                modified["R"].b *= 2.2
                modified["S"].b *= 2.0
                modified["R"].a *= 0.85
                # LBBB: absent Q waves, broad notched R in lateral leads
                if disease == "lbbb":
                    modified["Q"].a *= 0.3
                    modified["R"].a *= 1.1
                # RBBB: rsR' pattern, wide S in lateral leads
                if disease == "rbbb":
                    modified["S"].b *= 2.5
                    modified["S"].a *= 1.4

            # --- Hyperkalemia (progressive) ---
            if disease == "hyperkalemia":
                k_level = patient_state.get("potassium_level", 5.5)
                if k_level < 6.0:
                    # Mild: peaked T-waves
                    modified["T"].a *= 1.8
                    modified["T"].b *= 0.5
                elif k_level < 7.0:
                    # Moderate: PR prolongation, P-wave flattening, QRS widening
                    modified["T"].a *= 2.2
                    modified["T"].b *= 0.4
                    modified["P"].a *= 0.3
                    modified["P"].b *= 1.5
                    modified["Q"].b *= 1.8
                    modified["S"].b *= 1.8
                else:
                    # Severe: sine wave pattern
                    modified["T"].a *= 2.5
                    modified["T"].b *= 0.3
                    modified["P"].a *= 0.1
                    modified["Q"].b *= 2.5
                    modified["S"].b *= 2.5
                    modified["R"].b *= 2.0

            # --- Hypokalemia ---
            if disease == "hypokalemia":
                modified["T"].a *= 0.4
                modified["T"].b *= 1.4
                modified["ST"] = _WaveParams(a=0.06, b=0.04, theta0=1.18)
                modified["U"] = _WaveParams(a=0.12, b=0.08, theta0=1.90)

            # --- Long QT syndrome ---
            if disease in ("lqts", "long_qt"):
                # Prolonged QT: T-wave attractor shifted later
                modified["T"].theta0 += 0.15
                modified["T"].b *= 1.3
                # Torsades risk: beat-to-beat T-wave alternans
                modified["T"].a *= 0.8

            # --- Short QT syndrome ---
            if disease in ("sqts", "short_qt"):
                modified["T"].theta0 -= 0.12
                modified["T"].b *= 0.7
                modified["T"].a *= 1.2

            # --- Brugada syndrome ---
            if disease == "brugada":
                # Coved ST elevation in V1-V2, T-wave inversion
                modified["ST"] = _WaveParams(a=0.15, b=0.06, theta0=1.15)
                modified["T"].a *= -0.8
                modified["T"].b *= 0.7

            # --- Atrial fibrillation ---
            if disease == "af":
                # Irregular RR, no P-waves, fibrillatory baseline
                modified["P"].a *= 0.05
                modified["P"].b *= 3.0

            # --- Ventricular tachycardia ---
            if disease == "vt":
                # Wide QRS, loss of normal morphology
                modified["R"].b *= 3.0
                modified["S"].b *= 3.0
                modified["Q"].a *= 0.3
                modified["P"].a *= 0.2

            # --- Complete heart block ---
            if disease == "complete_heart_block":
                # AV dissociation: P-waves independent of QRS
                modified["P"].a *= 0.8
                modified["P"].b *= 1.5

            # --- Pulmonary embolism ---
            if disease == "pe":
                # S1Q3T3 pattern, right heart strain
                modified["Q"].a *= 1.5
                modified["T"].a *= -0.6
                modified["R"].a *= 1.2  # Right axis deviation

            # --- Sepsis ---
            if disease == "sepsis":
                # Sinus tachycardia, non-specific ST-T changes
                modified["T"].a *= 0.7
                modified["T"].b *= 1.1

            # --- Drug toxicity (e.g., digoxin) ---
            if disease == "drug_toxicity":
                # Scooped ST depression ("Salvador Dali moustache")
                modified["ST"] = _WaveParams(a=-0.10, b=0.04, theta0=1.18)
                modified["T"].a *= 0.5

        return modified

    # -- HRV modulation ---------------------------------------------------

    def _apply_hrv_modulation(
        self,
        omega: float,
        hrv_params: Dict[str, Any],
        respiratory_rate: float,
        duration: float,
        n_samples: int,
    ) -> NDArray[np.float64]:
        """Return an array of instantaneous angular velocities modulated by HRV.

        Parameters
        ----------
        omega : float
            Base angular velocity (rad/s) for the mean heart rate.
        hrv_params : dict
            Keys ``rsa_amplitude``, ``lf_amplitude``, ``hf_amplitude``,
            ``vlf_amplitude``, ``ulf_amplitude`` (each in bpm deviation).
        respiratory_rate : float
            Respiratory frequency in Hz (typically 0.15-0.40).
        duration : float
            Total signal duration in seconds.
        n_samples : int
            Number of output samples.
        """
        t = np.linspace(0.0, duration, n_samples, endpoint=False)

        # Convert amplitude from bpm to rad/s factor
        bpm_to_rad = _PI2 / 60.0

        # RSA (respiratory sinus arrhythmia)
        rsa_amp = hrv_params.get("rsa_amplitude", 6.0) * bpm_to_rad
        rsa = rsa_amp * np.sin(_PI2 * respiratory_rate * t)

        # LF component
        lf_freq = 0.095
        lf_amp = hrv_params.get("lf_amplitude", 3.5) * bpm_to_rad
        lf = lf_amp * np.sin(_PI2 * lf_freq * t + self.rng.uniform(0, _PI2))

        # HF component
        hf_freq = respiratory_rate
        hf_amp = hrv_params.get("hf_amplitude", 6.0) * bpm_to_rad
        hf = hf_amp * np.sin(_PI2 * hf_freq * t + self.rng.uniform(0, _PI2))

        # VLF component
        vlf_freq = 0.02
        vlf_amp = hrv_params.get("vlf_amplitude", 2.0) * bpm_to_rad
        vlf = vlf_amp * np.sin(_PI2 * vlf_freq * t + self.rng.uniform(0, _PI2))

        # ULF circadian modulation
        ulf_freq = 1.0 / 86400.0
        ulf_amp = hrv_params.get("ulf_amplitude", 10.0) * bpm_to_rad
        ulf = ulf_amp * np.sin(_PI2 * ulf_freq * t)

        modulated = omega + rsa + lf + hf + vlf + ulf
        return modulated

    # -- McSharry ODE integration -----------------------------------------

    def _mcsharry_ode(
        self,
        t: float,
        y: NDArray[np.float64],
        omega_t: float,
        waves: Dict[str, _WaveParams],
        z0: float,
    ) -> NDArray[np.float64]:
        """McSharry ODE right-hand side.

        y = [theta, z]
        """
        theta = y[0]
        z = y[1]

        dtheta_sum = 0.0
        dz_sum = 0.0
        for w in waves.values():
            dt = theta - w.theta0
            g = w.a * dt * np.exp(-dt * dt / (2.0 * w.b * w.b))
            dtheta_sum += g
            dz_sum += w.a * np.exp(-dt * dt / (2.0 * w.b * w.b))

        dtheta = omega_t - dtheta_sum * 0.1
        dz = -dz_sum - (z - z0)
        return np.array([dtheta, dz])

    # -- Generate ---------------------------------------------------------

    def generate(
        self,
        duration_seconds: float,
        heart_rate_bpm: float = 72.0,
        hrv_params: Optional[Dict[str, Any]] = None,
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate an ECG signal.

        Parameters
        ----------
        duration_seconds : float
            Duration of the output signal in seconds.
        heart_rate_bpm : float
            Mean heart rate in beats per minute.
        hrv_params : dict, optional
            HRV band amplitudes (bpm).  Defaults to healthy adult values.
        patient_state : dict, optional
            Patient demographics and disease list.

        Returns
        -------
        ecg_signal : ndarray of shape (n_samples,)
        """
        if patient_state is None:
            patient_state = {}
        if hrv_params is None:
            hrv_params = {}

        fs = self.fs
        n_samples = int(duration_seconds * fs)
        t_array = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        # Mean angular velocity
        omega_mean = _PI2 * (heart_rate_bpm / 60.0)

        # Respiratory rate from patient state or default
        resp_rate = patient_state.get("respiratory_rate_bpm", 16.0) / 60.0

        # Default HRV amplitudes
        hrv_defaults = {
            "rsa_amplitude": 6.0,
            "lf_amplitude": 3.5,
            "hf_amplitude": 6.0,
            "vlf_amplitude": 2.0,
            "ulf_amplitude": 10.0,
        }
        hrv_defaults.update(hrv_params)
        hrv_params = hrv_defaults

        # Demographic HRV scaling
        age = patient_state.get("age", 40)
        if age > 65:
            hrv_params["rsa_amplitude"] *= 0.5
            hrv_params["lf_amplitude"] *= 0.5
            hrv_params["hf_amplitude"] *= 0.5
            hrv_params["vlf_amplitude"] *= 0.6
        if age > 75:
            hrv_params["rsa_amplitude"] *= 0.7
            hrv_params["lf_amplitude"] *= 0.7

        # Build wave parameters
        waves = self._default_waves()
        waves = self._modify_waves(waves, patient_state)

        z0 = 0.0

        # Modulate omega at full resolution, then resample via ODE integration
        omega_full = self._apply_hrv_modulation(
            omega_mean, hrv_params, resp_rate, duration_seconds, n_samples
        )

        # Integrate ODE sample-by-sample (Euler-Maruyama for speed)
        ecg = np.zeros(n_samples)
        theta = self.rng.uniform(0, _PI2)
        z = z0

        for i in range(n_samples):
            omega_i = omega_full[i]
            # Euler step (dt = 1/fs)
            dt_step = 1.0 / fs
            dy = self._mcsharry_ode(0.0, np.array([theta, z]), omega_i, waves, z0)
            theta = theta + dy[0] * dt_step
            z = z + dy[1] * dt_step
            theta = _wrap_phase(theta)
            ecg[i] = z

        # --- Demographic waveform shaping ---
        age = patient_state.get("age", 40)
        sex = patient_state.get("sex", "male")
        bmi = patient_state.get("bmi", 25.0)

        # Age: slight baseline wander increase
        if age > 65:
            bw_freq = self.rng.uniform(0.15, 0.30)
            bw = 0.03 * np.sin(_PI2 * bw_freq * t_array)
            ecg += bw

        # Female: slightly shorter QT
        if sex == "female":
            pass  # Already encoded in T-wave theta0 shift

        # High BMI: baseline wander and smoothing
        if bmi > 30:
            bw = 0.05 * np.sin(_PI2 * 0.2 * t_array + self.rng.uniform(0, _PI2))
            ecg += bw
            # Low-pass effect: apply mild smoothing
            b_smooth = np.ones(5) / 5.0
            ecg = np.convolve(ecg, b_smooth, mode="same")

        # Normalize to physiological range [-0.5, 1.5] mV
        ecg_min, ecg_max = ecg.min(), ecg.max()
        if ecg_max - ecg_min > 1e-10:
            ecg = (ecg - ecg_min) / (ecg_max - ecg_min) * 2.0 - 0.5
        else:
            ecg[:] = 0.0

        # Additive noise
        noise_power = self.rng.uniform(0.005, 0.015)
        ecg += self.rng.normal(0.0, noise_power, n_samples)

        # 60 Hz powerline interference (optional, 10% chance)
        if self.rng.random() < 0.10:
            pl_amp = self.rng.uniform(0.01, 0.04)
            ecg += pl_amp * np.sin(_PI2 * 60.0 * t_array + self.rng.uniform(0, _PI2))

        return ecg.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Accelerometer Sensor – 3-axis at 52 Hz
# ---------------------------------------------------------------------------

class AccelerometerSensor:
    """3-axis accelerometer simulation at 52 Hz.

    Produces gravity-dominated Z signal with activity-dependent oscillatory
    components scaled by BMI.
    """

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["accelerometer_hz"]

    def generate(
        self,
        duration_seconds: float,
        activity_state: str = "rest",
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate 3-axis accelerometer data.

        Returns
        -------
        accel : ndarray of shape (3, n_samples)
            Axes: [X, Y, Z] in m/s^2.
        """
        if patient_state is None:
            patient_state = {}
        n_samples = int(duration_seconds * self.fs)
        t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        bmi = patient_state.get("bmi", 25.0)
        bmi_scale = 1.0 / (1.0 + 0.02 * (bmi - 22.0))
        bmi_scale = np.clip(bmi_scale, 0.5, 1.5)

        accel = np.zeros((3, n_samples), dtype=np.float64)

        # Gravity: primarily Z axis
        tilt_x = self.rng.uniform(-0.05, 0.05)
        tilt_y = self.rng.uniform(-0.05, 0.05)
        accel[0, :] = 9.81 * np.sin(tilt_x)
        accel[1, :] = 9.81 * np.sin(tilt_y)
        accel[2, :] = 9.81 * np.cos(tilt_x) * np.cos(tilt_y)

        # Activity-specific signals
        if activity_state == "rest":
            accel += self.rng.normal(0.0, 0.02 * bmi_scale, accel.shape)

        elif activity_state == "walking":
            freq = self.rng.uniform(1.5, 2.5)
            amp = self.rng.uniform(0.1, 0.3) * 9.81 * bmi_scale
            accel[0, :] += amp * 0.3 * np.sin(_PI2 * freq * t + self.rng.uniform(0, _PI2))
            accel[1, :] += amp * 0.2 * np.sin(_PI2 * freq * t + self.rng.uniform(0, _PI2))
            accel[2, :] += amp * 0.8 * np.abs(np.sin(_PI2 * freq * t))
            accel += self.rng.normal(0.0, 0.04 * bmi_scale, accel.shape)

        elif activity_state == "running":
            freq = self.rng.uniform(2.5, 4.0)
            amp = self.rng.uniform(0.3, 0.8) * 9.81 * bmi_scale
            accel[0, :] += amp * 0.4 * np.sin(_PI2 * freq * t)
            accel[1, :] += amp * 0.3 * np.sin(_PI2 * freq * t + 0.5)
            accel[2, :] += amp * 1.0 * np.abs(np.sin(_PI2 * freq * t))
            accel += self.rng.normal(0.0, 0.08 * bmi_scale, accel.shape)

        elif activity_state == "stairs":
            freq = self.rng.uniform(1.0, 2.0)
            amp = self.rng.uniform(0.2, 0.5) * 9.81 * bmi_scale
            accel[0, :] += amp * 0.5 * np.sin(_PI2 * freq * t)
            accel[2, :] += amp * 0.9 * np.abs(np.sin(_PI2 * freq * t))
            accel += self.rng.normal(0.0, 0.06 * bmi_scale, accel.shape)

        elif activity_state == "falling":
            spike_center = int(0.5 * n_samples)
            spike_width = int(0.1 * n_samples)
            spike_start = max(0, spike_center - spike_width)
            spike_end = min(n_samples, spike_center + spike_width)
            spike = np.zeros(n_samples)
            spike_t = np.linspace(-1.0, 1.0, spike_end - spike_start)
            spike_val = self.rng.uniform(3.0, 10.0) * 9.81
            spike[spike_start:spike_end] = spike_val * np.exp(-8.0 * spike_t ** 2)
            accel[0, :] += spike * self.rng.uniform(-0.3, 0.3)
            accel[1, :] += spike * self.rng.uniform(-0.3, 0.3)
            accel[2, :] += spike
            accel += self.rng.normal(0.0, 0.1, accel.shape)

        elif activity_state == "arm_movement":
            freq1 = self.rng.uniform(0.8, 1.5)
            freq2 = self.rng.uniform(2.0, 3.5)
            amp1 = self.rng.uniform(0.1, 0.3) * 9.81 * bmi_scale
            amp2 = self.rng.uniform(0.05, 0.15) * 9.81 * bmi_scale
            accel[0, :] += amp1 * np.sin(_PI2 * freq1 * t) + amp2 * np.sin(_PI2 * freq2 * t)
            accel[1, :] += amp1 * 0.8 * np.cos(_PI2 * freq1 * t)
            accel += self.rng.normal(0.0, 0.05 * bmi_scale, accel.shape)

        return accel.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Gyroscope Sensor – 3-axis angular velocity at 52 Hz
# ---------------------------------------------------------------------------

class GyroscopeSensor:
    """3-axis gyroscope simulation at 52 Hz."""

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["gyroscope_hz"]

    def generate(
        self,
        duration_seconds: float,
        activity_state: str = "rest",
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate 3-axis gyroscope data.

        Returns
        -------
        gyro : ndarray of shape (3, n_samples)
            Angular velocity [X, Y, Z] in rad/s.
        """
        if patient_state is None:
            patient_state = {}
        n_samples = int(duration_seconds * self.fs)
        t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        gyro = np.zeros((3, n_samples), dtype=np.float64)

        if activity_state == "rest":
            gyro += self.rng.normal(0.0, 0.005, gyro.shape)

        elif activity_state == "walking":
            freq = self.rng.uniform(1.5, 2.5)
            amp = self.rng.uniform(0.1, 0.4)
            gyro[0, :] += amp * np.sin(_PI2 * freq * t)
            gyro[1, :] += amp * 0.5 * np.sin(_PI2 * freq * t + 0.3)
            gyro += self.rng.normal(0.0, 0.02, gyro.shape)

        elif activity_state == "running":
            freq = self.rng.uniform(2.5, 4.0)
            amp = self.rng.uniform(0.3, 0.8)
            gyro[0, :] += amp * np.sin(_PI2 * freq * t)
            gyro[1, :] += amp * 0.6 * np.sin(_PI2 * freq * t + 0.4)
            gyro[2, :] += amp * 0.3 * np.sin(_PI2 * freq * 0.5 * t)
            gyro += self.rng.normal(0.0, 0.04, gyro.shape)

        elif activity_state == "stairs":
            freq = self.rng.uniform(1.0, 2.0)
            amp = self.rng.uniform(0.2, 0.5)
            gyro[0, :] += amp * np.sin(_PI2 * freq * t)
            gyro[2, :] += amp * 0.4 * np.sin(_PI2 * freq * t + 0.2)
            gyro += self.rng.normal(0.0, 0.03, gyro.shape)

        elif activity_state == "falling":
            spike_center = int(0.5 * n_samples)
            spike_width = int(0.1 * n_samples)
            spike_start = max(0, spike_center - spike_width)
            spike_end = min(n_samples, spike_center + spike_width)
            spike = np.zeros(n_samples)
            spike_t = np.linspace(-1.0, 1.0, spike_end - spike_start)
            spike_val = self.rng.uniform(3.0, 8.0)
            spike[spike_start:spike_end] = spike_val * np.exp(-8.0 * spike_t ** 2)
            gyro[0, :] += spike * self.rng.uniform(-0.5, 0.5)
            gyro[1, :] += spike * self.rng.uniform(-0.5, 0.5)
            gyro[2, :] += spike
            gyro += self.rng.normal(0.0, 0.05, gyro.shape)

        elif activity_state == "arm_movement":
            freq1 = self.rng.uniform(0.8, 1.5)
            freq2 = self.rng.uniform(2.0, 3.5)
            amp = self.rng.uniform(0.3, 1.0)
            gyro[0, :] += amp * np.sin(_PI2 * freq1 * t)
            gyro[1, :] += amp * 0.7 * np.sin(_PI2 * freq2 * t)
            gyro += self.rng.normal(0.0, 0.03, gyro.shape)

        return gyro.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. PPG Sensor – Photoplethysmography at 50 Hz
# ---------------------------------------------------------------------------

class PPGSensor:
    """Photoplethysmography simulation at 50 Hz.

    Generates a PPG waveform with systolic peak, dicrotic notch, and
    diastolic runoff using a Gaussian attractor model.  Pulses are
    synchronised with ECG R-peaks via a pulse transit time (PTT).
    """

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["ppg_hz"]

    @staticmethod
    def _ppg_attractors() -> Dict[str, _WaveParams]:
        return {
            "systolic": _WaveParams(a=1.0, b=0.06, theta0=0.25),
            "dicrotic": _WaveParams(a=-0.25, b=0.04, theta0=0.55),
            "diastolic": _WaveParams(a=0.15, b=0.12, theta0=0.75),
        }

    def _modify_ppg_attractors(
        self,
        attractors: Dict[str, _WaveParams],
        patient_state: Dict[str, Any],
    ) -> Dict[str, _WaveParams]:
        """Modify PPG morphology for demographics and disease."""
        modified = {k: _WaveParams(v.a, v.b, v.theta0) for k, v in attractors.items()}
        age = patient_state.get("age", 40)
        bmi = patient_state.get("bmi", 25.0)
        diseases = patient_state.get("diseases", [])

        # Age: reduced arterial compliance → dicrotic notch less prominent
        if age > 60:
            modified["dicrotic"].a *= 0.5
            modified["dicrotic"].b *= 1.3
            modified["systolic"].b *= 1.15

        # High BMI: reduced perfusion, broader pulses
        if bmi > 30:
            modified["systolic"].a *= 0.85
            modified["systolic"].b *= 1.2
            modified["dicrotic"].a *= 0.7

        # Disease effects
        for disease in diseases:
            if disease in ("acs_unstable_angina", "acs_nstemi", "acs_stemi", "ischemia"):
                # Reduced perfusion, blunted dicrotic notch
                modified["systolic"].a *= 0.8
                modified["dicrotic"].a *= 0.4

            if disease in ("dcm", "hcm"):
                # Reduced cardiac output → low amplitude, slow upstroke
                modified["systolic"].a *= 0.7
                modified["systolic"].b *= 1.4

            if disease == "hyperkalemia":
                # Reduced peripheral perfusion
                modified["systolic"].a *= 0.6

            if disease == "sepsis":
                # Vasodilation: wide dicrotic, reduced amplitude
                modified["systolic"].a *= 0.65
                modified["dicrotic"].a *= 0.3
                modified["dicrotic"].b *= 1.5

        return modified

    def generate(
        self,
        duration_seconds: float,
        ecg_signal: Optional[NDArray[np.float64]] = None,
        ecg_hz: float = 130.0,
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate a PPG signal.

        Parameters
        ----------
        duration_seconds : float
        ecg_signal : ndarray, optional
            ECG signal used for R-peak detection.  If None, heart rate is
            inferred from patient_state.
        ecg_hz : float
            Sampling rate of the ECG signal.
        patient_state : dict, optional

        Returns
        -------
        ppg : ndarray of shape (n_samples,) normalised to [0, 1].
        """
        if patient_state is None:
            patient_state = {}
        n_samples = int(duration_seconds * self.fs)
        t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        # --- Detect R-peak locations from ECG ---
        if ecg_signal is not None and len(ecg_signal) > 0:
            ecg_ds_factor = max(1, int(ecg_hz / self.fs))
            ecg_ds = ecg_signal[::ecg_ds_factor][:n_samples]
            if len(ecg_ds) < n_samples:
                ecg_ds = np.pad(ecg_ds, (0, n_samples - len(ecg_ds)))
            # Simple R-peak detection: threshold + refractory
            ecg_thresh = np.mean(ecg_ds) + 0.6 * np.std(ecg_ds)
            refractory_samples = int(0.25 * self.fs)
            r_peak_indices: List[int] = []
            i = 0
            while i < len(ecg_ds):
                if ecg_ds[i] > ecg_thresh:
                    r_peak_indices.append(i)
                    i += refractory_samples
                else:
                    i += 1
        else:
            hr_bpm = patient_state.get("heart_rate_bpm", 72.0)
            rr_samples = int(60.0 / hr_bpm * self.fs)
            r_peak_indices = list(range(rr_samples, n_samples, rr_samples))

        if len(r_peak_indices) < 2:
            r_peak_indices = list(range(int(60.0 / 72.0 * self.fs), n_samples,
                                        int(60.0 / 72.0 * self.fs)))

        # PTT (pulse transit time) in samples
        ptt_ms = self.rng.uniform(200.0, 300.0)
        ptt_samples = int(ptt_ms / 1000.0 * self.fs)

        # Build PPG using Gaussian attractors around each pulse
        attractors = self._modify_ppg_attractors(self._ppg_attractors(), patient_state)
        ppg = np.zeros(n_samples, dtype=np.float64)

        for r_idx in r_peak_indices:
            pulse_start = r_idx + ptt_samples
            if pulse_start >= n_samples:
                continue
            # Window around pulse
            win_len = int(0.8 * self.fs)
            pulse_end = min(pulse_start + win_len, n_samples)
            t_pulse = np.arange(pulse_start, pulse_end) / self.fs

            for wave in attractors.values():
                centre = (pulse_start + pulse_end) / 2.0 / self.fs
                dt_val = t_pulse - centre
                ppg[pulse_start:pulse_end] += wave.a * np.exp(
                    -dt_val ** 2 / (2.0 * wave.b ** 2)
                )

        # Normalise to [0, 1]
        ppg_min, ppg_max = ppg.min(), ppg.max()
        if ppg_max - ppg_min > 1e-10:
            ppg = (ppg - ppg_min) / (ppg_max - ppg_min)
        else:
            ppg[:] = 0.5

        # Perfusion index (PI): ratio of pulsatile to non-pulsatile component
        # Typical range 0.02% (poor perfusion) to 20% (strong signal)
        perfusion_index = patient_state.get("perfusion_index", 2.0)
        pi_scale = np.clip(perfusion_index / 2.0, 0.1, 5.0)
        ppg *= pi_scale

        # Respiratory-induced amplitude variation (10-15%)
        resp_rate = patient_state.get("respiratory_rate_bpm", 16.0) / 60.0
        resp_amp = self.rng.uniform(0.10, 0.15)
        resp_var = resp_amp * np.sin(_PI2 * resp_rate * t + self.rng.uniform(0, _PI2))
        ppg *= 1.0 + resp_var

        # Respiratory-induced baseline wander
        resp_baseline = 0.05 * np.sin(_PI2 * resp_rate * 0.5 * t + self.rng.uniform(0, _PI2))
        ppg += resp_baseline

        # Skin tone effect: darker skin reduces PPG amplitude by 10-30%
        skin_tone = patient_state.get("skin_tone", 0.0)
        if skin_tone > 0.3:
            attenuation = 0.10 + 0.20 * min((skin_tone - 0.3) / 0.7, 1.0)
            ppg *= 1.0 - attenuation

        # Age effect: reduced arterial compliance
        age = patient_state.get("age", 40)
        if age > 60:
            compliance_factor = 1.0 - 0.005 * (age - 60)
            ppg *= max(compliance_factor, 0.6)

        # Motion artifact on PPG
        activity = patient_state.get("activity_state", "rest")
        if activity != "rest":
            motion_freq = self.rng.uniform(0.5, 3.0)
            motion_amp = {"walking": 0.08, "running": 0.20, "stairs": 0.12,
                          "arm_movement": 0.15}.get(activity, 0.05)
            ppg += motion_amp * np.sin(_PI2 * motion_freq * t + self.rng.uniform(0, _PI2))
            ppg += motion_amp * 0.3 * self.rng.normal(0, 1, n_samples) * (
                1.0 + 0.5 * np.sin(_PI2 * 2.0 * t)
            )

        # Ambient light interference (occasional)
        if self.rng.random() < 0.08:
            ambient_freq = self.rng.choice([50.0, 60.0, 100.0, 120.0])
            ambient_amp = self.rng.uniform(0.01, 0.05)
            ppg += ambient_amp * np.sin(_PI2 * ambient_freq * t + self.rng.uniform(0, _PI2))

        # Measurement noise
        noise_power = self.rng.uniform(0.005, 0.02)
        ppg += self.rng.normal(0.0, noise_power, n_samples)

        # Clamp
        ppg = np.clip(ppg, 0.0, 1.0)

        return ppg.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. SpO2 Sensor – Beer-Lambert derived from PPG
# ---------------------------------------------------------------------------

class SpO2Sensor:
    """SpO2 estimation derived from dual-wavelength PPG via Beer-Lambert law."""

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["spo2_hz"]

    def generate(
        self,
        duration_seconds: float,
        ppg_signal: Optional[NDArray[np.float64]] = None,
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate SpO2 signal.

        Uses Beer-Lambert law with dual-wavelength (red 660nm, IR 940nm)
        photoplethysmography.  The R-ratio of normalised AC/DC components
        maps to SpO2 via an empirical calibration curve.

        Returns
        -------
        spo2 : ndarray of shape (n_samples,) in percent.
        """
        if patient_state is None:
            patient_state = {}
        n_samples = int(duration_seconds * self.fs)
        t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        baseline_spo2 = patient_state.get("baseline_spo2", 98.0)
        diseases = patient_state.get("diseases", [])

        # Downsample PPG to SpO2 rate
        ppg_fs = self.config.simulation.sampling_rates["ppg_hz"]
        if ppg_signal is not None and len(ppg_signal) > 0:
            ppg_ds_factor = max(1, int(ppg_fs / self.fs))
            ppg_ds = ppg_signal[::ppg_ds_factor][:n_samples]
            if len(ppg_ds) < n_samples:
                ppg_ds = np.pad(ppg_ds, (0, n_samples - len(ppg_ds)))
        else:
            ppg_ds = 0.5 + 0.2 * np.sin(_PI2 * 1.2 * t)

        # Beer-Lambert: AC/DC ratio for red and IR channels
        # Extinction coefficients (approximate, arbitrary units)
        eps_HbO2_red = 0.5
        eps_Hb_red = 1.5
        eps_HbO2_ir = 0.8
        eps_Hb_ir = 0.9

        # DC baseline (tissue, venous blood)
        dc_red = 0.5 + 0.1 * self.rng.normal(0, 1)
        dc_ir = 0.6 + 0.1 * self.rng.normal(0, 1)
        dc_red = max(dc_red, 1e-6)
        dc_ir = max(dc_ir, 1e-6)

        # AC component (pulsatile arterial blood)
        ac_red = np.abs(ppg_ds * (0.8 + 0.2 * self.rng.normal(0, 1, n_samples))) + 1e-9
        ac_ir = np.abs(ppg_ds * (1.0 + 0.1 * self.rng.normal(0, 1, n_samples))) + 1e-9

        # R-ratio
        r_ratio = (ac_red / dc_red) / (ac_ir / dc_ir)

        # Empirical calibration curve: SpO2 = 110 - 25*R (simplified)
        # More accurate: polynomial fit from lookup table
        spo2 = 110.0 - 25.0 * r_ratio

        # Add respiratory-induced desaturation events (cyclic)
        resp_rate = patient_state.get("respiratory_rate_bpm", 16.0) / 60.0
        resp_desat = 2.0 * np.sin(_PI2 * resp_rate * t)
        spo2 += resp_desat

        # Motion artifact on SpO2 (correlated with accelerometer)
        activity = patient_state.get("activity_state", "rest")
        if activity in ("running", "stairs"):
            motion_noise = self.rng.uniform(1.0, 3.0) * np.sin(
                _PI2 * self.rng.uniform(0.1, 0.5) * t
            )
            spo2 += motion_noise

        # Disease-specific effects
        for disease in diseases:
            if disease == "respiratory_failure":
                spo2 -= self.rng.uniform(5.0, 15.0)
                spo2 += 3.0 * np.sin(_PI2 * 0.05 * t)
            if disease == "pe":
                spo2 -= self.rng.uniform(3.0, 10.0)
            if disease in ("acs_stemi", "acs_nstemi"):
                spo2 -= self.rng.uniform(1.0, 5.0)
            if disease == "sepsis":
                spo2 -= self.rng.uniform(2.0, 8.0)
            if disease == "anemia":
                spo2 -= self.rng.uniform(0.0, 3.0)

        # Clamp physiological range
        spo2 = np.clip(spo2, 60.0, 100.0)

        # Measurement noise (oximeter accuracy ~2% in clinical range)
        spo2 += self.rng.normal(0.0, 0.5, n_samples)

        # Quantise to 1% steps (typical display resolution)
        spo2 = np.round(spo2)

        return spo2.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Temperature Sensor – skin temperature at 1 Hz
# ---------------------------------------------------------------------------

class TemperatureSensor:
    """Skin temperature sensor simulation at 1 Hz."""

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["temperature_hz"]

    def generate(
        self,
        duration_seconds: float,
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate skin temperature signal.

        Models basal skin temperature with circadian rhythm, exercise
        response, fever, and environmental effects.

        Returns
        -------
        temp : ndarray of shape (n_samples,) in degrees Celsius.
        """
        if patient_state is None:
            patient_state = {}
        n_samples = int(duration_seconds * self.fs)
        t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        baseline_temp = patient_state.get("skin_temperature", 34.5)
        diseases = patient_state.get("diseases", [])

        temp = np.full(n_samples, baseline_temp, dtype=np.float64)

        # Circadian modulation: core temp peaks ~18:00, troughs ~04:00
        # Skin temp lags core by ~1-2 hours
        circadian_period = 86400.0
        temp += 0.5 * np.sin(_PI2 * t / circadian_period - math.pi / 3.0)

        # Post-prandial rise (~0.3C after meals, ~4h cycle)
        meal_freq = 1.0 / 14400.0
        temp += 0.15 * np.sin(_PI2 * meal_freq * t + self.rng.uniform(0, _PI2))

        # Exercise effect: delayed rise in skin temperature
        activity = patient_state.get("activity_state", "rest")
        if activity in ("walking", "running", "stairs"):
            exercise_offset = {"walking": 1.0, "running": 2.5, "stairs": 1.5}
            base_offset = exercise_offset.get(activity, 0.0)
            # Delayed onset (exponential approach to steady state)
            tau = 300.0  # time constant ~5 min
            for i in range(n_samples):
                elapsed = t[i]
                temp[i] += base_offset * (1.0 - np.exp(-elapsed / tau))

        # Fever response with delay from core temperature
        for disease in diseases:
            if disease in ("sepsis", "fever", "infection"):
                core_temp = patient_state.get("core_temperature", 38.5)
                skin_core_coupling = 0.7
                fever_delay_min = self.rng.uniform(10.0, 30.0)
                fever_delay_s = fever_delay_min * 60.0
                for i in range(n_samples):
                    if t[i] > fever_delay_s:
                        temp[i] += (core_temp - 37.0) * skin_core_coupling
            if disease == "hypothermia":
                temp -= self.rng.uniform(2.0, 5.0)
            if disease in ("thyrotoxicosis", "hyperthyroid"):
                temp += self.rng.uniform(0.5, 1.5)
            if disease in ("hypothyroid", "myxedema"):
                temp -= self.rng.uniform(0.5, 2.0)

        # Skin impedance drift effect on temperature reading
        # (temperature sensors less affected but still present)
        impedance_drift = 0.01 * np.sin(_PI2 * 0.001 * t)
        temp += impedance_drift

        # Sensor noise (typical accuracy ~0.1C)
        temp += self.rng.normal(0.0, 0.1, n_samples)

        return temp.astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Respiration Sensor – respiratory signal at 25 Hz
# ---------------------------------------------------------------------------

class RespirationSensor:
    """Respiratory signal simulation at 25 Hz.

    Models thoracic impedance variation driven by tidal volume with
    breath-to-breath variability and pathological breathing patterns.
    """

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.fs = config.simulation.sampling_rates["respiration_hz"]

    def generate(
        self,
        duration_seconds: float,
        patient_state: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float64]:
        """Generate respiratory impedance-like signal.

        Models thoracic impedance variation driven by tidal volume with
        realistic breath-to-breath variability and pathological patterns.

        Returns
        -------
        resp : ndarray of shape (n_samples,) arbitrary units centred on 0.
        """
        if patient_state is None:
            patient_state = {}
        n_samples = int(duration_seconds * self.fs)
        t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

        resp_rate_bpm = patient_state.get("respiratory_rate_bpm", 16.0)
        resp_rate_hz = resp_rate_bpm / 60.0
        tidal_volume = patient_state.get("tidal_volume_ml", 450.0)
        diseases = patient_state.get("diseases", [])

        # Tidal volume normalisation (relative amplitude)
        tv_norm = tidal_volume / 450.0

        # Breath-to-breath variability (CV ~15-25%)
        cv = self.rng.uniform(0.15, 0.25)
        # Random walk phase noise for irregular breathing
        phase_noise = np.cumsum(self.rng.normal(0, cv * 0.01, n_samples))
        phase = _PI2 * resp_rate_hz * t + phase_noise

        # Inspiration is slightly longer than expiration (I:E ~ 1:1.5)
        # Use arcsin transform to create asymmetric waveform
        raw_sin = np.sin(phase)
        resp = tv_norm * np.sign(raw_sin) * np.abs(raw_sin) ** 0.8

        # Disease-specific patterns
        for disease in diseases:
            if disease == "heart_failure_severe":
                # Cheyne-Stokes: crescendo-decrescendo modulation
                cs_period = self.rng.uniform(30.0, 60.0)
                cs_mod = 0.5 * (1.0 + np.sin(_PI2 * t / cs_period))
                resp *= cs_mod

            if disease == "acidosis":
                # Kussmaul: deep and rapid
                resp *= 2.0
                resp_rate_hz *= 1.5

            if disease == "respiratory_failure":
                # Shallow rapid breathing (tachypnea)
                resp *= 0.4
                resp_rate_hz *= 1.8

            if disease == "obstructive_apnea":
                # Periodic breathing with apnea pauses
                apnea_period = self.rng.uniform(15.0, 30.0)
                apnea_window = 3.0
                mask = np.ones(n_samples)
                for i in range(n_samples):
                    cycle_pos = t[i] % apnea_period
                    if cycle_pos > (apnea_period - apnea_window):
                        mask[i] = 0.0
                resp *= mask

            if disease == "neuromuscular":
                # Progressive weakening: decreasing tidal volume
                decline = np.linspace(1.0, 0.3, n_samples)
                resp *= decline

            if disease == "pneumonia":
                # Tachypnea with reduced tidal volume, splinting
                resp *= 0.6
                resp_rate_hz *= 1.6
                # Add sharp inspiratory gasps
                n_gaspers = int(duration_seconds * resp_rate_hz * 0.3)
                for _ in range(n_gaspers):
                    gasp_idx = self.rng.integers(0, n_samples)
                    gasp_width = int(0.05 * self.fs)
                    gasp_end = min(gasp_idx + gasp_width, n_samples)
                    resp[gasp_idx:gasp_end] *= 2.0

            if disease == "asthma":
                # Prolonged expiration, wheezing artifact
                resp_rate_hz *= 0.8
                # High-frequency wheeze overlay
                wheeze_freq = self.rng.uniform(200.0, 400.0)
                wheeze_amp = 0.05 * tv_norm
                wheeze = wheeze_amp * np.sin(_PI2 * wheeze_freq * t)
                # Wheeze only during expiration
                exp_mask = (np.sin(phase) < 0).astype(float)
                resp += wheeze * exp_mask

            if disease == "copd":
                # Air trapping: elevated baseline, slow expiration
                resp += 0.1 * tv_norm
                resp_rate_hz *= 0.9

        # Additive noise (mechanical / electrode)
        noise_level = 0.02 * tv_norm
        resp += self.rng.normal(0.0, noise_level, n_samples)

        return resp.astype(np.float32)


# ---------------------------------------------------------------------------
# 8. Sensor Artifact Model
# ---------------------------------------------------------------------------

class SensorArtifactModel:
    """Comprehensive artifact injection for all sensor modalities.

    Models ADC quantisation, clock drift, jitter, packet loss, Bluetooth
    latency, lead-vector changes, skin impedance drift, motion artifacts,
    battery effects, saturation, and compression artifacts.
    """

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.sensor_cfg = config.sensor

    def _quantise_adc(self, signal_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply ADC quantisation (12-bit, range ±3.2 mV)."""
        levels = 2 ** self.sensor_cfg.adc_resolution_bits
        vref = self.sensor_cfg.adc_vref
        v_range = 2.0 * (vref / 2.0)
        step = v_range / levels
        quantised = np.round(signal_arr / step) * step
        return quantised

    def _apply_clock_drift(
        self, signal_arr: NDArray[np.float64], hz: float, time_hours: float
    ) -> NDArray[np.float64]:
        """Apply cumulative clock drift (ppm)."""
        drift_ppm = self.sensor_cfg.clock_drift_ppm
        drift_fraction = drift_ppm * 1e-6 * time_hours * 3600.0

        if signal_arr.ndim == 1:
            n_samples = len(signal_arr)
            original_time = np.arange(n_samples) / hz
            drifted_time = original_time * (1.0 + drift_fraction)
            new_indices = drifted_time * hz
            valid = (new_indices >= 0) & (new_indices < n_samples - 1)
            result = np.zeros_like(signal_arr)
            idx_low = np.floor(new_indices[valid]).astype(int)
            idx_high = np.minimum(idx_low + 1, n_samples - 1)
            frac = new_indices[valid] - idx_low
            result[valid] = signal_arr[idx_low] * (1.0 - frac) + signal_arr[idx_high] * frac
            return result
        else:
            result = np.empty_like(signal_arr)
            for axis_idx in range(signal_arr.shape[0]):
                result[axis_idx] = self._apply_clock_drift(
                    signal_arr[axis_idx], hz, time_hours
                )
            return result

    def _apply_jitter_1d(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Apply timestamp jitter (Gaussian σ=2ms) to a 1-D array."""
        jitter_s = self.sensor_cfg.timestamp_jitter_ms / 1000.0
        n_samples = len(signal_arr)
        t = np.arange(n_samples) / hz
        t_jittered = t + self.rng.normal(0, jitter_s, n_samples)
        new_indices = t_jittered * hz
        valid = (new_indices >= 0) & (new_indices < n_samples - 1)
        result = np.zeros_like(signal_arr)
        idx_low = np.floor(new_indices[valid]).astype(int)
        idx_high = np.minimum(idx_low + 1, n_samples - 1)
        frac = new_indices[valid] - idx_low
        result[valid] = signal_arr[idx_low] * (1.0 - frac) + signal_arr[idx_high] * frac
        return result

    def _apply_jitter(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Apply timestamp jitter, handling multi-dimensional arrays."""
        if signal_arr.ndim == 1:
            return self._apply_jitter_1d(signal_arr, hz)
        result = np.empty_like(signal_arr)
        for i in range(signal_arr.shape[0]):
            result[i] = self._apply_jitter_1d(signal_arr[i], hz)
        return result

    def _apply_packet_loss_1d(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Simulate bursty packet loss (~2% overall rate) on a 1-D array."""
        loss_rate = self.sensor_cfg.packet_loss_rate
        n_samples = len(signal_arr)
        result = signal_arr.copy()

        n_bursts = int(n_samples * loss_rate / (0.05 * hz)) + 1
        for _ in range(n_bursts):
            burst_start = self.rng.integers(0, max(1, n_samples - 1))
            burst_len = self.rng.integers(1, max(2, int(0.05 * hz)))
            burst_end = min(burst_start + burst_len, n_samples)
            result[burst_start:burst_end] = 0.0

        return result

    def _apply_packet_loss(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Simulate bursty packet loss, handling multi-dimensional arrays."""
        if signal_arr.ndim == 1:
            return self._apply_packet_loss_1d(signal_arr, hz)
        result = np.empty_like(signal_arr)
        for i in range(signal_arr.shape[0]):
            result[i] = self._apply_packet_loss_1d(signal_arr[i], hz)
        return result

    def _apply_bluetooth_latency_1d(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Apply log-normal Bluetooth latency to a 1-D array."""
        mean_ms = self.sensor_cfg.bluetooth_latency_ms_mean
        std_ms = self.sensor_cfg.bluetooth_latency_ms_std
        mu = math.log(mean_ms ** 2 / math.sqrt(std_ms ** 2 + mean_ms ** 2))
        sigma = math.sqrt(math.log(1.0 + (std_ms / mean_ms) ** 2))
        n_samples = len(signal_arr)
        latencies_ms = self.rng.lognormal(mu, sigma, n_samples)
        latencies_s = latencies_ms / 1000.0
        result = np.zeros_like(signal_arr)
        shift_samples = (latencies_s * hz).astype(int)
        for i in range(n_samples):
            src = i - shift_samples[i]
            if 0 <= src < n_samples:
                result[i] = signal_arr[src]
        return result

    def _apply_bluetooth_latency(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Apply log-normal Bluetooth latency, handling multi-dimensional arrays."""
        if signal_arr.ndim == 1:
            return self._apply_bluetooth_latency_1d(signal_arr, hz)
        result = np.empty_like(signal_arr)
        for i in range(signal_arr.shape[0]):
            result[i] = self._apply_bluetooth_latency_1d(signal_arr[i], hz)
        return result

    def _apply_lead_vector_change(
        self, signal_arr: NDArray[np.float64], hz: float, time_hours: float
    ) -> NDArray[np.float64]:
        """Apply ±30% R-peak amplitude variation over time."""
        n_samples = signal_arr.shape[-1]
        t = np.arange(n_samples) / hz
        variation_rate = self.rng.uniform(0.01, 0.05)
        amplitude_mod = 1.0 + 0.3 * np.sin(_PI2 * variation_rate * t + self.rng.uniform(0, _PI2))
        if signal_arr.ndim == 1:
            return signal_arr * amplitude_mod
        else:
            return signal_arr * amplitude_mod[np.newaxis, :]

    def _apply_skin_impedance(
        self, signal_arr: NDArray[np.float64], hz: float, time_hours: float
    ) -> NDArray[np.float64]:
        """Apply skin impedance drift (electrode drying)."""
        z0 = self.rng.uniform(*self.sensor_cfg.skin_impedance_range)
        drying_rate = self.sensor_cfg.electrode_drying_rate
        n_samples = signal_arr.shape[-1]
        t = np.arange(n_samples) / hz
        impedance = z0 + drying_rate * t / 3600.0
        impedance_norm = impedance / np.max(impedance)
        attenuation = 1.0 / (1.0 + 0.01 * impedance_norm)
        if signal_arr.ndim == 1:
            return signal_arr * attenuation
        else:
            return signal_arr * attenuation[np.newaxis, :]

    def _apply_motion_artifacts(
        self,
        signal_arr: NDArray[np.float64],
        hz: float,
        activity_state: str,
    ) -> NDArray[np.float64]:
        """Apply motion artifacts: baseline wander, EMG, electrode motion."""
        n_samples = signal_arr.shape[-1]
        t = np.arange(n_samples) / hz
        result = signal_arr.copy()

        if activity_state == "rest":
            return result

        bw_amp = {"walking": 0.15, "running": 0.35, "stairs": 0.25, "arm_movement": 0.20}.get(
            activity_state, 0.1
        )
        bw_freq = self.rng.uniform(0.1, 0.5)
        bw = bw_amp * np.sin(_PI2 * bw_freq * t + self.rng.uniform(0, _PI2))
        if signal_arr.ndim == 1:
            result += bw
        else:
            result += bw[np.newaxis, :]

        emg_amp = {"walking": 0.05, "running": 0.12, "stairs": 0.08, "arm_movement": 0.07}.get(
            activity_state, 0.03
        )
        emg_env = np.abs(np.sin(_PI2 * 30.0 * t))
        emg_noise = emg_env * self.rng.normal(0, 1, n_samples)
        if signal_arr.ndim == 1:
            result += emg_amp * emg_noise
        else:
            result += emg_amp * emg_noise[np.newaxis, :]

        return result

    def _apply_battery_effects(
        self, signal_arr: NDArray[np.float64], patient_state: Dict[str, Any]
    ) -> NDArray[np.float64]:
        """Apply battery voltage effects on signal quality."""
        battery_level = patient_state.get("battery_level", 1.0)
        v_min, v_max = self.sensor_cfg.battery_voltage_range
        v_nominal = v_max

        if battery_level > 0.7:
            return signal_arr
        elif battery_level > 0.3:
            gain = 0.95 + 0.05 * battery_level
            return signal_arr * gain
        else:
            gain = 0.6 + 0.35 * battery_level
            result = signal_arr * gain
            noise_shape = signal_arr.shape
            noise = self.rng.normal(0, 0.02 * (1.0 - battery_level), noise_shape)
            result = result + noise
            return result

    def _apply_saturation(
        self, signal_arr: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Clip signal at ADC saturation limits."""
        threshold = self.sensor_cfg.sensor_saturation_threshold
        vref = self.sensor_cfg.adc_vref
        clip_level = threshold * vref / 2.0
        return np.clip(signal_arr, -clip_level, clip_level)

    def _apply_baseline_wander(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Apply 0.1-0.5 Hz baseline wander from respiration and movement."""
        n_samples = signal_arr.shape[-1]
        t = np.arange(n_samples) / hz
        bw_freq = self.rng.uniform(0.1, 0.5)
        bw_amp = self.rng.uniform(0.02, 0.08)
        bw = bw_amp * np.sin(_PI2 * bw_freq * t + self.rng.uniform(0, _PI2))
        if signal_arr.ndim == 1:
            return signal_arr + bw
        else:
            return signal_arr + bw[np.newaxis, :]

    def _apply_50hz_interference(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Apply powerline interference (50 or 60 Hz)."""
        if self.rng.random() > 0.12:
            return signal_arr
        pl_freq = self.rng.choice([50.0, 60.0])
        pl_amp = self.rng.uniform(0.005, 0.02)
        n_samples = signal_arr.shape[-1]
        t = np.arange(n_samples) / hz
        interference = pl_amp * np.sin(_PI2 * pl_freq * t + self.rng.uniform(0, _PI2))
        if signal_arr.ndim == 1:
            return signal_arr + interference
        else:
            return signal_arr + interference[np.newaxis, :]

    def _apply_electrode_pop(
        self, signal_arr: NDArray[np.float64], hz: float
    ) -> NDArray[np.float64]:
        """Simulate occasional electrode pop artifacts."""
        if self.rng.random() > 0.05:
            return signal_arr
        result = signal_arr.copy()
        n_samples = signal_arr.shape[-1]
        n_pops = self.rng.integers(1, 4)
        for _ in range(n_pops):
            pop_idx = self.rng.integers(0, max(1, n_samples - 1))
            pop_width = self.rng.integers(1, max(2, int(0.02 * hz)))
            pop_end = min(pop_idx + pop_width, n_samples)
            pop_amp = self.rng.uniform(0.3, 1.0) * self.rng.choice([-1.0, 1.0])
            if result.ndim == 1:
                result[pop_idx:pop_end] = pop_amp
            else:
                result[:, pop_idx:pop_end] = pop_amp
        return result

    def _apply_compression_artifacts(
        self, signal_arr: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Simulate lossy compression ringing artifacts."""
        if self.rng.random() > 0.15:
            return signal_arr
        n_samples = signal_arr.shape[-1]
        n_components = self.rng.integers(3, 8)
        result = signal_arr.copy()
        t = np.arange(n_samples) / 100.0
        for _ in range(n_components):
            freq = self.rng.uniform(10.0, 50.0)
            amp = self.rng.uniform(0.001, 0.01)
            phase = self.rng.uniform(0, _PI2)
            ringing = amp * np.sin(_PI2 * freq * t + phase)
            if signal_arr.ndim == 1:
                result += ringing
            else:
                result += ringing[np.newaxis, :]
        return result

    def apply_artifacts(
        self,
        signals_dict: Dict[str, NDArray[np.float64]],
        patient_state: Dict[str, Any],
        time_hours: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, NDArray[np.float64]]:
        """Apply all artifacts to a dictionary of signals.

        Parameters
        ----------
        signals_dict : dict
            Mapping of signal names to arrays.  Recognised keys include
            ``'ecg'``, ``'accelerometer'``, ``'gyroscope'``, ``'ppg'``,
            ``'spo2'``, ``'temperature'``, ``'respiration'``.
        patient_state : dict
            Patient context including ``activity_state``, ``battery_level``.
        time_hours : float
            Elapsed recording time in hours.
        rng : Generator, optional
            Override RNG for deterministic artifact injection.

        Returns
        -------
        modified : dict with same keys, artifact-contaminated arrays.
        """
        if rng is not None:
            self.rng = rng

        activity = patient_state.get("activity_state", "rest")
        modified: Dict[str, NDArray[np.float64]] = {}

        hz_map = {
            "ecg": self.config.simulation.sampling_rates["ecg_hz"],
            "ppg": self.config.simulation.sampling_rates["ppg_hz"],
            "spo2": self.config.simulation.sampling_rates["spo2_hz"],
            "temperature": self.config.simulation.sampling_rates["temperature_hz"],
            "respiration": self.config.simulation.sampling_rates["respiration_hz"],
            "accelerometer": self.config.simulation.sampling_rates["accelerometer_hz"],
            "gyroscope": self.config.simulation.sampling_rates["gyroscope_hz"],
        }

        for key, sig in signals_dict.items():
            if key not in hz_map:
                modified[key] = sig
                continue

            hz = hz_map[key]
            s = sig.astype(np.float64)

            # Universal artifacts (temp excluded from most biopotential effects)
            if key != "temperature":
                s = self._apply_clock_drift(s, hz, time_hours)
                s = self._apply_jitter(s, hz)
                s = self._apply_packet_loss(s, hz)
                s = self._apply_bluetooth_latency(s, hz)
                s = self._apply_skin_impedance(s, hz, time_hours)
                s = self._apply_battery_effects(s, patient_state)
                s = self._apply_compression_artifacts(s)
                s = self._apply_baseline_wander(s, hz)
                s = self._apply_50hz_interference(s, hz)

            # ECG / PPG specific
            if key in ("ecg", "ppg"):
                s = self._apply_lead_vector_change(s, hz, time_hours)
                s = self._apply_motion_artifacts(s, hz, activity)
                s = self._apply_electrode_pop(s, hz)

            # ADC quantisation only for raw biopotential signals
            if key in ("ecg", "ppg", "respiration"):
                s = self._quantise_adc(s)

            # Saturation only for raw biopotential signals
            if key in ("ecg", "ppg", "respiration"):
                s = self._apply_saturation(s)

            modified[key] = s.astype(np.float32)

        return modified


# ---------------------------------------------------------------------------
# 9. MultimodalSensorSuite – top-level orchestrator
# ---------------------------------------------------------------------------

class MultimodalSensorSuite:
    """Generates all sensor modalities from a coherent physiological state.

    ECG R-peaks define the cardiac timing.  PPG pulses follow R-peaks with
    a pulse transit time.  Accelerometer and gyroscope reflect the current
    activity.  All modalities share circadian phase and receive consistent
    cross-modal artifact contamination.
    """

    def __init__(self, config: OHCAConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.reproducibility.seed)
        self.ecg_sensor = ECGSensor(config, self.rng)
        self.accel_sensor = AccelerometerSensor(config, self.rng)
        self.gyro_sensor = GyroscopeSensor(config, self.rng)
        self.ppg_sensor = PPGSensor(config, self.rng)
        self.spo2_sensor = SpO2Sensor(config, self.rng)
        self.temp_sensor = TemperatureSensor(config, self.rng)
        self.resp_sensor = RespirationSensor(config, self.rng)
        self.artifact_model = SensorArtifactModel(config, self.rng)

    def _derive_patient_state(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields are present with physiological defaults."""
        state = dict(patient_state)
        state.setdefault("age", 40)
        state.setdefault("sex", "male")
        state.setdefault("bmi", 25.0)
        state.setdefault("heart_rate_bpm", 72.0)
        state.setdefault("respiratory_rate_bpm", 16.0)
        state.setdefault("tidal_volume_ml", 450.0)
        state.setdefault("skin_tone", 0.0)
        state.setdefault("skin_temperature", 34.5)
        state.setdefault("core_temperature", 37.0)
        state.setdefault("baseline_spo2", 98.0)
        state.setdefault("activity_state", "rest")
        state.setdefault("battery_level", 1.0)
        state.setdefault("diseases", [])
        state.setdefault("hrv_params", {})
        return state

    @staticmethod
    def _ecg_to_r_peak_signal(
        ecg: NDArray[np.float32],
        ecg_hz: float,
        output_hz: float,
        refractory_ms: float = 250.0,
    ) -> NDArray[np.float32]:
        """Convert an ECG signal to a binary R-peak indicator signal.

        Parameters
        ----------
        ecg : ndarray
            ECG signal.
        ecg_hz : float
            Sampling rate of the ECG signal.
        output_hz : float
            Desired output sampling rate.
        refractory_ms : float
            Minimum interval between R-peaks in milliseconds.

        Returns
        -------
        r_peak_signal : ndarray of shape (n_output_samples,) with 1 at
        R-peak locations and 0 elsewhere.
        """
        refractory_samples = int(refractory_ms / 1000.0 * ecg_hz)
        threshold = np.mean(ecg) + 0.6 * np.std(ecg)
        r_indices: List[int] = []
        i = 0
        while i < len(ecg):
            if ecg[i] > threshold:
                r_indices.append(i)
                i += refractory_samples
            else:
                i += 1
        n_output = int(len(ecg) / ecg_hz * output_hz)
        r_signal = np.zeros(n_output, dtype=np.float32)
        for idx in r_indices:
            out_idx = int(idx / ecg_hz * output_hz)
            if 0 <= out_idx < n_output:
                r_signal[out_idx] = 1.0
        return r_signal

    def generate_all(
        self,
        patient_state: Dict[str, Any],
        duration_seconds: float = 300.0,
        config: Optional[OHCAConfig] = None,
        apply_artifacts: bool = True,
        time_hours: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate all sensor modalities from a single patient state.

        Parameters
        ----------
        patient_state : dict
            Patient demographics, diseases, activity, and physiological
            parameters.
        duration_seconds : float
            Duration of each signal in seconds (default 300 s = 5 min).
        config : OHCAConfig, optional
            Override configuration.
        apply_artifacts : bool
            Whether to apply cross-modal artifacts.
        time_hours : float
            Elapsed recording time for drift artifacts.

        Returns
        -------
        dict with keys:
            ``'ecg'``, ``'accelerometer'``, ``'gyroscope'``, ``'ppg'``,
            ``'spo2'``, ``'temperature'``, ``'respiration'``,
            ``'r_peak_indices'``, ``'metadata'``.
        """
        if config is not None:
            self.config = config
        state = self._derive_patient_state(patient_state)

        hr_bpm = state["heart_rate_bpm"]
        resp_rate = state["respiratory_rate_bpm"]
        activity = state["activity_state"]
        hrv_params = state.get("hrv_params", {})

        # --- ECG ---
        ecg = self.ecg_sensor.generate(
            duration_seconds=duration_seconds,
            heart_rate_bpm=hr_bpm,
            hrv_params=hrv_params,
            patient_state=state,
        )

        # --- Detect R-peaks from ECG for synchronisation ---
        ecg_hz = self.config.simulation.sampling_rates["ecg_hz"]
        ecg_ds_factor = max(1, int(ecg_hz / 10.0))
        ecg_ds = ecg[::ecg_ds_factor]
        ecg_thresh = np.mean(ecg_ds) + 0.6 * np.std(ecg_ds)
        refractory = int(0.25 * 10.0)
        r_indices: List[int] = []
        i = 0
        while i < len(ecg_ds):
            if ecg_ds[i] > ecg_thresh:
                r_indices.append(i * ecg_ds_factor)
                i += refractory * ecg_ds_factor
            else:
                i += 1

        # --- PPG ---
        ppg = self.ppg_sensor.generate(
            duration_seconds=duration_seconds,
            ecg_signal=ecg,
            ecg_hz=ecg_hz,
            patient_state=state,
        )

        # --- SpO2 ---
        spo2 = self.spo2_sensor.generate(
            duration_seconds=duration_seconds,
            ppg_signal=ppg,
            patient_state=state,
        )

        # --- Temperature ---
        temperature = self.temp_sensor.generate(
            duration_seconds=duration_seconds,
            patient_state=state,
        )

        # --- Respiration ---
        respiration = self.resp_sensor.generate(
            duration_seconds=duration_seconds,
            patient_state=state,
        )

        # --- Accelerometer ---
        accelerometer = self.accel_sensor.generate(
            duration_seconds=duration_seconds,
            activity_state=activity,
            patient_state=state,
        )

        # --- Gyroscope ---
        gyroscope = self.gyro_sensor.generate(
            duration_seconds=duration_seconds,
            activity_state=activity,
            patient_state=state,
        )

        # --- Cross-modal artifact injection ---
        signals_dict: Dict[str, NDArray[np.float64]] = {
            "ecg": ecg,
            "ppg": ppg,
            "spo2": spo2,
            "temperature": temperature,
            "respiration": respiration,
            "accelerometer": accelerometer,
            "gyroscope": gyroscope,
        }

        if apply_artifacts:
            signals_dict = self.artifact_model.apply_artifacts(
                signals_dict, state, time_hours, self.rng
            )

        # --- Metadata ---
        metadata = {
            "duration_seconds": duration_seconds,
            "heart_rate_bpm": hr_bpm,
            "respiratory_rate_bpm": resp_rate,
            "activity_state": activity,
            "diseases": list(state.get("diseases", [])),
            "age": state["age"],
            "sex": state["sex"],
            "bmi": state["bmi"],
            "time_hours": time_hours,
            "sampling_rates": dict(self.config.simulation.sampling_rates),
            "ptt_ms": ptt_ms if "ptt_ms" in dir() else None,
        }

        return {
            "ecg": signals_dict["ecg"],
            "accelerometer": signals_dict["accelerometer"],
            "gyroscope": signals_dict["gyroscope"],
            "ppg": signals_dict["ppg"],
            "spo2": signals_dict["spo2"],
            "temperature": signals_dict["temperature"],
            "respiration": signals_dict["respiration"],
            "r_peak_indices": np.array(r_indices, dtype=np.int64),
            "r_peak_signal": self._ecg_to_r_peak_signal(
                signals_dict["ecg"], ecg_hz, ecg_hz
            ),
            "metadata": metadata,
        }

    def generate_ecg_only(
        self,
        patient_state: Dict[str, Any],
        duration_seconds: float = 300.0,
        hrv_params: Optional[Dict[str, Any]] = None,
    ) -> NDArray[np.float32]:
        """Generate only the ECG signal (faster for single-modality use)."""
        state = self._derive_patient_state(patient_state)
        return self.ecg_sensor.generate(
            duration_seconds=duration_seconds,
            heart_rate_bpm=state["heart_rate_bpm"],
            hrv_params=hrv_params or state.get("hrv_params", {}),
            patient_state=state,
        )

    def generate_ppg_only(
        self,
        patient_state: Dict[str, Any],
        duration_seconds: float = 300.0,
    ) -> NDArray[np.float32]:
        """Generate only the PPG signal."""
        state = self._derive_patient_state(patient_state)
        return self.ppg_sensor.generate(
            duration_seconds=duration_seconds,
            patient_state=state,
        )
