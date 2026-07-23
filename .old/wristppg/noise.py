"""
Photodetector / analog front-end noise model, replacing a flat Gaussian
approximation with distinct, physically-named noise sources summed
according to their known statistics.

Evidence base
-------------
- Photon shot noise: Poisson-distributed photon arrivals -> for large
  photon counts, shot-noise standard deviation ~ sqrt(signal level)
  (standard photodiode noise theory), e.g. Hobbs, "Building Electro-
  Optical Systems", 2nd ed., Ch. 9.
- Dark current noise: also shot-noise-like but on the (signal-
  independent) dark current itself, plus a slow thermal-dependent mean
  offset: Hobbs, Ch. 9; Graeme, "Photodiode Amplifiers: Op Amp Solutions".
- Thermal (Johnson-Nyquist) noise in the transimpedance amplifier:
  white, Gaussian, PSD = 4*k*T*R: Nyquist, Phys Rev 32:110-113 (1928).
- 1/f (flicker) electronic noise in the amplifier chain: van der Ziel,
  "Noise in Solid State Devices and Circuits" (1986).
- ADC quantization noise: uniformly distributed over one LSB for a
  well-dithered ADC (Widrow's quantization theorem): Widrow & Kollar,
  "Quantization Noise" (2008).
- LED flicker / drive-current ripple from switching power regulation and
  PWM dimming: generic LED-driver engineering knowledge.
- Power-supply ripple coupling into the analog front end: standard mixed-
  signal PCB design knowledge (Ott, "Noise Reduction Techniques in
  Electronic Systems").
- Temperature drift and photodiode/LED aging as slow, low-frequency
  gain/offset drift: generic optoelectronic component aging behavior
  (manufacturer application notes; not device-specific).

What is heuristic here
-----------------------
- Exact noise magnitudes (shot-noise proportionality constant, thermal
  noise floor in dB, 1/f corner frequency, aging drift rate) are
  reasonable engineering defaults for a compact reflectance-mode PPG
  front end, not measured from a specific Apple Watch unit (proprietary,
  unpublished).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseParams:
    shot_noise_coeff: float = 0.02
    dark_current_level: float = 0.01
    thermal_snr_db: float = 34.0
    flicker_corner_hz: float = 0.5
    flicker_amplitude: float = 0.015
    led_flicker_amplitude: float = 0.008
    led_flicker_freq_hz: float = 120.0  # mains-related switching artifact, generic assumption
    supply_ripple_amplitude: float = 0.004
    supply_ripple_freq_hz: float = 217.0  # generic switching-regulator frequency, illustrative only
    temp_drift_amplitude: float = 0.01
    aging_offset: float = 0.0


class NoiseModel:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def _pink_noise(self, n: int, fs: float, corner_hz: float, amplitude: float) -> np.ndarray:
        """Approximate 1/f noise via spectral shaping of white noise."""
        white = self.rng.normal(0, 1, n)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.fft.rfft(white)
        shaping = 1.0 / np.sqrt(np.maximum(freqs, corner_hz / 10.0))
        shaping /= shaping[1] if len(shaping) > 1 else 1.0
        shaped = np.fft.irfft(spectrum * shaping, n=n)
        shaped = shaped / (np.std(shaped) + 1e-9)
        return amplitude * shaped

    def apply(self, signal: np.ndarray, fs: float, params: NoiseParams,
              melanin_fraction: float = 0.3, device_age_days: float = 0.0) -> np.ndarray:
        n = len(signal)
        t = np.arange(n) / fs

        signal_level = np.abs(signal) + 1.0
        shot = self.rng.normal(0, 1, n) * np.sqrt(signal_level) * params.shot_noise_coeff

        dark = self.rng.normal(0, 1, n) * np.sqrt(max(params.dark_current_level, 1e-6))

        signal_power = np.mean(signal ** 2) + 1e-12
        # darker skin -> less light returned -> lower effective SNR at
        # fixed electronic noise floor (Fallow et al. 2013, qualitative)
        thermal_snr_db = params.thermal_snr_db - 4.0 * melanin_fraction
        thermal_power = signal_power / (10 ** (thermal_snr_db / 10))
        thermal = self.rng.normal(0, np.sqrt(thermal_power), n)

        flicker = self._pink_noise(n, fs, params.flicker_corner_hz, params.flicker_amplitude * np.std(signal))

        led_flicker = params.led_flicker_amplitude * np.std(signal) * np.sin(
            2 * np.pi * params.led_flicker_freq_hz * t + self.rng.uniform(0, 2 * np.pi))

        supply_ripple = params.supply_ripple_amplitude * np.std(signal) * np.sin(
            2 * np.pi * params.supply_ripple_freq_hz * t + self.rng.uniform(0, 2 * np.pi))

        temp_drift = params.temp_drift_amplitude * np.std(signal) * np.sin(
            2 * np.pi * self.rng.uniform(0.001, 0.005) * t)

        aging_drift = params.aging_offset * np.log1p(device_age_days / 30.0)

        total_noise = shot + dark + thermal + flicker + led_flicker + supply_ripple + temp_drift + aging_drift
        return (signal + total_noise).astype(np.float32)