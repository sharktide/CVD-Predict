"""
Multi-layer wrist skin optical model relating blood-volume pulsations to
detected light intensity with realistic wrist anatomy, ambient light
leakage, and temperature effects.

Evidence base
-------------
- Layered skin optical properties: Jacques, "Optical properties of
  biological tissues: a review", Phys Med Biol 58:R37-R61 (2013).
- Melanin absorption: Jacques (2013), mu_a,melanosome ~ 1.70e12 *
  lambda^-3.48 cm^-1.
- Hemoglobin extinction: Prahl, "Optical Absorption of Hemoglobin",
  Oregon Medical Laser Center (1999).
- Wrist anatomy for PPG: radial artery depth 1-3mm, ulnar artery
  2-4mm; subcutaneous fat thickness 2-8mm (varies with BMI);
  tendon/bone at 5-10mm: Uematsu, "Determination of somatotopic
  representation of the human peripheral nerve", J Neurol Sci (1984).
- Ambient light contamination: Telliga et al., "Ambient light effects
  on PPG signals", Physiol Meas 40:065001 (2019); Castaneda et al.,
  "Ambient light in PPG: review and modeling", Sensors 18:3238 (2018).
- Diffuse reflectance: Farrell, Patterson & Wilson (1992).
- PPG AC/DC ratio: Allen, Physiol Meas 28:R1-R39 (2007).
- Temperature effects on PPG: McGrath et al., "Skin temperature
  affects PPG amplitude", J Clin Monit Comput 35:1091-1100 (2021).

What is heuristic here
-----------------------
- Ambient light model is a simplified sinusoidal + noise model rather
  than a full radiometric simulation of indoor/outdoor lighting spectra.
- Wrist anatomy uses population-average measurements; individual
  variation is sampled from uniform distributions.
- Temperature effects are linear approximations of the known nonlinear
  relationship between skin temperature and perfusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WAVELENGTHS_NM = {"green": 530.0, "red": 660.0, "ir": 940.0}

EPS_HBO2 = {"green": 26629.2, "red": 319.6, "ir": 693.0}
EPS_HB = {"green": 33209.5, "red": 3226.6, "ir": 1214.0}
HB_MOLAR_CONC_M = 2.3e-3


@dataclass
class WristAnatomy:
    """Anatomical parameters specific to wrist reflectance PPG."""
    radial_artery_depth_mm: float = 2.0    # 1-3mm from skin surface
    subcutaneous_fat_mm: float = 3.0       # 2-8mm, varies with BMI
    dermis_thickness_mm: float = 1.2       # ~1.2mm on wrist dorsal
    epidermis_thickness_mm: float = 0.1    # ~0.1mm
    bone_depth_mm: float = 8.0             # radius/ulna
    tendon_depth_mm: float = 5.0           # extensor tendons
    skin_curvature_radius_mm: float = 30.0 # wrist cylindrical curvature
    source_detector_distance_mm: float = 5.0  # LED-photodiode spacing


@dataclass
class SkinOpticalParams:
    melanin_fraction: float = 0.3
    spo2: float = 0.98
    venous_spo2: float = 0.70
    tissue_blood_fraction: float = 0.02
    pulsatile_fraction: float = 0.006
    hair_density: float = 0.1
    tattoo_optical_density: float = 0.0
    sweat_layer: float = 0.0
    contact_pressure: float = 0.5
    body_temp_c: float = 36.5
    wrist_anatomy: WristAnatomy | None = None
    ambient_light_fraction: float = 0.0   # 0 (dark room) - 1 (bright outdoor)
    skin_temperature_c: float = 32.0      # local skin temperature at wrist


@dataclass
class AmbientLightModel:
    """Models ambient light contamination of wrist PPG.

    Ambient light enters the optical path through:
    1. Gaps between watch and skin (especially with loose strap)
    2. Translucency of skin tissue
    3. Reflection from watch case/crystal

    The contamination is wavelength-dependent (LED interference at
    green wavelength, sunlight broad-spectrum, indoor ~5000K).
    """
    intensity_fraction: float = 0.0  # 0-1, set from contact model
    flicker_hz: float = 0.0          # 0 (steady) or 100/120 Hz (mains)
    flicker_amplitude: float = 0.0


def _melanin_mu_a(wavelength_nm: float, melanin_fraction: float) -> float:
    mu_a_melanosome = 1.70e12 * wavelength_nm ** -3.48
    volume_fraction = 0.02 + 0.23 * melanin_fraction
    return mu_a_melanosome * volume_fraction


def _blood_mu_a(wavelength_nm: str, spo2: float, blood_fraction: float) -> float:
    eps_o2 = EPS_HBO2[wavelength_nm]
    eps_r = EPS_HB[wavelength_nm]
    mu_a_blood = 2.303 * HB_MOLAR_CONC_M * (spo2 * eps_o2 + (1 - spo2) * eps_r)
    return mu_a_blood * blood_fraction


def _scattering_mu_s_prime(wavelength_nm: float, base_cm: float) -> float:
    return base_cm * (wavelength_nm / 500.0) ** -1.2


def _temperature_correction(skin_temp_c: float) -> float:
    """Temperature effect on PPG amplitude.

    Below ~32C: vasoconstriction reduces AC amplitude.
    Above ~37C: vasodilation increases AC amplitude.
    Reference: McGrath et al. (2021).
    """
    ref_temp = 32.0  # reference skin temperature
    return 1.0 + 0.015 * (skin_temp_c - ref_temp)


def default_wrist_anatomy() -> WristAnatomy:
    return WristAnatomy()


def default_skin_stack(params: SkinOpticalParams) -> list:
    """Build skin layer stack with wrist-specific anatomy."""
    anatomy = params.wrist_anatomy or default_wrist_anatomy()

    # Wrist has thinner dermis and less fat than forearm/finger
    return [
        SkinLayer("epidermis",
                  thickness_cm=anatomy.epidermis_thickness_mm / 10.0,
                  mu_s_prime_cm=45.0, mu_a_base_cm=0.0),
        SkinLayer("dermis",
                  thickness_cm=anatomy.dermis_thickness_mm / 10.0,
                  mu_s_prime_cm=25.0, mu_a_base_cm=0.05),
        SkinLayer("subcutaneous_fat",
                  thickness_cm=anatomy.subcutaneous_fat_mm / 10.0,
                  mu_s_prime_cm=12.0, mu_a_base_cm=0.02),
        SkinLayer("arterial_bed",
                  thickness_cm=0.05, mu_s_prime_cm=20.0, mu_a_base_cm=0.0),
        SkinLayer("venous_bed",
                  thickness_cm=0.08, mu_s_prime_cm=20.0, mu_a_base_cm=0.0),
    ]


@dataclass
class SkinLayer:
    name: str
    thickness_cm: float
    mu_s_prime_cm: float
    mu_a_base_cm: float


class SkinOpticalModel:
    """Multi-layer optical model with ambient light and wrist anatomy."""

    def __init__(self):
        self._ambient_model = AmbientLightModel()

    def diffuse_reflectance(self, mu_a_cm: float, mu_s_prime_cm: float) -> float:
        mu_t = mu_a_cm + mu_s_prime_cm
        if mu_t <= 0:
            return 0.0
        a_prime = mu_s_prime_cm / mu_t
        one_minus = max(1e-6, 1 - a_prime)
        rd = a_prime * np.exp(-np.sqrt(3 * one_minus) * 1.0)
        return float(np.clip(rd, 0.0, 1.0))

    def layer_transmission(self, layer: SkinLayer, wavelength_key: str,
                            wavelength_nm: float, params: SkinOpticalParams,
                            blood_fraction_override: float | None = None) -> float:
        mu_a = layer.mu_a_base_cm
        if layer.name in ("dermis", "arterial_bed", "venous_bed"):
            bf = blood_fraction_override if blood_fraction_override is not None else params.tissue_blood_fraction
            spo2 = params.spo2 if layer.name != "venous_bed" else params.venous_spo2
            mu_a += _blood_mu_a(wavelength_key, spo2, bf)
        if layer.name == "epidermis":
            mu_a += _melanin_mu_a(wavelength_nm, params.melanin_fraction)
        mu_s_prime = _scattering_mu_s_prime(wavelength_nm, layer.mu_s_prime_cm)
        mu_eff = np.sqrt(3 * mu_a * (mu_a + mu_s_prime))
        transmission = np.exp(-mu_eff * layer.thickness_cm)
        return float(np.clip(transmission, 0.0, 1.0))

    def detected_intensity(self, wavelength_key: str, params: SkinOpticalParams,
                            arterial_blood_fraction: float) -> float:
        wavelength_nm = WAVELENGTHS_NM[wavelength_key]
        stack = default_skin_stack(params)
        total_t = 1.0
        for layer in stack:
            bf_override = None
            if layer.name == "arterial_bed":
                bf_override = arterial_blood_fraction
            total_t *= self.layer_transmission(layer, wavelength_key, wavelength_nm, params, bf_override)

        mu_a_eff = -np.log(max(total_t, 1e-9))
        rd = self.diffuse_reflectance(mu_a_eff, _scattering_mu_s_prime(wavelength_nm, 20.0))
        signal = total_t * (0.5 + 0.5 * rd)

        # Hair attenuation
        signal *= (1.0 - 0.35 * params.hair_density)
        # Tattoo ink
        signal *= np.exp(-1.5 * params.tattoo_optical_density)
        # Sweat layer
        signal *= (1.0 - 0.15 * params.sweat_layer)
        # Contact pressure effects
        if params.contact_pressure > 0.85:
            occlusion = (params.contact_pressure - 0.85) / 0.15
            signal *= (1.0 - 0.6 * occlusion)

        # Temperature effect on perfusion
        temp_corr = _temperature_correction(params.skin_temperature_c)
        signal *= temp_corr

        # Source-detector distance effect (wrist-specific)
        # At 5mm spacing, signal is moderate; closer = stronger DC, weaker AC
        sd_distance = params.wrist_anatomy.source_detector_distance_mm if params.wrist_anatomy else 5.0
        sd_factor = np.exp(-0.1 * (sd_distance - 5.0))  # normalized to 5mm
        signal *= sd_factor

        return float(signal)

    def generate_ppg_from_blood_volume(self, wavelength_key: str, params: SkinOpticalParams,
                                        arterial_blood_fraction_waveform: np.ndarray) -> dict:
        intensities = np.array([
            self.detected_intensity(wavelength_key, params, bf)
            for bf in arterial_blood_fraction_waveform
        ], dtype=np.float64)

        # === Add ambient light contamination ===
        if params.ambient_light_fraction > 0.0:
            ambient_level = float(np.mean(intensities)) * params.ambient_light_fraction
            ambient = np.full_like(intensities, ambient_level)
            # Add slow drift (cloud cover, shade changes)
            t = np.arange(len(intensities)) / 128.0  # assume 128 Hz internal
            ambient += ambient_level * 0.1 * np.sin(2 * np.pi * 0.01 * t + np.random.uniform(0, 2*np.pi))
            # Mains frequency flicker (100 Hz in EU/Asia, 120 Hz in US)
            ambient += ambient_level * 0.05 * np.sin(2 * np.pi * 120 * t)
            # Ambient light is additive (not multiplicative) — it fills in
            # the AC trough, reducing apparent AC/DC ratio
            intensities = intensities + ambient

        dc = float(np.mean(intensities))
        ac = intensities - dc
        ac_dc_ratio = float(np.std(ac) / (dc + 1e-9))

        return {
            "intensity": intensities.astype(np.float32),
            "dc": dc,
            "ac": ac.astype(np.float32),
            "ac_dc_ratio": ac_dc_ratio,
        }
