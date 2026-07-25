"""
Physiological sub-models for the OHCA Predictor virtual patient simulator.

Each model implements continuous human physiology using ODEs, SDEs, and
stochastic processes appropriate for simulating patient state evolution
over 24-72 hour windows.

Models:
    CircadianModel        - Kronauer/Punjabi-type SCN oscillator with light input
    ActivityModel         - Time-inhomogeneous Markov chain for physical activity
    ANSModel              - Autonomic nervous system with RSA and baroreflex
    RespiratoryModel      - Chemoreceptor-driven respiratory dynamics
    ThermoregulationModel - Core/skin temperature regulation with fever
    HydrationModel        - Fluid/electrolyte homeostasis

Mathematical Framework:
    All stochastic models use Euler-Maruyama integration:
        X(t+dt) = X(t) + f(X,t)*dt + sigma*sqrt(dt)*N(0,1)
    where f is the drift term, sigma is the diffusion coefficient,
    and dt is the time step in hours.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ===========================================================================
# Circadian Model
# ===========================================================================

class CircadianModel:
    """Kronauer-type two-process circadian oscillator with melatonin feedback.

    The suprachiasmatic nucleus (SCN) is modeled as a nonlinear activator-
    inhibitor oscillator driven by photic input and modulated by melatonin
    feedback.

    Differential Equations::

        dS/dt = (1/tau_c) * (S_max*(1-S)*A(t,L) - S*I(M) + sigma_s*dW_s)
        dM/dt = (1/tau_m) * (M_max*D(t)*S_night - M + sigma_m*dW_m)

    where:
        S    - circadian phase variable [0, 1] (1 = peak alertness)
        M    - melatonin concentration [0, 1]
        tau_c - intrinsic period (hours), patient-specific ~24.0 +/- 0.5
        A(t,L) = max(0, sin(2pi*(hour-14)/24)) * (1 + gamma*ln(1+L/L_ref))
        I(M) = 1 + beta * M * 10   (melatonin inhibition)
        D(t) = max(0, -sin(2pi*(hour-14)/24))   (darkness function)
        S_night = max(0, S - 0.5)   (melatonin suppression by light-active SCN)

    References:
        Kronauer et al. (1999) Math. Biosci. 155: 1-22.
        Pandey et al. (2016) J. Biol. Rhythms 31: 48-58.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        period_hours: float = 24.0,
        light_lux: float = 250.0,
    ) -> None:
        self.rng = rng
        self.tau_c: float = float(np.clip(period_hours, 23.5, 24.5))
        self.light_lux: float = light_lux

        # Oscillator parameters
        self.S_max: float = 1.03
        self.beta: float = 0.0075
        self.gamma: float = 0.5
        self.kappa: float = 0.15

        # Melatonin parameters
        self.tau_m: float = 2.0
        self.M_max: float = 1.0

        # Noise amplitudes
        self.sigma_s: float = 0.02
        self.sigma_m: float = 0.01

        # State variables
        self.S: float = 0.5
        self.M: float = 0.1
        self.light_adapted: float = 0.0
        self.time_of_day: float = 0.0

    def initial_state(self, rng: np.random.Generator) -> None:
        """Randomize initial circadian state."""
        self.S = float(rng.uniform(0.3, 0.7))
        self.M = float(rng.uniform(0.05, 0.2))
        self.light_adapted = 0.0
        self.time_of_day = float(rng.uniform(0.0, 24.0))

    def _light_activator(self, hour: float, light_level: float) -> float:
        """Light-dependent activator: A(t,L).

        Peak activation occurs at ~14:00 (biological day).  Light level is
        compressed logarithmically (Weber-Fechner).
        """
        phase = 2.0 * np.pi * (hour - 14.0) / 24.0
        day_signal = max(0.0, np.sin(phase))
        light_factor = 1.0 + self.gamma * np.log1p(light_level / 100.0)
        return day_signal * light_factor

    def _darkness_function(self, hour: float) -> float:
        """Darkness function D(t) for melatonin production."""
        phase = 2.0 * np.pi * (hour - 14.0) / 24.0
        return max(0.0, -np.sin(phase))

    def update(
        self,
        dt: float,
        hour_of_day: float,
        light_lux: Optional[float] = None,
    ) -> None:
        """Advance circadian oscillator by *dt* hours (Euler-Maruyama).

        Args:
            dt: Time step in hours.
            hour_of_day: Continuous time of day [0, 24).
            light_lux: Ambient illuminance (lux).  ``None`` uses default.
        """
        if light_lux is None:
            light_lux = self.light_lux

        # Slow light adaptation
        self.light_adapted += (light_lux - self.light_adapted) * self.kappa * dt
        self.light_adapted = max(0.0, self.light_adapted)

        # Activator / inhibitor
        activator = self._light_activator(hour_of_day, self.light_adapted)
        inhibitor = 1.0 + self.beta * self.M * 10.0

        # Circadian oscillator SDE
        drift_S = (1.0 / self.tau_c) * (
            self.S_max * (1.0 - self.S) * activator - self.S * inhibitor
        )
        self.S += drift_S * dt + self.sigma_s * np.sqrt(dt) * self.rng.standard_normal()

        # Melatonin SDE
        darkness = self._darkness_function(hour_of_day)
        melatonin_suppression = max(0.0, self.S - 0.5)
        drift_M = (1.0 / self.tau_m) * (
            self.M_max * darkness * (1.0 - melatonin_suppression) - self.M
        )
        self.M += drift_M * dt + self.sigma_m * np.sqrt(dt) * self.rng.standard_normal()

        self.S = float(np.clip(self.S, 0.0, 1.0))
        self.M = float(np.clip(self.M, 0.0, 1.0))
        self.time_of_day = hour_of_day

    @property
    def circadian_alertness(self) -> float:
        """Current circadian alertness [0, 1]."""
        return float(np.clip(self.S, 0.0, 1.0))

    @property
    def melatonin_level(self) -> float:
        """Current melatonin concentration [0, 1]."""
        return float(np.clip(self.M, 0.0, 1.0))

    @property
    def is_daytime(self) -> bool:
        """True when biological day (S > 0.5)."""
        return self.S > 0.5


# ===========================================================================
# Activity Model
# ===========================================================================

class ActivityModel:
    """Time-inhomogeneous Markov chain for physical activity.

    State space::

        0 = sleeping
        1 = resting
        2 = light_activity
        3 = moderate_activity
        4 = vigorous_activity

    Transition probabilities are element-wise multiplied by a circadian
    modifier (time-of-day) and a fitness modifier (patient-specific) before
    row-normalisation::

        P_ij(t) = base_P_ij * circadian_mod(i,j,t) * fitness_mod(j)
        P_ij(t) /= sum_j P_ij(t)

    References:
        Atkinson et al. (2007) Med. Sci. Sports Exerc. 39: 1401-1409.
    """

    STATES = [
        "sleeping",
        "resting",
        "light_activity",
        "moderate_activity",
        "vigorous_activity",
    ]

    BASE_TRANSITIONS = np.array(
        [
            [0.950, 0.040, 0.010, 0.000, 0.000],  # sleeping
            [0.020, 0.850, 0.100, 0.030, 0.000],  # resting
            [0.010, 0.150, 0.700, 0.120, 0.020],  # light_activity
            [0.005, 0.080, 0.200, 0.650, 0.065],  # moderate_activity
            [0.001, 0.050, 0.100, 0.250, 0.599],  # vigorous_activity
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        rng: np.random.Generator,
        fitness_level: float = 0.5,
    ) -> None:
        self.rng = rng
        self.fitness_level: float = float(np.clip(fitness_level, 0.0, 1.0))
        self.state_idx: int = 1
        self.time_in_state: float = 0.0
        self.total_time_in_states: np.ndarray = np.zeros(5, dtype=np.float64)

    def initial_state(self, rng: np.random.Generator) -> None:
        """Set initial activity state to resting."""
        self.state_idx = 1
        self.time_in_state = 0.0
        self.total_time_in_states = np.zeros(5, dtype=np.float64)

    def _get_circadian_modifier(self, hour: float) -> np.ndarray:
        """5x5 element-wise modifier driven by time-of-day.

        Night (~22:00-06:00): sleeping self-transition boosted, activity
        transitions suppressed.  Afternoon (~12:00-18:00): activity boosted.
        """
        mod = np.ones((5, 5), dtype=np.float64)

        night_factor = 0.5 * (1.0 - np.cos(2.0 * np.pi * (hour - 2.0) / 24.0))
        day_factor = 1.0 - night_factor

        # Sleeping self-transition: higher at night
        mod[0, 0] = 1.0 + 0.5 * night_factor
        mod[0, 1] = 1.0 - 0.3 * night_factor

        # Resting -> sleeping: higher at night
        mod[1, 0] = 1.0 + 2.0 * night_factor
        mod[1, 1] = 1.0 + 0.3 * night_factor

        # Any state -> sleeping: higher at night
        for i in range(2, 5):
            mod[i, 0] = 1.0 + 3.0 * night_factor

        # Vigorous activity: only during the day
        for i in range(5):
            mod[i, 4] *= day_factor

        # Moderate activity: suppressed at night
        for i in range(5):
            mod[i, 3] *= 0.3 + 0.7 * day_factor

        return mod

    def _get_fitness_modifier(self) -> np.ndarray:
        """5x5 element-wise modifier based on patient fitness."""
        mod = np.ones((5, 5), dtype=np.float64)

        vig_scale = 0.2 + 1.8 * self.fitness_level
        for i in range(5):
            mod[i, 4] *= vig_scale

        mod[4, 1] *= 1.0 + 0.5 * self.fitness_level
        mod[4, 2] *= 1.0 + 0.3 * self.fitness_level
        mod[3, 4] *= 0.3 + 1.7 * self.fitness_level

        return mod

    def update(self, dt: float, hour_of_day: float) -> str:
        """Advance the Markov chain by *dt* hours.

        Args:
            dt: Time step in hours.
            hour_of_day: Current time of day [0, 24).

        Returns:
            Name of the new activity state.
        """
        circ = self._get_circadian_modifier(hour_of_day)
        fit = self._get_fitness_modifier()

        trans = self.BASE_TRANSITIONS * circ * fit
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        trans = trans / row_sums

        new_state = int(self.rng.choice(5, p=trans[self.state_idx]))

        self.total_time_in_states[self.state_idx] += dt
        if new_state != self.state_idx:
            self.time_in_state = 0.0
            self.state_idx = new_state
        else:
            self.time_in_state += dt

        return self.STATES[self.state_idx]

    @property
    def state(self) -> str:
        """Current activity state name."""
        return self.STATES[self.state_idx]

    @property
    def activity_level(self) -> float:
        """Numeric activity level [0, 4]."""
        return float(self.state_idx)


# ===========================================================================
# Autonomic Nervous System Model
# ===========================================================================

class ANSModel:
    """Autonomic nervous system with RSA and baroreceptor reflex.

    Two competing neural populations are modeled::

        dsymp/dt = -(symp - s_target)/tau_s + noise_s
        dpara/dt = -(para - p_target)/tau_p + noise_p

    Heart-rate contribution::

        HR = HR_base + k_s*symp - k_p*para + baroreflex

    Baroreceptor reflex (first-order + integral)::

        baroreflex = -K_baro*(MAP - MAP_set) - K_int*integral(MAP_error)

    Respiratory Sinus Arrhythmia (RSA) is implicit: parasympathetic tone
    modulates HR at the respiratory frequency.

    References:
        Task Force of ESC/NASPE (1996) Circulation 93: 1043-1065.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

        # Tones [0, 1]
        self.sympathetic_tone: float = 0.3
        self.parasympathetic_tone: float = 0.7

        # Time constants (hours)
        self.tau_symp: float = 0.1
        self.tau_para: float = 0.05

        # Resting setpoints
        self.s_rest: float = 0.3
        self.p_rest: float = 0.7

        # HR gain parameters (bpm per unit tone)
        self.k_symp_hr: float = 20.0
        self.k_para_hr: float = 25.0

        # Baroreflex
        self.k_baro: float = 0.5
        self.k_baro_int: float = 0.01
        self.map_setpoint: float = 93.0
        self.baroreceptor_integral: float = 0.0

        # Diffusion
        self.sigma_symp: float = 0.01
        self.sigma_para: float = 0.01

    def initial_state(self, rng: np.random.Generator) -> None:
        """Randomize initial autonomic tones."""
        self.sympathetic_tone = float(rng.uniform(0.2, 0.4))
        self.parasympathetic_tone = float(rng.uniform(0.6, 0.8))
        self.baroreceptor_integral = 0.0

    def update(
        self,
        dt: float,
        activity_level: float,
        mean_arterial_pressure: float,
        resp_rate_bpm: float,
        circadian_alertness: float,
    ) -> Tuple[float, float]:
        """Update ANS state.

        Args:
            dt: Time step in hours.
            activity_level: Numeric activity [0, 4].
            mean_arterial_pressure: MAP in mmHg.
            resp_rate_bpm: Respiratory rate (breaths min^-1).
            circadian_alertness: Circadian signal [0, 1].

        Returns:
            (sympathetic_drive_hr, parasympathetic_drive_hr) in bpm.
        """
        activity_drive = activity_level / 4.0
        circ_mod = 1.0 - circadian_alertness

        # Sympathetic target
        s_target = float(np.clip(
            self.s_rest + 0.6 * activity_drive - 0.1 * circ_mod, 0.0, 1.0
        ))
        drift_s = -(self.sympathetic_tone - s_target) / self.tau_symp
        self.sympathetic_tone += (
            drift_s * dt
            + self.sigma_symp * np.sqrt(dt) * self.rng.standard_normal()
        )

        # Parasympathetic target
        p_target = float(np.clip(
            self.p_rest - 0.5 * activity_drive + 0.2 * circ_mod, 0.0, 1.0
        ))
        drift_p = -(self.parasympathetic_tone - p_target) / self.tau_para
        self.parasympathetic_tone += (
            drift_p * dt
            + self.sigma_para * np.sqrt(dt) * self.rng.standard_normal()
        )

        self.sympathetic_tone = float(np.clip(self.sympathetic_tone, 0.0, 1.0))
        self.parasympathetic_tone = float(np.clip(self.parasympathetic_tone, 0.0, 1.0))

        # Baroreceptor reflex
        map_error = mean_arterial_pressure - self.map_setpoint
        self.baroreceptor_integral += map_error * self.k_baro_int * dt
        self.baroreceptor_integral = float(np.clip(
            self.baroreceptor_integral, -10.0, 10.0
        ))
        baroreflex_effect = (
            -self.k_baro * map_error - 0.1 * self.baroreceptor_integral
        )

        # HR contributions
        symp_drive = self.k_symp_hr * self.sympathetic_tone
        para_drive = (
            -self.k_para_hr * self.parasympathetic_tone + baroreflex_effect
        )

        return float(symp_drive), float(para_drive)

    @property
    def sympathetic_parasympathetic_ratio(self) -> float:
        """Sympathetic / parasympathetic ratio (>1 = sympathodominance)."""
        if self.parasympathetic_tone < 1e-6:
            return 100.0
        return self.sympathetic_tone / self.parasympathetic_tone


# ===========================================================================
# Respiratory Model
# ===========================================================================

class RespiratoryModel:
    """Chemoreceptor-driven respiratory dynamics.

    Central chemoreceptors (medulla) respond to CO2 / pH::

        f_co2 = K_co2 * (PCO2 - 40) * (1 + K_pH * max(0, 7.4 - pH))

    Peripheral chemoreceptors (carotid bodies) respond to hypoxia::

        f_o2 = K_o2 * 100 / (PO2 + PO2_offset)

    Respiratory rate follows with first-order inertia::

        dRR/dt = K_rr * (RR_set - RR) + sigma * dW

    Cheyne-Stokes: sinusoidal modulation of RR with period ~45-90 s.
    Kussmaul: deep, rapid breathing when pH < 7.2.

    References:
        Grodins et al. (1954) J. Appl. Physiol. 7: 283-308.
        Khoo (1990) in *Modeling and Analysis of Complex Systems*.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

        # Baseline values
        self.resp_rate: float = 14.0
        self.tidal_volume: float = 0.5
        self.minute_ventilation: float = 7.0

        # Chemoreceptor parameters
        self.pco2_setpoint: float = 40.0
        self.k_co2: float = 1.5
        self.k_ph: float = 0.5
        self.k_o2: float = 0.3
        self.po2_offset: float = 30.0

        # Setpoints and inertia
        self.rr_baseline: float = 14.0
        self.tv_baseline: float = 0.5
        self.k_rr: float = 2.0
        self.k_tv: float = 0.3

        # Variability
        self.rr_variability: float = 0.5

        # Cheyne-Stokes
        self.cheyne_stokes_period: float = 60.0
        self.cheyne_stokes_amplitude: float = 0.0
        self.cheystokes_active: bool = False

        # Kussmaul
        self.kussmaul_depth_factor: float = 0.0

        # Noise
        self.sigma_rr: float = 0.2

        # Blood gas approximations
        self.pco2: float = 40.0
        self.po2: float = 90.0
        self.blood_ph: float = 7.4

    def initial_state(self, rng: np.random.Generator) -> None:
        """Randomize initial respiratory state."""
        self.resp_rate = float(np.clip(rng.normal(14.0, 1.5), 8.0, 25.0))
        self.tidal_volume = float(np.clip(rng.normal(0.5, 0.05), 0.3, 1.2))
        self.minute_ventilation = self.resp_rate * self.tidal_volume
        self.pco2 = float(rng.normal(40.0, 2.0))
        self.po2 = float(rng.normal(90.0, 5.0))
        self.blood_ph = float(rng.normal(7.4, 0.02))
        self.cheyne_stokes_amplitude = 0.0
        self.kussmaul_depth_factor = 0.0

    def update(
        self,
        dt: float,
        metabolic_rate: float,
        blood_ph: float,
        activity_level: float,
        spo2_percent: float,
        circadian_alertness: float,
        time_seconds: float,
    ) -> None:
        """Advance respiratory state.

        Args:
            dt: Time step in hours.
            metabolic_rate: Multiplier [0.8, 2.5].
            blood_ph: Arterial pH.
            activity_level: Numeric activity [0, 4].
            spo2_percent: Oxygen saturation [%].
            circadian_alertness: Circadian signal [0, 1].
            time_seconds: Elapsed simulation time (s).
        """
        self.blood_ph = blood_ph

        # Estimate PCO2 from ventilation and metabolism
        co2_production = 200.0 * metabolic_rate
        self.minute_ventilation = max(self.resp_rate * self.tidal_volume, 0.1)
        self.pco2 = float(np.clip(
            co2_production / self.minute_ventilation * 2.0, 15.0, 80.0
        ))

        # Estimate PO2 from SpO2
        self.po2 = float(np.clip(spo2_percent * 1.0 + 10.0, 20.0, 120.0))

        # Central chemoreceptor drive
        f_co2 = self.k_co2 * (self.pco2 - self.pco2_setpoint)
        f_ph = f_co2 * self.k_ph * max(0.0, 7.4 - self.blood_ph)

        # Peripheral chemoreceptor drive
        f_o2 = self.k_o2 * 100.0 / (self.po2 + self.po2_offset)

        # Activity and circadian drives
        f_activity = activity_level * 2.0
        f_circadian = -1.0 * (1.0 - circadian_alertness)

        # Setpoint
        rr_setpoint = float(np.clip(
            self.rr_baseline + f_co2 + f_ph + f_o2 + f_activity + f_circadian,
            6.0,
            40.0,
        ))

        tv_setpoint = float(np.clip(
            self.tv_baseline * (1.0 + self.k_tv * (metabolic_rate - 1.0)),
            0.2,
            2.5,
        ))

        # Kussmaul breathing when acidotic
        if self.blood_ph < 7.2:
            self.kussmaul_depth_factor = min(
                1.0, (7.2 - self.blood_ph) / 0.3
            )
            tv_setpoint *= 1.0 + 1.5 * self.kussmaul_depth_factor
        else:
            self.kussmaul_depth_factor *= max(0.0, 1.0 - dt)

        # Euler-Maruyama for RR
        drift_rr = self.k_rr * (rr_setpoint - self.resp_rate)
        self.resp_rate += (
            drift_rr * dt
            + self.sigma_rr * np.sqrt(dt) * self.rng.standard_normal()
        )

        # Euler-Maruyama for TV
        drift_tv = 2.0 * (tv_setpoint - self.tidal_volume)
        self.tidal_volume += (
            drift_tv * dt
            + 0.01 * np.sqrt(dt) * self.rng.standard_normal()
        )

        # Cheyne-Stokes modulation
        if self.cheystokes_active and self.cheyne_stokes_amplitude > 0:
            cs_phase = 2.0 * np.pi * time_seconds / self.cheyne_stokes_period
            self.resp_rate += self.cheyne_stokes_amplitude * np.sin(cs_phase)

        self.resp_rate = float(np.clip(self.resp_rate, 4.0, 45.0))
        self.tidal_volume = float(np.clip(self.tidal_volume, 0.1, 3.0))
        self.minute_ventilation = self.resp_rate * self.tidal_volume

    @property
    def spo2_from_po2(self) -> float:
        """Estimate SpO2 from PO2 via Hill equation for O2-Hb."""
        p50 = 26.8
        n_hill = 2.7
        if self.po2 <= 0:
            return 0.0
        sat = self.po2 ** n_hill / (p50 ** n_hill + self.po2 ** n_hill)
        return float(np.clip(sat * 100.0, 50.0, 100.0))


# ===========================================================================
# Thermoregulation Model
# ===========================================================================

class ThermoregulationModel:
    """Two-compartment core-skin thermal model.

    Core temperature::

        dTc/dt = (M + M_shiver - H_cs - H_resp) / (m_c * c)

    Skin temperature::

        dTs/dt = (H_cs - H_se - Q_sweat) / (m_s * c)

    Heat fluxes::

        H_cs = K_cs * vasodilation_factor * (Tc - Ts)   (countercurrent)
        H_se = K_se * (Ts - T_env)                      (radiation/convection)
        H_resp = 0.1 * M                                (respiratory loss)
        M_shiver = K_sh * max(0, Tset - Tc)
        Q_sweat = K_sw * max(0, Tc - Tset)

    Fever is modeled as a pyrogen-driven setpoint shift::

        dTset/dt = k_on * (inflammatory * 3 - Tset_shift)
                   when inflammatory > 0.3, else decay toward 0.

    References:
        Werner (1981) in *Physiology and Pharmacology of Temperature
        Regulation*, pp. 229-251.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

        # Temperatures
        self.core_temp: float = 37.0
        self.skin_temp: float = 33.0

        # Setpoint
        self.core_setpoint: float = 37.0

        # Thermal parameters
        self.mass_core: float = 50.0
        self.mass_skin: float = 8.0
        self.specific_heat: float = 3490.0
        self.conductance_core_skin: float = 50.0
        self.conductance_skin_env: float = 10.0

        # Thermoregulatory gains
        self.k_sweat: float = 100.0
        self.k_shiver: float = 80.0
        self.k_vasodilate: float = 2.0

        # Fever dynamics
        self.fever_setpoint_shift: float = 0.0
        self.fever_onset_rate: float = 0.1
        self.fever_resolution_rate: float = 0.05

        # Environment
        self.ambient_temp: float = 22.0

        # Noise
        self.sigma_core: float = 0.005
        self.sigma_skin: float = 0.01

        # Basal metabolic rate (W)
        self.basal_metabolic_rate: float = 80.0

    def initial_state(self, rng: np.random.Generator) -> None:
        """Randomize initial thermal state."""
        self.core_temp = float(np.clip(rng.normal(37.0, 0.2), 36.0, 37.8))
        self.skin_temp = float(np.clip(rng.normal(33.0, 0.5), 30.0, 35.0))
        self.fever_setpoint_shift = 0.0

    def update(
        self,
        dt: float,
        metabolic_rate: float,
        sympathetic_tone: float,
        inflammatory_state: float,
        circadian_alertness: float,
        ambient_temp: Optional[float] = None,
    ) -> None:
        """Advance thermal state.

        Args:
            dt: Time step in hours.
            metabolic_rate: Multiplier [0.8, 2.5].
            sympathetic_tone: ANS sympathetic tone [0, 1].
            inflammatory_state: Inflammatory state [0, 1].
            circadian_alertness: Circadian signal [0, 1].
            ambient_temp: Environment temperature (deg C), uses default if None.
        """
        if ambient_temp is not None:
            self.ambient_temp = ambient_temp

        dt_s = dt * 3600.0  # convert to seconds for thermal calc

        # Fever response
        if inflammatory_state > 0.3:
            target_fever = inflammatory_state * 3.0
            rate = (
                self.fever_onset_rate
                if target_fever > self.fever_setpoint_shift
                else self.fever_resolution_rate
            )
            self.fever_setpoint_shift += (
                rate * (target_fever - self.fever_setpoint_shift) * dt
            )
        else:
            self.fever_setpoint_shift *= max(
                0.0, 1.0 - self.fever_resolution_rate * dt
            )
        self.fever_setpoint_shift = float(
            np.clip(self.fever_setpoint_shift, 0.0, 4.0)
        )

        # Circadian temperature modulation (~0.5 deg C oscillation)
        circadian_temp_mod = 0.3 * (circadian_alertness - 0.5)

        effective_setpoint = (
            self.core_setpoint + self.fever_setpoint_shift + circadian_temp_mod
        )

        # Metabolic heat production
        M = self.basal_metabolic_rate * metabolic_rate

        # Shivering
        temp_deficit = effective_setpoint - self.core_temp
        M_shiver = self.k_shiver * max(0.0, temp_deficit)

        total_heat = M + M_shiver

        # Core-skin heat transfer (countercurrent)
        vaso_factor = 1.0 + self.k_vasodilate * (1.0 - sympathetic_tone)
        H_cs = self.conductance_core_skin * vaso_factor * (
            self.core_temp - self.skin_temp
        )

        # Respiratory heat loss (~10% of metabolic heat)
        H_resp = 0.1 * total_heat

        # Skin-environment heat transfer
        H_se = self.conductance_skin_env * (
            self.skin_temp - self.ambient_temp
        )

        # Sweating
        temp_excess = self.core_temp - effective_setpoint
        Q_sweat = self.k_sweat * max(0.0, temp_excess) * (
            1.0 - sympathetic_tone * 0.3
        )

        # Update core temperature (Euler-Maruyama)
        dTc = (total_heat - H_cs - H_resp) / (
            self.mass_core * self.specific_heat
        )
        self.core_temp += dTc * dt_s + self.sigma_core * np.sqrt(
            dt_s
        ) * self.rng.standard_normal()

        # Update skin temperature
        dTs = (H_cs - H_se - Q_sweat) / (
            self.mass_skin * self.specific_heat
        )
        self.skin_temp += dTs * dt_s + self.sigma_skin * np.sqrt(
            dt_s
        ) * self.rng.standard_normal()

        self.core_temp = float(np.clip(self.core_temp, 34.0, 42.0))
        self.skin_temp = float(np.clip(self.skin_temp, 25.0, 40.0))

    @property
    def temperature_gradient(self) -> float:
        """Core-skin temperature gradient."""
        return self.core_temp - self.skin_temp

    @property
    def fever_active(self) -> bool:
        """Whether patient has active fever (>37.5 deg C)."""
        return self.core_temp > 37.5


# ===========================================================================
# Hydration Model
# ===========================================================================

class HydrationModel:
    """Fluid and electrolyte homeostasis model.

    Hydration level::

        dH/dt = (intake - output) / V_total

    Fluid output components:
        - Insensible loss: ~800 mL/day (skin + respiratory)
        - Renal output: proportional to GFR
        - Sweating: proportional to thermoregulatory demand

    Blood potassium::

        dK/dt = dietary_K + cellular_release - renal_excretion - transcellular_shift

    Blood sodium::

        dNa/dt = dietary_Na + ADH_effect - renal_excretion

    Blood glucose::

        dG/dt = hepatic_production - peripheral_uptake + glycogenolysis

    Troponin (marker of myocardial injury)::

        dT/dt = release_rate * ischemic_burden - clearance_rate * T

    BNP (marker of cardiac strain)::

        dB/dt = release_rate * cardiac_stress - clearance_rate * B

    Lactate::

        dL/dt = anaerobic_production - hepatic_clearance

    pH::

        dpH/dt = ventilation_effect - metabolic_acid + buffering

    References:
        Guyton & Hall - Textbook of Medical Physiology, 14th ed.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

        # Hydration
        self.hydration_level: float = 0.75
        self.total_body_water: float = 42.0

        # Fluid balance rates (L/hour)
        self.intake_rate: float = 0.1
        self.insensible_loss: float = 0.033
        self.renal_loss: float = 0.04
        self.sweat_rate: float = 0.0

        # Electrolytes (mmol/L)
        self.blood_potassium: float = 4.0
        self.blood_sodium: float = 140.0
        self.blood_calcium: float = 2.3

        # Glucose (mg/dL)
        self.blood_glucose: float = 90.0

        # pH
        self.blood_ph: float = 7.4

        # Lactate (mmol/L)
        self.lactate: float = 1.0

        # Troponin (ng/L)
        self.troponin: float = 5.0

        # BNP (pg/mL)
        self.bnp: float = 30.0

        # Renal parameters
        self.gfr: float = 120.0
        self.potassium_reabsorption: float = 0.90
        self.sodium_reabsorption: float = 0.99

        # Metabolic
        self.hepatic_glucose_production: float = 2.0
        self.insulin_sensitivity: float = 1.0

        # Noise std devs
        self.sigma_potassium: float = 0.02
        self.sigma_sodium: float = 0.3
        self.sigma_calcium: float = 0.02
        self.sigma_glucose: float = 2.0
        self.sigma_ph: float = 0.005
        self.sigma_lactate: float = 0.05
        self.sigma_troponin: float = 0.5
        self.sigma_bnp: float = 2.0

    def initial_state(self, rng: np.random.Generator) -> None:
        """Randomize initial hydration and electrolyte state."""
        self.hydration_level = float(rng.uniform(0.65, 0.85))
        self.blood_potassium = float(rng.normal(4.0, 0.3))
        self.blood_sodium = float(rng.normal(140.0, 3.0))
        self.blood_calcium = float(rng.normal(2.3, 0.15))
        self.blood_glucose = float(rng.normal(90.0, 10.0))
        self.blood_ph = float(rng.normal(7.4, 0.03))
        self.lactate = float(rng.normal(1.0, 0.3))
        self.troponin = float(rng.uniform(1.0, 15.0))
        self.bnp = float(rng.normal(30.0, 15.0))

    def update(
        self,
        dt: float,
        metabolic_rate: float,
        activity_level: float,
        sympathetic_tone: float,
        inflammatory_state: float,
        ischemic_burden: float,
        myocardial_irritability: float,
        ejection_fraction: float,
    ) -> None:
        """Advance hydration and electrolyte state.

        Args:
            dt: Time step in hours.
            metabolic_rate: Multiplier.
            activity_level: Numeric activity [0, 4].
            sympathetic_tone: ANS sympathetic tone [0, 1].
            inflammatory_state: Inflammatory state [0, 1].
            ischemic_burden: Myocardial ischemia [0, 1].
            myocardial_irritability: Myocardial irritability [0, 1].
            ejection_fraction: Cardiac EF [0.15, 0.75].
        """
        # Sweat rate from activity
        self.sweat_rate = 0.01 * activity_level

        # Fluid balance
        total_output = self.insensible_loss + self.renal_loss + self.sweat_rate
        net_fluid = (self.intake_rate - total_output) * dt
        self.hydration_level += net_fluid / self.total_body_water
        self.hydration_level = float(np.clip(self.hydration_level, 0.0, 1.0))

        # Dehydration affects renal function
        renal_function = min(1.0, self.hydration_level / 0.6)

        # ---- Potassium ---------------------------------------------------
        dietary_k = 0.1 * self.hydration_level
        renal_k_excretion = (
            (1.0 - self.potassium_reabsorption) * self.gfr / 120.0 * 0.5
        )
        acidosis_shift = max(0.0, (7.4 - self.blood_ph) * 2.0)
        insulin_shift = -0.3 * self.insulin_sensitivity * sympathetic_tone
        ischemia_release = 0.5 * ischemic_burden

        dk = (
            dietary_k
            - renal_k_excretion * renal_function
            + acidosis_shift
            + insulin_shift
            + ischemia_release
        ) * dt
        self.blood_potassium += dk + self.sigma_potassium * np.sqrt(dt) * self.rng.standard_normal()
        self.blood_potassium = float(np.clip(self.blood_potassium, 2.5, 7.5))

        # ---- Sodium ------------------------------------------------------
        dietary_na = 0.5
        adh_effect = 0.3 * (1.0 - self.hydration_level)
        renal_na_excretion = (
            (1.0 - self.sodium_reabsorption) * 0.3 * renal_function
        )

        dna = (dietary_na - renal_na_excretion * renal_function - adh_effect) * dt
        self.blood_sodium += dna + self.sigma_sodium * np.sqrt(dt) * self.rng.standard_normal()
        self.blood_sodium = float(np.clip(self.blood_sodium, 120.0, 165.0))

        # ---- Calcium -----------------------------------------------------
        dca = 0.01 * (2.3 - self.blood_calcium) * dt
        self.blood_calcium += dca + self.sigma_calcium * np.sqrt(dt) * self.rng.standard_normal()
        self.blood_calcium = float(np.clip(self.blood_calcium, 1.0, 4.0))

        # ---- Glucose -----------------------------------------------------
        hepatic_prod = self.hepatic_glucose_production * (
            1.0 + 0.5 * sympathetic_tone
        )
        peripheral_uptake = (
            1.5 * self.insulin_sensitivity * (self.blood_glucose / 90.0)
        )
        stress_glycogenolysis = 1.0 * sympathetic_tone * inflammatory_state

        dg = (hepatic_prod - peripheral_uptake + stress_glycogenolysis) * dt
        self.blood_glucose += dg + self.sigma_glucose * np.sqrt(dt) * self.rng.standard_normal()
        self.blood_glucose = float(np.clip(self.blood_glucose, 30.0, 600.0))

        # ---- pH ----------------------------------------------------------
        co2_washout = 0.01 * (self.gfr / 120.0 - 1.0) * 10.0
        metabolic_acid = 0.05 * (self.lactate - 1.0)
        metabolic_buffer = 0.02 * (7.4 - self.blood_ph)

        dph = (co2_washout - metabolic_acid + metabolic_buffer) * dt
        self.blood_ph += dph + self.sigma_ph * np.sqrt(dt) * self.rng.standard_normal()
        self.blood_ph = float(np.clip(self.blood_ph, 6.8, 7.8))

        # ---- Lactate -----------------------------------------------------
        lactate_production = (
            0.3 * ischemic_burden + 0.1 * max(0.0, metabolic_rate - 1.0)
        )
        lactate_clearance = 0.5 * self.lactate * renal_function

        dlact = (lactate_production - lactate_clearance) * dt
        self.lactate += dlact + self.sigma_lactate * np.sqrt(dt) * self.rng.standard_normal()
        self.lactate = float(np.clip(self.lactate, 0.3, 20.0))

        # ---- Troponin ----------------------------------------------------
        troponin_release = (
            5.0 * ischemic_burden + 2.0 * myocardial_irritability
        )
        troponin_clearance = 0.05 * self.troponin

        dtrop = (troponin_release - troponin_clearance) * dt
        self.troponin += dtrop + self.sigma_troponin * np.sqrt(dt) * self.rng.standard_normal()
        self.troponin = float(np.clip(self.troponin, 0.0, 1000.0))

        # ---- BNP ---------------------------------------------------------
        cardiac_stress = max(0.0, (1.0 - ejection_fraction) * 2.0)
        bnp_release = 10.0 * cardiac_stress
        bnp_clearance = 0.3 * self.bnp

        dbnp = (bnp_release - bnp_clearance) * dt
        self.bnp += dbnp + self.sigma_bnp * np.sqrt(dt) * self.rng.standard_normal()
        self.bnp = float(np.clip(self.bnp, 0.0, 5000.0))
