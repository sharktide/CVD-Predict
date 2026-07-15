"""
Rhythm generator: produces a beat-by-beat sequence (RR intervals, beat
type, and hemodynamic modifiers) that downstream cardiac/vascular models
consume to produce physiologically consistent PPG morphology changes,
rather than hand-drawn arrhythmia waveform templates.

Evidence base
-------------
- AFib: irregularly irregular RR intervals with loss of the atrial
  contribution to ventricular filling ("atrial kick"), reducing preload
  and hence beat-to-beat stroke volume: Kotecha & Piccini, "Atrial
  fibrillation in heart failure: what should we do?", Eur Heart J
  36:3250-7 (2015); RR interval statistics (near-random, Poisson-like
  irregularity with characteristic short-term correlation) reviewed in
  Fung, Chan et al., "Poincare plot analysis of AF", PLoS ONE (2015).
- PVC (premature ventricular contraction): early ectopic beat with a
  short coupling interval, reduced stroke volume (incomplete filling)
  and a full compensatory pause (RR after PVC ~ 2x normal), because the
  ectopic beat resets the ventricle but not the sinus node timing:
  Surawicz & Knilans, "Chou's Electrocardiography in Clinical Practice",
  6th ed., Ch. 17.
- PAC (premature atrial contraction): early beat with normal/near-normal
  morphology but a shorter, non-fully-compensatory pause (resets the
  sinus node): Surawicz & Knilans, Ch. 16.
- Bigeminy/trigeminy: regular alternation of every 2nd/3rd beat being
  ectopic (typically PVC), producing a strong pulse-amplitude
  alternation pattern peripherally: standard clinical ECG/PPG literature
  (e.g., Solosenko, Petrenas & Marozas, "PPG-based method for premature
  ventricular contraction detection...", Physiol Meas 36:2445 (2015)).
- SVT (supraventricular tachycardia): sudden-onset, regular, rapid
  rhythm (150-250 bpm), abrupt onset/offset: Josephson, "Clinical Cardiac
  Electrophysiology", 5th ed.
- VT (ventricular tachycardia): regular wide-complex rapid rhythm
  (100-250 bpm) with markedly reduced, sometimes progressively falling,
  stroke volume due to loss of coordinated contraction: Josephson (as
  above).
- Bradycardia: sustained HR < 60 bpm, may be sinus or due to conduction
  disease.
- Heart block (2nd/3rd degree AV block): intermittent or complete failure
  of atrial impulses to reach the ventricle, producing dropped beats or
  a slow, dissociated ventricular escape rhythm and hence a "pulse
  deficit" (heart rate on ECG > palpable/optical pulse rate): Josephson,
  Ch. on conduction disease.
- Sinus pauses: transient absence of a sinus beat (RR interval markedly
  longer than baseline, no compensatory beat): standard Holter-monitor
  literature.
- Pulse deficit: any beat with markedly reduced stroke volume may fail
  to generate a detectable peripheral pulse at all (common in AFib and
  frequent ectopy): Emergency medicine/cardiology teaching literature
  (e.g., Zeng et al., "Pulse deficit in atrial fibrillation", 2017).

What is heuristic here
-----------------------
- Exact coupling-interval distributions, escape-rhythm rates, and
  block ratios are set to clinically plausible ranges (e.g. PVC coupling
  ~0.5-0.7x baseline RR) rather than fit to a specific ECG database.
- We do not model true atrial vs ventricular electrical activation;
  effects are approximated purely at the level of RR timing and the
  resulting stroke-volume / preload consequences fed to the cardiac
  model (which is the layer that actually determines PPG morphology).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Beat:
    rr_s: float                 # interval from previous beat (s)
    beat_type: str              # "sinus", "pvc", "pac", "escape", "dropped"
    sv_scale: float = 1.0       # multiplier on stroke volume (preload/contraction effect)
    contractility_scale: float = 1.0
    perfusion_ok: bool = True   # False -> pulse deficit, no detectable peripheral pulse
    label_rhythm: str = "sinus"


RHYTHM_TYPES = (
    "sinus", "afib", "pvc_isolated", "pac_isolated", "bigeminy", "trigeminy",
    "svt", "vt", "bradycardia", "heart_block_2", "heart_block_3", "sinus_pause",
)


@dataclass
class ArrhythmiaConfig:
    rhythm: str = "sinus"
    base_hr_bpm: float = 72.0
    duration_s: float = 120.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


class RhythmGenerator:
    """Produces a list[Beat] covering the requested duration for a given
    rhythm label, with realistic RR-interval statistics and per-beat
    hemodynamic modifiers.
    """

    def generate(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        method = getattr(self, f"_gen_{cfg.rhythm}", None)
        if method is None:
            raise ValueError(f"Unknown rhythm '{cfg.rhythm}'. Options: {RHYTHM_TYPES}")
        return method(cfg)

    # ------------------------------------------------------------------
    def _base_rr(self, hr_bpm: float) -> float:
        return 60.0 / hr_bpm

    def _n_beats(self, cfg: ArrhythmiaConfig, hr_bpm: float) -> int:
        return int(cfg.duration_s / self._base_rr(hr_bpm)) + 5

    def _gen_sinus(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        beats = []
        for _ in range(n):
            rr = base_rr * (1 + rng.normal(0, 0.02))
            beats.append(Beat(rr_s=max(rr, 0.3), beat_type="sinus", label_rhythm="sinus"))
        return beats

    def _gen_afib(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        beats = []
        prev_rr = base_rr
        for _ in range(n):
            # irregularly-irregular: short-term correlated noise (AR(1)-like)
            # with a high overall coefficient of variation (>15%), consistent
            # with AFib RR statistics literature.
            innovation = rng.normal(0, 0.28 * base_rr)
            rr = 0.4 * prev_rr + 0.6 * base_rr + innovation
            rr = float(np.clip(rr, 0.3, 1.8))
            prev_rr = rr
            # loss of atrial kick -> reduced, variable preload/SV
            sv_scale = np.clip(rng.normal(0.80, 0.12), 0.4, 1.05)
            # very short RR (rapid ventricular response) beats may not
            # generate a palpable/optical pulse at all (pulse deficit)
            perfusion_ok = rr > 0.35 * base_rr / max(cfg.base_hr_bpm / 72.0, 0.5)
            perfusion_ok = perfusion_ok and rng.random() > 0.05
            beats.append(Beat(rr_s=rr, beat_type="afib", sv_scale=sv_scale,
                               perfusion_ok=bool(perfusion_ok), label_rhythm="afib"))
        return beats

    def _ectopic_run(self, cfg: ArrhythmiaConfig, every_n: int, kind: str) -> List[Beat]:
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        beats = []
        for i in range(n):
            if every_n is not None and (i + 1) % every_n == 0:
                beats.extend(self._make_ectopic_pair(rng, base_rr, kind))
            else:
                rr = base_rr * (1 + rng.normal(0, 0.02))
                beats.append(Beat(rr_s=max(rr, 0.3), beat_type="sinus", label_rhythm=kind))
        return beats

    def _make_ectopic_pair(self, rng, base_rr: float, kind: str) -> List[Beat]:
        if kind in ("pvc_isolated", "bigeminy", "trigeminy"):
            coupling = base_rr * rng.uniform(0.45, 0.65)
            compensatory = base_rr * rng.uniform(1.35, 1.55)  # full compensatory pause
            ectopic = Beat(rr_s=coupling, beat_type="pvc",
                            sv_scale=float(np.clip(rng.normal(0.55, 0.10), 0.2, 0.8)),
                            contractility_scale=1.15,  # ectopic beats often hypercontractile/early
                            label_rhythm=kind)
            follow = Beat(rr_s=compensatory, beat_type="sinus",
                           sv_scale=float(np.clip(rng.normal(1.20, 0.08), 1.0, 1.5)),  # enhanced post-pause filling
                           label_rhythm=kind)
            return [ectopic, follow]
        else:  # pac
            coupling = base_rr * rng.uniform(0.65, 0.85)
            pause = base_rr * rng.uniform(0.95, 1.10)  # non-fully-compensatory (resets sinus node)
            ectopic = Beat(rr_s=coupling, beat_type="pac",
                            sv_scale=float(np.clip(rng.normal(0.85, 0.08), 0.6, 1.0)),
                            label_rhythm=kind)
            follow = Beat(rr_s=pause, beat_type="sinus", label_rhythm=kind)
            return [ectopic, follow]

    def _gen_pvc_isolated(self, cfg): return self._ectopic_run(cfg, every_n=10, kind="pvc_isolated")
    def _gen_pac_isolated(self, cfg): return self._ectopic_run(cfg, every_n=12, kind="pac_isolated")
    def _gen_bigeminy(self, cfg): return self._ectopic_run(cfg, every_n=2, kind="bigeminy")
    def _gen_trigeminy(self, cfg): return self._ectopic_run(cfg, every_n=3, kind="trigeminy")

    def _gen_svt(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        onset = rng.integers(n // 6, n // 3)
        svt_hr = rng.uniform(150, 220)
        svt_rr = self._base_rr(svt_hr)
        offset = onset + rng.integers(n // 4, n // 2)
        beats = []
        for i in range(n):
            if onset <= i < offset:
                rr = svt_rr * (1 + rng.normal(0, 0.015))  # regular, rapid
                sv_scale = 0.75  # reduced filling time -> reduced SV
                beats.append(Beat(rr_s=max(rr, 0.25), beat_type="svt", sv_scale=sv_scale, label_rhythm="svt"))
            else:
                rr = base_rr * (1 + rng.normal(0, 0.02))
                beats.append(Beat(rr_s=max(rr, 0.3), beat_type="sinus", label_rhythm="svt"))
        return beats

    def _gen_vt(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        onset = rng.integers(n // 6, n // 3)
        vt_hr = rng.uniform(120, 220)
        vt_rr = self._base_rr(vt_hr)
        run_len = rng.integers(6, max(7, n // 4))
        offset = min(onset + run_len, n)
        beats = []
        for i in range(n):
            if onset <= i < offset:
                rr = vt_rr * (1 + rng.normal(0, 0.03))
                # progressively degrading stroke volume during sustained VT
                frac = (i - onset) / max(offset - onset, 1)
                sv_scale = float(np.clip(0.55 - 0.25 * frac + rng.normal(0, 0.05), 0.15, 0.6))
                beats.append(Beat(rr_s=max(rr, 0.2), beat_type="vt", sv_scale=sv_scale,
                                   perfusion_ok=sv_scale > 0.2, label_rhythm="vt"))
            else:
                rr = base_rr * (1 + rng.normal(0, 0.02))
                beats.append(Beat(rr_s=max(rr, 0.3), beat_type="sinus", label_rhythm="vt"))
        return beats

    def _gen_bradycardia(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        rng = cfg.rng
        brady_hr = min(cfg.base_hr_bpm, rng.uniform(35, 55))
        base_rr = self._base_rr(brady_hr)
        n = self._n_beats(cfg, brady_hr)
        beats = []
        for _ in range(n):
            rr = base_rr * (1 + rng.normal(0, 0.04))
            sv_scale = float(np.clip(1.15 + rng.normal(0, 0.05), 0.9, 1.4))  # longer filling -> larger SV (Frank-Starling)
            beats.append(Beat(rr_s=max(rr, 0.4), beat_type="sinus", sv_scale=sv_scale, label_rhythm="bradycardia"))
        return beats

    def _gen_heart_block_2(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        """Mobitz-like 2nd-degree block: every Nth atrial impulse is
        dropped (no ventricular beat -> no pulse), producing a genuine
        pulse deficit pattern."""
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        drop_every = int(rng.integers(3, 5))
        beats = []
        for i in range(n):
            if (i + 1) % drop_every == 0:
                beats.append(Beat(rr_s=2 * base_rr * (1 + rng.normal(0, 0.02)),
                                   beat_type="dropped", perfusion_ok=False, label_rhythm="heart_block_2"))
            else:
                rr = base_rr * (1 + rng.normal(0, 0.02))
                beats.append(Beat(rr_s=max(rr, 0.3), beat_type="sinus", label_rhythm="heart_block_2"))
        return beats

    def _gen_heart_block_3(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        """Complete (3rd-degree) block: atrial and ventricular activity
        are dissociated; ventricle driven by a slow escape rhythm,
        independent of (and typically slower than) the sinus rate."""
        rng = cfg.rng
        escape_hr = rng.uniform(25, 45)
        base_rr = self._base_rr(escape_hr)
        n = self._n_beats(cfg, escape_hr)
        beats = []
        for _ in range(n):
            rr = base_rr * (1 + rng.normal(0, 0.06))  # escape rhythms are less regular
            sv_scale = float(np.clip(1.25 + rng.normal(0, 0.08), 0.9, 1.6))
            beats.append(Beat(rr_s=max(rr, 0.5), beat_type="escape", sv_scale=sv_scale, label_rhythm="heart_block_3"))
        return beats

    def _gen_sinus_pause(self, cfg: ArrhythmiaConfig) -> List[Beat]:
        rng = cfg.rng
        base_rr = self._base_rr(cfg.base_hr_bpm)
        n = self._n_beats(cfg, cfg.base_hr_bpm)
        pause_indices = set(rng.choice(n, size=max(1, n // 40), replace=False).tolist())
        beats = []
        for i in range(n):
            if i in pause_indices:
                rr = base_rr * rng.uniform(1.8, 3.0)  # no compensatory beat, just a long gap
                beats.append(Beat(rr_s=rr, beat_type="sinus", label_rhythm="sinus_pause"))
            else:
                rr = base_rr * (1 + rng.normal(0, 0.02))
                beats.append(Beat(rr_s=max(rr, 0.3), beat_type="sinus", label_rhythm="sinus_pause"))
        return beats