"""
MAAA — Layer 3: Human State Monitoring (Monitoraggio dello Stato Umano)

Stima continua dello stato cognitivo ed emotivo dell'utente da segnali multimodali:
  - Stress Detection → voce (pitch, tremor, speech rate) + pupil dilation
  - Cognitive Overload Estimation → blink rate, fixation duration, saccade velocity
  - Freezing / Indecision Detection → calo rispetto alla baseline personale
  - Panic Estimation → agitazione motoria, voce ad alta pitch, ambiente rumoroso
  - Attentional Collapse Detection → gaze disperso, blink rate anomalo

ATTENZIONE — questo NON è un rilevatore di stress validato. È un proxy di
entropia cognitiva: somme pesate di scalari sintetici prodotti da
``layers/l1_perception.py``. Nessun sensore reale, nessun modello addestrato,
nessuna validazione su soggetti umani. I pesi e le soglie sono scelti a mano e
calibrati sull'intervallo che il generatore sintetico può produrre
(vedi ``config/default.yaml`` → ``human_state_thresholds`` e
``tests/test_reachability.py``).

Output: HumanStateFrame con score 0–1 per ogni dimensione cognitiva
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Optional

import maaa_config
from layers.l1_perception import PerceptionFrame

logger = logging.getLogger("maaa.l3_human_state")


def _threshold(name: str, default: float) -> float:
    return maaa_config.human_state_threshold(name, default)


class CognitiveState(Enum):
    """Simplified categorical classification of user's cognitive state."""
    CALM          = "calm"
    ALERT         = "alert"           # Heightened attention, functional
    STRESSED      = "stressed"        # Elevated arousal, still effective
    OVERLOADED    = "overloaded"      # Too much info, decision quality drops
    FROZEN        = "frozen"          # Paralysis, unable to act
    PANICKING     = "panicking"       # Fight-or-flight, irrational
    COLLAPSED     = "collapsed"       # Attentional/cognitive collapse


@dataclass
class HumanStateFrame:
    """
    Continuous multidimensional estimation of the user's cognitive state.
    All scores are 0.0 (absent) to 1.0 (maximum intensity).
    """
    timestamp: float

    # Continuous dimensions
    stress_score: float           # 0 = calm, 1 = extreme stress
    cognitive_overload: float     # 0 = clear mind, 1 = overloaded
    panic_score: float            # 0 = calm, 1 = full panic
    freeze_score: float           # 0 = moving, 1 = frozen/paralysed
    attentional_collapse: float   # 0 = focused, 1 = attention gone
    arousal: float                # general physiological arousal

    # Categorical classification (derived)
    state: CognitiveState

    # Derived capacity scores
    decision_capacity: float      # 0–1, how well user can make decisions
    receptivity: float            # 0–1, how well user can receive guidance

    # Signal-level diagnostics
    voice_stress_index: float
    gaze_stability: float
    motor_agitation: float

    # Trend (delta from previous frame, for escalation detection)
    stress_delta: float = 0.0
    panic_delta: float = 0.0

    @property
    def is_critical(self) -> bool:
        """User is in a state where standard guidance will not work."""
        return (self.panic_score > _threshold("is_critical_panic", 0.75) or
                self.freeze_score > _threshold("is_critical_freeze", 0.65) or
                self.attentional_collapse > _threshold("is_critical_attentional", 0.75))

    @property
    def needs_immediate_override(self) -> bool:
        """Emergency: guidance must be simplified to single imperative.

        Thresholds live in config/default.yaml. The previous hard-coded gates
        (panic > 0.85, overload > 0.90) sat above everything the pipeline can
        produce, so this property was always False.
        """
        return (self.panic_score > _threshold("immediate_override_panic", 0.78) or
                self.cognitive_overload > _threshold("immediate_override_overload", 0.80))

    def summary(self) -> str:
        return (f"State={self.state.value} "
                f"stress={self.stress_score:.2f} "
                f"overload={self.cognitive_overload:.2f} "
                f"panic={self.panic_score:.2f} "
                f"freeze={self.freeze_score:.2f} "
                f"receptivity={self.receptivity:.2f}")


def _classify_state(stress: float, overload: float,
                    panic: float, freeze: float,
                    attentional: float) -> CognitiveState:
    """Rule-based classifier — production replaces with trained model on labelled data.

    Gates come from config/default.yaml (``maaa.human_state_thresholds``) and
    are calibrated against the ranges the synthetic generator can reach;
    tests/test_reachability.py asserts every state below is attainable.
    """
    if panic > _threshold("panicking", 0.75):
        return CognitiveState.PANICKING
    if attentional > _threshold("collapsed", 0.75):
        return CognitiveState.COLLAPSED
    if freeze > _threshold("frozen", 0.65):
        return CognitiveState.FROZEN
    if overload > _threshold("overloaded", 0.70):
        return CognitiveState.OVERLOADED
    if stress > _threshold("stressed", 0.60):
        return CognitiveState.STRESSED
    if stress > _threshold("alert", 0.30) or overload > _threshold("alert", 0.30):
        return CognitiveState.ALERT
    return CognitiveState.CALM


def _compute_decision_capacity(state: CognitiveState,
                               stress: float, overload: float) -> float:
    """Decision capacity drops sharply at extremes of arousal (Yerkes-Dodson)."""
    base = {
        CognitiveState.CALM:       0.95,
        CognitiveState.ALERT:      0.90,
        CognitiveState.STRESSED:   0.70,
        CognitiveState.OVERLOADED: 0.45,
        CognitiveState.FROZEN:     0.20,
        CognitiveState.PANICKING:  0.15,
        CognitiveState.COLLAPSED:  0.05,
    }[state]
    return max(0.0, base - overload * 0.1 - stress * 0.05)


def _compute_receptivity(state: CognitiveState, overload: float) -> float:
    """How well the user can receive and process guidance right now."""
    base = {
        CognitiveState.CALM:       1.00,
        CognitiveState.ALERT:      0.90,
        CognitiveState.STRESSED:   0.70,
        CognitiveState.OVERLOADED: 0.35,
        CognitiveState.FROZEN:     0.50,   # frozen user CAN receive if prompt is right
        CognitiveState.PANICKING:  0.20,
        CognitiveState.COLLAPSED:  0.10,
    }[state]
    return max(0.0, base - overload * 0.15)


class PersonalBaseline:
    """Running estimate of this user's resting motor / ocular / speech activity.

    Freezing is a drop below the person's *own* baseline, not absolute
    stillness. The previous formula rewarded stillness in absolute terms
    ((1-motor)*0.4 + (1-saccade)*0.35 + speech_low*0.25), which scored a calm,
    motionless user at ~0.63 and tripped the 0.65 FROZEN gate on roughly a
    quarter of the ticks of a scenario with zero injected stress.

    The baseline absorbs only low-arousal frames and holds still while a
    candidate freeze is in progress, so a freeze episode cannot quietly become
    the new normal. Until WARMUP_FRAMES frames have been seen the baseline is
    not ``ready`` and freeze_score stays 0 — the monitor reports "not enough
    evidence" rather than guessing.

    Known limitation: while freeze_raw stays above FREEZE_HOLD the baseline
    does not track genuine long-term changes in the user's resting activity.
    """

    WARMUP_FRAMES = 15
    ALPHA = 0.05                # EMA weight once warmed up
    CALM_STRESS_MAX = 0.45      # above this the frame is not baseline material
    FREEZE_HOLD = 0.35          # above this a freeze may be under way — hold the baseline

    def __init__(self):
        self.motor: Optional[float] = None
        self.saccade: Optional[float] = None
        self.speech_wpm: Optional[float] = None
        self.samples = 0

    @property
    def ready(self) -> bool:
        return self.samples >= self.WARMUP_FRAMES

    def update(self, stress: float, motor: float, saccade_norm: float,
               speech_wpm: float, freeze_raw: float = 0.0):
        if self.ready and (stress > self.CALM_STRESS_MAX or
                           freeze_raw > self.FREEZE_HOLD):
            return
        self.samples += 1
        alpha = self.ALPHA if self.ready else 1.0 / self.samples
        if self.motor is None:
            self.motor, self.saccade, self.speech_wpm = motor, saccade_norm, speech_wpm
            return
        self.motor += alpha * (motor - self.motor)
        self.saccade += alpha * (saccade_norm - self.saccade)
        self.speech_wpm += alpha * (speech_wpm - self.speech_wpm)

    @staticmethod
    def _drop(current: float, base: Optional[float], floor: float) -> float:
        """Fractional drop of ``current`` below ``base``, 0 when at or above it."""
        if base is None or base <= floor:
            return 0.0
        return max(0.0, min(1.0, (base - current) / base))

    def drops(self, motor: float, saccade_norm: float,
              speech_wpm: float) -> tuple[float, float, float]:
        if not self.ready:
            return 0.0, 0.0, 0.0
        return (self._drop(motor, self.motor, 0.005),
                self._drop(saccade_norm, self.saccade, 0.02),
                self._drop(speech_wpm, self.speech_wpm, 20.0))


class L3HumanStateMonitor:
    """
    Layer 3 — Human State Monitor.

    Fuses multimodal biosignals from Layer 1 into continuous cognitive state estimates.
    Maintains a rolling window for trend detection (escalation / de-escalation).

    Production extensions:
      - OpenFace 2 micro-expression analysis on AR video
      - Affectiva SDK for emotion recognition
      - Heart rate estimation from facial blood-flow (rPPG)
      - Skin conductance from wearable sensor
    """

    WINDOW_SIZE = 30   # frames (~1 second at 30fps)

    # Voice-pitch normalisation window (Hz). Calibrated against the synthetic
    # generator in l1_perception.MicrophoneAdapter, which emits
    # N(150 + panic*100, 20): ~150 Hz at rest, ~250 Hz at maximum panic.
    PITCH_BASE_HZ = 140.0
    PITCH_SPAN_HZ = 110.0

    # Motor agitation normalisation: lateral acceleration (m/s^2, gravity
    # excluded) and angular rate (rad/s) at maximum injected panic.
    LATERAL_ACC_REF = 3.0
    GYRO_REF        = 1.2

    # Ocular normalisation. These used to be round numbers (500 deg/s, 20 bpm)
    # that the generator never approaches: saccade velocity tops out at ~400
    # and blink deviation at ~9 bpm, so gaze_instability saturated at 0.73 and
    # cognitive_overload / attentional_collapse could not reach their gates.
    SACCADE_REF  = 400.0    # deg/s at maximum injected stress
    FIXATION_REF = 400.0    # ms; longer than this counts as a full fixation
    # Blink rate is only abnormal outside the physiological resting band
    # (l1_perception documents 15–20 bpm as normal, <10 or >30 as stressed).
    # Without the dead band, sensor noise around the resting rate alone put
    # cognitive_overload above the ALERT gate on ~17% of resting frames.
    BLINK_RESTING_BPM = 16.0
    BLINK_TOLERANCE   = 4.0
    BLINK_REF         = 5.0

    # Arousal at which the freeze gate opens fully (see freeze block below).
    FREEZE_AROUSAL_REF = 0.30

    def __init__(self):
        self._stress_history:  deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._panic_history:   deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._freeze_history:  deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self.baseline = PersonalBaseline()
        self._tick = 0
        logger.info("[L3] Human State Monitor initialized")

    def process(self, perception: PerceptionFrame) -> HumanStateFrame:
        self._tick += 1
        ts = time.time()

        eye   = perception.eye
        audio = perception.audio
        imu   = perception.imu

        # ── Voice stress index ────────────────────────────────────────────────
        # High pitch, tremor, fast speech → elevated stress.
        # The pitch term must come from the microphone. It used to be computed
        # from eye.pupil_diameter_mm, so the "voice stress index" published by
        # the REST API was 40% pupil diameter and voice_pitch_hz was never read.
        pitch_norm  = min(1.0, max(0.0,
            (audio.voice_pitch_hz - self.PITCH_BASE_HZ) / self.PITCH_SPAN_HZ))
        tremor_idx  = audio.voice_tremor
        rate_stress = min(1.0, max(0.0, (audio.speech_rate_wpm - 130) / 120))
        voice_stress = (pitch_norm * 0.4 + tremor_idx * 0.35 + rate_stress * 0.25)

        # ── Gaze stability index ──────────────────────────────────────────────
        # Low fixation, high saccade velocity → instability
        fixation_norm  = min(1.0, eye.fixation_duration_ms / self.FIXATION_REF)
        saccade_norm   = min(1.0, eye.saccade_velocity / self.SACCADE_REF)
        blink_abnormal = max(0.0,
            abs(eye.blink_rate_per_min - self.BLINK_RESTING_BPM) - self.BLINK_TOLERANCE
        ) / self.BLINK_REF
        gaze_instability = saccade_norm * 0.5 + (1.0 - fixation_norm) * 0.35 + min(1.0, blink_abnormal) * 0.15
        gaze_stability = 1.0 - gaze_instability

        # ── Motor agitation ───────────────────────────────────────────────────
        # Agitation is lateral movement and head rotation, not the magnitude of
        # the acceleration vector: |a| is dominated by gravity and stays near
        # 9.8 m/s^2 however much the user thrashes, so the old
        # (|a| - 9.8) / 15 term measured ~0.02 even at maximum injected panic
        # and capped panic_score at 0.66.
        lateral_acc = math.sqrt(imu.accel_x ** 2 + imu.accel_y ** 2)
        gyro_mag = math.sqrt(imu.gyro_x ** 2 + imu.gyro_y ** 2 + imu.gyro_z ** 2)
        motor_agitation = min(1.0, max(0.0,
            0.5 * (lateral_acc / self.LATERAL_ACC_REF) +
            0.5 * (gyro_mag / self.GYRO_REF)))

        # ── Pupil dilation (arousal) ──────────────────────────────────────────
        pupil_arousal = min(1.0, max(0.0, (eye.pupil_diameter_mm - 3.0) / 5.0))

        # ── Composite scores ──────────────────────────────────────────────────
        stress_score = (
            voice_stress        * 0.35 +
            gaze_instability    * 0.25 +
            pupil_arousal       * 0.20 +
            motor_agitation     * 0.20
        )
        stress_score = min(1.0, stress_score)

        # Cognitive overload: high blink rate, short fixations, many micro-saccades
        overload = (
            min(1.0, blink_abnormal) * 0.4 +
            (1.0 - fixation_norm)    * 0.35 +
            saccade_norm             * 0.25
        )
        overload = min(1.0, overload)

        # Panic: fast speech + high motor + high stress + loud environment
        ambient_stress = min(1.0, (perception.audio.ambient_db - 45.0) / 50.0)
        panic_score = (
            voice_stress    * 0.35 +
            motor_agitation * 0.30 +
            stress_score    * 0.20 +
            ambient_stress  * 0.15
        )
        panic_score = min(1.0, panic_score)

        # Attentional collapse: dispersed gaze, abnormal blink, poor fixation
        attentional = (
            gaze_instability           * 0.50 +
            min(1.0, blink_abnormal)   * 0.30 +
            stress_score               * 0.20
        )
        attentional = min(1.0, attentional)

        arousal = (stress_score + panic_score + pupil_arousal) / 3.0

        # ── Freeze: drop below THIS user's baseline, while aroused ────────────
        motor_drop, saccade_drop, speech_drop = self.baseline.drops(
            motor_agitation, saccade_norm, audio.speech_rate_wpm)
        freeze_raw = (motor_drop   * 0.40 +
                      saccade_drop * 0.35 +
                      speech_drop  * 0.25)
        # Tonic immobility is stillness *under arousal*; a relaxed user who
        # simply stops moving is not frozen.
        arousal_gate = min(1.0, arousal / self.FREEZE_AROUSAL_REF)
        freeze_score = freeze_raw * arousal_gate
        # Freeze and panic are mutually exclusive: high panic suppresses freeze
        freeze_score = max(0.0, freeze_score - panic_score * 0.8)
        self.baseline.update(stress_score, motor_agitation, saccade_norm,
                             audio.speech_rate_wpm, freeze_raw)

        # ── Classification ────────────────────────────────────────────────────
        state = _classify_state(stress_score, overload, panic_score,
                                freeze_score, attentional)
        decision_capacity = _compute_decision_capacity(state, stress_score, overload)
        receptivity = _compute_receptivity(state, overload)

        # ── Trends ────────────────────────────────────────────────────────────
        self._stress_history.append(stress_score)
        self._panic_history.append(panic_score)
        self._freeze_history.append(freeze_score)

        stress_delta = 0.0
        panic_delta  = 0.0
        if len(self._stress_history) >= 5:
            recent = list(self._stress_history)
            stress_delta = recent[-1] - sum(recent[-5:-1]) / 4
            panic_recent = list(self._panic_history)
            panic_delta  = panic_recent[-1] - sum(panic_recent[-5:-1]) / 4

        frame = HumanStateFrame(
            timestamp=ts,
            stress_score=round(stress_score, 3),
            cognitive_overload=round(overload, 3),
            panic_score=round(panic_score, 3),
            freeze_score=round(freeze_score, 3),
            attentional_collapse=round(attentional, 3),
            arousal=round(arousal, 3),
            state=state,
            decision_capacity=round(decision_capacity, 3),
            receptivity=round(receptivity, 3),
            voice_stress_index=round(voice_stress, 3),
            gaze_stability=round(gaze_stability, 3),
            motor_agitation=round(motor_agitation, 3),
            stress_delta=round(stress_delta, 4),
            panic_delta=round(panic_delta, 4),
        )

        return frame

    def get_trend(self) -> str:
        """Summarize recent trend."""
        if len(self._stress_history) < 10:
            return "insufficient_data"
        recent = list(self._stress_history)
        delta = recent[-1] - recent[0]
        if delta > 0.15:   return "escalating"
        if delta < -0.15:  return "de-escalating"
        return "stable"

    @property
    def mean_freeze(self) -> float:
        """Mean freeze score over the rolling window (0.0 when empty)."""
        if not self._freeze_history:
            return 0.0
        return sum(self._freeze_history) / len(self._freeze_history)

    @property
    def tick(self) -> int:
        return self._tick
