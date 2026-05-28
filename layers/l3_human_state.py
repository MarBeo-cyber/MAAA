"""
MAAA — Layer 3: Human State Monitoring (Monitoraggio dello Stato Umano)

Stima continua dello stato cognitivo ed emotivo dell'utente da segnali multimodali:
  - Stress Detection → voce (pitch, tremor, speech rate) + pupil dilation
  - Cognitive Overload Estimation → blink rate, fixation duration, saccade velocity
  - Freezing / Indecision Detection → ridotta velocità saccadica, immobilità
  - Panic Estimation → movimento rapido, voce ad alta pitch, respirazione accelerata
  - Attentional Collapse Detection → gaze disperso, blink rate anomalo

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

from layers.l1_perception import PerceptionFrame

logger = logging.getLogger("maaa.l3_human_state")


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
        return (self.panic_score > 0.75 or
                self.freeze_score > 0.70 or
                self.attentional_collapse > 0.80)

    @property
    def needs_immediate_override(self) -> bool:
        """Emergency: guidance must be simplified to single imperative."""
        return self.panic_score > 0.85 or self.cognitive_overload > 0.90

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
    """Rule-based classifier — production replaces with trained model on labelled data."""
    if panic > 0.75:
        return CognitiveState.PANICKING
    if attentional > 0.75:
        return CognitiveState.COLLAPSED
    if freeze > 0.65:
        return CognitiveState.FROZEN
    if overload > 0.70:
        return CognitiveState.OVERLOADED
    if stress > 0.60:
        return CognitiveState.STRESSED
    if stress > 0.30 or overload > 0.30:
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

    def __init__(self):
        self._stress_history:  deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._panic_history:   deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._freeze_history:  deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._prev_frame: Optional[HumanStateFrame] = None
        self._tick = 0
        logger.info("[L3] Human State Monitor initialized")

    def process(self, perception: PerceptionFrame) -> HumanStateFrame:
        self._tick += 1
        ts = time.time()

        eye   = perception.eye
        audio = perception.audio
        imu   = perception.imu

        # ── Voice stress index ────────────────────────────────────────────────
        # High pitch, tremor, fast speech → elevated stress
        pitch_norm  = min(1.0, max(0.0, (eye.pupil_diameter_mm - 2.0) / 6.0))
        tremor_idx  = audio.voice_tremor
        rate_stress = min(1.0, max(0.0, (audio.speech_rate_wpm - 130) / 120))
        voice_stress = (pitch_norm * 0.4 + tremor_idx * 0.35 + rate_stress * 0.25)

        # ── Gaze stability index ──────────────────────────────────────────────
        # Low fixation, high saccade velocity → instability
        fixation_norm  = min(1.0, eye.fixation_duration_ms / 400.0)
        saccade_norm   = min(1.0, eye.saccade_velocity / 500.0)
        blink_abnormal = abs(eye.blink_rate_per_min - 16.0) / 20.0
        gaze_instability = saccade_norm * 0.5 + (1.0 - fixation_norm) * 0.35 + min(1.0, blink_abnormal) * 0.15
        gaze_stability = 1.0 - gaze_instability

        # ── Motor agitation ───────────────────────────────────────────────────
        total_acc = imu.total_acceleration
        motor_agitation = min(1.0, max(0.0, (total_acc - 9.8) / 15.0))

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

        # Freeze: low motor + low saccade velocity + low speech rate
        speech_low = min(1.0, max(0.0, (130.0 - audio.speech_rate_wpm) / 90.0))
        freeze_score = (
            (1.0 - motor_agitation)   * 0.40 +
            (1.0 - saccade_norm)      * 0.35 +
            speech_low                * 0.25
        )
        # Freeze and panic are mutually exclusive: high panic suppresses freeze
        freeze_score = max(0.0, freeze_score - panic_score * 0.8)

        # Attentional collapse: dispersed gaze, abnormal blink, poor fixation
        attentional = (
            gaze_instability           * 0.50 +
            min(1.0, blink_abnormal)   * 0.30 +
            stress_score               * 0.20
        )
        attentional = min(1.0, attentional)

        arousal = (stress_score + panic_score + pupil_arousal) / 3.0

        # ── Classification ────────────────────────────────────────────────────
        state = _classify_state(stress_score, overload, panic_score,
                                freeze_score, attentional)
        decision_capacity = _compute_decision_capacity(state, stress_score, overload)
        receptivity = _compute_receptivity(state, overload)

        # ── Trends ────────────────────────────────────────────────────────────
        self._stress_history.append(stress_score)
        self._panic_history.append(panic_score)

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

        self._prev_frame = frame
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
    def tick(self) -> int:
        return self._tick
