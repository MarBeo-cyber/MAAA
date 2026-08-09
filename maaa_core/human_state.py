from __future__ import annotations

from .models import SensorFrame, HumanState


class HumanStateEstimator:
    """L3 — cognitive-entropy proxy over caller-supplied scalars.

    No sensor is consumed and no state is detected. ``stress`` and
    ``attention_collapse`` are identity passthroughs of ``frame.audio_stress``
    and ``frame.gaze_fixation_risk``; ``overload`` and ``panic`` are two fixed
    weighted sums of the same three numbers. Those three numbers are invented
    by whoever constructs the SensorFrame.

    ``HumanState.cognitive_entropy`` is the single quantity this layer exists
    to produce: one scalar summarising how much the guidance should be
    simplified. It is a design proxy, not a measurement, and it has never been
    validated against a human subject.

    A real implementation would need voice-stress features, HRV/GSR, eye
    tracking and motion patterns — none of which exist here.
    """

    def estimate(self, frame: SensorFrame) -> HumanState:
        stress = max(0.0, min(1.0, frame.audio_stress))
        overload = max(0.0, min(1.0, 0.55 * frame.audio_stress + 0.45 * frame.gaze_fixation_risk))
        panic = max(0.0, min(1.0, 0.50 * frame.audio_stress + 0.50 * frame.imu_instability))
        attention_collapse = max(0.0, min(1.0, frame.gaze_fixation_risk))
        return HumanState(stress=round(stress, 4), overload=round(overload, 4), panic=round(panic, 4), attention_collapse=round(attention_collapse, 4))
