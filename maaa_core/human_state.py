from __future__ import annotations

from .models import SensorFrame, HumanState


class HumanStateEstimator:
    """L3 - estimates stress, overload, panic and attentional collapse.

    Prototype: uses simulated audio stress, IMU instability and gaze fixation.
    Production: voice stress, HRV/GSR from PAAA, eye tracking, micro-expression
    and motion pattern features.
    """

    def estimate(self, frame: SensorFrame) -> HumanState:
        stress = max(0.0, min(1.0, frame.audio_stress))
        overload = max(0.0, min(1.0, 0.55 * frame.audio_stress + 0.45 * frame.gaze_fixation_risk))
        panic = max(0.0, min(1.0, 0.50 * frame.audio_stress + 0.50 * frame.imu_instability))
        attention_collapse = max(0.0, min(1.0, frame.gaze_fixation_risk))
        return HumanState(stress=round(stress, 4), overload=round(overload, 4), panic=round(panic, 4), attention_collapse=round(attention_collapse, 4))
