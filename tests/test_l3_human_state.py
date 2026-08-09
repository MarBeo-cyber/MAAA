"""L3 human-state regressions: false positives, wrong inputs, saturation."""

from __future__ import annotations

import math

from layers.l1_perception import (L1EmbodiedPerception, SceneCondition,
                                  AudioFrame, EyeTrackingData, IMUData,
                                  PerceptionFrame, VideoFrame, DepthMap, GPSData)
from layers.l3_human_state import L3HumanStateMonitor, CognitiveState


def _calm_run(ticks: int):
    l1 = L1EmbodiedPerception(True)
    l3 = L3HumanStateMonitor()
    l1.inject_scenario(SceneCondition.NORMAL, stress=0.0, panic=0.0,
                       obstruction=0.0, emergency_sounds=False)
    return [l3.process(l1.capture()) for _ in range(ticks)]


def test_calm_baseline_does_not_produce_frozen_false_positives():
    """The freeze formula used to reward absolute stillness: a calm, motionless
    user scored ~0.63 and crossed the 0.65 FROZEN gate on ~25% of ticks of a
    NORMAL scenario with zero injected stress, writing a spurious human_crisis
    row each time. Target from the working paper: < 5% false alarms."""
    frames = _calm_run(1000)
    frozen = sum(1 for f in frames if f.state is CognitiveState.FROZEN)
    rate = frozen / len(frames)
    assert rate < 0.05, f"FROZEN false-positive rate {rate:.1%} (was ~25%)"


def test_calm_baseline_is_classified_calm():
    frames = _calm_run(1000)
    calm = sum(1 for f in frames if f.state is CognitiveState.CALM)
    assert calm / len(frames) > 0.90, (
        f"only {calm}/1000 calm frames classified CALM")


def test_freeze_needs_a_drop_below_the_users_own_baseline():
    """Absolute stillness alone is not freezing; a drop below the personal
    baseline while aroused is."""
    l1 = L1EmbodiedPerception(True)
    l3 = L3HumanStateMonitor()
    l1.inject_scenario(SceneCondition.NORMAL, 0.0, 0.0, 0.0, False)
    for _ in range(60):
        l3.process(l1.capture())
    calm_freeze = max(l3.process(l1.capture()).freeze_score for _ in range(100))

    l1.inject_scenario(SceneCondition.SMOKY, stress=0.9, panic=0.0,
                       obstruction=0.3, emergency_sounds=True, freeze=1.0)
    frozen_freeze = max(l3.process(l1.capture()).freeze_score for _ in range(200))

    assert calm_freeze < 0.40, calm_freeze
    assert frozen_freeze > 0.65, frozen_freeze


def test_freeze_is_zero_before_the_baseline_is_established():
    """Without a baseline the monitor must report no evidence, not a guess."""
    l1 = L1EmbodiedPerception(True)
    l3 = L3HumanStateMonitor()
    l1.inject_scenario(SceneCondition.SMOKY, stress=0.9, panic=0.0,
                       obstruction=0.3, emergency_sounds=True, freeze=1.0)
    first = l3.process(l1.capture())
    assert not l3.baseline.ready
    assert first.freeze_score == 0.0


def _frame(pitch_hz: float, pupil_mm: float) -> PerceptionFrame:
    """Perception frame in which only voice pitch and pupil diameter vary."""
    ts = 0.0
    return PerceptionFrame(
        timestamp=ts,
        video=VideoFrame(ts, 1920, 1080, 0.85, 0.75, 0.1, 0.02, 0.01,
                         SceneCondition.NORMAL),
        depth=DepthMap(ts, 1.0, 5.0, 3.0),
        imu=IMUData(ts, 0.1, 0.1, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gps=GPSData(ts, 45.0, 9.0, 50.0, 3.0),
        eye=EyeTrackingData(ts, 0.5, 0.5, 15.0, pupil_mm, 300.0, 100.0, 2.5),
        audio=AudioFrame(ts, True, pitch_hz, 0.0, 130.0, 45.0, False, False, False),
    )


def test_voice_stress_index_reads_voice_pitch_not_pupil_diameter():
    """The pitch component of voice_stress was computed from
    eye.pupil_diameter_mm; audio.voice_pitch_hz was generated and never read,
    so the 'voice stress index' published by /human was 40% pupil diameter."""
    l3 = L3HumanStateMonitor()
    low_pitch = l3.process(_frame(pitch_hz=150.0, pupil_mm=4.0)).voice_stress_index
    high_pitch = l3.process(_frame(pitch_hz=250.0, pupil_mm=4.0)).voice_stress_index
    assert high_pitch > low_pitch + 0.2, (low_pitch, high_pitch)

    # And changing the pupil alone must not move the *voice* index at all.
    small_pupil = l3.process(_frame(pitch_hz=150.0, pupil_mm=2.5)).voice_stress_index
    big_pupil = l3.process(_frame(pitch_hz=150.0, pupil_mm=7.5)).voice_stress_index
    assert small_pupil == big_pupil, (small_pupil, big_pupil)


def test_motor_agitation_excludes_gravity():
    """(|a| - 9.8)/15 is dominated by gravity: it read ~0.02 even at maximum
    injected panic, which capped panic_score at 0.66."""
    l3 = L3HumanStateMonitor()
    still = _frame(150.0, 4.0)
    agitated = _frame(150.0, 4.0)
    agitated.imu = IMUData(0.0, 2.0, 1.5, 9.8, 0.8, 0.6, 0.5, 0.0, 0.0, 0.0)

    still_motor = l3.process(still).motor_agitation
    agitated_motor = l3.process(agitated).motor_agitation
    # The total acceleration barely differs...
    assert abs(agitated.imu.total_acceleration - still.imu.total_acceleration) < 1.0
    # ...but the agitation must.
    assert agitated_motor > 0.5, agitated_motor
    assert still_motor < 0.1, still_motor


def test_blink_rate_within_the_resting_band_is_not_abnormal():
    """Sensor noise around the 15-16 bpm resting rate alone used to push
    cognitive_overload over the ALERT gate on ~17% of resting frames."""
    l3 = L3HumanStateMonitor()
    f = _frame(150.0, 4.0)
    f.eye = EyeTrackingData(0.0, 0.5, 0.5, 15.0, 4.0, 300.0, 100.0, 2.5)
    resting = l3.process(f).cognitive_overload
    assert resting < 0.30, resting


def test_needs_immediate_override_is_reachable():
    """panic > 0.85 / overload > 0.90 sat above everything the pipeline could
    produce, so this property was always False."""
    l1 = L1EmbodiedPerception(True)
    l3 = L3HumanStateMonitor()
    l1.inject_scenario(SceneCondition.COLLAPSED, stress=1.0, panic=1.0,
                       obstruction=1.0, emergency_sounds=True)
    assert any(l3.process(l1.capture()).needs_immediate_override
               for _ in range(400))


def test_freeze_history_is_recorded():
    """_freeze_history was declared and never appended to."""
    l1 = L1EmbodiedPerception(True)
    l3 = L3HumanStateMonitor()
    l1.inject_scenario(SceneCondition.NORMAL, 0.0, 0.0, 0.0, False)
    for _ in range(40):
        l3.process(l1.capture())
    assert len(l3._freeze_history) > 0
    assert not math.isnan(l3.mean_freeze)
