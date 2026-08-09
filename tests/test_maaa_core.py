from maaa_core.models import (SensorFrame, AutopoieticStatus, RiskLevel,
                              OutputChannel)
from maaa_core.orchestrator import MAAAOrchestrator
from maaa_core.human_state import HumanStateEstimator


def test_critical_risk_guides_to_safe_path():
    maaa = MAAAOrchestrator(max_words=9)
    frame = SensorFrame(
        detected_objects=["scala", "porta_est"],
        blocked_paths=["scala"],
        safe_paths=["porta_est"],
        hazards={"scala": 0.94, "porta_est": 0.10},
        audio_stress=0.6,
        gaze_fixation_risk=0.4,
    )
    plan = maaa.process(frame, AutopoieticStatus(True, 70, 0.9, 80))
    assert plan.priority.value == "CRITICAL"
    assert "porta_est" in plan.message
    assert len(plan.message.split()) <= 9


def test_failsafe_overrides_regulatory_plan():
    maaa = MAAAOrchestrator()
    frame = SensorFrame(hazards={"corridoio": 0.2}, safe_paths=["uscita"])
    plan = maaa.process(frame, AutopoieticStatus(system_ok=False, battery_pct=70, sensor_integrity=0.9, latency_ms=80))
    assert plan.mode.value == "failsafe"
    assert plan.reason == "autopoietic_failsafe"


def test_human_state_entropy_range():
    estimator = HumanStateEstimator()
    state = estimator.estimate(SensorFrame(audio_stress=0.9, imu_instability=0.7, gaze_fixation_risk=0.8))
    assert 0 <= state.cognitive_entropy <= 1
    assert state.cognitive_entropy > 0.6


def test_memory_records_events():
    maaa = MAAAOrchestrator()
    frame = SensorFrame(hazards={"area_sinistra": 0.91}, safe_paths=["porta_est"])
    maaa.process(frame, AutopoieticStatus(True, 70, 0.9, 70))
    assert len(maaa.memory.events) == 1


def test_low_risk_observe_only():
    maaa = MAAAOrchestrator()
    frame = SensorFrame(hazards={"corridoio": 0.12}, safe_paths=["uscita"])
    plan = maaa.process(frame, AutopoieticStatus(True, 70, 0.95, 60))
    assert plan.mode.value == "observe"


# ── Human override, NFR-05 (SensorFrame.user_command was declared, never read) ──

def test_user_command_mute_latches_until_resume():
    maaa = MAAAOrchestrator()
    status = AutopoieticStatus(True, 70, 0.95, 60)
    hazards = {"corridoio": 0.50}

    muted = maaa.process(SensorFrame(hazards=hazards, safe_paths=["uscita"],
                                     user_command="mute"), status)
    assert muted.reason == "user_override_mute"
    assert muted.channels == [OutputChannel.LOG]

    # No further command: the mute is still in force.
    still = maaa.process(SensorFrame(hazards=hazards, safe_paths=["uscita"]), status)
    assert still.reason == "user_override_mute"

    resumed = maaa.process(SensorFrame(hazards={"corridoio": 0.70},
                                       safe_paths=["uscita"],
                                       user_command="resume"), status)
    assert resumed.reason == "high_risk"


def test_user_command_stop_suppresses_only_this_frame():
    maaa = MAAAOrchestrator()
    status = AutopoieticStatus(True, 70, 0.95, 60)
    stopped = maaa.process(SensorFrame(hazards={"corridoio": 0.70},
                                       safe_paths=["uscita"],
                                       user_command="stop"), status)
    assert stopped.reason == "user_override_stop"
    following = maaa.process(SensorFrame(hazards={"corridoio": 0.70},
                                         safe_paths=["uscita"]), status)
    assert following.reason == "high_risk"


def test_mute_cannot_silence_critical_guidance():
    """docs/SAFETY.md: the user can switch off advice, not the danger warning."""
    maaa = MAAAOrchestrator()
    status = AutopoieticStatus(True, 70, 0.95, 60)
    maaa.process(SensorFrame(hazards={"corridoio": 0.10}, user_command="mute"), status)
    plan = maaa.process(SensorFrame(hazards={"scala": 0.94},
                                    safe_paths=["porta_est"]), status)
    assert plan.priority is RiskLevel.CRITICAL
    assert plan.reason == "critical_risk"


def test_unknown_user_command_is_ignored_and_logged(caplog):
    maaa = MAAAOrchestrator()
    status = AutopoieticStatus(True, 70, 0.95, 60)
    with caplog.at_level("WARNING", logger="maaa.core.orchestrator"):
        plan = maaa.process(SensorFrame(hazards={"corridoio": 0.70},
                                        safe_paths=["uscita"],
                                        user_command="launch"), status)
    assert plan.reason == "high_risk"
    assert any("unknown user_command" in r.getMessage() for r in caplog.records)


def test_risk_engine_is_a_classifier_over_supplied_hazards():
    """Documented behaviour: the score is the caller's largest hazard value,
    put in a band. Nothing is estimated."""
    from maaa_core.perception import SceneGraphBuilder
    from maaa_core.risk import RiskEstimationEngine
    graph = SceneGraphBuilder().build(SensorFrame(hazards={"a": 0.42, "b": 0.77}))
    score, level, item = RiskEstimationEngine().score(graph)
    assert (score, level.value, item) == (0.77, "HIGH", "b")
