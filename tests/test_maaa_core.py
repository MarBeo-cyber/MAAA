from maaa_core.models import SensorFrame, AutopoieticStatus
from maaa_core.orchestrator import MAAAOrchestrator
from maaa_core.regulatory import RegulatoryEngine
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
