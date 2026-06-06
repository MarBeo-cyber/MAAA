from maaa_core.models import SensorFrame, AutopoieticStatus
from maaa_core.orchestrator import MAAAOrchestrator

maaa = MAAAOrchestrator(max_words=9)

frame = SensorFrame(
    detected_objects=["scala_principale", "corridoio_nord", "porta_est"],
    blocked_paths=["scala_principale"],
    safe_paths=["porta_est"],
    hazards={
        "scala_principale": 0.91,
        "area_sinistra": 0.88,
        "corridoio_nord": 0.42,
        "porta_est": 0.10,
    },
    imu_instability=0.35,
    audio_stress=0.78,
    gaze_fixation_risk=0.70,
)

status = AutopoieticStatus(system_ok=True, battery_pct=72, sensor_integrity=0.92, latency_ms=84)
plan = maaa.process(frame, status=status)

print("MAAA guidance plan")
print("------------------")
print("mode:", plan.mode.value)
print("priority:", plan.priority.value)
print("message:", plan.message)
print("channels:", [c.value for c in plan.channels])
print("reason:", plan.reason)
print("memory_events:", len(maaa.memory.events))
