"""Minimal reference-core demo (maaa_core).

Every number below is typed in by hand. The "risk 0.91 CRITICAL" the demo
prints is the 0.91 written into `hazards` three lines above it — the pipeline
classifies that number into a band, it does not estimate it. See README.md.

For the full runtime (five layers, REST API, SQLite memory) run:
    python main_maaa.py demo
"""

import os
import sys

# Allow `python examples/run_demo.py` from a clean clone without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maaa_core.models import SensorFrame, AutopoieticStatus     # noqa: E402
from maaa_core.orchestrator import MAAAOrchestrator             # noqa: E402

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
