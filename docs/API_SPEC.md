# API Specification

## SensorFrame

Input to the runtime loop.

```json
{
  "detected_objects": ["scala_principale", "porta_est"],
  "blocked_paths": ["scala_principale"],
  "safe_paths": ["porta_est"],
  "hazards": {"scala_principale": 0.91, "porta_est": 0.10},
  "imu_instability": 0.35,
  "audio_stress": 0.78,
  "gaze_fixation_risk": 0.70
}
```

## GuidancePlan

Output of the runtime loop.

```json
{
  "mode": "stop",
  "message": "STOP. Evita scala_principale. Vai verso porta_est.",
  "channels": ["voice", "ar", "haptic"],
  "priority": "CRITICAL",
  "reason": "critical_risk"
}
```
