# API Specification

**All data returned by this API is derived from synthetic sensor input.**
Risk scores, human-state scores and positions are generated, not measured.

This document describes the API that exists: the Flask app in
`api/maaa_api.py`, served by `python main_maaa.py server` on port 5002.
Contract tests: `tests/test_api.py`.

The previous version of this file described a JSON API over `maaa_core`. That
API has never existed — `maaa_core` has no serialisation layer at all. Its
data model is documented at the end of this file as Python objects.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | Version, tick count, session summary, full last snapshot |
| GET | `/snapshot` | Full last pipeline snapshot |
| GET | `/human` | Layer 3 output |
| GET | `/risk` | Layer 2 risk map with per-object detail |
| GET | `/guidance` | Layer 4 output: message, AR overlay, haptic, filter log |
| GET | `/memory/working` | Last 20 working-memory events (60 s window) |
| GET | `/memory/episodic` | Session event counts and 20 most recent |
| GET | `/memory/recall` | Cosine recall of similar past situations |
| GET | `/session` | Layer 5 session summary |
| GET | `/override` | Current human override and accepted commands |
| POST | `/override` | Set or clear the human override |
| POST | `/scenario/<name>` | Inject a simulated scenario |
| POST | `/tick/n/<n>` | Run n pipeline ticks (capped at 200) |

`/human`, `/risk`, `/guidance` and `/memory/recall` return `503` with
`{"error": "no_data"}` before the first tick.

---

## GET /status

```json
{
  "agent": "MAAA v0.4.0",
  "tick": 12,
  "running": false,
  "simulation_mode": true,
  "data_source": "synthetic",
  "override": null,
  "session": { "...": "see GET /session" },
  "last":    { "...": "see GET /snapshot" }
}
```

`data_source` is always `"synthetic"` in this build.

## GET /snapshot

Nested view of one pipeline cycle: `tick`, `latency_ms`, `risk`, `human`,
`guidance`, `predictions`, `causal_summary`, `health`, `l4_stats`,
`output_stats`. The `health` block is the part worth reading closely:

```json
{
  "health": {
    "ok": true,
    "battery_pct": null,
    "battery_source": "unavailable",
    "failsafe": false,
    "warnings": [],
    "degraded_reasons": []
  }
}
```

`battery_pct` is `null` and `battery_source` is `"unavailable"` unless a
source has been attached via `L5AutopoieticContinuity.set_battery_source()`.
There is no battery-management driver in this repository. `main_maaa.py`
attaches one labelled `"simulated"`; never treat that as a reading.

`degraded_reasons` is empty when healthy, otherwise carries strings such as
`"l2_error:RuntimeError: model not loaded"`, `"l3_heartbeat_stale:2.4s"`,
`"memory_error:database is locked"` or `"battery_critical:0%"`. A non-empty
list means the failsafe is active.

## GET /human

```json
{
  "state": "collapsed",
  "stress_score": 0.768,
  "cognitive_overload": 0.9,
  "panic_score": 0.687,
  "freeze_score": 0.0,
  "attentional_collapse": 0.899,
  "arousal": 0.736,
  "decision_capacity": 0.0,
  "receptivity": 0.0,
  "voice_stress": 0.556,
  "gaze_stability": 0.109,
  "motor_agitation": 1.0,
  "stress_delta": 0.0302,
  "panic_delta": -0.0154,
  "is_critical": true,
  "needs_override": true,
  "trend": "stable"
}
```

`state` ∈ `calm | alert | stressed | overloaded | frozen | panicking | collapsed`.
All scores are 0–1. `trend` ∈ `insufficient_data | escalating | stable | de-escalating`.

`voice_stress` is computed from `audio.voice_pitch_hz`, `voice_tremor` and
`speech_rate_wpm`. (It used to be 40% pupil diameter — the pitch term read the
eye tracker.)

## GET /risk

```json
{
  "global_risk": 0.825,
  "global_level": "CRITICAL",
  "structural": 0.220,
  "time_to_act_s": 20.99,
  "passable_bearings": [90.0, 135.0, 225.0, 270.0],
  "recommended_path": 0.382,
  "objects": [
    {
      "id": "debris_l",
      "category": "debris",
      "dist_m": 1.131,
      "bearing": 26.86,
      "risk": 1.0,
      "level": "CRITICAL",
      "action": "Evitare — rischio crollo (debris)",
      "is_exit": false
    }
  ]
}
```

`global_level` ∈ `SAFE | LOW | MEDIUM | HIGH | CRITICAL`, from the bands in
`config/default.yaml`. `time_to_act_s` is `null` below a global risk of 0.7.

## GET /guidance

```json
{
  "urgency": "CRITICAL",
  "voice_message": "Non muoverti.",
  "full_message": "Non muoverti.",
  "suppressed": false,
  "reason": "",
  "filters": ["urgency=CRITICAL", "words=2",
              "relevance=bypassed:critical", "timing=pass:ok"],
  "delay_ms": 0.0,
  "ar_overlay": {
    "active": true,
    "text": "⚠ PERICOLO",
    "color": "red",
    "path_deg": 0.382,
    "dangers": [26.86, 327.59, 91.33]
  },
  "haptic": { "active": true, "pattern": "SOS", "intensity": 1.0 },
  "stats": { "total_cycles": 12, "outputs_delivered": 12,
             "outputs_suppressed": 0, "delivery_rate": 1.0 }
}
```

`urgency` ∈ `SILENT | AMBIENT | NORMAL | ELEVATED | CRITICAL`.
`color` ∈ `green | yellow | orange | red`.
`pattern` ∈ `none | single | double | SOS`.

When `suppressed` is `true`, `voice_message` is empty, `reason` names the
filter that blocked it (`relevance_filter`, `timing:too_soon (1.2s < 3.0s)`,
`user_override_mute`, …) and `full_message` holds the candidate that was not
delivered. Suppressed cycles are logged to episodic memory too (NFR-08).

## GET /session

```json
{
  "session_id": "e13f3218",
  "tick": 12,
  "uptime_s": 0.043,
  "working_mem_size": 14,
  "episodic_events": {"guidance_delivered": 12, "human_crisis": 1, "risk_critical": 1},
  "autobio_memories": 0,
  "failsafe_active": false,
  "avg_latency_ms": 0.333
}
```

Episodic event types: `guidance_delivered`, `guidance_suppressed`,
`risk_critical`, `human_crisis`, `failsafe_activated`, `failsafe_cleared`.

## POST /scenario/&lt;name&gt;

```bash
curl -XPOST localhost:5002/scenario/collapsed
# {"scenario": "collapsed", "injected": true}
```

| name | scene | stress | panic | obstruction | sounds | freeze |
|---|---|---|---|---|---|---|
| `normal` | NORMAL | 0.1 | 0.0 | 0.0 | no | 0.0 |
| `smoky` | SMOKY | 0.5 | 0.3 | 0.2 | yes | 0.0 |
| `dark` | DARK | 0.4 | 0.2 | 0.1 | no | 0.0 |
| `dusty` | DUSTY | 0.4 | 0.2 | 0.3 | no | 0.0 |
| `obstructed` | OBSTRUCTED | 0.4 | 0.1 | 0.8 | no | 0.0 |
| `collapsed` | COLLAPSED | 0.8 | 0.7 | 0.6 | yes | 0.0 |
| `panic` | COLLAPSED | 0.9 | 0.95 | 0.7 | yes | 0.0 |
| `frozen` | SMOKY | 0.9 | 0.0 | 0.3 | yes | 1.0 |

An unknown name returns `400`:

```json
{"error": "unknown scenario 'bogus'",
 "available": ["normal", "smoky", "dark", "collapsed", "panic",
               "dusty", "obstructed", "frozen"]}
```

## GET/POST /override

Human override (NFR-05, docs/SAFETY.md).

```bash
curl -XPOST localhost:5002/override -H 'content-type: application/json' \
     -d '{"command":"mute"}'
# {"override": "mute", "applied": "mute"}
```

Commands: `mute` (silence guidance below CRITICAL), `resume` / `clear` (remove
the override). `null` also clears. An unknown command returns `400` with the
`available` list.

**`mute` cannot silence CRITICAL urgency.** The user can switch off advice, not
the danger warning. See docs/SAFETY.md.

## POST /tick/n/&lt;n&gt;

```json
{"ticks_run": 3, "total_ticks": 15}
```

`n` is capped at 200. Serialised behind a lock.

---

## `maaa_core` data model (no HTTP surface)

The minimal reference core is used in-process. Its two dataclasses:

```python
SensorFrame(
    detected_objects=["scala_principale", "porta_est"],
    blocked_paths=["scala_principale"],
    safe_paths=["porta_est"],
    hazards={"scala_principale": 0.91, "porta_est": 0.10},
    imu_instability=0.35,
    audio_stress=0.78,
    gaze_fixation_risk=0.70,
    user_command=None,          # "mute" | "resume" | "stop"
)

GuidancePlan(
    mode=GuidanceMode.STOP,     # observe | warn | guide | stop | failsafe
    message="STOP. Evita scala_principale. Vai verso porta_est.",
    channels=[OutputChannel.VOICE, OutputChannel.AR, OutputChannel.HAPTIC],
    target_path="porta_est",
    avoid="scala_principale",
    priority=RiskLevel.CRITICAL,
    max_words=9,
    reason="critical_risk",
)
```

`hazards` values are supplied by the caller. `priority` is the largest of them
put in a band — the `CRITICAL` above is the `0.91` on the line before it, not
an inference. See docs/ARCHITECTURE.md.
