# MAAA Architecture

**All sensor input is synthetic.** Everything below describes what the code in
this repository does, not what a deployed system would do. Where a layer is a
placeholder, it says so.

## Core axiom

The MAAA is not a simulation of life. It simulates functions that make living
systems capable of persisting, learning and cooperating.

The design goal is *cognitive entropy reduction*: in an emergency the limiting
resource is the operator's attention, so the system's job is to say less, not
more. Every filter in Layer 4 exists to withhold information.

## Two implementations

| | `layers/` (+ `core/`, `api/`, `main_maaa.py`) | `maaa_core/` |
|---|---|---|
| Role | reference runtime | minimal reference core |
| Logical LOC | ~1,950 | ~260 |
| Shared | risk bands, brevity limit, failsafe thresholds, version — all via `maaa_config` |
| Not shared | data model. `layers` works on `PerceptionFrame`/`CognitionFrame`/`HumanStateFrame`; `maaa_core` on `SensorFrame`/`SceneGraph`/`HumanState` |

They are deliberately separate. `maaa_core` is the readable statement of the
loop; `layers` is the one with the memory, the API and the filters. Both are
live: `examples/run_demo.py` drives the first, `main_maaa.py` the second.

## Five layers

| Layer | Name | Role | What exists here |
|---|---|---|---|
| L1 | Embodied Perception | First-person sensing through AR/wearable devices | Synthetic generator for 6 scene conditions. `simulation_mode=False` raises `HardwareUnavailableError` |
| L2 | Situational Cognition | Scene graph, risk estimation, causal state | Object lookup table + heuristic risk score. No detector, no depth model, no SLAM |
| L3 | Human State Monitoring | Stress, overload, panic, freeze, attentional collapse | Weighted sums over synthetic scalars — a proxy, not a measurement |
| L4 | Symbiotic Regulation | Output filtering, timing, brevity, multimodal guidance | Four filters; 22 curated Italian messages selected deterministically |
| L5 | Autopoietic Continuity | Failsafe, memory, recovery, continuity of the human-system relation | Three-level memory (working / SQLite / vector), heartbeat health, failsafe + recovery |

## Runtime loop (reference runtime)

```text
L1.capture()      → PerceptionFrame     ─┐ per-stage heartbeat + latency
L2.process()      → CognitionFrame       │ reported to L5; a stage that
L3.process()      → HumanStateFrame      │ raises marks itself not-ok
L4.regulate()     → GuidanceOutput      ─┘
OutputDispatcher.dispatch()
L5.process()      → SystemHealth  (memory write + health + failsafe)
```

## L2 — what "risk estimation" means here

There is no estimator. Two things happen:

**Per object**, `_compute_risk` looks the category up in a fixed table
(`debris` 0.85, `door` 0.10, …), adds a proximity term
`max(0, 1 - d/10) * 0.2` and an environmental term
`(1 - env_quality) * 0.15`, plus gaussian noise. Hand-chosen constants.

**Aggregate**, `_build_risk_map`:

```text
env_degradation = 0.40·smoke + 0.20·dust + 0.40·(1 − env_quality)
proximity_w(d)  = clamp(1.5 / max(d, 0.3), 0, 1)
hazard          = max over objects of (risk_probability · proximity_w(distance))
global_risk     = 0.45·env_degradation + 0.55·hazard
```

The second term is why a CRITICAL hazard 1.2 m away can drive the aggregate on
its own. The previous formula was a flat weighted sum in which the worst object
contributed at most 0.25, so a hazard you were standing next to was diluted by
the average haze and `global_risk` could not exceed ~0.744 under any injected
scenario — putting the whole CRITICAL band out of reach.

In `maaa_core`, the equivalent component is
`RiskEstimationEngine` — a **threshold classifier over caller-supplied hazard
scores**. `SceneGraphBuilder` copies `SensorFrame.hazards` into the graph
unchanged; `score()` returns the largest value and puts it in a band. Given
`{"a": 0.42, "b": 0.77}` it returns `(0.77, HIGH, "b")`. The `0.91 CRITICAL` in
`examples/run_demo.py` is the `0.91` typed into that file three lines earlier.

## L3 — what "human state monitoring" means here

A **cognitive-entropy proxy over simulated scalars**. There is no Stress
Detector, no Overload Estimator and no Panic-Freezing Detector as separate
modules; there are five weighted sums over the synthetic eye, audio and IMU
values produced by L1.

```text
voice_stress  = 0.40·pitch_norm + 0.35·tremor + 0.25·rate_stress
gaze_instab.  = 0.50·saccade_norm + 0.35·(1 − fixation_norm) + 0.15·blink_abnormal
motor_agit.   = 0.50·(lateral_accel / 3.0) + 0.50·(|gyro| / 1.2)
stress        = 0.35·voice + 0.25·gaze_instab. + 0.20·pupil_arousal + 0.20·motor
overload      = 0.40·blink_abnormal + 0.35·(1 − fixation_norm) + 0.25·saccade_norm
panic         = 0.35·voice + 0.30·motor + 0.20·stress + 0.15·ambient
attentional   = 0.50·gaze_instab. + 0.30·blink_abnormal + 0.20·stress
freeze        = (0.40·motor_drop + 0.35·saccade_drop + 0.25·speech_drop)
                × min(1, arousal/0.30) − 0.8·panic
```

Three of these had inputs that made them structurally unable to move:

- `pitch_norm` was computed from `eye.pupil_diameter_mm`. The microphone's
  `voice_pitch_hz` was generated and never read, so the "voice stress index"
  published by `/human` was 40% pupil diameter. It now reads
  `(voice_pitch_hz − 140) / 110`.
- `motor_agitation` was `(|a| − 9.8) / 15`. `|a|` is dominated by gravity and
  stays near 9.8 m/s² however much the user thrashes, so it read ~0.02 at
  maximum injected panic and capped `panic_score` at 0.66. It now uses lateral
  acceleration and angular rate.
- Saccade velocity was divided by 500 °/s and blink deviation by 20 bpm, neither
  of which the generator approaches, so `gaze_instability` saturated at 0.73 and
  overload/attentional could not reach their gates. The references are now the
  values the generator actually produces, and blink abnormality has a ±4 bpm
  dead band around the resting rate.

### Freezing is relative, not absolute

`freeze` used to be `0.40·(1−motor) + 0.35·(1−saccade) + 0.25·speech_low` — it
rewarded stillness. A calm, motionless user scored ~0.63 and crossed the 0.65
FROZEN gate on ~25% of ticks of a scenario with zero injected stress, writing a
spurious `human_crisis` row to SQLite each time.

It is now the fractional **drop below the user's own baseline**, gated on
arousal (tonic immobility is stillness *under* arousal; a relaxed user who
stops moving is not frozen). `PersonalBaseline` is an EMA that absorbs only
low-stress frames and holds still while a candidate freeze is in progress, so a
freeze episode cannot become the new normal. Before the baseline has warmed up
(15 frames) `freeze_score` is 0 — the monitor reports "not enough evidence"
rather than guessing.

**Known limitation:** while `freeze_raw` stays above 0.35 the baseline does not
track genuine long-term changes in the user's resting activity.

## L4 — the four filters

1. **Relevance** — passes only what changes the optimal action (risk delta
   > 0.15, bearing change > 20°, or a new critical object). In a sustained
   emergency the risk plateaus and this filter silences most cycles; that is
   intended, and since the fix to L5 every suppressed cycle is still logged.
2. **Timing** — minimum interval by urgency (3.0 / 1.5 / 0.5 s), plus
   receptivity gates at panic > 0.90 and attentional collapse > 0.85.
3. **Brevity** — 9-word maximum from `config/default.yaml`. `generate()` selects
   one of 22 curated Italian messages deterministically from the risk map;
   `shorten()` falls back to the same set rather than truncating mid-sentence.
4. **Urgency** — SILENT / AMBIENT / NORMAL / ELEVATED / CRITICAL from the shared
   risk bands plus human-state gates, with anti-alarm-fatigue hysteresis: a
   level is held until the risk drops below 70% of its peak.

**Human override** (`set_override("mute"|"resume")`, `POST /override`) silences
guidance below CRITICAL. It cannot silence CRITICAL — see docs/SAFETY.md.

## L5 — continuity

Three memory levels, all real:

| Level | Storage | Lifetime |
|---|---|---|
| Working | thread-locked deque, 60 s rolling eviction | volatile |
| Episodic | SQLite, indexed, parameterised inserts | session |
| Autobiographical | JSON + numpy cosine over 16-dim vectors | cross-session |

Autobiographical vectors are keyed by `blake2b` rather than `hash()`. Python's
string hashing is salted per process, so the same state summary embedded in
three runs produced 0.8 / 0.7 / 0.32 and anything persisted to disk was
incomparable after a restart.

### Health and failsafe

Each pipeline stage reports a heartbeat to L5 with its latency; a stage that
raises reports `ok=False`. `SystemHealth` derives `sensor_ok` / `l2_ok` /
`l3_ok` / `l4_ok` from heartbeat freshness, error state and per-stage latency,
and `memory_ok` from a live SQLite probe. Any of these, a critical loop latency
or a battery below 5% populates `degraded_reasons`, which activates the
failsafe. After `recovery_cycles` healthy cycles the failsafe clears.
Activation and clearing are both written to episodic memory.

These five flags used to be hardcoded `True`, so `is_degraded` was always False
and `_activate_failsafe` was dead code — contradicting FR-08, NFR-07 and
SAFETY.md.

**Battery** is not fabricated. There is no battery-management driver in this
repository, so `battery_pct` is `None` and `battery_source` is `"unavailable"`
until a source is attached with `set_battery_source()`. `main_maaa.py` attaches
one labelled `"simulated"`. The previous build reported
`100 − uptime_hours × 25` through `/status` as if it were a reading.

## Calibration

Gates live in `config/default.yaml`. They are calibrated against the ranges the
synthetic generator can actually produce — measured over 1,000 ticks per
scenario, with the pipeline as shipped:

| Signal | Max observed (before) | Max observed (after) | Gate |
|---|---|---|---|
| `global_risk` | 0.744 | **0.872** | CRITICAL ≥ 0.80 |
| `panic_score` | 0.681 | **0.945** | PANICKING > 0.75 |
| `freeze_score` (frozen scenario) | 0.178 | **0.901** | FROZEN > 0.65 |
| `cognitive_overload` | 0.799 | **0.956** | OVERLOADED > 0.70 |
| `attentional_collapse` | 0.780 | **0.960** | COLLAPSED > 0.75 |
| `stress_score` | 0.916 | **0.957** | STRESSED > 0.60 |

On a calm NORMAL scenario the same signals peak at risk 0.184, stress 0.239,
panic 0.168, freeze 0.191 — every gate clear by a wide margin.
`tests/test_reachability.py` fails if any state or urgency level stops being
reachable; `tests/test_l3_human_state.py` fails if the calm baseline starts
producing FROZEN false positives above 5%.

## Key design rule

LLM reasoning is not allowed in the Tier-0 emergency loop. The emergency loop
uses deterministic, local, prevalidated functions. Nothing in this repository
calls a language model, and `config/default.yaml` sets
`llm_in_primary_emergency_loop: false`.
