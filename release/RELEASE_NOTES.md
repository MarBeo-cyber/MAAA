# MAAA Release Notes

## v0.4.0 — audit remediation

Corrective release. No new capability; the point is that the documents and the
code now say the same thing, and the safety-critical paths can execute.

### Honesty

- README line 1, every document header and `/status` now state that all sensor
  input is synthetic.
- "Risk Estimation Engine" is documented as what it is: a threshold classifier
  over caller-supplied hazard scores. "Human State Monitoring" as a
  cognitive-entropy proxy over simulated scalars.
- Removed the edge-first stack claim (YOLOv9, ORB-SLAM3, Depth Anything,
  OpenFace). None of it is present; `requirements.txt` is pyyaml, numpy, flask
  and pytest.
- Removed the unverified NFR figures and the verification-method column naming
  benchmarks that were never run. Each requirement now carries a status.
- `docs/API_SPEC.md` documents the Flask API that exists, replacing a spec for
  a JSON API over `maaa_core` that never did.
- The web page is labelled a storyboard: a hand-timed animation with hardcoded
  strings that never calls the Python.

### Safety paths made reachable

Over 2,000 ticks at maximum injected severity, the previous build peaked at
`global_risk` 0.744 against a 0.80 gate and `panic_score` 0.681 against 0.85.
`UrgencyLevel.CRITICAL`, the `FERMATI.` templates, the SOS haptic, the red AR
overlay, `CognitiveState.PANICKING` and the relevance-filter bypass could never
execute.

- Aggregate risk now combines environmental degradation with a
  proximity-weighted worst-case hazard, so a critical object 1.2 m away can
  drive it. Peak 0.872.
- `motor_agitation` no longer measures `|a| − 9.8` (dominated by gravity). Peak
  `panic_score` 0.945.
- Ocular normalisation references match what the generator produces; blink
  abnormality has a dead band around the resting rate.
- `tests/test_reachability.py` fails if any state or urgency level becomes
  unreachable again.

### False positives

Freezing was scored as absolute stillness, so a calm user crossed the 0.65
FROZEN gate on ~25% of ticks and wrote a spurious `human_crisis` row each time.
It is now the drop below the user's own baseline, gated on arousal. Measured
FROZEN false-positive rate on 1,000 calm ticks: **0.0%**.

### Failsafe

Health flags were hardcoded `True`, so `is_degraded` was always False and
`_activate_failsafe` was dead code. They now derive from per-stage heartbeats,
latency budgets, a SQLite probe and the attached battery source. Failsafe
activates, is recorded in episodic memory, and clears after healthy cycles.
Battery is no longer fabricated: `battery_pct` is `null` with
`battery_source: "unavailable"` until a source is attached.

### Silent fabrication stopped

With `simulation_mode=False` every adapter raised nothing and returned
synthetic data; `_init_hardware` logged "Hardware connected" unconditionally.
All six adapters now raise `HardwareUnavailableError` and there is no fallback
to simulation.

### Wrong outputs fixed

- `voice_stress` reads `audio.voice_pitch_hz`, not `eye.pupil_diameter_mm`.
- Autobiographical embeddings use `blake2b`, not process-salted `hash()`.
- The relevance filter's reason reports the real risk delta, not `0.00`.
- Suppressed guidance is logged (NFR-08): ~98% of cycles previously vanished.
- `L5.close()` no longer queries the store after closing it.

### Dead code that changed behaviour

- `OBJECT_TEMPLATES` gained DARK, DUSTY and OBSTRUCTED entries; an unmapped
  scene now logs a warning instead of silently modelling a safe room.
- `BrevityFilter.TEMPLATES` — 18 curated messages nothing referenced — are now
  the messages the system speaks. `shorten()` uses both its parameters.
- `SensorFrame.user_command` is implemented as a real override, with
  `POST /override` in the REST API (NFR-05).
- `_freeze_history` is populated; `_prev_frame` removed.

### Repository

- Single version string in `maaa_config.VERSION` (was v0.3 in the README and
  v1.0 in the code).
- `config/default.yaml` is now read; `maaa_core` and `layers` share their risk
  bands, brevity limit and failsafe thresholds through it.
- Added `pyproject.toml` and `.gitignore`. `python examples/run_demo.py` and a
  bare `pytest` both work from a clean clone.
- `requirements.txt` declares numpy and flask, which were always imported.
- De-duplicated the 1 MB `MAAA.mp4` (identical copy in `web/` removed).
- Test count 5 → 101.

---

## v0.3

- Reframed the working paper as an architectural integration of MAAA v1.2.
- Added five-layer implementation mapping.
- Added functional and non-functional requirements.
- Added deterministic regulatory engine.
- Added continuity/failsafe engine.
- Added WAAA-inspired memory pruning pattern.
- Added CLI demo, tests and the HTML sensory demo.
