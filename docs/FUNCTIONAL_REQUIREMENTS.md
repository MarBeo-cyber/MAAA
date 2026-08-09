# Functional Requirements

All sensor input is synthetic. The "Status" column says what the code in this
repository actually does against each requirement. "Simulated" means the
behaviour is implemented and exercised, but over generated data.

| ID | Requirement | Status | Where |
|---|---|---|---|
| FR-01 | Acquire multimodal wearable inputs from AR, IMU, audio and optional eye tracking | **Simulated.** Six adapters generate synthetic frames. No driver exists; `simulation_mode=False` raises `HardwareUnavailableError` rather than returning synthetic data | `layers/l1_perception.py` |
| FR-02 | Build a scene graph of detected hazards, paths and objects | **Simulated.** Objects come from `OBJECT_TEMPLATES`, one entry per scene condition, with noise. No object detector runs | `layers/l2_cognition.py` |
| FR-03 | Estimate structural, environmental and walkability risks | **Heuristic.** Fixed per-category base risks plus proximity and environment terms; aggregate is `0.45·env_degradation + 0.55·proximity-weighted worst hazard`. Not a learned model | `layers/l2_cognition.py` |
| FR-04 | Estimate human cognitive entropy from stress/load proxies | **Proxy only.** Weighted sums over synthetic scalars. Never validated against a human subject | `layers/l3_human_state.py`, `maaa_core/human_state.py` |
| FR-05 | Produce short multimodal guidance plans | **Implemented.** Voice message ≤ 9 words, AR overlay, haptic pattern, per-channel activation | `layers/l4_regulation.py` |
| FR-06 | Apply relevance, timing, brevity and urgency filters | **Implemented.** All four run every cycle; the filter log records which fired and why | `layers/l4_regulation.py` |
| FR-07 | Preserve episodic continuity and avoid contradictory guidance | **Implemented.** Working memory (60 s), SQLite episodic store, cross-session vector recall; the relevance filter suppresses repeats | `layers/l5_continuity.py` |
| FR-08 | Enter failsafe mode under sensor degradation, battery collapse or excessive latency | **Implemented.** Health flags derive from per-stage heartbeats, latency budgets, a SQLite probe and the attached battery source. Failsafe activates and clears, and both are logged | `layers/l5_continuity.py`, `core/maaa_agent.py` |
| FR-09 | Operate offline-first for critical functions | **By construction.** Nothing makes a network call; the REST API is an optional local server. Not otherwise verified | — |
| FR-10 | Support mesh/multi-agent extension in later releases | **Not implemented.** No multi-agent code exists | — |

## Human override

`docs/SAFETY.md` and NFR-05 require human override to always be available. It
is implemented in both trees:

- `maaa_core`: `SensorFrame.user_command` accepts `mute` (latching), `resume`
  and `stop` (one frame). Read by `MAAAOrchestrator.process`.
- `layers`: `L4SymbioticRegulation.set_override()` and `POST /override`.

Neither can silence CRITICAL urgency. That limit is deliberate and is stated in
docs/SAFETY.md.

## Not implemented

Named in earlier documents, absent from the code:

- Object detection, depth estimation, SLAM, micro-expression analysis
- Any trained model of any kind
- Physiological sensing (HRV, GSR, rPPG)
- Mesh / multi-agent operation
- Any hardware integration
