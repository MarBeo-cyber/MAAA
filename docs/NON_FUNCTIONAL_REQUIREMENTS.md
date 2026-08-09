# Non-Functional Requirements

All sensor input is synthetic, so most of these are **design intents, not
measurements**. The previous version of this document carried target figures
(latency budgets, availability, endurance, false-alarm rates) and a
verification column naming methods — "benchmark su hardware target", "test
fault injection", "test batteria", "Disponibilità >99.5%", "Autonomia >4 ore" —
none of which were ever run. Those figures are removed. What remains is either
enforced by code or explicitly marked unverified.

| ID | Requirement | Status |
|---|---|---|
| NFR-01 | Tier-0 voice/AR/haptic guidance latency | **Unverified.** No hardware, no benchmark. The budget lives in `config/default.yaml` (`latency.tier0_guidance_ms`) and is not enforced anywhere |
| NFR-02 | Tier-1 cognitive loop latency | **Partially enforced.** Per-stage latency is measured every cycle and a stage over `failsafe.stage_latency_max_ms` marks itself not-ok, which degrades health. The absolute figure is meaningless without target hardware |
| NFR-03 | Offline critical operation | **By construction.** No code path makes a network call. Not otherwise verified |
| NFR-04 | Raw video cloud upload forbidden by default | **By construction.** No video exists and nothing is uploaded. `config/default.yaml` sets `raw_video_cloud_upload: false` |
| NFR-05 | Human override always available | **Implemented and tested.** `SensorFrame.user_command` in `maaa_core`; `set_override()` and `POST /override` in `layers`. Scope limit: override cannot silence CRITICAL urgency (see docs/SAFETY.md) |
| NFR-06 | Output brevity, 7–9 words in emergency mode | **Enforced and tested.** Limit from `config/default.yaml` (`max_words: 9`); `BrevityFilter.shorten()` falls back to a curated template rather than truncating |
| NFR-07 | Graceful degradation | **Implemented and tested.** Health derives from per-stage heartbeats, latency, a SQLite probe and the battery source; failsafe activates on degradation and clears after `recovery_cycles` healthy cycles |
| NFR-08 | Auditability — every guidance event logged | **Implemented and tested.** Both delivered and suppressed guidance are written to working and episodic memory, with the filter log and the suppression reason. `tests/test_l5_continuity.py` asserts `logged == total_cycles` |

## False alarms

The working paper set a target of <5% false alarms. That target was never
measured; when it was, the FROZEN false-positive rate on a calm NORMAL
scenario was **~25%** (246–281 of 1,000 ticks across runs), each writing a
`human_crisis` row to SQLite.

After rebasing the freeze detector on deviation from the user's own baseline,
the measured rate is **0.0%** FROZEN over 1,000 calm ticks, with ~99% of frames
classified CALM. `tests/test_l3_human_state.py` fails if it rises above 5%.

This is a measurement over synthetic input. It says the detector no longer
fires on its own noise floor; it says nothing about performance on a human.

## Verification status

| Claim | Verified? | How |
|---|---|---|
| Every cognitive state reachable | Yes | `tests/test_reachability.py` |
| Every urgency level AMBIENT…CRITICAL reachable | Yes | `tests/test_reachability.py` |
| Every risk band LOW…CRITICAL reachable | Yes | `tests/test_reachability.py` |
| FROZEN false-positive rate < 5% on calm input | Yes | `tests/test_l3_human_state.py`, 1,000 ticks |
| Failsafe activates on layer failure / battery collapse | Yes | `tests/test_l5_continuity.py` |
| Every guidance cycle logged | Yes | `tests/test_l5_continuity.py` |
| No synthetic data returned in hardware mode | Yes | `tests/test_l1_hardware_mode.py` |
| REST API contract | Yes | `tests/test_api.py` |
| Latency on target hardware | **No** | No target hardware exists |
| Availability, endurance, battery life | **No** | No hardware, no battery driver |
| Human-state accuracy | **No** | Would require human subjects and ethics approval |
