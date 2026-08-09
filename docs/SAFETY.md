# Safety Notes

The MAAA is a high-risk assistive architecture. It must remain advisory until
validated.

**Nothing in this repository has been validated.** All sensor input is
synthetic, no model has been trained or evaluated, and no figure in these
documents comes from a measurement on real hardware or a real person. This is
an architecture demonstrator.

## Safety constraints

- No generative LLM in the primary emergency loop.
- No medical or psychological diagnosis from stress signals.
- No raw video cloud upload by default.
- No automatic actuation of physical systems.
- Human override must always be available.
- Failsafe mode must be deterministic and locally available.

## How each constraint stands in the code

| Constraint | Status |
|---|---|
| No LLM in the emergency loop | Nothing in the repository calls a language model. `config/default.yaml` sets `llm_in_primary_emergency_loop: false` |
| No diagnosis | Layer 3 emits proxy scores, never a diagnosis. The docstrings say so explicitly |
| No raw video upload | No video exists and no code path makes a network call |
| No physical actuation | The output dispatcher prints. The production hooks are commented out |
| Human override | Implemented in both trees; see below |
| Deterministic local failsafe | Implemented in `layers/l5_continuity.py`, driven by heartbeats, latency, a store probe and the battery source. No network, no randomness |

## The limits of the human override

Override is implemented as:

- `maaa_core`: `SensorFrame.user_command` ∈ `mute` (latching) / `resume` /
  `stop` (single frame), read by `MAAAOrchestrator.process`.
- `layers`: `L4SymbioticRegulation.set_override()`, exposed as `POST /override`.

**A mute does not silence CRITICAL urgency, and does not suppress the
failsafe.** The user can switch off advice; they cannot switch off the
collision warning or the notice that the system itself is degraded.

This is a deliberate limit on "human override must always be available", and it
is a design position, not an oversight. State it to anyone who reviews this
system: the override is a volume control on guidance, not a kill switch on
danger reporting. A build where silence can be latched through a critical
hazard is not one this architecture should ship.

## Known unsafe-to-rely-on properties

Listed so no one mistakes them for solved:

- **Freeze and panic detection are proxies.** They fire on generated numbers.
  The measured false-positive rate (0% FROZEN on 1,000 calm synthetic ticks)
  describes the detector's noise floor, not its accuracy on a person.
- **Risk is heuristic.** Per-category constants and a proximity term. It is not
  a model of structural collapse.
- **Battery is not measured.** `battery_pct` is `null` unless a source is
  attached. `main_maaa.py` attaches one labelled `"simulated"`.
- **Latency figures are meaningless without target hardware.** The per-stage
  budget is enforced relative to a configured number, not a benchmark.
- **Recovery from failsafe is time-based**, not evidence-based: after
  `recovery_cycles` healthy cycles the failsafe clears. It does not verify that
  the underlying fault was repaired.
