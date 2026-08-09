# MAAA — Metacognitive Autopoietic Adaptive Agent

**Every input this system consumes is synthetic.** There are no sensors, no
device drivers, no trained models and no real-world data anywhere in this
repository. It is an architecture demonstrator: an executable statement of a
design, not a working assistive device.

> Not a simulation of life, but a simulation of the functions that make life
> capable of persisting, learning and cooperating.

Version 0.4.0. (Earlier trees called themselves v0.3 in the README and v1.0 in
the code; there is now one version string, in `maaa_config.VERSION`.)

## Scope

This prototype is a software architecture demonstrator. It does not control
real safety equipment, does not diagnose human states, and must not be used
for real emergency guidance without validation and certification.

To put that more concretely, because the previous README did not:

- **No perception.** Objects come from a lookup table keyed by scene condition
  (`layers/l2_cognition.py`, `OBJECT_TEMPLATES`) with gaussian noise on
  distance and bearing. No image is ever processed.
- **No sensors.** `layers/l1_perception.py` generates every IMU, depth, gaze,
  audio and GPS value from `random.gauss`. With `simulation_mode=False` every
  adapter raises `HardwareUnavailableError` — it will not hand you synthetic
  data dressed up as a reading.
- **No human-state measurement.** Layer 3 computes weighted sums over those
  synthetic scalars. It has never been validated against a human subject.
- **No SLAM, no depth estimation.** The SLAM pose is a random walk.
- **No performance figures.** Nothing in this repository has been benchmarked
  on target hardware, fault-injected, or measured for availability or
  endurance. Where the requirement documents used to state numbers, they now
  state that the number is unverified.

## What is here

Two implementations of the same five-layer idea. They are both real, both
imported by different entry points, and they now share their configuration.

| | `layers/` + `core/` + `api/` + `main_maaa.py` | `maaa_core/` |
|---|---|---|
| Role | **Reference runtime** | **Minimal reference core** |
| Size | ~1,950 logical LOC | ~260 logical LOC |
| Loop | 5 layers, 10 Hz, snapshots | one function call |
| Memory | working / episodic (SQLite) / autobiographical (numpy cosine) | one pruned event list |
| I/O | Flask REST API, 13 endpoints | none |
| Entry point | `python main_maaa.py demo` | `python examples/run_demo.py` |

`maaa_core/` is the smallest complete statement of the loop: one class per
layer, no threads, no database, readable end to end in ten minutes. `layers/`
is where the behaviour actually lives. Both read `config/default.yaml` through
`maaa_config`, so the risk bands, the 9-word brevity limit and the failsafe
thresholds are defined once.

### The five layers

| Layer | Name | What it does here |
|---|---|---|
| L1 | Embodied Perception | Generates synthetic frames for six scene conditions |
| L2 | Situational Cognition | Table-driven scene graph; heuristic risk score per object; aggregate risk |
| L3 | Human State Monitoring | Cognitive-entropy proxy: weighted sums over synthetic scalars |
| L4 | Symbiotic Regulation | Four filters (relevance, timing, brevity, urgency) over a curated message set |
| L5 | Autopoietic Continuity | Three-level memory, heartbeat-driven health, failsafe and recovery |

The two components the previous README named "Risk Estimation Engine" and
"Human State Monitoring" are described honestly in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): a **threshold classifier over
caller-supplied hazard scores** and a **cognitive-entropy proxy over simulated
scalars**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python examples/run_demo.py        # minimal core, one frame
python main_maaa.py demo           # full runtime, four-phase scenario
python main_maaa.py server         # REST API on :5002
pytest -q                          # 101 tests
```

Both entry points work from a clean clone with no install step. `pip install -e .`
also works if you prefer the package on your path.

### REST API

`python main_maaa.py server` serves 13 endpoints on `:5002`, including a human
override channel. Full contract in [docs/API_SPEC.md](docs/API_SPEC.md).

```bash
curl -s localhost:5002/status | jq '.agent, .data_source, .last.health'
curl -s -XPOST localhost:5002/scenario/collapsed
curl -s -XPOST localhost:5002/override -H 'content-type: application/json' \
     -d '{"command":"mute"}'
```

## Calibration, and why it matters

An audit of the previous tree found that the entire CRITICAL emergency
response was unreachable: over 2,000 ticks at maximum injected severity the
aggregate risk peaked at **0.744** against a 0.80 gate and panic at **0.681**
against 0.85. `UrgencyLevel.CRITICAL`, the `FERMATI.` templates, the SOS
haptic and the red AR overlay could never execute. A demo phase titled
"CROLLO STRUTTURALE — PANICO" printed `[ELEVATED]`. Meanwhile a calm user
tripped the FROZEN gate on ~25% of ticks.

Three normalisation constants were the cause — motor agitation measured
`|a| - 9.8` (dominated by gravity), saccade velocity was divided by a
reference the generator never approaches, and freezing was scored as absolute
stillness rather than a drop below the user's own baseline. They are fixed,
and `tests/test_reachability.py` now fails if any state or urgency level
becomes unreachable again. Measured ranges are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#calibration).

## Repository layout

```
maaa_config.py          single config + version source (reads config/default.yaml)
config/default.yaml     risk bands, gates, brevity limit, failsafe thresholds
layers/                 L1–L5 reference runtime
core/maaa_agent.py      8-step pipeline, per-stage heartbeats
api/maaa_api.py         Flask REST API
main_maaa.py            CLI: demo | server | both | tick <n>
maaa_core/             minimal reference core
examples/run_demo.py    minimal-core demo
web/                    AR guidance storyboard (a hand-timed animation, not a client)
docs/                   architecture, requirements, API spec, safety, references
tests/                  101 tests
```

`web/MAAA_AR_Guidance_Demo_v0_3.html` is a **storyboard**: a hand-timed
`setTimeout` animation with hardcoded strings. It does not call the Python and
is not driven by any pipeline output.

## Documents

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the layers, the formulas, the calibration
- [docs/FUNCTIONAL_REQUIREMENTS.md](docs/FUNCTIONAL_REQUIREMENTS.md) — FR-01…FR-10 with implementation status
- [docs/NON_FUNCTIONAL_REQUIREMENTS.md](docs/NON_FUNCTIONAL_REQUIREMENTS.md) — NFR-01…NFR-08, verified vs unverified
- [docs/API_SPEC.md](docs/API_SPEC.md) — the REST API that exists
- [docs/SAFETY.md](docs/SAFETY.md) — safety constraints and the limits of the override
- [docs/references.md](docs/references.md) — the literature the design draws on
