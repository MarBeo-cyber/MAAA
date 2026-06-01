# MAAA — Technical Architecture Reference

## Five-Layer Architecture

### L1 — Embodied Perception

Shared perception between agent and human via AR wearable:

| Sensor | Technology | Function |
|---|---|---|
| AR glasses | Ray-Ban Meta Smart Glasses | First-person video 30fps, audio 16kHz, microphone |
| Depth camera | Intel RealSense / ToF sensor | Depth estimation, obstacle detection |
| IMU | Accelerometer + gyroscope | Orientation, fall detection, stability |
| Eye tracker | Tobii / integrated AR | Attention focus, cognitive load proxy |
| Microphone | Directional array | User voice, ambient sounds, risk detection |
| Edge compute | NVIDIA Jetson / Snapdragon XR | Local inference <50ms |

---

### L2 — Situational Cognition

```
Video frame (30fps)
    ↓
Object detection (YOLOv9) + semantic segmentation (SAM)
    ↓
Scene graph update (Neo4j / custom GNN)
    ↓
Risk estimation: structural + environmental + traversability
    ↓
Dynamic causal risk map
```

Risk dimensions:
- **Structural:** wall/ceiling/floor/stair stability probability
- **Environmental:** smoke, gas, water, fire, obstacles
- **Traversability:** exit accessibility, corridor passability

---

### L3 — Human State Monitoring

Continuous estimation of the user's cognitive state:

| Signal | Method | Output |
|---|---|---|
| Voice analysis | Stress detection (pitch, rate, pauses) | Stress index [0–1] |
| Micro-expressions | OpenFace 2 / Affectiva | Anxiety, confusion, panic |
| Eye tracking | Fixation duration, saccade rate | Cognitive overload estimate |
| Behavioural | Freezing detection, indecision | Action capability estimate |
| IMU | Movement quality, tremor | Physical state |

---

### L4 — Symbiotic Regulation (Regulatory Engine)

Four filters applied to every output before delivery:

```
Candidate output
    → Relevance filter  (changes action plan? if NO: discard)
    → Timing filter     (user cognitively ready? if NO: queue)
    → Brevity filter    (compress to ≤9 words, imperative syntax)
    → Urgency filter    (escalate tone only if risk increases)
    → Output dispatch   (voice, AR overlay, haptic)
```

Output channels:
- **Voice:** primary channel; max 9 words; imperative; calm tone
- **AR overlay:** danger highlights, direction arrows, exit markers
- **Haptic:** vibration pattern for critical alerts

---

### L5 — Autopoietic Continuity

MAAA monitors its own operational integrity:

- Self-monitoring of each layer's health
- Episodic autobiographical memory (VectorDB: Weaviate/Qdrant)
- Graceful degradation: voice-only if AR fails
- Human override: any gesture/voice command immediately honoured
- Watchdog: automatic module restart on failure
- Audit log: every decision logged for post-event review

---

## Processing Pipeline

| Step | Phase | Target Latency |
|---|---|---|
| 1 | Sensor acquisition | — |
| 2 | Edge pre-processing | <20ms |
| 3 | Semantic perception (L2) | <50ms (**Tier 0**) |
| 4 | Risk estimation | <50ms (**Tier 0**) |
| 5 | Human state update (L3) | <200ms (Tier 1) |
| 6 | Regulatory engine (L4) | <200ms (Tier 1) |
| 7 | Output dispatch | <50ms (Tier 0 for voice/haptic) |
| 8 | Autopoietic check (L5) | <1s (Tier 2) |

---

## Multi-Agent Coordination

Multiple MAAA instances (worn by different operators) coordinate via peer-to-peer mesh:

- Partial environment map sharing (distributed SLAM fusion)
- Risk information sharing across areas
- Operator status broadcasting (active / in difficulty / disconnected)
- Cooperative routing to safety exits

---

## Technology Stack

**Vision:** YOLOv9, SAM, DINO, Depth Anything v2  
**SLAM:** ORB-SLAM3, OpenVINS  
**Scene Graph:** Neo4j / custom GNN  
**Human State:** OpenFace 2, Affectiva, custom voice stress  
**LLM Core:** LLaMA local / API LLM  
**Episodic Memory:** VectorDB (Weaviate/Qdrant), text-embedding-3-large  
**AR Rendering:** Unity XR / WebXR  
**Orchestration:** LangGraph, CrewAI  
**Connectivity:** 5G / WiFi 6E + BLE (offline-first)
