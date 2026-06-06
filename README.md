[![AURA Framework](https://img.shields.io/badge/AURA-Level%202%20%7C%20MAAA-1F3864)](https://github.com/MarBeo-cyber/AURA)

# MAAA — Metacognitive Autopoietic Adaptive Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

> *Il MAAA non informa — stabilizza e guida l'attenzione umana.*

Il MAAA è la seconda generazione di agenti autopoietici. Mentre il WAAA monitora l'ambiente attraverso sensori e preserva la propria coerenza interna, il MAAA condivide lo spazio percettivo dell'utente tramite dispositivi AR indossabili, creando una forma di **shared embodiment** uomo-macchina.

Progettato per scenari ad alto rischio — emergenza sismica, disaster recovery, supporto decisionale in condizioni critiche — dove il tempo decisionale collassa e il sovraccarico cognitivo è massimo.

---

## Conceptual Genealogy — Artificial Ontogenesis

| Agent | Core Function | Biological Analogy |
|---|---|---|
| WAAA | Weak autopoietic perception | Sensory reflex calibration |
| **MAAA** | **Metacognitive embodied cognition in emergency** | **Acute stress response and stabilisation** |
| PAAA | Personal neurofunctional continuity | Homeostasis / immune surveillance |
| SAAA | Sapient learning consolidation | Myelination / synaptic plasticity |

*The WAAA → MAAA → PAAA → SAAA progression constitutes an artificial ontogenesis: development by stages analogous to biological cognitive maturation.*

**Repositories:**
- WAAA: https://github.com/MarBeo-cyber/waaa
- MAAA: https://github.com/MarBeo-cyber/MAAA  
- PAAA: https://github.com/MarBeo-cyber/PAAA
- SAAA: https://github.com/MarBeo-cyber/SAAA

---

## Five-Layer Architecture

```
L1  Embodied Perception      ← AR glasses (video, audio, depth), IMU, eye tracking
L2  Situational Cognition    ← scene graph, object detection, risk estimation
L3  Human State Monitoring   ← stress, cognitive overload, freezing detection
L4  Symbiotic Regulation     ← adaptive guidance, regulatory engine, output timing
L5  Autopoietic Continuity   ← self-monitoring, episodic memory, system integrity
```

---

## Primary Use Case: Earthquake Emergency

In a seismic emergency:
- Decision time collapses
- Cognitive overload is maximal
- Panic degrades reasoning
- Spatial orientation is compromised

The MAAA intervenes as an **incarnated cognitive stabilisation system**:
- perceives the environment through AR glasses
- builds a dynamic causal risk map of the scene
- monitors the user's cognitive state
- delivers calibrated, essential guidance (max 7–9 words, imperative syntax)

Example output: *"Non guardare a sinistra. Procedi avanti 4 metri. Poi gira a destra."*

---

## Cognitive Entropy Reduction

The MAAA does not maximise information transmitted. It **minimises cognitive chaos**.

Every output passes four filters before delivery:
1. **Relevance filter** — only information that changes the action plan
2. **Timing filter** — output only when the user can receive it (cognitive state monitored)
3. **Brevity filter** — max 7–9 words, simple imperative syntax
4. **Urgency filter** — escalation only if risk state increases

---

## Quick Start

```bash
git clone https://github.com/MarBeo-cyber/MAAA.git
cd MAAA
pip install -r requirements.txt
pip install -e .
python main_maaa.py
```

---

## Integration with PAAA

When the MAAA is active in an emergency scenario, it can access (with prior user consent) the PAAA longitudinal neurofunctional baseline to better calibrate real-time cognitive state monitoring — distinguishing genuine stress-induced degradation from the user's individual variability.

---

## Project Structure

```
MAAA/
├── core/
│   └── maaa_agent.py       Main orchestrator (8-step pipeline, <200ms target)
├── layers/
│   ├── l1_perception.py    Embodied Perception
│   ├── l2_cognition.py     Situational Cognition + Risk Engine
│   ├── l3_human_state.py   Human State Monitoring
│   ├── l4_regulation.py    Symbiotic Regulation + Regulatory Engine
│   └── l5_continuity.py    Autopoietic Continuity
├── memory/                 Episodic autobiographical memory
├── sensors/                AR glasses, IMU, audio input
└── api/                    REST API for multi-agent coordination
```

---

## Scientific Positioning

| Area | References | MAAA Contribution |
|---|---|---|
| Embodied AI | Varela, Maturana, Brooks | First implementation of shared embodiment human-AI via AR |
| Extended Mind | Clark & Chalmers (1998) | MAAA as physical external metacognitive organ |
| Autopoiesis | Varela & Maturana (1972) | Distributed and symbiotic human-machine autopoiesis |
| Cognitive Load Theory | Sweller (1988) | First AI system designed for active cognitive entropy reduction |
| Human-AI Teaming | Klein, Woods, DARPA XAI | Adaptive co-cognition in high-risk scenarios |

---

## Citation

```bibtex
@software{maaa2025,
  title  = {MAAA: Metacognitive Autopoietic Adaptive Agent},
  author = {Beozzi, Marco Giuseppe},
  year   = {2025},
  url    = {https://github.com/MarBeo-cyber/MAAA},
  note   = {Part of the WAAA → MAAA → PAAA → SAAA artificial ontogenesis}
}
```
