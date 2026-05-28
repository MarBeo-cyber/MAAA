# 🧠 MAAA — Metacognitive Autopoietic Adaptive Agent

> **Sistema AI incarnato per la stabilizzazione cognitiva in emergenza — architettura a 5 layer con regulatory engine, human state monitoring e memoria autobiografica episodica.**

[![CI](https://github.com/YOUR_ORG/maaa/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/maaa/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Predecessor: WAAA](https://img.shields.io/badge/Predecessor-WAAA-orange)](https://github.com/YOUR_ORG/waaa)

---

## Cos'è il MAAA?

Il MAAA è la seconda generazione di agenti autopoietici, successore del [WAAA](https://github.com/YOUR_ORG/waaa). Mentre il WAAA monitora e preserva **la propria capacità percettiva**, il MAAA condivide il medesimo spazio percettivo dell'utente tramite dispositivi AR indossabili, creando una forma di **shared embodiment uomo-macchina**.

**Principio guida:** non massimizzare l'informazione trasmessa, ma **minimizzare l'entropia cognitiva**.

> *"Se il WAAA rappresenta il primo vagito di un agente autopoietico, il MAAA ne rappresenta la naturale evoluzione — la lallazione, i primi suoni (mmmmm, mma, maaa)."*

---

## Architettura a 5 Layer

```
┌──────────────────────────────────────────────────────────────────┐
│  L5  Autopoietic Continuity   Self-monitoring · Memoria · Failsafe│
├──────────────────────────────────────────────────────────────────┤
│  L4  Symbiotic Regulation     Regulatory Engine · 4 Filtri       │
├──────────────────────────────────────────────────────────────────┤
│  L3  Human State Monitoring   Stress · Panico · Freeze · Overload│
├──────────────────────────────────────────────────────────────────┤
│  L2  Situational Cognition    Scene Graph · Risk Engine · SLAM   │
├──────────────────────────────────────────────────────────────────┤
│  L1  Embodied Perception      AR · IMU · GPS · EyeTracker · Mic  │
└──────────────────────────────────────────────────────────────────┘
```

### I 4 Filtri del Regulatory Engine (L4)

| Filtro | Funzione |
|--------|---------|
| **Rilevanza** | Solo informazioni che cambiano il piano d'azione |
| **Timing** | Output solo quando l'utente è in grado di recepirlo |
| **Brevità** | Massimo 7–9 parole, sintassi imperativa semplice |
| **Urgenza** | Escalation del tono solo se il rischio aumenta (anti-alarm-fatigue) |

### Memoria a 3 Livelli

| Livello | Tipo | Retention | Utilizzo |
|---------|------|-----------|---------|
| Working Memory | Contesto 60s | Volatile / RAM | Decisioni real-time |
| Episodica | Sessione corrente | SQLite on-device | Coerenza narrativa |
| Autobiografica | Profilo utente | VectorDB (numpy/Qdrant) | Personalizzazione |

---

## Struttura del Progetto

```
maaa/
├── main_maaa.py              # Entry point (demo / server / both)
├── core/
│   └── maaa_agent.py         # Pipeline orchestrator — 8 step real-time loop
├── layers/
│   ├── l1_perception.py      # AR glasses · IMU · GPS · EyeTracker · Mic
│   ├── l2_cognition.py       # Scene graph · Object detection · Risk engine
│   ├── l3_human_state.py     # Stress · Panic · Freeze · Overload detection
│   ├── l4_regulation.py      # Regulatory engine — 4 filters
│   └── l5_continuity.py      # Memory · Self-monitoring · Failsafe
├── api/
│   └── maaa_api.py           # REST API — Flask :5002
└── tests/
    └── test_layers.py        # Test suite (pytest)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_ORG/maaa.git
cd maaa

# Install (core dependencies only: Flask + numpy)
pip install -r requirements.txt

# Demo — 4 fasi scenario terremoto (55 secondi)
python main_maaa.py demo

# REST API
python main_maaa.py server

# Server + Demo simultanei
python main_maaa.py both
```

---

## Demo — Scenario Emergenza Terremoto

```
FASE 1 — AMBIENTE NORMALE (10s)
  Rischio: LOW (0.14) | Umano: calm | receptivity=0.97
  ○ [SUPPRESSO] relevance_filter  ← il filtro rilevanza non produce output inutili

FASE 2 — FUMO E STRESS (15s)
  Rischio: HIGH (0.74) | Umano: alert | receptivity=0.83
  ⚡ [ELEVATED] "Uscita a 88°, 5 metri. Muoviti."
  AR: Uscita → (orange) → 88°  |  ⚡ Haptic: double

FASE 3 — CROLLO STRUTTURALE — PANICO (20s)
  Rischio: CRITICAL (0.91) | Umano: panicking | receptivity=0.20
  🔴 [CRITICAL] "PERICOLO IMMINENTE. Fermati."
  ○ [SUPPRESSO] panic_peak_receptivity_zero  ← timing filter protegge dal flooding

FASE 4 — RECOVERY (10s)
  Rischio: LOW (0.12) | Umano: calm | receptivity=0.97
  ▶ [NORMAL] "Ambiente stabile. Continua a monitorare."
```

---

## REST API

```bash
# Avvia server (background tick loop automatico a 10 Hz)
python main_maaa.py server

# Stato completo
curl http://localhost:5002/status

# Stato cognitivo utente
curl http://localhost:5002/human

# Mappa del rischio
curl http://localhost:5002/risk

# Ultimo output guidance
curl http://localhost:5002/guidance

# Memoria working (ultimi 60s)
curl http://localhost:5002/memory/working

# Recall autobiografica — situazioni simili a quella attuale
curl http://localhost:5002/memory/recall

# Inietta scenario di emergenza
curl -X POST http://localhost:5002/scenario/collapsed
curl -X POST http://localhost:5002/scenario/panic
curl -X POST http://localhost:5002/scenario/normal

# Esegui 50 tick manualmente
curl -X POST http://localhost:5002/tick/n/50
```

---

## Hardware Target (produzione)

| Componente | Tecnologia |
|------------|-----------|
| Occhiali AR | Ray-Ban Meta Smart Glasses |
| Depth Camera | Intel RealSense D435 / ToF |
| IMU | Integrato occhiali + wristband |
| Eye Tracker | Tobii / integrato AR |
| Edge compute | NVIDIA Jetson Orin / Snapdragon XR |
| Connettività | 5G + WiFi 6E + BLE |

---

## Differenze da WAAA

| | WAAA | MAAA |
|--|------|------|
| **Soggetto** | Il sistema stesso | Sistema + utente umano |
| **Obiettivo** | Preservare capacità percettiva propria | Preservare cognizione condivisa |
| **Output** | Dati interni / REST API | Voce + AR overlay + haptic |
| **Memoria** | VectorBiography (eventi sistema) | 3 livelli (working/episodica/autobio) |
| **Sensori** | Webcam + sensori digitali | AR + IMU + GPS + eye + mic + depth |
| **Regulatory** | Goal switching (RandomForest) | 4-filter Regulatory Engine |

---

## Contribuire

Priority areas:
- [ ] Integrazione YOLOv9 reale per object detection
- [ ] Integrazione ORB-SLAM3 per localizzazione
- [ ] OpenFace 2 per analisi micro-espressioni
- [ ] LLM-based regulatory engine (Claude API / LLaMA locale)
- [ ] Qdrant VectorDB per autobiographical memory scalabile
- [ ] Multi-agent mesh: coordinamento MAAA ↔ MAAA in scenario allargato
- [ ] Hardware HAL per Ray-Ban Meta / Snapdragon XR

---

## Riferimenti Teorici

| Teoria | Autori | Applicazione MAAA |
|--------|--------|------------------|
| Embodied AI | Varela, Maturana, Brooks | Shared embodiment tramite AR |
| Extended Mind | Clark & Chalmers (1998) | MAAA come organo metacognitivo esterno |
| Autopoiesi | Varela & Maturana (1972) | Autopoiesi distribuita uomo-macchina |
| Cognitive Load Theory | Sweller (1988) | Regulatory Engine — riduzione entropia |
| Human-AI Teaming | Klein, Woods, DARPA XAI | Co-cognizione adattiva |

---

## Licenza

MIT — vedi [LICENSE](LICENSE).

*Marco G. Beozzi — Sviluppato in collaborazione con Claude (Anthropic)*
