# MAAA v0.3 Architecture

## Core axiom

The MAAA is not a simulation of life. It simulates functions that make living systems capable of persisting, learning and cooperating.

## Five layers

| Layer | Name | Role |
|---|---|---|
| L1 | Embodied Perception | First-person sensing through AR/wearable devices |
| L2 | Situational Cognition | Scene graph, object detection, risk estimation, causal state |
| L3 | Human State Monitoring | Stress, overload, panic, attentional collapse |
| L4 | Symbiotic Regulation | Output filtering, timing, brevity, AR/audio/haptic guidance |
| L5 | Autopoietic Continuity | Failsafe, memory, recovery, continuity of human-system relation |

## Runtime loop

```text
SensorFrame
 -> EmbodiedPerceptionAdapter
 -> SceneGraphBuilder
 -> RiskEstimationEngine
 -> HumanStateEstimator
 -> RegulatoryEngine
 -> AutopoieticContinuityEngine
 -> GuidancePlan
```

## Key design rule

LLM reasoning is not allowed in the Tier-0 emergency loop. The emergency loop uses deterministic, local, prevalidated functions.
