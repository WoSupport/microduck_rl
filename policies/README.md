# Microduck Deployable Policies

This directory stores exported, verified ONNX policies ready for deployment on the physical Microduck biped robot or rehearsal in MuJoCo simulation.

All policies in this directory follow the **Strict 61D Observation Contract**:
$$\text{Obs (61D)} = [\underbrace{48\text{D base proprioception}}_{\text{joint pos/vel, IMU, last actions}}, \underbrace{13\text{D command block}}_{\text{twist}(3), \text{head\_pose}(4), \text{body\_pose}(6)}]$$
Observation normalization is baked directly into the ONNX computational graph (`EmpiricalNormalization`), so policies accept raw observations without external runtime scaling.

---

## Policies

### 1. `butt_wiggle_policy.onnx` — The Duck Butt-Wiggle ("Preen Shake")
* **File:** `butt_wiggle_policy.onnx` (776 KB)
* **Behavior:** High-frequency lateral hip-roll oscillation (~3 Hz) with locked-forward head gaze (vestibulo-ocular reflex) and firmly planted webbed feet without sliding or drifting.
* **Architecture:** MLP Policy `[61] -> 512 -> ELU -> 256 -> ELU -> 128 -> ELU -> [14]`
* **Control Frequency:** 50 Hz (20 ms tick), identical to hardware `robotd`.
* **Actuators:** 14× Robotis Dynamixel XL330 active servos (5 left leg, 4 neck/head, 5 right leg).
* **QA Validation:** Evaluated and approved (**THUMBS UP / PASS**) by independent multimodal vision model audit; 0 resets/falls over 12s/600-step continuous simulation.

---

## How to Deploy on Physical Hardware

Deploy onto a real Microduck biped robot running `robotd` via `robotctl`:

```bash
# Add the policy to the robot's policy registry
sudo robotctl policy add butt-wiggle policies/butt_wiggle_policy.onnx

# Trigger the Butt-Wiggle gesture
robotctl robot do butt-wiggle
```

---

## How to Rehearse in Simulation

Rehearse policy execution on CPU or GPU with the BAM M6 voltage-controlled actuator model:

```bash
uv run scripts/infer_policy.py --walking policies/butt_wiggle_policy.onnx --new-cmd-obs
```
