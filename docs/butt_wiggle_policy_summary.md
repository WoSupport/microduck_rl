# Policy `butt_wiggle` — The Duck Butt-Wiggle ("Preen Shake")

**Goal**: The Microduck stands planted on its webbed feet and performs high-frequency lateral hip-roll oscillation (~2.5–3.5 Hz) while keeping its head stabilized and gaze locked forward (vestibulo-ocular reflex).

- **Task ID**: `Mjlab-ButtWiggle-Flat-MicroDuck`
- **Config Module**: `src/mjlab_microduck/tasks/microduck_butt_wiggle_env_cfg.py`
- **Deployable ONNX**: `policies/butt_wiggle_policy.onnx` (776 KB)
- **Observation Contract**: Strict 61D (`actor` obs: 48D proprioception + 13D commands: `twist(3)`, `head_pose(4)`, `body_pose(6)`). Hot-swappable at runtime.
- **Actuator Model**: BAM M6 voltage-controlled Dynamixel XL330 with battery sag (6.5V–8.2V) and gear backlash.

---

## Kinematics & Posture Targets

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Nominal Standing Height** | `STAND_Z = 0.126` m | Foot sole to trunk base |
| **Lateral Hip Roll Amplitude** | $\pm 0.22$ rad ($\approx \pm 12.6^\circ$) | Hip roll servos (`left_hip_roll`, `right_hip_roll`) |
| **Target Frequency** | 2.5 – 3.5 Hz (default 3.0 Hz) | Commanded dynamically via phase reference |
| **Gaze Pitch Envelope** | Level ($\le 5^\circ$ error) | Head/jaw link in world coordinates |

---

## Reward Formulation & Mathematical Invariants

### 1. Multiplicative Core Goal (The Anti-Freeze Invariant)
To prevent the optimizer from exploiting stationary standing postures:
$$R_{\text{composite}} = R_{\text{upright}} \times R_{\text{gaze\_lock}} \times R_{\text{wiggle}}$$

Where $R_{\text{wiggle}}$ is formulated via Pythagorean linear projection:
$$r_{\text{pos}} = \frac{dq_l \sin(\phi) + dq_r \sin(\phi)}{2 A}, \quad r_{\text{vel}} = \frac{\dot{q}_l \cos(\phi) + \dot{q}_r \cos(\phi)}{4\pi f A}$$
$$R_{\text{wiggle}} = \text{clamp}(r_{\text{pos}} + 0.5 r_{\text{vel}} + 0.5 r_\omega, 0, 1)$$

* **Full 3 Hz Oscillation:** Achieves $\approx 1.0$ continuously throughout the periodic cycle.
* **Stationary Standing:** $r_{\text{pos}} \approx 0, r_{\text{vel}} \approx 0 \implies R_{\text{composite}} \equiv \mathbf{0.0000}$! Standing still earns zero reward.

### 2. Drift Suppression & Foot Anchoring
* `butt_wiggle_lin_vel`: $-1.0 \|v_{\text{base}}\|^2$ penalizes translation.
* `butt_wiggle_feet_air`: $-0.5$ penalizes foot lifting.
* `foot_slip`: $-0.5$ penalizes tangential sliding.
* Removed disruptive environment perturbations (`push_robot` shocks and random CoM shifts).

### 3. Action Regularization Curriculum
Action rate penalty (`action_rate_l2`) follows a 3-stage schedule:
* Iterations 0 – 500: $-0.0002$ (allows unconstrained exploration of 3 Hz roll).
* Iterations 500 – 1000: $-0.0005$ (consolidates limit cycle).
* Iterations 1000 – 1200: $-0.0010$ (eliminates servo chatter and torque spikes).

---

## Verification & QA Results

* **Simulation Rollout:** 600 continuous steps @ 50 Hz (12.0s) with 0 resets, 0 falls.
* **Vision Model Audit:** Evaluated by independent multimodal vision subagent (`Model="pro"`).
* **Verdict:** **THUMBS UP (PASS)**
  * Vigorous alternating roll oscillation observed.
  * Head locked level and forward (vestibulo-ocular reflex).
  * Feet planted firmly without translation drift.
