# Microduck Training Recipes

This directory contains executable recipes and orchestration scripts for training Microduck RL policies locally or on cloud compute.

---

## The Duck Butt-Wiggle ("Preen Shake") Recipe

* **Task ID:** `Mjlab-ButtWiggle-Flat-MicroDuck`
* **Config Module:** `src/mjlab_microduck/tasks/microduck_butt_wiggle_env_cfg.py`
* **MDP Functions:** `src/mjlab_microduck/tasks/mdp.py` (`ButtWigglePhaseCommand`, `butt_wiggle_composite`, `butt_wiggle_lin_vel_penalty`)

---

### Option 1: Local GPU Training

Train directly on a machine equipped with a CUDA GPU:

```bash
# 1. First run a 5-iteration smoke test (64 envs) to verify config and dependencies
uv run train Mjlab-ButtWiggle-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 2. Launch production training (4,096 parallel environments, 1,200 iterations)
uv run train Mjlab-ButtWiggle-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1200 \
  --agent.save-interval 100 \
  --agent.run-name butt_wiggle_production \
  --agent.wandb-project ducky_butt_wiggle

# 3. Export trained checkpoint to ONNX with baked-in normalizer
uv run -m mjlab_microduck.export Mjlab-ButtWiggle-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/butt_wiggle/<run_name>/model_1199.pt \
  --onnx-file policies/butt_wiggle_policy.onnx \
  --video True --video-length 600
```

---

### Option 2: Cloud GPU Training via Modal

Modal executes the full training pipeline in an isolated cloud container on NVIDIA L4 GPU, automatically exports the ONNX policy, renders a 12-second off-screen evaluation video, and saves checkpoints to persistent Modal Volumes.

```bash
# 1. Verify active profile and credit balance (Zero Out-of-Pocket Invariant)
modal billing summary

# 2. Run the production orchestrator
python recipes/run_production.py
```

#### Modal Recipe Components:
* `recipes/modal_train_butt_wiggle.py`:
  * Sets up Debian-slim container with MuJoCo Warp, EGL headless rendering, and `uv` virtualenv.
  * Implements `train_production` with 4,096 parallel environments on NVIDIA L4 GPU.
  * Exports ONNX model with baked-in `EmpiricalNormalization`.
  * Renders off-screen close-up `.mp4` evaluation video (600 steps @ 50 Hz).
* `recipes/run_production.py`:
  * Connects to Modal app, triggers remote execution, and streams artifacts back locally.

---

### Key Hyperparameters & Formulation

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Control Frequency** | 50 Hz (20 ms) | Matches hardware `robotd` tick |
| **PPO Rollout Steps** | 24 steps/env | 98,304 transitions per iteration |
| **Total Iterations** | 1,200 | ~118M physical transitions |
| **Goal Reward** | Multiplicative ($R_{\text{upright}} \times R_{\text{gaze}} \times R_{\text{wiggle}}$, weight 5.0) | Zero compromise basin: standing still earns strictly 0.000 |
| **Target Frequency** | 2.5 – 3.5 Hz | Commanded via `twist` command slot |
| **Drift Penalties** | Linear velocity ($-1.0 \|v\|^2$), Feet air ($-0.5$) | Prevents translation and slipping |
| **Action Rate Schedule** | $-0.0002 \to -0.0005 \to -0.0010$ | Staged curriculum prevents servo chatter without trapping exploration |
