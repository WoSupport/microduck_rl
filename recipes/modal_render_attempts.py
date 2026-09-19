"""Modal runner to render honest, unmasked progress animations for all Microduck attempts.
- Exactly the same length for every attempt (600 frames = 12.00 seconds @ 50 Hz).
- auto_reset = False and terminations.clear(): if a policy fails or falls over, it stays down.
- Zero masking, zero auto-resets.
- Uniform close-up 3/4 camera framing, no debug visualizer arrows.
"""

import os
import glob
import time
import subprocess
import modal

app = modal.App("microduck-render-progress")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libgles2", "libglib2.0-0", "curl", "build-essential")
    .run_commands(
        "curl -fsSL https://astral.sh/uv/install.sh | sh",
    )
    .env({
        "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    })
    .add_local_dir(
        "/home/ubuntu/vibeduck/microduck_rl",
        remote_path="/root/microduck_rl",
        ignore=[".venv", "__pycache__", ".git"],
        copy=True,
    )
    .add_local_dir(
        "/home/ubuntu/vibeduck/all_checkpoints",
        remote_path="/root/checkpoints",
        copy=True,
    )
    .run_commands(
        "cd /root/microduck_rl && uv sync",
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=900,
)
def render_all_runs(video_length: int = 600) -> dict:
    """Render all 9 attempts with exact same duration and NO auto-resets."""
    import os
    import sys
    import glob
    import subprocess

    worker_script = f"""
import os
import sys
import glob
import time
import torch
import numpy as np
import imageio.v3 as iio
from dataclasses import asdict

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

runs = [
    {{"code": "v1", "pt": "/root/checkpoints/v1_model.pt", "push": False}},
    {{"code": "v2", "pt": "/root/checkpoints/v2_model.pt", "push": False}},
    {{"code": "v3", "pt": "/root/checkpoints/v3_model.pt", "push": False}},
    {{"code": "v4", "pt": "/root/checkpoints/v4_model.pt", "push": False}},
    {{"code": "v5", "pt": "/root/checkpoints/v5_model.pt", "push": False}},
    {{"code": "v6", "pt": "/root/checkpoints/v6_model.pt", "push": False}},
    {{"code": "v7", "pt": "/root/checkpoints/v7_model.pt", "push": True}},
    {{"code": "v8", "pt": "/root/checkpoints/v8_model.pt", "push": False}},
    {{"code": "v9", "pt": "/root/checkpoints/v9_model.pt", "push": False}},
]

out_dir = "/tmp/rendered_attempts"
os.makedirs(out_dir, exist_ok=True)
device = "cuda:0"

for run in runs:
    code = run["code"]
    pt_path = run["pt"]
    print(f"\\n==========================================")
    print(f"Rendering {{code}} from {{pt_path}} (length={video_length}, no-reset)")
    print(f"==========================================")

    task_id = "Mjlab-ButtWiggle-Flat-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    # 1 env, NO auto reset, NO terminations
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False
    env_cfg.terminations.clear()

    # Uniform Camera framing (close-up 3/4 view)
    env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.ASSET_BODY
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.body_name = "trunk_base"
    env_cfg.viewer.distance = 0.55
    env_cfg.viewer.elevation = -12.0
    env_cfg.viewer.azimuth = 145.0
    env_cfg.viewer.width = 640
    env_cfg.viewer.height = 480

    if run["push"]:
        print("  [v7] Enabling push perturbation events to reflect v7 training conditions...")
        from mjlab.managers import EventTermCfg
        from mjlab.tasks.velocity import mdp as vel_mdp
        env_cfg.events["push_robot"] = EventTermCfg(
            func=vel_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={{"velocity_range": {{"x": (-0.4, 0.4), "y": (-0.4, 0.4)}}}},
        )
    else:
        env_cfg.events.pop("push_robot", None)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env.manager_visualizers = {{}}
    env.update_visualizers = lambda visualizer: None

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(pt_path, map_location=device)
    policy = runner.get_inference_policy(device=device)

    wrapped_env.reset()

    frames = []
    frame0 = env.render()
    if frame0 is not None:
        frames.append(frame0)

    t0 = time.time()
    for step_idx in range({video_length} - 1):
        with torch.no_grad():
            obs = wrapped_env.get_observations()
            act = policy(obs)
            wrapped_env.step(act)
            f = env.render()
            if f is not None:
                frames.append(f)

    step_time = time.time() - t0
    print(f"  Rendered {{len(frames)}} frames in {{step_time:.2f}}s ({{len(frames)/step_time:.1f}} fps)")

    mp4_path = os.path.join(out_dir, f"{{code}}_honest_12s.mp4")
    iio.imwrite(mp4_path, frames, fps=50, codec="libx264")
    print(f"  Saved video: {{mp4_path}} ({{os.path.getsize(mp4_path)/1e6:.2f}} MB)")

    env.close()

print("ALL RENDERS COMPLETE")
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/microduck_rl/src"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    proc = subprocess.run(
        ["/root/microduck_rl/.venv/bin/python", "-c", worker_script],
        cwd="/root/microduck_rl",
        env=env,
        capture_output=True,
        text=True,
    )
    print("STDOUT:\n", proc.stdout)
    if proc.stderr:
        print("STDERR:\n", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Rendering failed with exit code {proc.returncode}")

    results = {}
    out_dir = "/tmp/rendered_attempts"
    for f in sorted(glob.glob(f"{out_dir}/*_honest_12s.mp4")):
        code = os.path.basename(f).split("_")[0]
        with open(f, "rb") as fh:
            results[code] = fh.read()

    return results
