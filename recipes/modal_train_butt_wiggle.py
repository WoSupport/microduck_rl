"""Modal runner for training Microduck Duck Butt-Wiggle ("Preen Shake") policy.

Requirements:
- Target task: Mjlab-ButtWiggle-Flat-MicroDuck
- Hardware: NVIDIA L4 (fallback A10G)
- Strict 61D observation contract
- Multiplicative composite reward for high-frequency hip-roll oscillation + gaze-locked head
- WandB logging to ducky_butt_wiggle
- ONNX export with baked-in normalizer
- Headless video rendering (.mp4) for visual inspection
- Save to Modal volume: microduck-models
"""

import glob
import os
import shutil
import subprocess
from datetime import datetime
import modal

app = modal.App("microduck-butt-wiggle")
volume = modal.Volume.from_name("microduck-models", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

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
    .run_commands(
        "cd /root/microduck_rl && uv sync",
    )
)

WANDB_UTILS_CONTENT = """# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    \"\"\"Summary writer for W&B.\"\"\"

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs=flush_secs)
        run_name = os.path.split(log_dir)[-1]
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config={"log_dir": log_dir},
            settings=wandb.Settings(),
        )
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        wandb.config.update({"train_cfg": train_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)
"""


def patch_rsl_rl_wandb():
    """Ensure rsl_rl has clean WandbSummaryWriter without start_method='thread'."""
    files = glob.glob("/root/microduck_rl/.venv/**/rsl_rl/utils/wandb_utils.py", recursive=True)
    for p in files:
        try:
            with open(p, "w") as f:
                f.write(WANDB_UTILS_CONTENT)
            print(f"[PATCH] Successfully wrote pristine WandbSummaryWriter to {p}")
        except Exception as e:
            print(f"[WARN] Failed to write {p}: {e}")



@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    secrets=[wandb_secret],
    volumes={"/models": volume},
)
def smoke_test():
    """5-iteration smoke test with 64 envs to verify training and export integrity."""
    print("=== STARTING 5-ITERATION SMOKE TEST (64 envs) ===")
    patch_rsl_rl_wandb()

    cmd = [
        "/root/microduck_rl/.venv/bin/train",
        "Mjlab-ButtWiggle-Flat-MicroDuck",
        "--env.scene.num-envs", "64",
        "--agent.max-iterations", "5",
        "--agent.save-interval", "5",
        "--agent.run-name", "smoke_test_butt_wiggle",
        "--agent.wandb-project", "ducky_butt_wiggle",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/microduck_rl/src"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    proc = subprocess.run(cmd, cwd="/root/microduck_rl", env=env, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Smoke test failed with exit code {proc.returncode}")
    print("=== SMOKE TEST TRAINING PASSED! ===")

    # Test export and video recording on smoke test checkpoint
    candidates = glob.glob("/root/microduck_rl/logs/rsl_rl/butt_wiggle/*smoke_test_butt_wiggle*")
    if candidates:
        log_dir = sorted(candidates)[-1]
        pts = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
        if pts:
            latest_pt = pts[-1]
            onnx_path = f"{log_dir}/smoke_policy.onnx"
            export_cmd = [
                "/root/microduck_rl/.venv/bin/python",
                "-m", "mjlab_microduck.export",
                "Mjlab-ButtWiggle-Flat-MicroDuck",
                "--checkpoint-file", latest_pt,
                "--onnx-file", onnx_path,
                "--video", "True",
                "--video-length", "50",
            ]
            print("Testing export & video rendering...")
            exp_proc = subprocess.run(export_cmd, cwd="/root/microduck_rl", env=env)
            if exp_proc.returncode != 0:
                raise RuntimeError(f"Export in smoke test failed with exit code {exp_proc.returncode}")
            print("=== SMOKE TEST EXPORT & VIDEO PASSED! ===")

    print("=== ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY! ===")
    return True


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,  # 2 hours max (1500 iters takes ~1h 15m)
    secrets=[wandb_secret],
    volumes={"/models": volume},
)
def train_production(
    num_envs: int = 4096,
    max_iterations: int = 1500,
    save_interval: int = 100,
    run_name: str = "v1_butt_wiggle_preen_shake",
):
    """Full production training run on NVIDIA L4."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    full_run_name = f"{timestamp}_{run_name}"
    print(f"=== STARTING PRODUCTION TRAINING: {full_run_name} (envs={num_envs}, iters={max_iterations}) ===")
    patch_rsl_rl_wandb()

    cmd = [
        "/root/microduck_rl/.venv/bin/train",
        "Mjlab-ButtWiggle-Flat-MicroDuck",
        "--env.scene.num-envs", str(num_envs),
        "--agent.max-iterations", str(max_iterations),
        "--agent.save-interval", str(save_interval),
        "--agent.run-name", full_run_name,
        "--agent.wandb-project", "ducky_butt_wiggle",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/microduck_rl/src"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    proc = subprocess.run(cmd, cwd="/root/microduck_rl", env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    # Find checkpoint directory
    log_dir = f"/root/microduck_rl/logs/rsl_rl/butt_wiggle/{full_run_name}"
    if not os.path.exists(log_dir):
        candidates = glob.glob(f"/root/microduck_rl/logs/rsl_rl/butt_wiggle/*{run_name}*")
        if candidates:
            log_dir = sorted(candidates)[-1]

    print(f"Log directory: {log_dir}")
    latest_pt = f"{log_dir}/model_{max_iterations - 1}.pt"
    if not os.path.exists(latest_pt):
        pts = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=os.path.getmtime)
        if pts:
            latest_pt = pts[-1]

    print(f"Latest checkpoint: {latest_pt}")

    # Export policy to ONNX with baked-in normalizer
    onnx_path = f"{log_dir}/butt_wiggle_policy.onnx"
    export_cmd = [
        "/root/microduck_rl/.venv/bin/python",
        "-m", "mjlab_microduck.export",
        "Mjlab-ButtWiggle-Flat-MicroDuck",
        "--checkpoint-file", latest_pt,
        "--onnx-file", onnx_path,
    ]
    print("Exporting ONNX policy...")
    export_proc = subprocess.run(export_cmd, cwd="/root/microduck_rl", env=env)
    print(f"Export finished with code {export_proc.returncode}")

    # Render close-up evaluation video without debug visualizer arrows
    print("Rendering close-up evaluation video...")
    video_dir = f"{log_dir}/videos/play"
    render_script = f"""
import os, glob, torch, mjlab.tasks
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

task_id = "Mjlab-ButtWiggle-Flat-MicroDuck"
env_cfg = load_env_cfg(task_id, play=True)
agent_cfg = load_rl_cfg(task_id)

env_cfg.scene.num_envs = 1
env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.ASSET_BODY
env_cfg.viewer.entity_name = "robot"
env_cfg.viewer.body_name = "trunk_base"
env_cfg.viewer.distance = 0.55
env_cfg.viewer.elevation = -12.0
env_cfg.viewer.azimuth = 145.0
env_cfg.viewer.width = 640
env_cfg.viewer.height = 480

video_dir = "{video_dir}"
os.makedirs(video_dir, exist_ok=True)

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env.manager_visualizers = {{}}
env.update_visualizers = lambda visualizer: None

env = VideoRecorder(env, video_folder=video_dir, step_trigger=lambda step: step == 0, video_length=600, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(task_id) or OnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
runner.load("{latest_pt}", map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")

env.reset()
dones_count = 0
for step in range(600):
    with torch.no_grad():
        obs = env.get_observations()
        act = policy(obs)
        _, _, d, _ = env.step(act)
        if d.any():
            dones_count += 1

env.close()
print(f"Evaluation video render complete! Total resets: {{dones_count}}")
"""
    subprocess.run(
        ["/root/microduck_rl/.venv/bin/python", "-c", render_script],
        cwd="/root/microduck_rl",
        env=env,
    )

    # Save to Modal volume
    dest_vol_dir = f"/models/butt_wiggle/{full_run_name}"
    os.makedirs(dest_vol_dir, exist_ok=True)

    # Copy params, checkpoint, ONNX, and videos
    if os.path.exists(f"{log_dir}/params"):
        shutil.copytree(f"{log_dir}/params", f"{dest_vol_dir}/params", dirs_exist_ok=True)
    if os.path.exists(latest_pt):
        shutil.copy2(latest_pt, dest_vol_dir)
    if os.path.exists(onnx_path):
        shutil.copy2(onnx_path, dest_vol_dir)

    video_files = glob.glob(f"{log_dir}/**/*.mp4", recursive=True)
    print(f"Found video files: {video_files}")
    for vf in video_files:
        shutil.copy2(vf, dest_vol_dir)

    volume.commit()
    print(f"=== COMMITTED TO MODAL VOLUME: {dest_vol_dir} ===")

    # Read back ONNX bytes and video bytes to return directly
    onnx_bytes = None
    if os.path.exists(onnx_path):
        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()

    video_bytes = None
    eval_videos = sorted(glob.glob(f"{video_dir}/*.mp4"), key=os.path.getmtime)
    if eval_videos:
        with open(eval_videos[-1], "rb") as f:
            video_bytes = f.read()
    elif video_files and os.path.exists(video_files[0]):
        with open(video_files[0], "rb") as f:
            video_bytes = f.read()

    return {
        "run_name": full_run_name,
        "latest_checkpoint": latest_pt,
        "volume_dest": dest_vol_dir,
        "onnx_size": len(onnx_bytes) if onnx_bytes else 0,
        "video_bytes": video_bytes,
        "onnx_bytes": onnx_bytes,
    }


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/models": volume},
)
def render_evaluation_video(
    checkpoint_path: str = "/models/butt_wiggle/2026-09-17_05-47-07_v1_butt_wiggle_preen_shake/model_1499.pt",
    video_length: int = 500,
):
    """Step the trained policy and render an evaluation .mp4 video."""
    import os
    import glob
    import subprocess

    script = f"""
import os
import glob
from dataclasses import asdict
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
device = "cuda:0"

task_id = "Mjlab-ButtWiggle-Flat-MicroDuck"
env_cfg = load_env_cfg(task_id, play=True)
agent_cfg = load_rl_cfg(task_id)

env_cfg.scene.num_envs = 1
env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.ASSET_BODY
env_cfg.viewer.entity_name = "robot"
env_cfg.viewer.body_name = "trunk_base"
env_cfg.viewer.distance = 0.55
env_cfg.viewer.elevation = -12.0
env_cfg.viewer.azimuth = 145.0
env_cfg.viewer.width = 640
env_cfg.viewer.height = 480

video_dir = "/tmp/butt_wiggle_eval_video"
os.makedirs(video_dir, exist_ok=True)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
# Disable debug visualizers so large force/momentum arrows are completely removed
env.manager_visualizers = {{}}
env.update_visualizers = lambda visualizer: None

env = VideoRecorder(
    env,
    video_folder=video_dir,
    step_trigger=lambda step: step == 0,
    video_length={video_length},
    disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(task_id) or OnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load("{checkpoint_path}", map_location=device)
policy = runner.get_inference_policy(device=device)

env.reset()
for _ in range({video_length}):
    with torch.no_grad():
        obs = env.get_observations()
        act = policy(obs)
        env.step(act)

env.close()
print("Stepping and video recording complete!")
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/microduck_rl/src"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    proc = subprocess.run(
        ["/root/microduck_rl/.venv/bin/python", "-c", script],
        cwd="/root/microduck_rl",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Render failed with code {proc.returncode}")

    mp4_files = glob.glob("/tmp/butt_wiggle_eval_video/**/*.mp4", recursive=True)
    print(f"Rendered videos: {mp4_files}")
    if not mp4_files:
        raise RuntimeError("No video file was produced!")

    video_path = mp4_files[0]
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    dest_path = "/models/butt_wiggle/2026-09-17_05-47-07_v1_butt_wiggle_preen_shake/butt_wiggle_eval.mp4"
    with open(dest_path, "wb") as f:
        f.write(video_bytes)
    volume.commit()
    print(f"Saved to volume: {dest_path}")

    return video_bytes

