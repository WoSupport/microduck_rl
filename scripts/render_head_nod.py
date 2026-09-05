#!/usr/bin/env python3
"""Render deterministic rollout video and keyframe snapshots for MicroDuck head nod policy."""

from __future__ import annotations

import argparse
import math
import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio

from mjlab_microduck.tasks.microduck_head_nod_env_cfg import HEAD_NOD_PERIOD

DEF_POS = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,  # Left leg
    0.3491, 0.3491, 0.0, 0.0,                # Neck & Head
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,   # Right leg
], dtype=np.float32)

JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


class ActorMLP(nn.Module):
    def __init__(self, obs_dim=61, act_dim=14, hidden_dims=(512, 256, 128)):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(1, obs_dim))
        self.register_buffer("obs_std", torch.ones(1, obs_dim))
        self.has_normalizer = False

        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.has_normalizer:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)
        return self.net(x)


def load_actor_from_checkpoint(ckpt_path: Path | str) -> ActorMLP:
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = ActorMLP(obs_dim=61, act_dim=14)
    state_dict = data.get("actor_state_dict", data.get("model_state_dict", {}))

    if "obs_normalizer._mean" in state_dict and "obs_normalizer._std" in state_dict:
        actor.obs_mean.copy_(state_dict["obs_normalizer._mean"].view(1, -1))
        actor.obs_std.copy_(state_dict["obs_normalizer._std"].view(1, -1))
        actor.has_normalizer = True

    mlp_dict = {}
    for k, v in state_dict.items():
        if k.startswith("mlp."):
            mlp_dict[k[4:]] = v
        elif k.startswith("actor.mlp."):
            mlp_dict[k[10:]] = v
        elif k.startswith("actor."):
            mlp_dict[k[6:]] = v

    if mlp_dict:
        actor.net.load_state_dict(mlp_dict, strict=False)
    actor.eval()
    return actor


def render_head_nod_rollout(
    ckpt_path: str | Path,
    output_mp4: str | Path | None = None,
    output_gif: str | Path | None = None,
    keyframes_dir: str | Path | None = None,
    duration_s: float = 4.5,
    fps: int = 50,
    camera_view: str = "perspective",
    width: int = 640,
    height: int = 480,
) -> None:
    scene_xml = Path(__file__).resolve().parent.parent / "src/mjlab_microduck/robot/microduck/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    actor = load_actor_from_checkpoint(ckpt_path)

    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.125
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    joint_qpos_adrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
    joint_dof_adrs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]

    for idx, adr in enumerate(joint_qpos_adrs):
        data.qpos[adr] = DEF_POS[idx]
    data.ctrl[:] = DEF_POS
    mujoco.mj_forward(model, data)

    # Renderer setup
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    def update_camera():
        cam.lookat[0] = data.xpos[trunk_id, 0] + 0.05
        cam.lookat[1] = data.xpos[trunk_id, 1]
        cam.lookat[2] = data.xpos[trunk_id, 2] + 0.02
        if camera_view == "perspective":
            cam.distance = 0.55
            cam.elevation = -12.0
            cam.azimuth = 135.0
        elif camera_view == "side":
            cam.distance = 0.50
            cam.elevation = -5.0
            cam.azimuth = 90.0
        elif camera_view == "front":
            cam.distance = 0.50
            cam.elevation = -8.0
            cam.azimuth = 180.0

    update_camera()

    dt = 1.0 / fps
    total_steps = int(duration_s * fps)
    frames = []

    last_action = np.zeros(14, dtype=np.float32)
    last_joint_vel = np.zeros(14, dtype=np.float32)

    for step in range(total_steps):
        sim_time = step * dt
        phase = (sim_time / HEAD_NOD_PERIOD) % 1.0

        # Observations
        imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        if imu_id >= 0:
            adr = model.sensor_adr[imu_id]
            imu_ang_vel = data.sensordata[adr : adr + 3].copy().astype(np.float32)
        else:
            imu_ang_vel = data.qvel[3:6].copy().astype(np.float32)

        rot_base = data.xmat[1].reshape(3, 3)
        proj_g = (rot_base.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

        joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adrs], dtype=np.float32)
        rel_joint_pos = (joint_pos - DEF_POS).astype(np.float32)
        cur_joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adrs], dtype=np.float32)

        cmd = np.array([math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase), 0.0], dtype=np.float32)
        head_cmd = np.zeros(4, dtype=np.float32)
        body_cmd = np.zeros(6, dtype=np.float32)

        obs_raw = np.concatenate([
            imu_ang_vel.flatten(),
            proj_g.flatten(),
            rel_joint_pos.flatten(),
            last_joint_vel.flatten(),
            last_action.flatten(),
            cmd.flatten(),
            head_cmd.flatten(),
            body_cmd.flatten(),
        ]).astype(np.float32)
        last_joint_vel = cur_joint_vel.copy()

        obs_tensor = torch.from_numpy(obs_raw).float().unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_tensor).squeeze(0).cpu().numpy()
        last_action = action.copy()

        motor_targets = DEF_POS + action
        for j in range(14):
            data.ctrl[j] = motor_targets[j]

        # Step 0.02s
        for _ in range(10):
            mujoco.mj_step(model, data)

        update_camera()
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        frames.append(pixels)

    if output_mp4:
        Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(output_mp4, frames, fps=fps, codec="libx264")
        print(f"Saved rollout video to {output_mp4}")

    if output_gif:
        Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
        # Downsample frames for smooth GIF
        gif_frames = frames[::2]
        imageio.mimwrite(output_gif, gif_frames, fps=fps // 2, loop=0)
        print(f"Saved rollout GIF to {output_gif}")

    if keyframes_dir:
        k_dir = Path(keyframes_dir)
        k_dir.mkdir(parents=True, exist_ok=True)
        # Save key snapshots across the 3 big nods + pause sequence (2.0s cycle in 4.0s)
        snapshot_indices = [
            0,                                # Start / Nominal (0.0s)
            int((0.30 / duration_s) * total_steps), # Nod 1 Peak Down (-33.8°)
            int((0.50 / duration_s) * total_steps), # Nod 1 Recovery (0.0°)
            int((0.80 / duration_s) * total_steps), # Nod 2 Peak Down (-33.8°)
            int((1.00 / duration_s) * total_steps), # Nod 2 Recovery (0.0°)
            int((1.30 / duration_s) * total_steps), # Nod 3 Peak Down (-33.8°)
            int((1.50 / duration_s) * total_steps), # Nod 3 Recovery / Pause Start (0.0°)
            int((1.75 / duration_s) * total_steps), # Pause / Steady Eye-Level Rest
            min(int((2.00 / duration_s) * total_steps), total_steps - 1), # Cycle complete
        ]
        for i, idx in enumerate(snapshot_indices):
            idx = min(idx, len(frames) - 1)
            imageio.imwrite(k_dir / f"snapshot_{i}_{idx:04d}.png", frames[idx])
        print(f"Saved {len(snapshot_indices)} keyframe snapshots to {k_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render MicroDuck Head Nod Rollout")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--mp4", type=str, default=None, help="Path to output MP4")
    parser.add_argument("--gif", type=str, default=None, help="Path to output GIF")
    parser.add_argument("--keyframes-dir", type=str, default=None, help="Directory to save snapshot PNGs")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds (default 4.0s = 2 cycles)")
    parser.add_argument("--camera", type=str, default="perspective", choices=["perspective", "side", "front"])
    args = parser.parse_args()

    render_head_nod_rollout(
        ckpt_path=args.checkpoint,
        output_mp4=args.mp4,
        output_gif=args.gif,
        keyframes_dir=args.keyframes_dir,
        duration_s=args.duration,
        camera_view=args.camera,
    )


if __name__ == "__main__":
    main()
