"""Render deterministic rollout video and keyframe snapshots for MicroDuck velocity policy."""

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

DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)


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

    if mlp_dict:
        actor.net.load_state_dict(mlp_dict)
    actor.eval()
    return actor


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w**2 - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0
    return a - b + c


def render_velocity_rollout(
    ckpt_path: str | Path,
    output_mp4: str | Path | None = None,
    output_gif: str | Path | None = None,
    keyframes_dir: str | Path | None = None,
    cmd_vx: float = 0.25,
    cmd_vy: float = 0.0,
    cmd_vyaw: float = 0.0,
    duration_s: float = 5.0,
    fps: int = 50,
    camera_view: str = "side",
    width: int = 640,
    height: int = 480,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    xml_path = repo_root / "src/mjlab_microduck/robot/microduck/scene.xml"
    if not xml_path.exists():
        xml_path = repo_root / "src/mjlab_microduck/robot/microduck/scene_walk.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    actor = load_actor_from_checkpoint(ckpt_path)
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.115
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i in range(model.nu):
        jnt_adr = model.jnt_qposadr[model.actuator_trnid[i, 0]]
        data.qpos[jnt_adr] = DEFAULT_POSE[i]

    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = trunk_body_id

    if camera_view == "side":
        camera.distance = 0.70
        camera.elevation = -5.0
        camera.azimuth = 90.0
    elif camera_view == "front":
        camera.distance = 0.70
        camera.elevation = -5.0
        camera.azimuth = 0.0
    else:  # perspective
        camera.distance = 0.85
        camera.elevation = -15.0
        camera.azimuth = 45.0

    ctrl_dt = 0.02
    substeps = int(ctrl_dt / model.opt.timestep)
    total_steps = int(duration_s / ctrl_dt)

    frames = []
    last_action = np.zeros(14, dtype=np.float32)

    cmd_twist = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
    cmd_head = np.zeros(4, dtype=np.float32)
    cmd_body = np.zeros(6, dtype=np.float32)
    cmd_13d = np.concatenate([cmd_twist, cmd_head, cmd_body])

    saved_keyframes = []
    keyframe_steps = {
        0: "t0_start",
        int(total_steps * 0.25): "t25_step_left",
        int(total_steps * 0.50): "t50_step_right",
        int(total_steps * 0.75): "t75_mid_stride",
        total_steps - 1: "t100_final_walk",
    }

    if keyframes_dir:
        Path(keyframes_dir).mkdir(parents=True, exist_ok=True)

    for step in range(total_steps):
        quat = data.qpos[3:7].copy()
        proj_grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        base_ang_vel = data.qvel[3:6].copy()

        current_pos = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(14)], dtype=np.float32)
        joint_pos_rel = current_pos - DEFAULT_POSE
        joint_vel = np.array([data.qvel[model.jnt_dofadr[model.actuator_trnid[i, 0]]] for i in range(14)], dtype=np.float32)
        prev_act = last_action.copy()

        obs = np.concatenate([
            base_ang_vel,
            proj_grav,
            joint_pos_rel,
            joint_vel,
            prev_act,
            cmd_13d,
        ]).astype(np.float32)

        with torch.no_grad():
            action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE + action

        for _ in range(substeps):
            mujoco.mj_step(model, data)

        camera.lookat[:] = data.xpos[trunk_body_id]
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        frames.append(frame)

        if keyframes_dir and step in keyframe_steps:
            kf_name = f"{keyframe_steps[step]}.png"
            kf_path = os.path.join(str(keyframes_dir), kf_name)
            imageio.imwrite(kf_path, frame)
            saved_keyframes.append(kf_path)

    if output_mp4:
        Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_mp4), frames, fps=fps)
        print(f"Saved rollout MP4 to {output_mp4}")

    if output_gif:
        Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_gif), frames[::2], fps=fps // 2, loop=0)
        print(f"Saved rollout GIF to {output_gif}")

    return saved_keyframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mp4", type=str, default="rollout_walking.mp4")
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--keyframes-dir", type=str, default="keyframes_velocity")
    parser.add_argument("--vx", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--camera", type=str, default="side", choices=["perspective", "side", "front"])
    args = parser.parse_args()

    render_velocity_rollout(
        ckpt_path=args.checkpoint,
        output_mp4=args.mp4,
        output_gif=args.gif,
        keyframes_dir=args.keyframes_dir,
        cmd_vx=args.vx,
        duration_s=args.duration,
        camera_view=args.camera,
    )
