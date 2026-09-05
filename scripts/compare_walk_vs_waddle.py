"""Synchronized side-by-side comparison between Standard Walk and Gentle Waddle."""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_velocity_rollout import load_actor_from_checkpoint, quat_rotate_inverse, DEFAULT_POSE


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Convert [w, x, y, z] to roll, pitch, yaw in degrees."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def simulate_policy_rollout(
    ckpt_path: str,
    duration_s: float = 4.0,
    cmd_vx: float = 0.25,
    camera_view: str = "front",
    width: int = 640,
    height: int = 480,
):
    repo_root = Path(__file__).resolve().parents[1]
    xml_path = repo_root / "src/mjlab_microduck/robot/microduck/scene.xml"
    if not xml_path.exists():
        xml_path = repo_root / "src/mjlab_microduck/robot/microduck/scene_walk.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    actor = load_actor_from_checkpoint(ckpt_path)
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle")

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

    if camera_view == "front":
        camera.distance = 0.70
        camera.elevation = -5.0
        camera.azimuth = 0.0
    elif camera_view == "side":
        camera.distance = 0.70
        camera.elevation = -5.0
        camera.azimuth = 90.0
    else:  # perspective
        camera.distance = 0.85
        camera.elevation = -15.0
        camera.azimuth = 45.0

    ctrl_dt = 0.02
    substeps = int(ctrl_dt / model.opt.timestep)
    total_steps = int(duration_s / ctrl_dt)

    frames = []
    telemetry = {
        "time": [],
        "roll_deg": [],
        "pitch_deg": [],
        "base_z": [],
        "lf_z": [],
        "rf_z": [],
        "left_knee_rad": [],
        "right_knee_rad": [],
        "vx_actual": [],
    }

    last_action = np.zeros(14, dtype=np.float32)
    cmd_13d = np.concatenate([
        np.array([cmd_vx, 0.0, 0.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
    ])

    for step in range(total_steps):
        t = step * ctrl_dt
        quat = data.qpos[3:7].copy()
        roll, pitch, yaw = quat_to_euler_deg(quat)
        proj_grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        base_ang_vel = data.qvel[3:6].copy()

        current_pos = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(14)], dtype=np.float32)
        joint_pos_rel = current_pos - DEFAULT_POSE
        joint_vel = np.array([data.qvel[model.jnt_dofadr[model.actuator_trnid[i, 0]]] for i in range(14)], dtype=np.float32)

        # Log telemetry
        telemetry["time"].append(t)
        telemetry["roll_deg"].append(roll)
        telemetry["pitch_deg"].append(pitch)
        telemetry["base_z"].append(data.xpos[trunk_body_id][2])
        telemetry["lf_z"].append(data.xpos[left_foot_id][2] if left_foot_id >= 0 else 0.0)
        telemetry["rf_z"].append(data.xpos[right_foot_id][2] if right_foot_id >= 0 else 0.0)
        telemetry["left_knee_rad"].append(current_pos[3])
        telemetry["right_knee_rad"].append(current_pos[12])
        telemetry["vx_actual"].append(data.qvel[0])

        obs = np.concatenate([
            base_ang_vel,
            proj_grav,
            joint_pos_rel,
            joint_vel,
            last_action,
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

    return frames, {k: np.array(v) for k, v in telemetry.items()}


def create_side_by_side_video(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    telem_a: dict[str, np.ndarray],
    telem_b: dict[str, np.ndarray],
    title_a: str,
    title_b: str,
    output_mp4: str,
    output_gif: str,
    fps: int = 50,
):
    n_frames = min(len(frames_a), len(frames_b))
    composite_frames = []

    h, w, c = frames_a[0].shape
    header_h = 50
    footer_h = 36
    total_h = h + header_h + footer_h
    total_w = w * 2 + 6  # 6px divider

    for i in range(n_frames):
        # Create base canvas
        canvas = Image.new("RGB", (total_w, total_h), color=(15, 23, 42))  # dark slate bg
        draw = ImageDraw.Draw(canvas)

        img_a = Image.fromarray(frames_a[i])
        img_b = Image.fromarray(frames_b[i])

        canvas.paste(img_a, (0, header_h))
        canvas.paste(img_b, (w + 6, header_h))

        # Divider line
        draw.line([(w + 2, 0), (w + 2, total_h)], fill=(51, 65, 85), width=2)

        # Header A (Left)
        draw.rectangle([(0, 0), (w, header_h)], fill=(30, 41, 59))
        draw.text((16, 12), title_a, fill=(56, 189, 248))  # light blue
        roll_a = telem_a["roll_deg"][i]
        z_a = telem_a["base_z"][i] * 100.0
        draw.text((16, 30), f"Roll: {roll_a:+.1f}° | Height: {z_a:.1f}cm", fill=(148, 163, 184))

        # Header B (Right)
        draw.rectangle([(w + 6, 0), (total_w, header_h)], fill=(30, 41, 59))
        draw.text((w + 22, 12), title_b, fill=(251, 191, 36))  # amber
        roll_b = telem_b["roll_deg"][i]
        z_b = telem_b["base_z"][i] * 100.0
        draw.text((w + 22, 30), f"Roll: {roll_b:+.1f}° (Target: ±8°) | Height: {z_b:.1f}cm", fill=(148, 163, 184))

        # Footer
        t_sec = telem_a["time"][i]
        draw.rectangle([(0, total_h - footer_h), (total_w, total_h)], fill=(15, 23, 42))
        draw.text((total_w // 2 - 80, total_h - 26), f"Time: {t_sec:.2f}s | vx cmd = 0.25 m/s", fill=(203, 213, 225))

        composite_frames.append(np.array(canvas))

    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_mp4, composite_frames, fps=fps)
    print(f"Saved side-by-side MP4 to {output_mp4}")

    if output_gif:
        # Save every 2nd frame for web gif
        imageio.mimsave(output_gif, composite_frames[::2], fps=fps // 2, loop=0)
        print(f"Saved side-by-side GIF to {output_gif}")

    return composite_frames


def plot_telemetry_comparison(
    telem_std: dict[str, np.ndarray],
    telem_waddle: dict[str, np.ndarray],
    telem_deep: dict[str, np.ndarray],
    output_png: str,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = telem_std["time"]

    # 1. Roll Sway Angle
    ax = axes[0]
    ax.plot(t, telem_std["roll_deg"], label="Standard Walk (Erect)", color="#38bdf8", linewidth=1.8)
    ax.plot(t, telem_waddle["roll_deg"], label="Gentle Waddle (V1, 8° Target)", color="#f59e0b", linewidth=2.0)
    ax.plot(t, telem_deep["roll_deg"], label="Deep Crouch Waddle (V3, 11° Target)", color="#ec4899", linestyle="--", linewidth=1.5)
    ax.axhline(0, color="#64748b", linestyle=":", alpha=0.6)
    ax.set_ylabel("Torso Roll (°)", fontsize=11, fontweight="bold")
    ax.set_title("Kinematic Comparison: Standard Walk vs. Duck Waddles", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 2. Base Height (Crouch Depth)
    ax = axes[1]
    ax.plot(t, telem_std["base_z"] * 100.0, label="Standard Walk", color="#38bdf8", linewidth=1.8)
    ax.plot(t, telem_waddle["base_z"] * 100.0, label="Gentle Waddle (V1)", color="#f59e0b", linewidth=2.0)
    ax.plot(t, telem_deep["base_z"] * 100.0, label="Deep Crouch Waddle (V3)", color="#ec4899", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Base Height (cm)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 3. Knee Flexion (Left Knee)
    ax = axes[2]
    ax.plot(t, np.degrees(telem_std["left_knee_rad"]), label="Standard Walk Left Knee", color="#38bdf8", linewidth=1.8)
    ax.plot(t, np.degrees(telem_waddle["left_knee_rad"]), label="Gentle Waddle Left Knee", color="#f59e0b", linewidth=2.0)
    ax.plot(t, np.degrees(telem_deep["left_knee_rad"]), label="Deep Crouch Left Knee", color="#ec4899", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Left Knee Angle (°)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved telemetry comparison plot to {output_png}")


def main():
    ckpt_std = "logs/rsl_rl/velocity/v2_tuned_velocity_tracking/model_1200.pt"
    ckpt_waddle = "logs/rsl_rl/duck_walk/duck_walk/2026-09-04_22-15-54_v1_gentle_waddle/model_1499.pt"
    ckpt_deep = "logs/rsl_rl/duck_walk/duck_walk/2026-09-04_22-15-43_v3_deep_crouch/model_1499.pt"

    duration = 4.0

    print("Simulating Standard Walk (Front)...")
    frames_std_front, telem_std = simulate_policy_rollout(ckpt_std, duration_s=duration, camera_view="front")

    print("Simulating Gentle Waddle (Front)...")
    frames_waddle_front, telem_waddle = simulate_policy_rollout(ckpt_waddle, duration_s=duration, camera_view="front")

    print("Simulating Deep Crouch Waddle (Front)...")
    _, telem_deep = simulate_policy_rollout(ckpt_deep, duration_s=duration, camera_view="front")

    print("Generating Front View Side-by-Side Video...")
    create_side_by_side_video(
        frames_std_front,
        frames_waddle_front,
        telem_std,
        telem_waddle,
        title_a="Standard Walk (Erect / Neutral)",
        title_b="Gentle Waddle (v1: 8° Roll Sway)",
        output_mp4="renders/comparison_side_by_side_front.mp4",
        output_gif="renders/comparison_side_by_side_front.gif",
    )

    print("Simulating Standard Walk (Perspective)...")
    frames_std_persp, _ = simulate_policy_rollout(ckpt_std, duration_s=duration, camera_view="perspective")

    print("Simulating Gentle Waddle (Perspective)...")
    frames_waddle_persp, _ = simulate_policy_rollout(ckpt_waddle, duration_s=duration, camera_view="perspective")

    print("Generating Perspective View Side-by-Side Video...")
    create_side_by_side_video(
        frames_std_persp,
        frames_waddle_persp,
        telem_std,
        telem_waddle,
        title_a="Standard Walk (Erect / Neutral)",
        title_b="Gentle Waddle (v1: 8° Roll Sway)",
        output_mp4="renders/comparison_side_by_side_persp.mp4",
        output_gif="renders/comparison_side_by_side_persp.gif",
    )

    print("Generating Telemetry Comparison Plots...")
    plot_telemetry_comparison(
        telem_std,
        telem_waddle,
        telem_deep,
        output_png="renders/telemetry_comparison.png",
    )

    # Save synchronized keyframes at stride apex (t = 1.0s, t = 2.0s)
    kf_dir = Path("renders/keyframes_comparison")
    kf_dir.mkdir(parents=True, exist_ok=True)
    for step_idx, name in [(0, "t0_start"), (50, "t10_step_left"), (100, "t20_step_right"), (150, "t30_step_mid")]:
        img_std = Image.fromarray(frames_std_front[step_idx])
        img_waddle = Image.fromarray(frames_waddle_front[step_idx])
        # composite single image
        comp = Image.new("RGB", (640 * 2 + 4, 480))
        comp.paste(img_std, (0, 0))
        comp.paste(img_waddle, (644, 0))
        draw = ImageDraw.Draw(comp)
        draw.text((10, 10), f"Standard: Roll {telem_std['roll_deg'][step_idx]:+.1f}°", fill=(56, 189, 248))
        draw.text((654, 10), f"Gentle Waddle: Roll {telem_waddle['roll_deg'][step_idx]:+.1f}°", fill=(251, 191, 36))
        comp.save(kf_dir / f"{name}.png")

    print("All comparisons rendered successfully!")


if __name__ == "__main__":
    main()
