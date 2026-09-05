#!/usr/bin/env python3
"""Automated Physical Diagnostic Testbench for MicroDuck Head Nod Policy.

Evaluates trained checkpoints across deterministic rollouts against hard physical assertions:
1. Dual-Foot Grounding: Both feet continuously grounded (>= 95% of episode).
2. Deep Passionate Nod Down: Peak downward head pitch in world frame <= -22° (target is ~ -32°).
3. Head Nod Range of Motion: Total pitch excursion >= 25° (confirming active, energetic nodding).
4. Sagittal Alignment: Head yaw and roll drift <= 6° (nodding purely sagittal, facing ahead).
5. Torso Upright Posture: Torso world pitch tilt <= 10° (confirming no backward-lean reward gaming).
6. Anti-Collapse / Standing Height: Trunk z >= 0.090m throughout rollout.

Usage:
    uv run scripts/diagnose_head_nod.py --checkpoint <path_to_model.pt> --trials 5
"""

import argparse
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import mujoco

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


def load_policy(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))

    actor = ActorMLP(obs_dim=61, act_dim=14).to(device)

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


def run_diagnostics(
    checkpoint_path: str,
    trials: int = 5,
    duration_s: float = 4.5,
    dt: float = 0.02,
):
    print(f"\n========================================================")
    print(f" MicroDuck Head Nod Physical Testbench")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Trials: {trials} | Duration: {duration_s}s | Step dt: {dt}s")
    print(f"========================================================\n")

    scene_xml = Path(__file__).resolve().parent.parent / "src/mjlab_microduck/robot/microduck/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    device = "cpu"
    actor = load_policy(checkpoint_path, device=device)

    # Identifiers
    cam_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
    lf_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    joint_qpos_adrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
    joint_dof_adrs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]

    steps_per_trial = int(duration_s / dt)
    results = []

    for trial in range(trials):
        mujoco.mj_resetData(model, data)
        freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        qpos_adr = model.jnt_qposadr[freejoint_id]
        data.qpos[qpos_adr + 0] = 0.0
        data.qpos[qpos_adr + 1] = 0.0
        data.qpos[qpos_adr + 2] = 0.120
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        for idx, adr in enumerate(joint_qpos_adrs):
            data.qpos[adr] = DEF_POS[idx]
        data.ctrl[:] = DEF_POS
        mujoco.mj_forward(model, data)

        dual_contact_count = 0
        min_pitch_deg = 90.0  # Most negative pitch (deepest down-nod)
        max_pitch_deg = -90.0 # Most positive pitch (highest rebound)
        max_yaw_drift_deg = 0.0
        max_roll_drift_deg = 0.0
        max_torso_pitch_deg = 0.0
        min_torso_z = 1.0

        last_action = np.zeros(14, dtype=np.float32)
        last_joint_vel = np.zeros(14, dtype=np.float32)
        pause_devs = []

        for step in range(steps_per_trial):
            sim_time = step * dt
            phase = (sim_time / HEAD_NOD_PERIOD) % 1.0

            # 1. base_ang_vel (3D)
            imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
            if imu_id >= 0:
                adr = model.sensor_adr[imu_id]
                imu_ang_vel = data.sensordata[adr : adr + 3].copy().astype(np.float32)
            else:
                imu_ang_vel = data.qvel[3:6].copy().astype(np.float32)

            # 2. projected gravity (3D)
            rot_base = data.xmat[1].reshape(3, 3)
            proj_g = (rot_base.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

            # 3. joint_pos (14D)
            joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adrs], dtype=np.float32)
            rel_joint_pos = (joint_pos - DEF_POS).astype(np.float32)

            # 4. joint_vel (14D)
            cur_joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adrs], dtype=np.float32)

            # 5. actions (14D): last_action
            # 6. command (3D): [cos(2pi*phase), sin(2pi*phase), 0]
            cmd = np.array([math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase), 0.0], dtype=np.float32)

            # 7. head_cmd (4D), 8. body_cmd (6D)
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

            obs_tensor = torch.from_numpy(obs_raw).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action = actor(obs_tensor).squeeze(0).cpu().numpy()
            last_action = action.copy()

            # Apply action
            motor_targets = DEF_POS + action
            for j in range(14):
                data.ctrl[j] = motor_targets[j]

            # Step physics for 0.02s
            for _ in range(10):
                mujoco.mj_step(model, data)

            # 1. World-frame head pitch: forward camera vector elevation relative to ground
            cam_mat = data.site_xmat[cam_site].reshape(3, 3)
            cam_forward = cam_mat[:, 0]
            horiz_forward = math.hypot(cam_forward[0], cam_forward[1])
            pitch_deg = math.degrees(math.atan2(cam_forward[2], max(horiz_forward, 1e-6)))

            if pitch_deg < min_pitch_deg:
                min_pitch_deg = pitch_deg
            if pitch_deg > max_pitch_deg:
                max_pitch_deg = pitch_deg

            # Track pitch deviation during pause phase (phase >= 0.75)
            if phase >= 0.75:
                pause_devs.append(abs(pitch_deg))

            # 2. Sagittal alignment: head yaw and roll relative to default
            head_yaw_deg = math.degrees(abs(rel_joint_pos[7]))
            head_roll_deg = math.degrees(abs(rel_joint_pos[8]))
            if head_yaw_deg > max_yaw_drift_deg:
                max_yaw_drift_deg = head_yaw_deg
            if head_roll_deg > max_roll_drift_deg:
                max_roll_drift_deg = head_roll_deg

            # 3. Torso pitch tilt in world frame
            torso_pitch_deg = math.degrees(math.atan2(-rot_base[0, 2], rot_base[2, 2]))
            if abs(torso_pitch_deg) > max_torso_pitch_deg:
                max_torso_pitch_deg = abs(torso_pitch_deg)

            # 4. Torso height
            torso_z = data.xpos[1, 2]
            if torso_z < min_torso_z:
                min_torso_z = torso_z

            # 5. Dual foot contact check (checking body IDs for ankle_left and ankle_right)
            left_hit = False
            right_hit = False
            for c_idx in range(data.ncon):
                con = data.contact[c_idx]
                b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[con.geom1])
                b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[con.geom2])
                if b1 == "ankle_left" or b2 == "ankle_left":
                    left_hit = True
                if b1 == "ankle_right" or b2 == "ankle_right":
                    right_hit = True
            if left_hit and right_hit:
                dual_contact_count += 1

        dual_contact_pct = (dual_contact_count / steps_per_trial) * 100.0
        pitch_stroke_deg = max_pitch_deg - min_pitch_deg
        mean_pause_dev = float(np.mean(pause_devs)) if pause_devs else 0.0

        results.append({
            "dual_contact_pct": dual_contact_pct,
            "min_pitch_deg": min_pitch_deg,
            "max_pitch_deg": max_pitch_deg,
            "pitch_stroke_deg": pitch_stroke_deg,
            "max_yaw_drift_deg": max_yaw_drift_deg,
            "max_roll_drift_deg": max_roll_drift_deg,
            "max_torso_pitch_deg": max_torso_pitch_deg,
            "min_torso_z": min_torso_z,
            "mean_pause_dev": mean_pause_dev,
        })

    print("--- Diagnostic Audit Summary ---")
    avg_dual_contact = np.mean([r["dual_contact_pct"] for r in results])
    avg_min_pitch = np.mean([r["min_pitch_deg"] for r in results])
    avg_stroke = np.mean([r["pitch_stroke_deg"] for r in results])
    avg_yaw_drift = np.mean([r["max_yaw_drift_deg"] for r in results])
    avg_roll_drift = np.mean([r["max_roll_drift_deg"] for r in results])
    avg_torso_pitch = np.mean([r["max_torso_pitch_deg"] for r in results])
    min_torso_height = np.min([r["min_torso_z"] for r in results])
    avg_pause_dev = np.mean([r["mean_pause_dev"] for r in results])

    print(f"1. Dual Foot Grounding:      {avg_dual_contact:.1f}% (Assertion: >= 95.0%)")
    print(f"2. Peak Downward Nod Pitch:  {avg_min_pitch:.1f}° (Assertion: <= -22.0°)")
    print(f"3. Nod Range of Motion:      {avg_stroke:.1f}° (Assertion: >= 25.0°)")
    print(f"4. Pause Pitch Deviation:    {avg_pause_dev:.1f}° (Assertion: <= 5.0°)")
    print(f"5. Max Head Yaw Drift:       {avg_yaw_drift:.1f}° (Assertion: <= 6.0°)")
    print(f"6. Max Head Roll Drift:      {avg_roll_drift:.1f}° (Assertion: <= 6.0°)")
    print(f"7. Max Torso Pitch Tilt:     {avg_torso_pitch:.1f}° (Assertion: <= 10.0°)")
    print(f"8. Min Torso Height:         {min_torso_height:.3f}m (Assertion: >= 0.090m)")

    # Assertions
    assertions = [
        ("Dual-Foot Grounding >= 95%", avg_dual_contact >= 95.0),
        ("Peak Down-Nod Pitch <= -22°", avg_min_pitch <= -22.0),
        ("Nod Stroke >= 25°", avg_stroke >= 25.0),
        ("Pause Pitch Dev <= 5.0°", avg_pause_dev <= 5.0),
        ("Head Yaw Drift <= 6°", avg_yaw_drift <= 6.0),
        ("Head Roll Drift <= 6°", avg_roll_drift <= 6.0),
        ("Torso Pitch Tilt <= 10°", avg_torso_pitch <= 10.0),
        ("Torso Height >= 0.090m", min_torso_height >= 0.090),
    ]

    all_passed = True
    print("\n--- Physical Assertion Checks ---")
    for desc, passed in assertions:
        status = "PASSED [OK]" if passed else "FAILED [X]"
        if not passed:
            all_passed = False
        print(f"  {desc:<35}: {status}")

    print("\nFinal Verdict:", "PASSED - AUTHENTIC 3-BIG-NOD + PAUSE" if all_passed else "REJECTED - FAILED ASSERTIONS")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="MicroDuck Head Nod Diagnostic Testbench")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--trials", type=int, default=5, help="Number of test rollouts")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds (default 4.0s = 2 cycles)")
    args = parser.parse_args()

    success = run_diagnostics(args.checkpoint, trials=args.trials, duration_s=args.duration)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
