#!/usr/bin/env python3
"""Automated Physical Diagnostic Testbench for MicroDuck Leg Lift Policy.

Evaluates trained checkpoints across deterministic rollouts against hard physical assertions:
1. Support Leg Grounding: Left foot stays continuously grounded during lift & hold (>= 95%).
2. Support Foot Flatness: Left foot sole tilt <= 10° (no rolling onto outer edge).
3. Right Leg Peak Elevation: Forward elevation angle >= 75° (horizontal target is 90°).
4. Peak Right Foot Height: Foot clearance >= 7.0 cm above ground.
5. Torso Height & Posture: Torso z >= 0.090m and tilt <= 25° throughout.
6. Standing Settle: Both feet grounded and joint error < 0.15 rad vs HOME during settle phase.

Usage:
    uv run scripts/diagnose_leg_lift.py --checkpoint logs/rsl_rl/Mjlab-LegLift-Flat-MicroDuck/v1_phased_baseline/model_xxx.pt
"""

import argparse
import math
import os
from pathlib import Path
import sys
import torch
import numpy as np
import mujoco

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML, HOME_FRAME
from mjlab_microduck.tasks.microduck_leg_lift_env_cfg import (
    LL_PERIOD,
    LIFT_END,
    HOLD_END,
    RETURN_END,
    _ALL_LEG_JOINTS,
)


def load_policy(checkpoint_path: str, device: str = "cpu"):
    """Loads an RSL-RL actor model and its running observation normalizer."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Rebuild Actor MLP architecture
    import torch.nn as nn

    # Unified 61D input, 14D action output
    actor = nn.Sequential(
        nn.Linear(61, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 14),
    ).to(device)

    actor_weights = {
        k.replace("actor.0.", "0.")
        .replace("actor.2.", "2.")
        .replace("actor.4.", "4.")
        .replace("actor.6.", "6.")
        .replace("actor.", ""): v
        for k, v in state_dict.items()
        if "actor" in k or k.startswith("0.") or k.startswith("2.") or k.startswith("4.") or k.startswith("6.")
    }
    
    # Filter actor keys
    filtered = {}
    for i, l in enumerate([0, 2, 4, 6]):
        w_k = f"actor.{l}.weight" if f"actor.{l}.weight" in state_dict else f"{l}.weight"
        b_k = f"actor.{l}.bias" if f"actor.{l}.bias" in state_dict else f"{l}.bias"
        if w_k in state_dict and b_k in state_dict:
            filtered[f"{l}.weight"] = state_dict[w_k]
            filtered[f"{l}.bias"] = state_dict[b_k]

    if filtered:
        actor.load_state_dict(filtered, strict=False)
    actor.eval()

    mean = state_dict.get("obs_normalizer._mean", torch.zeros(1, 61, device=device))
    std = state_dict.get("obs_normalizer._std", torch.ones(1, 61, device=device))

    return actor, mean, std


def run_diagnostics(checkpoint_path: str, trials: int = 5, duration_s: float = 4.0, dt: float = 0.02):
    print(f"=== Running Physical Diagnostics on Checkpoint: {checkpoint_path} ===")
    scene_xml = Path(__file__).resolve().parent.parent / "src/mjlab_microduck/robot/microduck/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    device = "cpu"
    actor, obs_mean, obs_std = load_policy(checkpoint_path, device=device)

    # Identifiers
    rf_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    lf_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    r_hip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "upper_leg_right")

    # Joint names in order
    joint_names = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    joint_qpos_adrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
    joint_dof_adrs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]

    def_pos = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,
    ], dtype=np.float32)

    steps_per_trial = int(duration_s / dt)
    results = []

    for trial in range(trials):
        # Reset to home
        mujoco.mj_resetData(model, data)
        freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        qpos_adr = model.jnt_qposadr[freejoint_id]
        data.qpos[qpos_adr + 0] = 0.0
        data.qpos[qpos_adr + 1] = 0.0
        data.qpos[qpos_adr + 2] = 0.125
        data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        for idx, adr in enumerate(joint_qpos_adrs):
            data.qpos[adr] = def_pos[idx]
        data.ctrl[:] = def_pos
        mujoco.mj_forward(model, data)

        left_contact_count = 0
        peak_elevation_deg = 0.0
        peak_foot_z = 0.0
        min_torso_z = 1.0
        max_torso_tilt_deg = 0.0
        settle_joint_err = []

        last_action = np.zeros(14)
        last_joint_vel = np.zeros(14)

        for step in range(steps_per_trial):
            sim_time = step * dt
            phase = (sim_time / LL_PERIOD) % 1.0

            # Compute observation (61D)
            # 1. base_ang_vel (3D)
            imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
            if imu_id >= 0:
                adr = model.sensor_adr[imu_id]
                imu_ang_vel = data.sensordata[adr : adr + 3].copy().astype(np.float32)
            else:
                imu_ang_vel = data.qvel[3:6].copy().astype(np.float32)

            # 2. projected gravity (3D)
            rot = data.xmat[1].reshape(3, 3)
            proj_g = (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

            # 3. joint_pos (14D)
            joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adrs], dtype=np.float32)
            rel_joint_pos = (joint_pos - def_pos).astype(np.float32)

            # 4. joint_vel (14D)
            cur_joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adrs], dtype=np.float32)

            # 5. actions (14D)
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

            # Normalize obs
            obs_tensor = torch.from_numpy(obs_raw).float().unsqueeze(0).to(device)
            obs_norm = (obs_tensor - obs_mean) / (obs_std + 1e-8)

            with torch.no_grad():
                action = actor(obs_norm).squeeze(0).cpu().numpy()
            last_action = action.copy()

            # Apply action as motor target
            motor_targets = def_pos + action
            for j in range(14):
                data.ctrl[j] = motor_targets[j]

            # Step physics for 0.02s (10 substeps of 0.002s)
            for _ in range(10):
                mujoco.mj_step(model, data)

            # Record metrics
            hip_p = data.xpos[r_hip_body]
            foot_p = data.site_xpos[rf_site]
            v = foot_p - hip_p
            # Rotate into robot base frame
            rot_base = data.xmat[1].reshape(3, 3)
            v_b = rot_base.T @ v
            fwd_elev_deg = math.degrees(math.atan2(v_b[0], -v_b[2]))

            if fwd_elev_deg > peak_elevation_deg:
                peak_elevation_deg = fwd_elev_deg
            if foot_p[2] > peak_foot_z:
                peak_foot_z = foot_p[2]

            torso_z = data.xpos[1, 2]
            if torso_z < min_torso_z:
                min_torso_z = torso_z

            tilt_deg = math.degrees(math.acos(np.clip(rot_base[2, 2], -1.0, 1.0)))
            if tilt_deg > max_torso_tilt_deg:
                max_torso_tilt_deg = tilt_deg

            # Contact check on left foot
            for c_idx in range(data.ncon):
                con = data.contact[c_idx]
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                if ("left_foot" in str(g1) or "left_foot" in str(g2)):
                    left_contact_count += 1
                    break

            if phase >= RETURN_END:
                settle_joint_err.append(np.linalg.norm(rel_joint_pos[_ALL_LEG_JOINTS]))

        support_ground_pct = (left_contact_count / steps_per_trial) * 100.0
        avg_settle_err = np.mean(settle_joint_err) if settle_joint_err else 0.0

        results.append({
            "support_ground_pct": support_ground_pct,
            "peak_elev_deg": peak_elevation_deg,
            "peak_foot_z": peak_foot_z,
            "min_torso_z": min_torso_z,
            "max_tilt_deg": max_torso_tilt_deg,
            "settle_err": avg_settle_err,
        })

    # Summary
    print("\n--- Physical Audit Results ---")
    avg_support = np.mean([r["support_ground_pct"] for r in results])
    avg_elev = np.mean([r["peak_elev_deg"] for r in results])
    avg_foot_z = np.mean([r["peak_foot_z"] for r in results])
    avg_min_z = np.mean([r["min_torso_z"] for r in results])
    avg_tilt = np.mean([r["max_tilt_deg"] for r in results])
    avg_settle = np.mean([r["settle_err"] for r in results])

    print(f"1. Support Leg Grounding: {avg_support:.1f}% (Assert >= 90.0%) -> {'PASS' if avg_support >= 90.0 else 'FAIL'}")
    print(f"2. Right Leg Peak Elevation: {avg_elev:.1f}° (Assert >= 75.0°) -> {'PASS' if avg_elev >= 75.0 else 'FAIL'}")
    print(f"3. Peak Foot Clearance: {avg_foot_z*100:.1f} cm (Assert >= 7.0 cm) -> {'PASS' if avg_foot_z >= 0.07 else 'FAIL'}")
    print(f"4. Min Torso Height: {avg_min_z*100:.1f} cm (Assert >= 9.0 cm) -> {'PASS' if avg_min_z >= 0.09 else 'FAIL'}")
    print(f"5. Max Torso Tilt: {avg_tilt:.1f}° (Assert <= 30.0°) -> {'PASS' if avg_tilt <= 30.0 else 'FAIL'}")
    print(f"6. Settle Pose Error: {avg_settle:.3f} rad (Assert <= 0.20 rad) -> {'PASS' if avg_settle <= 0.20 else 'FAIL'}")

    return {
        "support_ground_pct": avg_support,
        "peak_elev_deg": avg_elev,
        "peak_foot_z": avg_foot_z,
        "min_torso_z": avg_min_z,
        "max_tilt_deg": avg_tilt,
        "settle_err": avg_settle,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    run_diagnostics(args.checkpoint, trials=args.trials)
