"""Microduck HEAD NOD environment — passionate, enthusiastic bipedal agreement gesture.

Biomechanical & MDP Design:
  1. Passionate Cadence & Trajectory: Rhythmic nodding at f ≈ 2.22 Hz (T = 0.45 s)
     with an asymmetric, emphatic down-nod (reaching ~ -32°) and energetic rebound
     recovery, expressing wholehearted agreement ("YES! ABSOLUTELY!").
  2. World-Frame Anti-Gaming Invariant (Rule 3): Head pitch is measured via the
     forward camera look vector in the gravity-aligned world frame (θ = atan2(vz, vx)).
     Cheating by leaning the trunk backward causes the world-frame pitch angle to drop,
     preventing base-tilt reward gaming.
  3. ZMP Stability & Continuous Dual-Foot Grounding (Rule 6): Both feet remain
     grounded (100% ground contact) with low horizontal slip velocity. MicroDuck's
     dual-foot support polygon safely accommodates the dynamic pitching moments.
  4. Zero Attempt-Tax / Delayed Regularization (Rule 7): Smoothness and action rate
     penalties start low (-0.05) during initial gesture discovery and ramp up via
     curriculum to ensure clean, jitter-free motor control.
  5. Oscillating Body Segments (Rule 8): Natural cyclic pitch dynamics pass unpenalized,
     while sagittal alignment keeps head yaw and roll centered facing forward.
  6. 61-D Observation Parity: Identical 61-D observation layout for hot-swappable
     policy deployment in robotd.
"""

import dataclasses
import math
from typing import Optional

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

HEAD_NOD_PERIOD = 2.00  # 2.00s sequence: 3 big nods (0.50s each) + 0.50s pause


def make_microduck_head_nod_env_cfg(
    play: bool = False,
    nod_period: float = HEAD_NOD_PERIOD,
) -> ManagerBasedRlEnvCfg:
    """Create MicroDuck natural compound head nodding environment configuration (3 big nods + pause)."""
    cfg = make_microduck_velocity_env_cfg(play=play)

    # 1. Replace twist velocity command with cyclic HeadNodPhaseCommand
    command = cfg.commands["twist"]
    cmd_dict = {
        k: v for k, v in vars(command).items()
        if k != "rel_turn_in_place_envs"
    }
    cfg.commands["twist"] = microduck_mdp.HeadNodPhaseCommandCfg(
        **{
            **cmd_dict,
            "class_type": microduck_mdp.HeadNodPhaseCommand,
            "period": nod_period,
            "randomize_phase": not play,
        }
    )

    # 2. Delete locomotion velocity & obsolete head pose tracking rewards
    # The head is actively driven by head_nod_pitch/velocity/alignment!
    for term in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
    ]:
        cfg.rewards.pop(term, None)

    # 3. Primary Objective: World-Frame Head Nod Pitch Tracking (Anti-Gaming Invariant)
    cfg.rewards["head_nod_pitch"] = RewardTermCfg(
        func=microduck_mdp.head_nod_world_pitch_track,
        weight=5.0,
        params={
            "command_name": "twist",
            "std": 0.18,
            "period": nod_period,
            "head_site_name": "head_camera",
        },
    )

    # 4. Head Pitch Angular Velocity Tracking (Expressive dynamic vigor)
    cfg.rewards["head_nod_velocity"] = RewardTermCfg(
        func=microduck_mdp.head_nod_velocity_track,
        weight=1.5,
        params={
            "command_name": "twist",
            "std": 1.2,
            "period": nod_period,
            "head_body_name": "jaw_soft",
        },
    )

    # 5. Sagittal Alignment: Keep head_yaw and head_roll zeroed (facing directly forward)
    cfg.rewards["head_nod_alignment"] = RewardTermCfg(
        func=microduck_mdp.head_nod_alignment,
        weight=2.5,
        params={"std": 0.22, "fine_std": 0.08, "fine_weight": 0.5},
    )

    # 6. Continuous Dual-Foot Ground Contact (ZMP Ground Stability)
    cfg.rewards["dual_feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.dual_feet_grounded,
        weight=2.5,
        params={"sensor_name": "feet_ground_contact"},
    )

    # 7. Foot Slip Penalty: Keep feet firmly planted without sliding
    cfg.rewards["foot_slip"].weight = -0.5
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.0  # Always active

    # 8. Trunk Upright Posture & Nominal Height
    cfg.rewards["upright"].weight = 2.5
    cfg.rewards["upright"].params["std"] = math.sqrt(0.04)  # ~11.5 deg tolerance

    cfg.rewards["trunk_height"] = RewardTermCfg(
        func=microduck_mdp.trunk_nominal_height,
        weight=1.5,
        params={"target_height": 0.112, "std": 0.02},
    )

    # 9. Leg Stance Posture: Stable bipedal standing base with compliant knees
    cfg.rewards["pose"].weight = 1.5

    # 10. Zero Attempt-Tax / Delayed Regularization (Rule 7)
    # Relax body_ang_vel so small reactive torso pitch oscillations are not penalized to death
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01

    # Remove curricula for removed velocity terms
    for cur_term in ["head_pose_bias_weight", "head_pose_range", "standing_envs"]:
        cfg.curriculum.pop(cur_term, None)

    # Action rate starts low during discovery
    cfg.rewards["action_rate_l2"].weight = -0.05

    # Action rate curriculum: ramp from -0.05 to -0.20 as nodding motion consolidates
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 400 * 24, "weight": -0.10},
                {"step": 800 * 24, "weight": -0.15},
                {"step": 1200 * 24, "weight": -0.20},
            ],
        },
    )

    return cfg


# Runner config
MicroduckHeadNodRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    wandb_project="ducky_head_nod",
    experiment_name="head_nod",
    run_name="v4_three_big_nods_pause",
    save_interval=100,
    max_iterations=2000,
)
