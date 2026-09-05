"""Microduck DUCK WALK environment — adorable waddling bipedal locomotion.

Biomechanical & MDP Design:
  1. Coordinated Waddle Roll: Instead of rigid upright posture (which strictly
     penalizes roll < 0.9°), pitch_upright preserves sagittal balance while
     waddle_roll rewards synchronized lateral sway (rolling toward the stance
     foot as the swing foot steps forward).
  2. Duck Crouch Stance: Lower Center of Mass (CoM) with knees flexed (0.20-0.35 rad)
     and splayed hip roll (+0.04 rad) providing a wide, cute, stable bipedal base.
  3. Dynamic Limit Cycle (Rule 6): Built on continuous alternating foot contacts
     rather than unstable static balance.
  4. Zero Attempt-Tax (Rule 7): Smoothness and angular velocity penalties start
     relaxed during early discovery, ramping via curriculum as the waddle solidifies.
  5. Oscillating Body Segments (Rule 8): Head bobbing occurs tax-free with an EMA
     bias penalty preventing DC droop.
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


def make_microduck_duck_walk_env_cfg(
    play: bool = False,
    waddle_angle_deg: float = 10.0,
    waddle_reward_weight: float = 2.5,
    crouch_knee_rad: float = 0.25,
    hip_splay_rad: float = 0.04,
) -> ManagerBasedRlEnvCfg:
    """Create MicroDuck duck walk environment configuration."""
    cfg = make_microduck_velocity_env_cfg(play=play)

    # 1. Sagittal pitch upright only — free the lateral roll to waddle
    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.pitch_upright,
        weight=2.0,
        params={
            "std": 0.22,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 2. Coordinated lateral waddle roll reward
    cfg.rewards["waddle_roll"] = RewardTermCfg(
        func=microduck_mdp.waddle_roll_reward,
        weight=waddle_reward_weight,
        params={
            "waddle_angle_rad": math.radians(waddle_angle_deg),
            "std_rad": math.radians(6.0),
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 3. Duck crouch posture reward (bent knees & wider hip roll stance)
    cfg.rewards["pose"].func = microduck_mdp.duck_walk_posture
    cfg.rewards["pose"].params["crouch_knee_rad"] = crouch_knee_rad
    cfg.rewards["pose"].params["hip_splay_rad"] = hip_splay_rad
    cfg.rewards["pose"].weight = 1.2

    # Wider tolerances on knees and hips to allow fluid waddling
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_yaw.*": 0.35,
        r".*hip_roll.*": 0.12,  # Allow lateral sway motion
        r".*hip_pitch.*": 0.45,
        r".*knee.*": 0.45,
        r".*ankle.*": 0.30,
    }

    # 4. Zero Attempt-Tax / Delayed Regularization (Rule 7)
    # Relax body_ang_vel so roll angular velocity is not suppressed
    cfg.rewards["body_ang_vel"].weight = -0.02  # was -0.05
    cfg.rewards["angular_momentum"].weight = -0.01  # was -0.02

    # Action rate starts low during discovery
    cfg.rewards["action_rate_l2"].weight = -0.05

    # Action rate curriculum: ramp from -0.05 to -0.20 by iteration 1500
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 500 * 24, "weight": -0.10},
                {"step": 1000 * 24, "weight": -0.15},
                {"step": 1500 * 24, "weight": -0.20},
            ],
        },
    )

    # 5. Locomotion velocity command ranges: focus on cute steady waddle
    cfg.commands["twist"].ranges.lin_vel_x = (0.15, 0.35)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.05, 0.05)
    cfg.commands["twist"].ranges.ang_vel_z = (-0.4, 0.4)

    return cfg


# Runner config template
MicroduckDuckWalkRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    wandb_project="ducky_duck_walk",
    experiment_name="duck_walk",
    run_name="v1_waddle",
    save_interval=200,
    max_iterations=2500,
)
