"""Microduck Leg Lift (Front 90°) task.

Episodic policy: starting from the canonical standing pose (HOME), the robot shifts
its Center of Mass (CoM) over the left foot, lifts the right leg forward until it
reaches 90° (horizontal to the ground), maintains single-leg standing balance at the peak,
smoothly lowers the right leg back down, and returns to the stable 2-feet standing pose.

Phased profile (LL_PERIOD = 4.0 s):
  - [0, LIFT_END)        0.0 -> 1.4 s:  Smooth weight shift & right leg forward lift
  - [LIFT_END, HOLD_END) 1.4 -> 1.8 s:  Hold 90° single-foot standing pose
  - [HOLD_END, RETURN_END) 1.8 -> 3.2 s:  Controlled leg lowering / return
  - [RETURN_END, 1.0)      3.2 -> 4.0 s:  Stable two-feet standing settle (HOME pose)

Obs layout is the unified 61D actor layout (proprioception + phase command + zero-padded
head/body slots) for hot-swappable policy switching.
"""

import math
from copy import deepcopy

# Symmetry — disabled (single-leg asymmetric task).
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to velocity / ground-pick) ───────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_JOINT_DAMPING_RANDOMIZATION   = False
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION     = False

# ── Ranges ───────────────────────────────────────────────────────────────────
COM_RANDOMIZATION_RANGE          = 0.003
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE           = (0.85, 1.15)
KD_RANDOMIZATION_RANGE           = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# ── Phase timings ─────────────────────────────────────────────────────────────
LL_PERIOD   = 4.0
LIFT_END    = 0.35   # 1.4s
HOLD_END    = 0.45   # 1.8s
RETURN_END  = 0.80   # 3.2s

# Joint groupings for 14-servo model:
# 0-4: left leg, 5-8: neck/head, 9-13: right leg
_LEFT_LEG_JOINTS  = [0, 1, 2, 3, 4]
_NECK_JOINTS      = [5, 6, 7, 8]
_RIGHT_LEG_JOINTS = [9, 10, 11, 12, 13]
_ALL_LEG_JOINTS   = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Peak pose overrides for coordinated single-leg balance and Straight Leg Max lift:
# Left leg shifts pelvis over support foot; Right leg flexes to +90° with straight knee
_PEAK_OVERRIDES = {
    1:  0.08,    # left_hip_roll (pelvis shift over support foot)
    2: -0.42,    # left_hip_pitch (support leg compliance)
    4:  0.42,    # left_ankle (support foot planted)
    9:  0.0,     # right_hip_yaw
    10: 0.0873,  # right_hip_roll
    11: 1.5708,  # right_hip_pitch (max forward limit +90°)
    12: 0.0,     # right_knee (straight knee)
    13: 0.0,     # right_ankle (straight neutral sole)
}


def make_microduck_leg_lift_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck Leg Lift (Front 90°) environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    support_foot_ground_cfg = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^left_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: remove walking-specific terms ────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: Leg Lift Objectives ──────────────────────────────────────────

    # Support foot must remain continuously grounded throughout the entire motion
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.leg_lift_support_foot_grounded,
        weight=4.0,
        params={"sensor_name": support_foot_ground_cfg.name, "command_name": "twist"},
    )

    # Support foot flat on the ground (penalize excessive tilt around ankle axis)
    cfg.rewards["support_foot_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["left_foot"]),
        },
    )

    # Lateral CoM shift over the left support foot
    cfg.rewards["com_shift"] = RewardTermCfg(
        func=microduck_mdp.leg_lift_com_over_support_foot_phased,
        weight=4.0,
        params={
            "foot_site_name": "left_foot",
            "std": 0.03,
            "command_name": "twist",
            "lift_end": LIFT_END,
            "hold_end": HOLD_END,
            "return_end": RETURN_END,
        },
    )

    # Trajectory pose tracking for all leg joints (coordinated single-leg support + right leg straight lift)
    cfg.rewards["leg_lift_pose"] = RewardTermCfg(
        func=microduck_mdp.leg_lift_pose_phased,
        weight=5.0,
        params={
            "std": 0.18,
            "command_name": "twist",
            "joint_indices": _ALL_LEG_JOINTS,
            "peak_overrides": _PEAK_OVERRIDES,
            "lift_end": LIFT_END,
            "hold_end": HOLD_END,
            "return_end": RETURN_END,
        },
    )

    # Right leg forward elevation angle tracking (0° -> 59.3° -> 0°) in world gravity frame
    cfg.rewards["leg_lift_elevation"] = RewardTermCfg(
        func=microduck_mdp.leg_lift_elevation_phased,
        weight=4.0,
        params={
            "hip_body_name": "upper_leg_right",
            "foot_site_name": "right_foot",
            "std": 0.15,
            "command_name": "twist",
            "lift_end": LIFT_END,
            "hold_end": HOLD_END,
            "return_end": RETURN_END,
        },
    )

    # Settle phase: both feet grounded
    cfg.rewards["both_feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.leg_lift_both_feet_grounded_phased,
        weight=4.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "return_end": RETURN_END,
        },
    )

    # Settle phase: return all leg joints cleanly to HOME standing pose
    cfg.rewards["settle_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _ALL_LEG_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RETURN_END,
        },
    )

    # Head posture matching HOME (strict lock to prevent heavy head floor dipping)
    cfg.rewards["pose_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.5,
        params={
            "std": 0.15,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # ── Rewards: balance and stability ─────────────────────────────────────────
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.5
    cfg.rewards["upright"].params["std"] = 0.40  # allows natural compliant tilt for dynamic balance over left foot

    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=2.0,
        params={
            "std": 0.03,
            "target_height": 0.115,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # ── Regularisation ────────────────────────────────────────────────────────
    cfg.rewards["action_rate_l2"].weight = -1.0
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-5e-3
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (unified 61D actor layout) ───────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # Sensor delays
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Observation noise
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].nan_policy = "sanitize"
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: cyclic phase encoding ────────────────────────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    cfg.commands["twist"] = microduck_mdp.LegLiftPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.LegLiftPhaseCommand, "period": LL_PERIOD}
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=microduck_mdp.leg_lift_fell_over,
        params={"max_tilt_rad": math.radians(45.0), "min_trunk_height": 0.085, "max_neck_pitch_error": 0.60},
        time_out=False,
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.115, 0.125)

    if ENABLE_VELOCITY_PUSHES:
        interval = (2.0, 4.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # ── Curriculum ────────────────────────────────────────────────────────────
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                    {"step": 2000 * 24, "range": 0.02},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckLegLiftRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="ducky_leg_lift",
    experiment_name="leg_lift",
    run_name="v1_phased_baseline",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
)
