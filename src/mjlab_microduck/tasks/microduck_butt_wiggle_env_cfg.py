"""Microduck Butt-Wiggle ("Preen Shake") environment.

The quintessential duck behavior:
- Planted webbed feet (both feet stay flat and grounded, zero slip).
- High-frequency hip-roll oscillation (~3 Hz).
- Gaze-locked head (vestibulo-ocular reflex: head stays level and motionless forward).
- Multiplicative composite reward prevents compromise basins.
- Strict 61D observation space contract for runtime compatibility.
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = False
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (4.0, 8.0)
VELOCITY_PUSH_RANGE = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 5.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

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
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_butt_wiggle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create environment configuration for Duck Butt-Wiggle ("Preen Shake")."""

    site_names = ["left_foot", "right_foot"]

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

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
    cfg.viewer.entity_name = "robot"
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 0.55
    cfg.viewer.elevation = -12.0
    cfg.viewer.azimuth = 145.0
    cfg.viewer.height = 480
    cfg.viewer.width = 640

    joint_pos_action = cfg.actions["joint_pos"]
    joint_pos_action.scale = {
        ".*hip_roll.*": 1.0,
        ".*neck.*": 1.0,
        ".*head.*": 1.0,
        ".*hip_yaw.*": 0.1,
        ".*hip_pitch.*": 0.1,
        ".*knee.*": 0.1,
        ".*ankle.*": 0.1,
    }

    # === CLEAN REWARDS ===
    # Remove locomotion-specific rewards
    for r in list(cfg.rewards.keys()):
        del cfg.rewards[r]

    # Multiplicative composite goal reward (linear projection quadrature)
    # R_composite = R_upright * R_gaze_lock * R_wiggle
    # Perfectly 1.0 at full 3 Hz oscillation, strictly 0.0 at rest (no compromise basin!)
    cfg.rewards["butt_wiggle_composite"] = RewardTermCfg(
        func=microduck_mdp.butt_wiggle_composite,
        weight=5.0,
        params={
            "command_name": "twist",
            "roll_amplitude": 0.22,
            "gaze_ori_std": 0.25,
            "gaze_vel_std": 1.5,
            "nominal_height": 0.126,
            "height_std": 0.03,
            "pitch_std": 0.15,
            "feet_sensor_name": feet_ground_cfg.name,
        },
    )

    # 5. Linear velocity drift penalty (kills all forward/backward/lateral drift!)
    cfg.rewards["butt_wiggle_lin_vel"] = RewardTermCfg(
        func=microduck_mdp.butt_wiggle_lin_vel_penalty,
        weight=-1.0,
        params={},
    )

    # Anti-jitter on knee/ankle pitches
    cfg.rewards["butt_wiggle_knee_penalty"] = RewardTermCfg(
        func=microduck_mdp.butt_wiggle_knee_pitch_penalty,
        weight=-0.001,
        params={},
    )

    # Penalize lifting feet
    cfg.rewards["butt_wiggle_feet_air"] = RewardTermCfg(
        func=microduck_mdp.butt_wiggle_feet_air_penalty,
        weight=-0.5,
        params={"sensor_name": feet_ground_cfg.name},
    )

    # Foot slip penalty
    cfg.rewards["foot_slip"] = RewardTermCfg(
        func=mdp.feet_slip,
        weight=-0.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        },
    )

    # Joint limits penalty
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limits if hasattr(microduck_mdp, "joint_pos_limits") else mdp.joint_pos_limits,
        weight=-1.0,
        params={},
    )

    # Self collisions penalty
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Action smoothness (curriculum ramps up gently from near-zero to allow high-frequency oscillation)
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.action_rate_l2 if hasattr(microduck_mdp, "action_rate_l2") else mdp.action_rate_l2,
        weight=-0.0002,
        params={},
    )

    # === OBSERVATIONS (Strict 61D layout) ===
    cfg.observations["critic"].terms.pop("foot_height", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.observations["actor"].terms.pop("base_lin_vel", None)

    # 1-ctrl-step lag on joint_vel
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
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

    # === COMMANDS ===
    # Twist slot: ButtWigglePhaseCommand ([cos(2πφ), sin(2πφ), freq])
    command = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = microduck_mdp.ButtWigglePhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.ButtWigglePhaseCommand,
            "freq_range": (2.5, 3.5),
            "default_freq": 3.0,
            "randomize_phase": True,
        }
    )

    # Head pose command: 4D deltas from neutral (kept small so input neurons stay alive)
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.02, 0.02),
            (-0.02, 0.02),
            (-0.02, 0.02),
            (-0.02, 0.02),
        ),
    )

    # Body pose command: 6D deltas from neutral (kept small so input neurons stay alive)
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.01, 0.01),
            (-0.01, 0.01),
            (-0.01, 0.01),
            (-0.02, 0.02),
            (-0.02, 0.02),
            (-0.02, 0.02),
        ),
    )

    # Add head + body commands to observations
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === EVENTS & DOMAIN RANDOMIZATION ===
    cfg.events.pop("push_robot", None)
    cfg.events.pop("base_com", None)
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.8, 1.2)

    cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.124, 0.128)
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

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
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if not play:
        cfg.events["reset_butt_wiggle_rsi"] = EventTermCfg(
            func=microduck_mdp.reset_butt_wiggle_rsi,
            mode="reset",
            params={
                "fraction": 0.30,
                "roll_amplitude": 0.18,
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if play:
        cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

    # === TERMINATIONS ===
    cfg.terminations["time_out"] = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(45.0)},  # 45 degrees envelope prevents falling while allowing dynamic oscillation
        time_out=False,
    )
    cfg.terminations["trunk_fall"] = TerminationTermCfg(
        func=microduck_mdp.trunk_fall,
        params={"min_height": 0.08},
        time_out=False,
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # === CURRICULUM ===
    for c in list(cfg.curriculum.keys()):
        del cfg.curriculum[c]

    # Smoothness curriculum: ramp action rate penalty gently from -0.0002 to -0.001
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.0002},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.0005},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.001},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.010},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.008},
                ],
            },
        )

    # Environment settings
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.episode_length_s = 10.0  # 10s episodes (500 steps)

    return cfg


from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


MicroduckButtWiggleRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=None,
    ),
    wandb_project="ducky_butt_wiggle",
    experiment_name="butt_wiggle",
    run_name="v1_butt_wiggle_preen_shake",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1500,
    clip_actions=1.0,
)

