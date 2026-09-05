from mjlab_microduck.tasks.microduck_leg_lift_env_cfg import (
    make_microduck_leg_lift_env_cfg,
    LL_PERIOD,
    LIFT_END,
    HOLD_END,
    RETURN_END,
)
from mjlab_microduck.tasks.mdp import LegLiftPhaseCommand


def test_leg_lift_cfg_rewards():
    """Verify Leg Lift task reward configuration."""
    cfg = make_microduck_leg_lift_env_cfg()
    r = cfg.rewards

    # Support foot grounded & flat
    assert "support_foot_grounded" in r
    assert r["support_foot_grounded"].weight == 4.0
    assert "support_foot_flat" in r
    assert r["support_foot_flat"].weight == -0.5
    assert "com_shift" in r
    assert r["com_shift"].weight == 4.0

    # Leg lift pose and elevation tracking
    assert "leg_lift_pose" in r
    assert r["leg_lift_pose"].weight == 5.0
    assert "leg_lift_elevation" in r
    assert r["leg_lift_elevation"].weight == 4.0

    # Settle phase dual foot contact and standing pose
    assert "both_feet_grounded" in r
    assert r["both_feet_grounded"].weight == 4.0
    assert "settle_pose_legs" in r
    assert r["settle_pose_legs"].weight == 6.0

    # Posture and stability
    assert "upright" in r
    assert r["upright"].weight == 1.5
    assert "height_stand" in r
    assert r["height_stand"].weight == 2.0
    assert "pose_neck" in r
    assert r["pose_neck"].weight == 2.5


def test_leg_lift_cfg_command_is_phase():
    """Verify command generator is LegLiftPhaseCommand with 4.0s period."""
    cfg = make_microduck_leg_lift_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is LegLiftPhaseCommand
    assert cmd.period == LL_PERIOD


def test_leg_lift_play_and_rough_variants():
    """Verify play and rough variants build cleanly."""
    cfg_play = make_microduck_leg_lift_env_cfg(play=True)
    assert "leg_lift_pose" in cfg_play.rewards

    cfg_rough = make_microduck_leg_lift_env_cfg(rough=True)
    assert "leg_lift_pose" in cfg_rough.rewards
