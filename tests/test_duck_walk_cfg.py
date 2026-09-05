"""Test configuration and observation layout invariants for MicroDuck Duck Walk task."""

import pytest
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_duck_walk_env_cfg import make_microduck_duck_walk_env_cfg


def test_duck_walk_env_cfg_creation():
    cfg = make_microduck_duck_walk_env_cfg()
    assert cfg is not None
    assert "robot" in cfg.scene.entities
    assert "nan_state" in cfg.terminations
    assert "upright" in cfg.rewards
    assert "waddle_roll" in cfg.rewards
    assert "pose" in cfg.rewards
    assert "head_pose_tracking" in cfg.rewards


def test_duck_walk_obs_action_contract():
    cfg = make_microduck_duck_walk_env_cfg()
    actor_terms = cfg.observations["actor"].terms

    # Check required 61D actor observation terms
    expected_terms = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]
    for term in expected_terms:
        assert term in actor_terms, f"Missing actor observation term: {term}"


def test_duck_walk_registered_task():
    cfg = load_env_cfg("Mjlab-DuckWalk-Flat-MicroDuck")
    assert cfg is not None
    assert cfg.scene.entities["robot"] is not None
