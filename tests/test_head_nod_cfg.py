"""Test configuration and observation layout invariants for MicroDuck Head Nod task."""

import math
import pytest
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_head_nod_env_cfg import make_microduck_head_nod_env_cfg


def test_head_nod_env_cfg_creation():
    cfg = make_microduck_head_nod_env_cfg()
    assert cfg is not None
    assert "robot" in cfg.scene.entities
    assert "nan_state" in cfg.terminations
    assert "upright" in cfg.rewards
    assert "head_nod_pitch" in cfg.rewards
    assert "head_nod_velocity" in cfg.rewards
    assert "head_nod_alignment" in cfg.rewards
    assert "dual_feet_grounded" in cfg.rewards
    assert "trunk_height" in cfg.rewards
    assert "pose" in cfg.rewards


def test_head_nod_obs_action_contract():
    cfg = make_microduck_head_nod_env_cfg()
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


def test_head_nod_registered_task():
    cfg = load_env_cfg("Mjlab-HeadNod-Flat-MicroDuck")
    assert cfg is not None
    assert cfg.scene.entities["robot"] is not None
    assert cfg.commands["twist"] is not None


def test_head_nod_trajectory_bounds():
    """Verify analytical nod trajectory produces 3 big emphatic nods + 0.50s pause."""
    from mjlab_microduck.tasks.mdp import _natural_head_nod_kinematics

    phases = torch.linspace(0, 1, 2000)
    target_pitch, target_omega = _natural_head_nod_kinematics(phases, period=2.00)

    min_pitch = torch.min(target_pitch).item()
    max_pitch = torch.max(target_pitch).item()

    # Peak nod down should be deeper than -30 degrees (-0.52 rad)
    assert min_pitch <= math.radians(-30.0), f"Expected min pitch <= -30 deg, got {math.degrees(min_pitch)} deg"
    # Rebound should reach horizontal (0 deg)
    assert abs(max_pitch) <= math.radians(0.01), f"Expected max pitch near 0 deg, got {math.degrees(max_pitch)} deg"
    # Stroke range of the nod should be at least 30 degrees
    assert (max_pitch - min_pitch) >= math.radians(30.0)

    # Check each of the 3 big nods and the pause
    w0, w1, w2, w3 = 0.0, 0.25, 0.50, 0.75
    m_1 = (phases >= w0) & (phases < w1)
    m_2 = (phases >= w1) & (phases < w2)
    m_3 = (phases >= w2) & (phases < w3)
    m_pause = phases >= w3

    peak_1 = torch.min(target_pitch[m_1]).item()
    peak_2 = torch.min(target_pitch[m_2]).item()
    peak_3 = torch.min(target_pitch[m_3]).item()

    # All 3 nods are big, deep down-nods (<= -30 degrees)
    assert peak_1 <= math.radians(-30.0), f"Expected peak 1 <= -30 deg, got {math.degrees(peak_1)} deg"
    assert peak_2 <= math.radians(-30.0), f"Expected peak 2 <= -30 deg, got {math.degrees(peak_2)} deg"
    assert peak_3 <= math.radians(-30.0), f"Expected peak 3 <= -30 deg, got {math.degrees(peak_3)} deg"

    # All 3 nods have identical depth within 0.1 deg
    assert abs(peak_1 - peak_2) < math.radians(0.1)
    assert abs(peak_2 - peak_3) < math.radians(0.1)

    # In the pause segment (last 25%), pitch is held identically at 0 deg and omega at 0 rad/s
    pause_pitch_max = torch.max(torch.abs(target_pitch[m_pause])).item()
    pause_omega_max = torch.max(torch.abs(target_omega[m_pause])).item()
    assert pause_pitch_max < 1e-4, f"Expected pause pitch = 0, got {pause_pitch_max}"
    assert pause_omega_max < 1e-4, f"Expected pause omega = 0, got {pause_omega_max}"

