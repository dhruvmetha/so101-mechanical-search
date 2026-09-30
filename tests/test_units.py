import numpy as np
import pytest

from so101sim import units

# Gripper joint range from the vendored MJCF, radians.
GRIPPER = (-0.174533, 1.7453292)


def test_joint_names_are_in_lerobot_order():
    assert units.JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )


def test_arm_joints_convert_radians_to_degrees():
    q = np.array([np.pi / 2, -np.pi / 4, 0.1, -0.2, 1.0, GRIPPER[0]])
    np.testing.assert_allclose(units.to_lerobot(q, GRIPPER)[:5], np.degrees(q[:5]))


@pytest.mark.parametrize("q_gripper, expected", [(GRIPPER[0], 0.0), (GRIPPER[1], 100.0)])
def test_gripper_range_ends_map_to_0_and_100(q_gripper, expected):
    q = np.zeros(6)
    q[5] = q_gripper
    assert units.to_lerobot(q, GRIPPER)[5] == pytest.approx(expected)


def test_round_trip_recovers_radians():
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, size=(100, 6))
    q[:, 5] = rng.uniform(*GRIPPER, size=100)
    for row in q:
        np.testing.assert_allclose(units.to_rad(units.to_lerobot(row, GRIPPER), GRIPPER), row, atol=1e-12)


def test_limits_convert_joint_ranges():
    ranges = np.array([[-1.0, 1.0]] * 5 + [list(GRIPPER)])
    low, high = units.lerobot_limits(ranges)
    np.testing.assert_allclose(low, [-np.degrees(1.0)] * 5 + [0.0])
    np.testing.assert_allclose(high, [np.degrees(1.0)] * 5 + [100.0])
