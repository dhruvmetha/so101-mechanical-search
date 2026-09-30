"""Convert between lerobot joint units and MJCF radians.

lerobot's SO-101 follower with use_degrees=True reports the five arm joints in degrees and the gripper
as 0 (closed) to 100 (open). On the real arm, degree zero is the midpoint of the calibration sweep. Here
there is no calibration, so zero is the model's zero and the gripper spans the MJCF joint range.
"""

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def to_lerobot(q, gripper_range):
    """MJCF radians, shape (..., 6), to lerobot units."""
    q = np.asarray(q, dtype=np.float64)
    lo, hi = gripper_range
    out = np.degrees(q)
    out[..., 5] = (q[..., 5] - lo) / (hi - lo) * 100.0
    return out


def to_rad(x, gripper_range):
    """lerobot units, shape (..., 6), to MJCF radians."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = gripper_range
    out = np.radians(x)
    out[..., 5] = lo + x[..., 5] / 100.0 * (hi - lo)
    return out


def lerobot_limits(joint_ranges):
    """Joint ranges in radians, shape (6, 2), to (low, high) in lerobot units."""
    joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
    gripper_range = joint_ranges[5]
    return to_lerobot(joint_ranges[:, 0], gripper_range), to_lerobot(joint_ranges[:, 1], gripper_range)
