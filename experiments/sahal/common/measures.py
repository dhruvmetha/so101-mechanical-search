"""What the studies measure: grasp conditions, tilt outcomes, contacts, clearance and simulator glitches."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import mujoco
import numpy as np

from so101sim import ik, units
from so101sim.shelf import TARGET, TOPPLE_DEG
from so101sim.shelf_checks import add_clearance
from so101sim.shelf_layout import GRASP_UP

LEAN_DEG = 5.0  # a cylinder tilting more than this leaned; more than TOPPLE_DEG (20) toppled
GRASP_CENTRE_TOLERANCE_M = 0.01  # red centre to jaw midpoint at the end of pick
GRASP_MIN_LIFT_M = 0.005  # red centre above its resting height at the end of pick
GRASP_MAX_TILT_DEG = 5.0
# A run is a simulator glitch, not physics, if the arm-cylinder contact force or the cylinder's speed goes past these.
# Pushes: normal ones peak at 11-26 N and 0.1-0.25 m/s; glitches reached 428 N and 12.9 kN, and 30-42 m/s
# (measured 2026-11-21). Grasps: the closed jaws normally squeeze with up to 137 N, so runs with a grasp use a
# higher force limit; their speed limit is the same.
GLITCH_CONTACT_FORCE_N = 100.0
GLITCH_GRASP_CONTACT_FORCE_N = 1000.0
GLITCH_SPEED_M_S = 1.0
CLEARANCE_SEARCH_M = 0.03
CLEARANCE_TOLERANCE_M = 1e-4


@dataclass(frozen=True)
class Contact:
    """One contact between the red cylinder and another geom during one physics substep."""

    geom: str
    height_above_board_m: float
    normal_force_n: float
    tangential_force_n: float
    slip_speed_m_s: float


def geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    """A geom's name, or `body#id` for the unnamed ones."""
    name = model.geom(geom_id).name
    return name if name else f"{model.body(model.geom_bodyid[geom_id]).name}#{geom_id}"


def contact_part(geom: str) -> str:
    """Which part of the gripper a geom label belongs to: jaw tip sphere, jaw box or mesh, or gripper body."""
    if "sph_tip" in geom:
        return "jaw tip sphere"
    if geom.startswith(("fixed_jaw", "moving_jaw")):
        return "jaw (box or mesh)"
    if geom.startswith("gripper#"):
        return "gripper body"
    return geom.split("#")[0]


def _point_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point: np.ndarray) -> np.ndarray:
    """World-frame linear velocity of `point` fixed to body `body_id`."""
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    return velocity[3:] + np.cross(velocity[:3], point - data.xpos[body_id])


def red_contacts(model: mujoco.MjModel, data: mujoco.MjData, red_geom: int, other_geoms: frozenset[int],
                 board_z: float, with_slip: bool = False) -> list[Contact]:
    """Every current contact between the red cylinder and a geom in `other_geoms`, with its forces.

    `with_slip` also computes the tangential slip speed at each contact, which costs two velocity calls.
    """
    out = []
    force = np.zeros(6)
    red_body = model.geom_bodyid[red_geom]
    for i in range(data.ncon):
        c = data.contact[i]
        other = c.geom2 if c.geom1 == red_geom else c.geom1 if c.geom2 == red_geom else None
        if other is None or other not in other_geoms:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        slip = 0.0
        if with_slip:
            relative = (_point_velocity(model, data, red_body, c.pos)
                        - _point_velocity(model, data, model.geom_bodyid[other], c.pos))
            normal = c.frame[:3]
            slip = float(np.linalg.norm(relative - (relative @ normal) * normal))
        out.append(Contact(geom_label(model, other), float(c.pos[2] - board_z), abs(float(force[0])),
                           float(np.linalg.norm(force[1:3])), slip))
    return out


def is_glitch(max_contact_force_n: float, max_speed_m_s: float, grasping: bool = False) -> bool:
    """True if a run's peak arm-cylinder force or cylinder speed is beyond anything physical here. `grasping` runs
    (the jaws squeeze the cylinder) use GLITCH_GRASP_CONTACT_FORCE_N instead of GLITCH_CONTACT_FORCE_N."""
    force_limit = GLITCH_GRASP_CONTACT_FORCE_N if grasping else GLITCH_CONTACT_FORCE_N
    return max_contact_force_n > force_limit or max_speed_m_s > GLITCH_SPEED_M_S


def tilt_outcome(max_tilt_deg: float, glitch: bool) -> str:
    """glitch, toppled (> 20 deg), leaned (> 5 deg) or slid."""
    if glitch:
        return "glitch"
    if max_tilt_deg > TOPPLE_DEG:
        return "toppled"
    return "leaned" if max_tilt_deg > LEAN_DEG else "slid"


def grasp_conditions(env, refused: str | None, failure: str | None) -> dict[str, bool | float]:
    """The grasp-success conditions at the end of pick, each on its own, plus the numbers behind them.

    Not refused; no failure during pick; both jaws on the red; red centre within 1 cm of the jaw midpoint; red lifted
    more than 5 mm off the board; tilt under 5 degrees.
    """
    inner = env.env
    pose = inner.cylinder_pose(TARGET)
    jaw_midpoint = ik.grasp_point(env.model, env.data)
    resting_z = inner.layout.board_z + GRASP_UP
    distance = float(np.linalg.norm(pose["pos"] - jaw_midpoint))
    lift = float(pose["pos"][2] - resting_z)
    return {
        "g_not_refused": refused is None,
        "g_no_failure": refused is None and failure is None,
        "g_both_jaws": bool(inner.held(TARGET)),
        "g_centred": distance <= GRASP_CENTRE_TOLERANCE_M,
        "g_lifted": lift > GRASP_MIN_LIFT_M,
        "g_upright": pose["tilt"] < GRASP_MAX_TILT_DEG,
        "red_jaw_distance_mm": round(distance * 1000, 1),
        "lift_mm": round(lift * 1000, 1),
        "pick_tilt_deg": round(pose["tilt"], 2),
    }


GRASP_CONDITION_KEYS = ("g_not_refused", "g_no_failure", "g_both_jaws", "g_centred", "g_lifted", "g_upright")
GRASP_CONDITION_NAMES = {"g_both_jaws": "not both jaws", "g_centred": "off-centre > 1 cm",
                         "g_lifted": "not lifted", "g_upright": "tilt >= 5 deg"}


def grasp_succeeded(conditions: dict) -> bool:
    return all(conditions[k] for k in GRASP_CONDITION_KEYS)


def first_failed_grasp_condition(conditions: dict) -> str | None:
    return next((GRASP_CONDITION_NAMES[k] for k in GRASP_CONDITION_NAMES if not conditions[k]), None)


def clearance_to_shelf(env, trajectory: np.ndarray) -> tuple[float, str | None, str | None]:
    """Smallest arm-to-shelf clearance along a lerobot trajectory, kinematic: (metres, arm geom, shelf geom).

    Uses the path check's own definition: the largest margin the shelf and table boxes can grow by
    (shelf_checks.add_clearance) before an arm geom touches them, found by bisection. Contact distances with a margin
    are not used: box-box pairs gave nonsense near the walls (-287 mm), the trap in .claude/rules/arm-and-motion.md.
    """
    inner = env.env

    def first_hit(margin: float) -> tuple[str, str] | None:
        model = add_clearance(copy.deepcopy(env.model), margin)
        data = mujoco.MjData(model)
        data.qpos[:] = env.data.qpos
        for row in trajectory:
            data.qpos[inner._qpos_adr] = units.to_rad(np.asarray(row, dtype=np.float64), inner._gripper_range)
            mujoco.mj_forward(model, data)
            for c in data.contact[: data.ncon]:
                arm, shelf = (c.geom1, c.geom2) if c.geom1 in inner._arm_ids else (c.geom2, c.geom1)
                if arm in inner._arm_ids and shelf in inner._shelf_ids:
                    return geom_label(model, arm), geom_label(model, shelf)
        return None

    touching = first_hit(0.0)
    if touching is not None:
        return 0.0, *touching
    low, high = 0.0, CLEARANCE_SEARCH_M
    hit = first_hit(high)
    if hit is None:
        return float("inf"), None, None
    while high - low > CLEARANCE_TOLERANCE_M:
        middle = 0.5 * (low + high)
        found = first_hit(middle)
        if found is None:
            low = middle
        else:
            high, hit = middle, found
    return low, *hit


def cell_stats(rows: list[dict]) -> dict[str, float]:
    """Topple rate, lean rate and max-tilt mean, median and standard error over the non-glitch rows of one cell."""
    valid = [r for r in rows if not _true(r["glitch"])]
    tilts = np.array([float(r["max_tilt_deg"]) for r in valid])
    n = len(valid)
    return {
        "n_runs": len(rows), "n_glitch": len(rows) - n, "n_valid": n,
        "topple_rate": float(np.mean([_true(r["toppled"]) for r in valid])) if n else float("nan"),
        "lean_rate": float(np.mean(tilts > LEAN_DEG)) if n else float("nan"),
        "mean_max_tilt_deg": float(tilts.mean()) if n else float("nan"),
        "median_max_tilt_deg": float(np.median(tilts)) if n else float("nan"),
        "standard_error_deg": float(tilts.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
    }


def _true(value) -> bool:
    """CSV-safe truth: True, 'True', 1 count; False, 'False', '', None do not."""
    return value is True or value == "True" or value == 1
