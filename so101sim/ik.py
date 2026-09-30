"""IK for the SO-101 tool point.

`solve_ik` is position only. The arm has 5 DOF, so a full 6-DOF pose target is overdetermined, and this
leaves wrist orientation to fall where it may.

`solve_topdown` is the one the bin-picking primitives use. Constraining the gripper to point straight
down costs 2 of the 5 DOF and leaves 3, which is one more than the 2 a table position needs, so jaw yaw
is partly controllable. Only partly: the yaw you get at a given point depends on the arm configuration
there, so the solver treats yaw as a soft objective and reports what it achieved.

`solve_side` is the shelf's: the gripper level, jaws horizontal, camera mount on top.

Both run on the MJCF, so there's no URDF frame to disagree with.
"""

import mujoco
import numpy as np

from so101sim.units import JOINT_NAMES

ARM_JOINTS = JOINT_NAMES[:5]

# The gripperframe site's x axis is the direction the gripper points, and its z axis is the direction the
# jaws close along. Measured, not assumed: sweeping wrist_roll rotates site z by exactly the roll angle
# and leaves site x fixed to 0.0000 degrees.
TOOL_AXIS, JAW_AXIS = 0, 2
DOWN = np.array([0.0, 0.0, -1.0])

# A cube ends up between the jaw tips, not at the site. In the site's own frame the midpoint sits
# [+1.4, 0.0, -18.6] mm at the closed stop and [-36.8, 0.0, +33.5] mm fully open, travelling 64.7 mm in
# between, because the moving jaw pivots rather than translating. Targeting the site would miss every
# grasp by a centimetre or more, and by a different amount at every jaw opening.
#
# Cross-checked against the so101 session's independent reading of the same MJCF on 2026-09-22.
JAW_TIP_GEOMS = ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")

# The wrist_roll axis passes through the jaw midpoint when the gripper sits at MJCF 0, to within 0.21 mm.
# Expressed in the site frame, that point is this offset from `gripperframe`, and the site itself is
# 12.00 mm off the roll axis. A tool defined here makes wrist_roll pure yaw with no position coupling,
# which is the right frame for a roll-only reorientation. Grasping does not get to use it: the jaws close
# on a 30 mm cube near MJCF 10, where the midpoint is 6.92 mm off axis, so yaw still drags the grasp
# point around a 13.8 mm circle and `solve_topdown` has to solve position and yaw together.
ROLL_AXIS_TCP = np.array([0.003073, 0.000018, -0.011950])


def solve_ik(model, target, q_init=None, site="gripperframe", tol=1e-4, restarts=8, **kwargs):
    """Damped least-squares IK with random restarts.

    Returns (q, err): the five arm joint angles in radians, clamped to joint limits, and the remaining
    distance to the target in metres.

    The first attempt starts from q_init, or the zero pose. The so101 session found seeding from a folded
    pose gave errors up to 74.6 mm on targets the zero seed reached exactly. Targets behind the base
    still trap the zero seed: pan swings toward the target and stops at its limit. Those get up to
    `restarts` more attempts from seeded random poses, so results are repeatable.
    """
    joint_ids = [model.joint(name).id for name in ARM_JOINTS]
    low, high = model.jnt_range[joint_ids].T
    rng = np.random.default_rng(0)
    seed = np.zeros(5) if q_init is None else np.asarray(q_init, dtype=np.float64)

    best_q, best_err = None, np.inf
    for _ in range(restarts + 1):
        q, err = _solve_from(model, target, seed, site, tol, **kwargs)
        if err < best_err:
            best_q, best_err = q, err
        if best_err < tol:
            break
        seed = rng.uniform(low, high)
    return best_q, best_err


def _solve_from(model, target, q_init, site, tol, max_iters=500, damping=1e-2, max_step=0.2):
    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).id for name in ARM_JOINTS]
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    low, high = model.jnt_range[joint_ids].T
    site_id = model.site(site).id
    target = np.asarray(target, dtype=np.float64)

    q = np.clip(np.array(q_init, dtype=np.float64), low, high)
    jacp = np.zeros((3, model.nv))
    for _ in range(max_iters):
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        err = target - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, dof_adr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        dq *= min(1.0, max_step / np.abs(dq).max())
        q = np.clip(q + dq, low, high)

    data.qpos[qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    return q, float(np.linalg.norm(target - data.site_xpos[site_id]))


def grasp_point(model, data):
    """World position midway between the jaw tips, which is where a grasped object ends up."""
    ids = [model.geom(g).id for g in JAW_TIP_GEOMS]
    return 0.5 * (data.geom_xpos[ids[0]] + data.geom_xpos[ids[1]])


def topdown_frame(yaw):
    """Gripper rotation for pointing straight down with the jaws closing along `yaw`."""
    x = DOWN
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def solve_topdown(model, target, yaw=0.0, gripper=0.0, q_init=None, restarts=5, pos_tol=1e-3,
                  yaw_weight=0.25, max_iters=200, damping=3e-2, max_step=0.15, flips=(0.0, np.pi)):
    """Point the gripper straight down with its jaw midpoint at `target`.

    `yaw` is the direction the jaws close, in radians about world z. A grasp is the same either way
    round, so the solver tries `yaw` and `yaw + pi` and keeps the better. `flips` lists the offsets it
    tries, in order; pass (0.0,) to hold the wrist to one of the two. `gripper` is the gripper joint
    angle in radians, held fixed, because the jaw midpoint shifts as the jaws open.

    Returns (q, info): five arm joint angles in radians, plus `pos_err` in metres and `tilt` and
    `yaw_err` in degrees. Position and pointing down are hard goals. Yaw is scaled by `yaw_weight`, so
    when the arm cannot do both the solver gives up yaw rather than the grasp point.
    """
    target = np.asarray(target, dtype=np.float64)
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = gripper
    rng = np.random.default_rng(0)
    low, high = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS]].T

    best = (None, None, np.inf)
    for cand_yaw in (yaw + f for f in flips):
        seed = (np.array([np.arctan2(target[1], target[0]), -0.6, 1.2, 1.2, 0.0])
                if q_init is None else np.asarray(q_init, dtype=np.float64))
        for _ in range(restarts + 1):
            q, info = _topdown_from(model, data, target, cand_yaw, seed, yaw_weight,
                                    max_iters, damping, max_step)
            score = info["pos_err"] * 1000 + info["tilt"] + yaw_weight * info["yaw_err"]
            if score < best[2]:
                best = (q, info, score)
            if info["pos_err"] < pos_tol and info["tilt"] < 1.0 and info["yaw_err"] < 5.0:
                break
            seed = np.concatenate([[np.arctan2(target[1], target[0])],
                                   rng.uniform(low[1:4], high[1:4]), [rng.uniform(low[4], high[4])]])
        if best[1]["pos_err"] < pos_tol and best[1]["tilt"] < 1.0 and best[1]["yaw_err"] < 5.0:
            break
    return best[0], best[1]


def _topdown_from(model, data, target, yaw, q_init, yaw_weight, max_iters, damping, max_step):
    joint_ids = [model.joint(n).id for n in ARM_JOINTS]
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    low, high = model.jnt_range[joint_ids].T
    site_id = model.site("gripperframe").id
    tips = [model.geom(g).id for g in JAW_TIP_GEOMS]

    quat_des = np.zeros(4)
    mujoco.mju_mat2Quat(quat_des, topdown_frame(yaw).flatten())
    quat_cur, rot_err = np.zeros(4), np.zeros(3)
    j0 = np.zeros((3, model.nv))
    j1 = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    # Index 0 of the local rotation error is spin about the tool axis, which is jaw yaw.
    w = np.array([yaw_weight, 1.0, 1.0])

    q = np.clip(np.array(q_init, dtype=np.float64), low, high)
    for _ in range(max_iters):
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        R = data.site_xmat[site_id].reshape(3, 3)
        e_pos = target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])
        mujoco.mju_mat2Quat(quat_cur, data.site_xmat[site_id])
        mujoco.mju_subQuat(rot_err, quat_des, quat_cur)
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(w * rot_err) < 1e-3:
            break
        mujoco.mj_jacGeom(model, data, j0, None, tips[0])
        mujoco.mj_jacGeom(model, data, j1, None, tips[1])
        mujoco.mj_jacSite(model, data, None, jacr, site_id)
        J = np.vstack([0.5 * (j0 + j1)[:, dof_adr], (R @ np.diag(w) @ R.T @ jacr)[:, dof_adr]])
        e = np.concatenate([e_pos, R @ (w * rot_err)])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
        dq *= min(1.0, max_step / max(np.abs(dq).max(), 1e-12))
        q = np.clip(q + dq, low, high)

    data.qpos[qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    R = data.site_xmat[site_id].reshape(3, 3)
    jaw = R[:, JAW_AXIS]
    yaw_delta = (yaw - np.arctan2(jaw[1], jaw[0]) + np.pi / 2) % np.pi - np.pi / 2
    return q, {
        "pos_err": float(np.linalg.norm(target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]]))),
        "tilt": float(np.degrees(np.arccos(np.clip(R[:, TOOL_AXIS] @ DOWN, -1.0, 1.0)))),
        "yaw_err": float(np.degrees(abs(yaw_delta))),
    }


# Level side grasp, for the shelf. The gripperframe's y axis points straight up, which keeps the tool axis (site x)
# and the jaw axis (site z) horizontal and puts the wrist camera mount on top. Rolled the other way, the mount hangs
# 8 to 9 cm under the gripper and hits a board 6 cm above the base, and the gripper gets only 4.5 cm into the bay
# (measured 2026-09-26). In this roll the moving jaw opens to the robot's right.
SIDE_UP_AXIS = 1
UP = np.array([0.0, 0.0, 1.0])


def pan_axis_xy(model):
    """Where shoulder_pan's vertical axis crosses the base plane, (0.0388, 0) on this MJCF.

    A level gripper can only point along a ray from here: with the tool level, lift, elbow and wrist flex keep the
    arm in one vertical plane, and only pan turns that plane.
    """
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return data.xanchor[model.joint("shoulder_pan").id][:2].copy()


def solve_side(model, target, gripper=0.0, q_init=None, restarts=6, pos_tol=1e-3, max_iters=300,
               damping=3e-2, max_step=0.15):
    """Level gripper, camera mount up, jaw midpoint at `target`.

    Position is 3 constraints and holding the gripperframe's y axis vertical is 2, which spends all 5 arm joints. So
    heading isn't a choice: the tool points along the ray from `pan_axis_xy`, a few degrees off because the jaw
    midpoint sits off the roll axis. `gripper` is the gripper joint in radians, held fixed, because the jaw midpoint
    moves 65 mm across the gripper's range.

    Returns (q, info): five arm angles in radians, plus `pos_err` in metres, `tilt` in degrees between the site's y
    axis and straight up, and `heading` in degrees, the tool axis's azimuth. Retries from seeded random poses, so
    results repeat run to run.
    """
    target = np.asarray(target, dtype=np.float64)
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = gripper
    mujoco.mj_kinematics(model, data)
    pan = data.xanchor[model.joint("shoulder_pan").id][:2].copy()
    phi = np.arctan2(target[1] - pan[1], target[0] - pan[0])
    low, high = model.jnt_range[[model.joint(n).id for n in ARM_JOINTS]].T
    rng = np.random.default_rng(0)
    # Lift + elbow + wrist flex = 0 keeps the tool level in this MJCF's signs, and roll -90 puts the mount up.
    seed = (np.array([phi, 0.0, 1.4, -1.4, -np.pi / 2]) if q_init is None
            else np.asarray(q_init, dtype=np.float64))

    best = (None, None, np.inf)
    for _ in range(restarts + 1):
        q, info = _side_from(model, data, target, seed, max_iters, damping, max_step)
        score = info["pos_err"] * 1000 + info["tilt"]
        if score < best[2]:
            best = (q, info, score)
        if info["pos_err"] < pos_tol and info["tilt"] < 1.0:
            break
        seed = np.concatenate([[phi], rng.uniform(low[1:4], high[1:4]), [-np.pi / 2 + rng.uniform(-0.3, 0.3)]])
    return best[0], best[1]


def _side_from(model, data, target, q_init, max_iters, damping, max_step):
    joint_ids = [model.joint(n).id for n in ARM_JOINTS]
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    low, high = model.jnt_range[joint_ids].T
    site_id = model.site("gripperframe").id
    tips = [model.geom(g).id for g in JAW_TIP_GEOMS]
    j0, j1, jacr = (np.zeros((3, model.nv)) for _ in range(3))

    q = np.clip(np.array(q_init, dtype=np.float64), low, high)
    for _ in range(max_iters):
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        up = data.site_xmat[site_id].reshape(3, 3)[:, SIDE_UP_AXIS]
        e_pos = target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])
        # The angular velocity that turns the site's y axis toward vertical. Its z part is always zero and heading is
        # free, so only the two horizontal rows constrain anything.
        e_rot = np.cross(up, UP)[:2]
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 1e-4:
            break
        mujoco.mj_jacGeom(model, data, j0, None, tips[0])
        mujoco.mj_jacGeom(model, data, j1, None, tips[1])
        mujoco.mj_jacSite(model, data, None, jacr, site_id)
        J = np.vstack([0.5 * (j0 + j1)[:, dof_adr], jacr[:2, dof_adr]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(5), np.concatenate([e_pos, e_rot]))
        dq *= min(1.0, max_step / max(np.abs(dq).max(), 1e-12))
        q = np.clip(q + dq, low, high)

    data.qpos[qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    R = data.site_xmat[site_id].reshape(3, 3)
    tool = R[:, TOOL_AXIS]
    return q, {
        "pos_err": float(np.linalg.norm(target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]]))),
        "tilt": float(np.degrees(np.arccos(np.clip(R[2, SIDE_UP_AXIS], -1.0, 1.0)))),
        "heading": float(np.degrees(np.arctan2(tool[1], tool[0]))),
    }
