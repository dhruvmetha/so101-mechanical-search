"""Grasps that lean back toward the base, for cubes too close to the base for a straight-down gripper.

Closer than about 0.15 m to the base, pointing the gripper straight down folds the arm until the wrist camera mount
hits the shoulder (e401, and the empty-bin calibration of 2026-09-30). Leaning the tool axis back toward the base
keeps the wrist farther out, so the arm folds less. This module solves IK for a lean angle and reports what a pose
collides with: the arm itself, the bin walls, and the seen scene.
"""
import numpy as np

import mujoco
from so101sim import ik

ARM = ik.ARM_JOINTS


def lean_frame(target_xy, yaw, lean):
    """Desired gripperframe rotation: straight down at `yaw`, then tipped by `lean` radians toward the base.

    The tip-over axis is horizontal and perpendicular to the ray from the base, so the tool axis swings from straight
    down to down-and-back along that ray.
    """
    r = np.array([target_xy[0], target_xy[1], 0.0])
    r /= np.linalg.norm(r)
    axis = np.array([-r[1], r[0], 0.0])  # horizontal, perpendicular to the ray
    down = ik.topdown_frame(yaw)
    tip = np.zeros(9)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, lean)
    mujoco.mju_quat2Mat(tip, quat)
    tip = tip.reshape(3, 3)
    frame = tip @ down
    # Pick the sign that makes the tool axis point back toward the base (negative along the ray).
    if frame[:, ik.TOOL_AXIS] @ r > 0:
        mujoco.mju_axisAngle2Quat(quat, axis, -lean)
        mujoco.mju_quat2Mat(tip.reshape(-1), quat)
        frame = tip.reshape(3, 3) @ down
    return frame


def solve_lean(model, target, yaw, lean, gripper=0.0, q_init=None, restarts=6, iters=300, damping=3e-2,
               max_step=0.15, flips=(0.0, np.pi)):
    """Jaw midpoint at `target` with the gripperframe at lean_frame. Returns (q, pos_err m, rot_err deg).

    `flips` are the yaw offsets tried in order. Yaw and yaw + pi close the jaws on the same faces but put the wrist
    camera mount on opposite sides, which decides whether the arm hits its own shoulder near the base.
    """
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = gripper
    joint_ids = [model.joint(n).id for n in ARM]
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    low, high = model.jnt_range[joint_ids].T
    site = model.site("gripperframe").id
    tips = [model.geom(g).id for g in ik.JAW_TIP_GEOMS]
    rng = np.random.default_rng(0)
    target = np.asarray(target, float)
    best = (None, np.inf, np.inf)
    for cand in (yaw + f for f in flips):
        R_des = lean_frame(target[:2], cand, lean)
        q_des = np.zeros(4)
        mujoco.mju_mat2Quat(q_des, R_des.flatten())
        seed = (np.array([np.arctan2(target[1], target[0]), -0.6, 1.2, 1.2, 0.0]) if q_init is None
                else np.asarray(q_init, float))
        for _ in range(restarts + 1):
            q = np.clip(seed.copy(), low, high)
            j0, j1, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv)), np.zeros((3, model.nv))
            q_cur, rot_err = np.zeros(4), np.zeros(3)
            for _ in range(iters):
                data.qpos[qpos_adr] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                R = data.site_xmat[site].reshape(3, 3)
                e_pos = target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])
                mujoco.mju_mat2Quat(q_cur, data.site_xmat[site])
                mujoco.mju_subQuat(rot_err, q_des, q_cur)
                if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                    break
                mujoco.mj_jacGeom(model, data, j0, None, tips[0])
                mujoco.mj_jacGeom(model, data, j1, None, tips[1])
                mujoco.mj_jacSite(model, data, None, jr, site)
                J = np.vstack([0.5 * (j0 + j1)[:, dof_adr], jr[:, dof_adr]])
                e = np.concatenate([e_pos, R @ rot_err])
                dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
                dq *= min(1.0, max_step / max(np.abs(dq).max(), 1e-12))
                q = np.clip(q + dq, low, high)
            data.qpos[qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            pos_err = float(np.linalg.norm(target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])))
            mujoco.mju_mat2Quat(q_cur, data.site_xmat[site])
            mujoco.mju_subQuat(rot_err, q_des, q_cur)
            rot = float(np.degrees(np.linalg.norm(rot_err)))
            if pos_err * 1000 + rot < best[1] * 1000 + best[2]:
                best = (q.copy(), pos_err, rot)
            if pos_err < 1e-3 and rot < 1.0:
                return best
            seed = np.concatenate([[np.arctan2(target[1], target[0])], rng.uniform(low[1:4], high[1:4]),
                                   [rng.uniform(low[4], high[4])]])
    return best


def _box_surface(n=7):
    """Unit-box surface grid in [-1, 1]^3: points with at least one coordinate at +-1."""
    lin = np.linspace(-1.0, 1.0, n)
    g = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    return g[np.abs(g).max(1) > 0.999]


class Clearance:
    """Signed distances from the moving end of the arm to the arm's own base links and to the bin walls.

    mj_geomDistance can't be used for this: for a box against a box it returned 0 mm for a pair 30 mm apart
    (2026-09-30). So the distal geoms are sampled as surface points (box faces, mesh vertices) and measured against
    the analytic distance field of each proximal box. Surface sampling at 7 per edge puts the error under 3 mm.
    """
    DISTAL = ("wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")
    PROXIMAL = ("shoulder", "upper_arm", "base_collision")
    WALLS = ("bin_wall_px", "bin_wall_nx", "bin_wall_py", "bin_wall_ny")

    def __init__(self, model):
        self.m = model
        ids = [model.body(b).id for b in self.DISTAL]
        self.distal = [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids and model.geom_contype[g]
                       and model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH)]
        ids = [model.body(b).id for b in self.PROXIMAL]
        self.prox = [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids and model.geom_contype[g]
                     and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        self.walls = [model.geom(n).id for n in self.WALLS]
        unit = _box_surface()
        self.local = {}
        for g in self.distal:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                self.local[g] = unit * model.geom_size[g]
            else:
                mid = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                self.local[g] = model.mesh_vert[a:a + n].astype(float)

    def points(self, data):
        """World points on every distal geom, and which geom each came from."""
        pts, owner = [], []
        for g, p in self.local.items():
            pts.append(data.geom_xpos[g] + p @ data.geom_xmat[g].reshape(3, 3).T)
            owner += [g] * len(p)
        return np.concatenate(pts), np.array(owner)

    def _sdf(self, data, pts, g):
        R = data.geom_xmat[g].reshape(3, 3)
        q = np.abs((pts - data.geom_xpos[g]) @ R) - self.m.geom_size[g]
        return np.linalg.norm(np.clip(q, 0, None), axis=1) + np.minimum(q.max(1), 0)

    def measure(self, data):
        """(self clearance m, wall clearance m, name of the worst self pair). Negative means penetration."""
        pts, owner = self.points(data)
        worst_self, pair = np.inf, ""
        for g in self.prox:
            dist = self._sdf(data, pts, g)
            k = int(np.argmin(dist))
            if dist[k] < worst_self:
                a = owner[k]
                worst_self = float(dist[k])
                pair = f"{self.m.geom(a).name or self.m.body(self.m.geom_bodyid[a]).name} vs " \
                       f"{self.m.geom(g).name or self.m.body(self.m.geom_bodyid[g]).name}"
        worst_wall = min(float(self._sdf(data, pts, g).min()) for g in self.walls)
        return worst_self, worst_wall, pair
