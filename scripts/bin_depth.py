"""How deep and how tall-walled a bin the SO-101 can work with straight-down grasps. Kinematics only.

scripts/bin_depth_dynamic.py runs the physics and draws the pictures. Results go to outputs/bin_depth_2026-09-30/ and
the write-up is docs/reports/2026-09-30-bin-depth.md.

    python scripts/bin_depth.py envelope   # reach along y = 0: both wrist rolls, self-clearance, holding, leaning
    python scripts/bin_depth.py lattice    # top-down IK down every column of a 1 cm grid, 5 mm steps in height
    python scripts/bin_depth.py bins       # score candidate bins against the lattice
    python scripts/bin_depth.py all

World z = 0 is the plane the arm's base stands on. A bin is its inner width W, its centre x (always at y = 0), the
height of its floor top, -F, and its wall height w above the floor, so the rim sits at -F + w. Two kinds:

    free    stands on a table at floor height, 4 mm walls, a 12 cm square riser from the table up to the base, the way
            so101sim/search_layout.py builds the current bin.
    flush   sunk into a table whose top is level with the rim, so everything outside the bin opening below the rim is
            solid. The riser, if any, runs from that tabletop up to the base.

A floor position passes when one of the four jaw yaws that close on an axis-aligned cube (0, 90, 180 and 270 degrees:
two jaw axes times two wrist rolls) has a straight-down path from travel height to the grasp where, at every 5 mm:
top-down IK lands within 1 mm and 1 degree with the jaw yaw within 5 degrees; every arm point stays 3 mm or more from
the bin, table and riser with the jaws open to GRIP_OPEN; and the arm stays 10 mm or more from itself, measured the way
demonstrator/reach.Clearance does. Those two margins are the ones behind the 54 of 55 empty-bin picks of 2026-09-30
(demonstrator/tools.py). IK uses the lattice's own continuation down each column, the way primitives._trace seeds
each step from the last.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread per worker; the pool is the parallelism

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "demonstrator"))

import so101sim  # noqa: E402,F401  before mujoco, which reads MUJOCO_GL at import
import mujoco  # noqa: E402

import reach  # noqa: E402  demonstrator/reach.py
from so101sim import SEARCH_SCENE_PATH, ik, primitives, search_layout as sl  # noqa: E402

OUT = REPO / "outputs" / "bin_depth_2026-09-30"
WORKERS = 12

CUBE_HALF = primitives.CUBE_HALF
WALL_T = sl.WALL_THICKNESS
FLOOR_T = sl.FLOOR_THICKNESS
# The riser under the base in so101sim/search_layout.py: 12 cm square, centred under base_collision.
RISER_LO = np.array([sl.RISER_X - sl.RISER_HALF, -sl.RISER_HALF])
RISER_HI = np.array([sl.RISER_X + sl.RISER_HALF, sl.RISER_HALF])

ENV_MARGIN = 0.003  # every arm point to the bin, table and riser
SELF_MARGIN = 0.010  # arm to itself
POS_TOL, TILT_TOL, YAW_TOL = 1e-3, 1.0, 5.0

GRIPS = {"plan": primitives.GRIP_PLAN, "hold": primitives.GRIP_HOLDING}
# The jaw opening the arm actually has while IK is solved at each grip value: open for the descent, closed on a cube
# for the carry.
JAW_AT = {"plan": primitives.GRIP_OPEN, "hold": primitives.GRIP_HOLDING}

# Envelope: along y = 0.
ENV_X = np.round(np.arange(0.0, 0.4501, 0.005), 4)
ENV_Z = np.round(np.arange(0.15, -0.3501, -0.005), 4)
LEANS = (15, 30, 45, -15, -30)  # degrees; positive tips the tool axis back toward the base, negative away from it

# Lattice for the bins.
LAT_X = np.round(np.arange(0.05, 0.3901, 0.01), 3)
LAT_Y = np.round(np.arange(-0.12, 0.1201, 0.01), 3)
LAT_Z = np.round(np.arange(0.10, -0.3001, -0.005), 4)
YAWS = (0.0, 90.0)  # world jaw yaw; each with flip 0 and 1 gives 0, 90, 180, 270

CATS = ("finger", "housing", "mount", "wrist", "forearm", "upper arm", "shoulder")
DISTAL = (0, 1, 2, 3)  # what demonstrator/reach.Clearance calls distal: wrist, gripper, camera mount, moving jaw


# ---- geometry --------------------------------------------------------------------------------------------------------

def _voxel(points, size=0.002):
    """One point per `size` voxel, keeping real surface points rather than voxel centres."""
    _, keep = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(keep)]


def _mesh_points(model, g, spacing):
    mid = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)
    f = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
    pts = [v]
    for a, b, c in v[f]:
        longest = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
        if longest <= spacing:
            continue
        k = int(np.ceil(longest / spacing))
        ij = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], float) / k
        pts.append(a + ij[:, :1] * (b - a) + ij[:, 1:] * (c - a))
    return np.concatenate(pts)


def _box_points(size, spacing):
    axes = [np.linspace(-s, s, max(2, int(np.ceil(2 * s / spacing)) + 1)) for s in size]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return g[(np.abs(np.abs(g) - size) < 1e-9).any(1)]


class ArmCloud:
    """Surface points on every collision geom of the moving arm, each geom tagged with the part it belongs to.

    Boxes get a surface grid and meshes their vertices plus a grid on every face, then each geom is thinned to one
    point per 2 mm voxel, so a point sits within about 2 mm of any spot on the surface. The finger meshes have faces
    60 mm across with vertices only at the corners (demonstrator/notes/d1005.md), which is why faces are sampled.
    Checks first bound each geom by a box or sphere and only measure its points when the bound comes near.
    """

    def __init__(self, model, spacing=0.003):
        self.m = model
        body = {n: model.body(n).id for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                                               "camera_mount", "moving_jaw_so101_v1")}
        by_body = {body["shoulder"]: 6, body["upper_arm"]: 5, body["lower_arm"]: 4, body["wrist"]: 3,
                   body["camera_mount"]: 2, body["moving_jaw_so101_v1"]: 0}
        self.geoms, self.local, self.cat = [], [], []
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            t = model.geom_type[g]
            if b == body["gripper"]:
                # The roll servo box and fixed_jaw_box1 are the housing above the fingers; the rest is the fixed jaw.
                c = 1 if (model.geom(g).name == "fixed_jaw_box1" or
                          (t == mujoco.mjtGeom.mjGEOM_BOX and model.geom(g).name == "")) else 0
            elif b in by_body:
                c = by_body[b]
            else:
                continue
            size = model.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                p = _box_points(size, spacing)
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                p = _mesh_points(model, g, spacing)
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                p = np.vstack([np.zeros(3), size[0] * np.vstack([np.eye(3), -np.eye(3)])])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                zs = np.linspace(-size[1] - size[0], size[1] + size[0], 5)
                p = np.array([[0, 0, z] for z in zs] + [[sx * size[0], sy * size[0], z] for z in zs[1:-1]
                                                         for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))])
            else:
                raise ValueError(f"no sampler for geom type {t}")
            self.geoms.append(g)
            self.local.append(_voxel(p))
            self.cat.append(c)
        lo = np.array([p.min(0) for p in self.local])
        hi = np.array([p.max(0) for p in self.local])
        # Corners of each geom's local bounding box, and a bounding sphere about its centre.
        signs = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        self.corners = lo[:, None, :] + signs[None] * (hi - lo)[:, None, :]
        self.centre = (lo + hi) / 2
        self.radius = np.linalg.norm(hi - lo, axis=1) / 2
        self.cat = np.array(self.cat)
        self.ids = np.array(self.geoms)
        self.distal = [k for k, c in enumerate(self.cat) if c in DISTAL]
        # The proximal boxes Clearance measures against: shoulder, upper arm, and the base's stand-in box.
        prox_bodies = {body["shoulder"], body["upper_arm"], model.body("base_collision").id}
        self.prox = [g for g in range(model.ngeom) if model.geom_bodyid[g] in prox_bodies and model.geom_contype[g]
                     and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]

    def points(self, data, k):
        g = self.geoms[k]
        return self.local[k] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]

    def world(self, data):
        """Every point, and the part each belongs to."""
        return (np.concatenate([self.points(data, k) for k in range(len(self.geoms))]),
                np.concatenate([[c] * len(p) for c, p in zip(self.cat, self.local)]))

    def self_clearance(self, data):
        """Distal points to the proximal boxes, as demonstrator/reach.Clearance.measure's first number.

        Exact below SELF_MARGIN + 2 cm; above that it may return a lower bound.
        """
        R = data.geom_xmat[self.ids].reshape(-1, 3, 3)
        cen = np.einsum("nij,nj->ni", R, self.centre) + data.geom_xpos[self.ids]
        best = np.inf
        for g in self.prox:
            size = self.m.geom_size[g]
            gap = np.linalg.norm(cen - data.geom_xpos[g], axis=1) - self.radius - np.linalg.norm(size)
            Rg = data.geom_xmat[g].reshape(3, 3)
            for k in self.distal:
                if gap[k] > SELF_MARGIN + 0.02:
                    best = min(best, float(gap[k]))
                    continue
                q = np.abs((self.points(data, k) - data.geom_xpos[g]) @ Rg) - size
                d = np.linalg.norm(np.clip(q, 0, None), axis=1) + np.minimum(q.max(1), 0)
                best = min(best, float(d.min()))
        return best

    def env_clearance(self, data, lo, hi, exact_below=ENV_MARGIN + 0.002):
        """Per part, the distance from the arm to the union of axis-aligned boxes lo/hi (B, 3). Negative inside.

        Exact where a part comes within `exact_below`; above that it may be a lower bound, the gap between the geom's
        world bounding box and the nearest obstacle box.
        """
        R = data.geom_xmat[self.ids].reshape(-1, 3, 3)
        corners = np.einsum("nij,nkj->nki", R, self.corners) + data.geom_xpos[self.ids][:, None, :]
        glo, ghi = corners.min(1), corners.max(1)
        sep = np.maximum(np.maximum(lo[None] - ghi[:, None], glo[:, None] - hi[None]), 0.0)  # (G, B, 3)
        gap = np.linalg.norm(sep, axis=2).min(1)
        out = np.full(len(CATS), np.inf)
        for k in range(len(self.geoms)):
            if gap[k] > exact_below:
                v = gap[k]
            else:
                v = aabb_sdf(self.points(data, k), lo, hi).min()
            out[self.cat[k]] = min(out[self.cat[k]], v)
        return out


def aabb_sdf(p, lo, hi):
    """Signed distance from points (N, 3) to the union of axis-aligned boxes lo/hi (B, 3). Negative inside."""
    c, e = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(p[:, None, :] - c[None]) - e[None]
    d = np.linalg.norm(np.maximum(q, 0), axis=2) + np.minimum(q.max(2), 0)
    return d.min(1)


def cat_min(values, cat):
    """Minimum of `values` per part, inf where a part has no points."""
    out = np.full(len(CATS), np.inf)
    np.minimum.at(out, cat, values)
    return out


# ---- bins ------------------------------------------------------------------------------------------------------------

def bin_boxes(kind, W, cx, F, w):
    """Obstacle boxes (lo, hi) for a bin, its table and the riser under the base."""
    h, zf = W / 2, -F
    rim = zf + w
    t = WALL_T
    boxes = [((cx - h - t, -h - t, zf - FLOOR_T), (cx + h + t, h + t, zf))]  # floor
    if kind == "free":
        table_top = zf - FLOOR_T
        boxes += [
            ((sl.TABLE_CENTER_X - sl.TABLE_HALF[0], -sl.TABLE_HALF[1], table_top - 2 * sl.TABLE_HALF[2]),
             (sl.TABLE_CENTER_X + sl.TABLE_HALF[0], sl.TABLE_HALF[1], table_top)),
            ((RISER_LO[0], RISER_LO[1], table_top), (RISER_HI[0], RISER_HI[1], 0.0)),
            ((cx + h, -h - t, zf), (cx + h + t, h + t, rim)),
            ((cx - h - t, -h - t, zf), (cx - h, h + t, rim)),
            ((cx - h - t, h, zf), (cx + h + t, h + t, rim)),
            ((cx - h - t, -h - t, zf), (cx + h + t, -h, rim)),
        ]
    elif kind == "flush":
        big = 1.0
        boxes += [
            ((cx + h, -big, rim - big), (big, big, rim)),
            ((-big, -big, rim - big), (cx - h, big, rim)),
            ((cx - h, h, rim - big), (cx + h, big, rim)),
            ((cx - h, -big, rim - big), (cx + h, -h, rim)),
        ]
        if rim < 0:
            boxes.append(((RISER_LO[0], RISER_LO[1], rim), (RISER_HI[0], RISER_HI[1], 0.0)))
    else:
        raise ValueError(kind)
    lo, hi = (np.array(v, float) for v in zip(*boxes))
    return lo, hi


def feasible(kind, W, cx, F, w):
    """A bin that doesn't overlap the riser, and a flush rim no higher than the base."""
    if cx - W / 2 - WALL_T < RISER_HI[0] - 1e-9:
        return False
    if kind == "flush" and -F + w > 1e-9:
        return False
    return True


# ---- IK columns ------------------------------------------------------------------------------------------------------

_MODEL = None
_CLOUD = None


def _init_worker():
    global _MODEL, _CLOUD
    _MODEL = mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH))
    _CLOUD = ArmCloud(_MODEL)


def solve_tilt(model, target, lean_deg, gripper=0.0, flip=0, q_init=None, restarts=4, iters=300, damping=3e-2,
               max_step=0.15):
    """A leaning grasp: jaw midpoint at `target`, the tool axis tipped `lean_deg` from straight down, jaws level.

    demonstrator/reach.solve_lean asks for a whole frame tipped along the ray from the world origin. On this arm the
    tool axis can only tip inside the arm's own vertical plane, which sits about 2 cm to the side of the pan axis and
    turns a few degrees away from that ray, so a full frame there is out of reach by 7 to 9 mm at a 15 to 30 degree lean
    with the jaws across the plane (measured at (0.20, 0, -0.05)). This solver asks only for what the arm can do: the
    tool axis at `lean_deg` from vertical (positive back toward the base, negative away), the jaw axis level, and the
    position. That is five equations for five joints. `flip` picks which way round the jaws sit, like primitives'.
    Returns (q, pos_err m, tilt_err deg, ok).
    """
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = gripper
    joint_ids = [model.joint(n).id for n in ik.ARM_JOINTS]
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    low, high = model.jnt_range[joint_ids].T
    site = model.site("gripperframe").id
    tips = [model.geom(g).id for g in ik.JAW_TIP_GEOMS]
    pan = ik.pan_axis_xy(model)
    target = np.asarray(target, float)
    u = target[:2] - pan
    u /= np.linalg.norm(u)
    want_tz = -np.cos(np.radians(lean_deg))
    rng = np.random.default_rng(0)
    phi = np.arctan2(u[1], u[0])
    roll = np.pi / 2 * (1 if flip == 0 else -1)
    # In this MJCF the tool points straight down when lift + elbow + wrist flex = 90 degrees, and each extra degree
    # tips it one degree back toward the base. Seeding the wrist that way picks the side of the lean; without it a
    # lean away from the base converged onto the mirror-image lean toward it.
    pitch = np.radians(90.0 + lean_deg)
    seed = (np.array([phi, -0.6, 1.2, pitch + 0.6 - 1.2, roll]) if q_init is None else np.asarray(q_init, float))
    j0, j1, jr = (np.zeros((3, model.nv)) for _ in range(3))
    best = (None, np.inf, np.inf, False)
    for _ in range(restarts + 1):
        q = np.clip(seed.copy(), low, high)
        for _ in range(iters):
            data.qpos[qpos_adr] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            R = data.site_xmat[site].reshape(3, 3)
            t, j = R[:, ik.TOOL_AXIS], R[:, ik.JAW_AXIS]
            e = np.concatenate([target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]]),
                                [want_tz - t[2], -j[2]]])
            if np.linalg.norm(e[:3]) < 1e-4 and np.abs(e[3:]).max() < 1e-4:
                break
            mujoco.mj_jacGeom(model, data, j0, None, tips[0])
            mujoco.mj_jacGeom(model, data, j1, None, tips[1])
            mujoco.mj_jacSite(model, data, None, jr, site)
            w = jr[:, dof_adr]
            J = np.vstack([0.5 * (j0 + j1)[:, dof_adr], t[1] * w[0] - t[0] * w[1], j[1] * w[0] - j[0] * w[1]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(5), e)
            dq *= min(1.0, max_step / max(np.abs(dq).max(), 1e-12))
            q = np.clip(q + dq, low, high)
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        R = data.site_xmat[site].reshape(3, 3)
        t, j = R[:, ik.TOOL_AXIS], R[:, ik.JAW_AXIS]
        pos_err = float(np.linalg.norm(target - 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])))
        tilt_err = abs(float(np.degrees(np.arccos(np.clip(-t[2], -1, 1)))) - abs(lean_deg))
        level = abs(float(np.degrees(np.arcsin(np.clip(j[2], -1, 1)))))
        side = float(t[:2] @ u)
        # Which way the jaws sit: flip 0 has the jaw axis turned +90 degrees from the ray, flip 1 -90.
        jaw_side = float(j[0] * -u[1] + j[1] * u[0])
        ok = (pos_err < POS_TOL and tilt_err < TILT_TOL and level < TILT_TOL
              and (abs(lean_deg) < 1e-9 or np.sign(side) == -np.sign(lean_deg))
              and (jaw_side > 0) == (flip == 0))
        if ok or pos_err * 1000 + tilt_err < best[1] * 1000 + best[2]:
            best = (q.copy(), pos_err, tilt_err, ok)
        if ok:
            return best
        lift, elbow = rng.uniform(low[1:3], high[1:3])
        seed = np.array([phi, lift, elbow, pitch - lift - elbow, roll + rng.uniform(-0.5, 0.5)])
    return best


def _solve(model, xyz, yaw, grip_rad, lean, seed, restarts, flip=0):
    if lean == 0:
        q, info = ik.solve_topdown(model, xyz, yaw=yaw, gripper=grip_rad, q_init=seed, restarts=restarts,
                                   flips=(0.0,))
        good = info["pos_err"] < POS_TOL and info["tilt"] < TILT_TOL and info["yaw_err"] < YAW_TOL
        return q, good, info["pos_err"]
    q, pos_err, _, good = solve_tilt(model, xyz, lean, gripper=grip_rad, flip=flip, q_init=seed, restarts=restarts)
    return q, good, pos_err


def column(task):
    """Solve down one column from the top, each step seeded from the last success, the way primitives._trace does.

    A seeded miss gets a fresh solve with four restarts, as _trace retries. Before the column's first success a fresh
    solve gets one restart, which only matters at the top edge of reach. Three misses in a row after the first
    success end the column. Returns per height: ok, the five joint angles, self clearance in m and, when asked,
    clearance to the riser column in m (a lower bound once past 2 cm).
    """
    x, y, yaw_deg, flip, grip, lean, zs, want_riser = task
    m, cloud = _MODEL, _CLOUD
    data = mujoco.MjData(m)
    adr = m.jnt_qposadr[[m.joint(n).id for n in ik.ARM_JOINTS]]
    gadr = m.jnt_qposadr[m.joint("gripper").id]
    grip_rad = primitives._grip_rad(m, GRIPS[grip])
    jaw_rad = primitives._grip_rad(m, JAW_AT[grip])
    yaw = np.radians(yaw_deg) + np.pi * flip
    n = len(zs)
    ok = np.zeros(n, bool)
    qs = np.full((n, 5), np.nan)
    self_c = np.full(n, np.nan)
    riser_c = np.full(n, np.nan)
    seed, started, misses = None, False, 0
    riser_lo = np.array([[RISER_LO[0], RISER_LO[1], -1.0]])
    riser_hi = np.array([[RISER_HI[0], RISER_HI[1], 0.0]])
    for k, z in enumerate(zs):
        q, good = None, False
        if seed is not None:
            q, good, _ = _solve(m, (x, y, z), yaw, grip_rad, lean, seed, 0, flip)
        if not good:
            q, good, _ = _solve(m, (x, y, z), yaw, grip_rad, lean, None, 4 if started else 1, flip)
        if good:
            ok[k], qs[k], seed, started, misses = True, q, q, True, 0
            data.qpos[adr] = q
            data.qpos[gadr] = jaw_rad
            mujoco.mj_kinematics(m, data)
            self_c[k] = cloud.self_clearance(data)
            if want_riser:
                riser_c[k] = cloud.env_clearance(data, riser_lo, riser_hi, exact_below=0.02).min()
        else:
            misses += 1
            if started and misses >= 3:
                break
    return task[:6], ok, qs, self_c, riser_c


def run_columns(tasks, workers=WORKERS, label=""):
    started = time.time()
    out = {}
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, (key, ok, qs, sc, rc) in enumerate(pool.imap_unordered(column, tasks, chunksize=2)):
            out[key] = (ok, qs, sc, rc)
            if (i + 1) % 200 == 0:
                print(f"  {label} {i + 1}/{len(tasks)} columns, {time.time() - started:.0f} s", flush=True)
    print(f"  {label} {len(tasks)} columns in {time.time() - started:.0f} s", flush=True)
    return out


# ---- envelope --------------------------------------------------------------------------------------------------------

def envelope(workers):
    tasks = []
    for x in ENV_X:
        for yaw in YAWS:
            for flip in (0, 1):
                for grip in GRIPS:
                    tasks.append((float(x), 0.0, yaw, flip, grip, 0, ENV_Z, True))
                # A lean tips the tool in the vertical plane through the base, so the jaws close across that plane
                # (yaw 90 on this line) and both fingertips stay level. It needs a direction to the base, so x > 0.
                for lean in LEANS:
                    if yaw == 90.0 and x > 0.02:
                        tasks.append((float(x), 0.0, yaw, flip, "plan", lean, ENV_Z, True))
    res = run_columns(tasks, workers, "envelope")
    keys = sorted(res)
    np.savez_compressed(OUT / "envelope.npz", x=ENV_X, z=ENV_Z,
                        keys=np.array([json.dumps(k) for k in keys]),
                        ok=np.stack([res[k][0] for k in keys]), q=np.stack([res[k][1] for k in keys]),
                        self_c=np.stack([res[k][2] for k in keys]), riser_c=np.stack([res[k][3] for k in keys]))
    summarize_envelope()


def load_envelope():
    d = np.load(OUT / "envelope.npz")
    keys = [tuple(json.loads(k)) for k in d["keys"]]
    idx = {k: i for i, k in enumerate(keys)}
    return d, idx


def _intervals(xs, mask):
    """Runs of True in mask as (x_first, x_last) pairs."""
    out, start = [], None
    for x, m in zip(xs, mask):
        if m and start is None:
            start = x
        if not m and start is not None:
            out.append((start, prev))
            start = None
        prev = x
    if start is not None:
        out.append((start, prev))
    return out


def envelope_masks(d, idx, grip="plan", lean=0, need_self=True, need_riser=False, all_yaws=False):
    """(n_x, n_z) mask of reachable points along y = 0: any roll, clear of self if asked; for both jaw axes if asked."""
    per_yaw = []
    for yaw in YAWS:
        m = np.zeros((len(d["x"]), len(d["z"])), bool)
        for flip in (0, 1):
            for i, x in enumerate(d["x"]):
                k = idx.get((float(x), 0.0, yaw, flip, grip, lean))
                if k is None:
                    continue
                good = d["ok"][k].copy()
                if need_self:
                    good &= np.nan_to_num(d["self_c"][k], nan=-1) >= SELF_MARGIN
                if need_riser:
                    good &= np.nan_to_num(d["riser_c"][k], nan=-1) >= ENV_MARGIN
                m[i] |= good
        per_yaw.append(m)
    return (per_yaw[0] & per_yaw[1]) if all_yaws else (per_yaw[0] | per_yaw[1])


def summarize_envelope():
    d, idx = load_envelope()
    xs, zs = d["x"], d["z"]
    rows = {}
    variants = {
        "IK only, any roll": dict(need_self=False),
        "one roll only (flip 0)": None,
        "clear of itself, best roll": dict(need_self=True),
        "clear of itself and a riser": dict(need_self=True, need_riser=True),
        "clear, both jaw axes": dict(need_self=True, all_yaws=True),
        "holding a cube, clear": dict(grip="hold", need_self=True),
        "clear, with leans": "lean",
    }
    masks = {}
    for name, kw in variants.items():
        if kw is None:
            m = np.zeros((len(xs), len(zs)), bool)
            for i, x in enumerate(xs):
                for yaw in YAWS:
                    k = idx[(float(x), 0.0, yaw, 0, "plan", 0)]
                    m[i] |= d["ok"][k] & (np.nan_to_num(d["self_c"][k], nan=-1) >= SELF_MARGIN)
        elif kw == "lean":
            m = envelope_masks(d, idx, need_self=True)
            for lean in LEANS:
                m |= envelope_masks(d, idx, lean=lean, need_self=True)
        else:
            m = envelope_masks(d, idx, **kw)
        masks[name] = m
        rows[name] = {f"{z:+.3f}": [(round(a, 3), round(b, 3)) for a, b in _intervals(xs, m[:, j])]
                      for j, z in enumerate(zs)}
    for lean in LEANS:
        m = envelope_masks(d, idx, lean=lean, need_self=True)
        masks[f"lean {lean:+d} deg, clear"] = m
        rows[f"lean {lean:+d} deg, clear"] = {f"{z:+.3f}": [(round(a, 3), round(b, 3)) for a, b in _intervals(xs, m[:, j])]
                                              for j, z in enumerate(zs)}
    (OUT / "envelope_intervals.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(OUT / "envelope_masks.npz", x=xs, z=zs, names=np.array(list(masks)),
                        masks=np.stack([masks[n] for n in masks]))
    # Deepest z with anything, per variant.
    print("x ranges along y = 0 (m from the base origin; the pan axis is at x = 0.0388)")
    head = ["z"] + list(masks)
    print(" | ".join(head))
    for j, z in enumerate(zs):
        if round(z * 1000) % 20:
            continue
        cells = []
        for name in masks:
            iv = _intervals(xs, masks[name][:, j])
            cells.append(" ".join(f"{a:.3f}-{b:.3f}" for a, b in iv) or "none")
        print(f"{z:+.3f} | " + " | ".join(cells))
    for name, m in masks.items():
        any_z = zs[m.any(0)]
        print(f"{name}: deepest {any_z.min() if len(any_z) else None}, highest {any_z.max() if len(any_z) else None}")


# ---- lattice ---------------------------------------------------------------------------------------------------------

HOLD_Z_MIN = -0.20  # carrying only happens at travel height, which stays above this for every candidate


def lattice(workers):
    zs = {"plan": LAT_Z, "hold": LAT_Z[LAT_Z >= HOLD_Z_MIN - 1e-9]}
    tasks = [(float(x), float(y), yaw, flip, grip, 0, zs[grip], False)
             for grip in GRIPS for x in LAT_X for y in LAT_Y for yaw in YAWS for flip in (0, 1)]
    res = run_columns(tasks, workers, "lattice")
    shape = (len(GRIPS), len(LAT_X), len(LAT_Y), len(YAWS), 2, len(LAT_Z))
    ok = np.zeros(shape, bool)
    q = np.full(shape + (5,), np.nan)
    sc = np.full(shape, np.nan)
    for (x, y, yaw, flip, grip, _), (o, qq, s, _) in res.items():
        i = (list(GRIPS).index(grip), int(np.argmin(np.abs(LAT_X - x))), int(np.argmin(np.abs(LAT_Y - y))),
             YAWS.index(yaw), flip)
        n = len(o)
        ok[i][:n], q[i][:n], sc[i][:n] = o, qq, s
    np.savez_compressed(OUT / "lattice.npz", x=LAT_X, y=LAT_Y, z=LAT_Z, ok=ok, q=q, self_c=sc)


# ---- scoring bins ----------------------------------------------------------------------------------------------------

KINDS = ("free", "flush")
WIDTHS = (0.22,)
CENTERS = (0.19, 0.21, 0.23, 0.25)
FLOORS = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24)  # F, floor top this far below the base
# Even walls make a flush bin with its rim level with the base plane (w = F) possible; 5 cm is today's wall.
WALLS = (0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24)

_LAT = None


def _init_scorer():
    global _LAT
    _init_worker()
    d = np.load(OUT / "lattice.npz")
    _LAT = {k: d[k] for k in d.files}


def _zi(z):
    i = int(np.argmin(np.abs(_LAT["z"] - z)))
    if abs(_LAT["z"][i] - z) > 1e-6:
        raise ValueError(f"z {z} is not on the lattice")
    return i


def _check_pose(data, adr, gadr, q, jaw_rad, lo, hi):
    """Clearance of one pose: (worst env clearance per part (7,), env points)."""
    data.qpos[adr] = q
    data.qpos[gadr] = jaw_rad
    mujoco.mj_kinematics(_MODEL, data)
    return _CLOUD.env_clearance(data, lo, hi)


def score(cand):
    """Every floor position of one bin: pass or the reason the best option failed; plus crossing at travel height."""
    kind, W, cx, F, w = cand
    L = _LAT
    m = _MODEL
    data = mujoco.MjData(m)
    adr = m.jnt_qposadr[[m.joint(n).id for n in ik.ARM_JOINTS]]
    gadr = m.jnt_qposadr[m.joint("gripper").id]
    jaw_open = primitives._grip_rad(m, primitives.GRIP_OPEN)
    jaw_hold = primitives._grip_rad(m, primitives.GRIP_HOLDING)
    lo, hi = bin_boxes(kind, W, cx, F, w)
    h, zf = W / 2, -F
    rim = zf + w
    travel = round(sl.travel_z(zf, rim) / 0.005) * 0.005
    k_top, k_bot = _zi(travel), _zi(zf + CUBE_HALF)
    levels = list(range(k_top, k_bot + 1))
    checked = sorted(set(levels[::2]) | {k_bot})  # every 1 cm, and the grasp itself
    xs, ys = L["x"], L["y"]
    reach_x = h - CUBE_HALF + 1e-9
    positions = [(i, j) for i, x in enumerate(xs) for j, y in enumerate(ys)
                 if abs(x - cx) <= reach_x and abs(y) <= reach_x]
    rows = []
    for i, j in positions:
        x, y = float(xs[i]), float(ys[j])
        # Try the rolls in order of self clearance at the grasp, which is cached, and stop at the first that passes.
        opts = [(a, f) for a in range(len(YAWS)) for f in (0, 1)]
        opts.sort(key=lambda o: -np.nan_to_num(L["self_c"][0, i, j, o[0], o[1], k_bot], nan=-1.0))
        passed, fails = None, []
        for a, f in opts:
            ok = L["ok"][0, i, j, a, f]
            if not ok[levels].all():
                fails.append((0, "reach", None, (a, f)))
                continue
            sc = L["self_c"][0, i, j, a, f, levels]
            if np.nanmin(sc) < SELF_MARGIN:
                fails.append((1, "self", float(np.nanmin(sc)), (a, f)))
                continue
            fail = None
            for k in checked[::-1]:  # the grasp first, where most misses happen, then upward every 1 cm
                worst = _check_pose(data, adr, gadr, L["q"][0, i, j, a, f, k], jaw_open, lo, hi)
                if worst.min() < ENV_MARGIN:
                    c = int(np.argmin(worst))
                    fail = (1, CATS[c], float(worst[c]), (a, f))
                    break
            if fail is not None:
                fails.append(fail)
                continue
            passed = ((a, f), float(np.nanmin(sc)))
            break
        # The reported reason is the option that came closest: a collision beats a missed reach, and among collisions
        # the one with the most clearance left.
        best = None if passed else max(fails, key=lambda t: (t[0], -1.0 if t[2] is None else t[2]))
        # Face-to-wall gaps, per side: near (-x, toward the base), far (+x), left (+y), right (-y).
        gaps = {"near": x - (cx - h) - CUBE_HALF, "far": cx + h - x - CUBE_HALF,
                "left": h - y - CUBE_HALF, "right": y + h - CUBE_HALF}
        # Carry: at travel height holding a cube, any option, clear of everything.
        carry = False
        for a in range(len(YAWS)):
            for f in (0, 1):
                if L["ok"][1, i, j, a, f, k_top] and L["self_c"][1, i, j, a, f, k_top] >= SELF_MARGIN:
                    worst = _check_pose(data, adr, gadr, L["q"][1, i, j, a, f, k_top], jaw_hold, lo, hi)
                    if worst.min() >= ENV_MARGIN:
                        carry = True
                        break
            if carry:
                break
        rows.append({"x": x, "y": y, "gaps": {k: round(v, 4) for k, v in gaps.items()},
                     "pass": passed is not None, "option": passed[0] if passed else best[3],
                     "self_mm": round(passed[1] * 1000, 1) if passed else None,
                     "why": None if passed else best[1],
                     "why_mm": None if passed or best[2] is None else round(best[2] * 1000, 1),
                     "carry": carry})
    return cand, {"travel_z": travel, "rim_z": rim, "floor_z": zf, "rows": rows}


def summarize_bin(cand, res):
    kind, W, cx, F, w = cand
    rows = res["rows"]
    n = len(rows)
    npass = sum(r["pass"] for r in rows)
    interior = [r for r in rows if min(r["gaps"].values()) >= 0.02 - 1e-9]
    band = {}
    for side in ("near", "far", "left", "right"):
        others = [s for s in ("near", "far", "left", "right") if s != side]
        # Positions nearest this wall and 2 cm or more from the other walls.
        line = [r for r in rows if all(r["gaps"][o] >= max(0.02 - 1e-9, r["gaps"][side]) for o in others)]
        fails = [r["gaps"][side] for r in line if not r["pass"]]
        # The widest failing gap along this wall, in mm: every position with a bigger gap passes.
        band[side] = round(max(fails) * 1000) if fails else -1
    whys = {}
    for r in rows:
        if not r["pass"]:
            whys[r["why"]] = whys.get(r["why"], 0) + 1
    return {"kind": kind, "W": W, "cx": cx, "F": F, "w": w, "rim_z": round(res["rim_z"], 4),
            "travel_z": round(res["travel_z"], 4), "positions": n, "pass": npass,
            "interior": len(interior), "interior_pass": sum(r["pass"] for r in interior),
            "carry_all": all(r["carry"] for r in rows), "carry_miss": sum(not r["carry"] for r in rows),
            "band_mm": band, "why": whys}


def candidates(kinds=KINDS, widths=WIDTHS, centers=CENTERS, floors=FLOORS, walls=WALLS):
    for kind in kinds:
        for W in widths:
            for cx in centers:
                for F in floors:
                    for w in walls:
                        if feasible(kind, W, float(cx), float(F), w):
                            yield (kind, float(W), float(cx), float(F), float(w))


def bins(workers, cands, tag=""):
    print(f"scoring {len(cands)} bins", flush=True)
    started = time.time()
    summary, detail = [], {}
    with mp.get_context("spawn").Pool(workers, initializer=_init_scorer) as pool:
        for i, (cand, res) in enumerate(pool.imap_unordered(score, cands, chunksize=1)):
            s = summarize_bin(cand, res)
            summary.append(s)
            detail[json.dumps(cand)] = res
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(cands)} bins, {time.time() - started:.0f} s", flush=True)
    summary.sort(key=lambda s: (s["kind"], s["W"], s["cx"], s["F"], s["w"]))
    (OUT / f"bins_summary{tag}.json").write_text(json.dumps(summary, indent=0))
    (OUT / f"bins_detail{tag}.json").write_text(json.dumps(detail))
    print(f"scored {len(cands)} bins in {time.time() - started:.0f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("envelope", "lattice", "bins", "summary", "all"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--kinds", nargs="+", default=KINDS)
    parser.add_argument("--widths", nargs="+", type=float, default=WIDTHS)
    parser.add_argument("--centers", nargs="+", type=float, default=CENTERS)
    parser.add_argument("--floors", nargs="+", type=float, default=FLOORS)
    parser.add_argument("--walls", nargs="+", type=float, default=WALLS)
    parser.add_argument("--tag", default="", help="suffix for the bins_summary/bins_detail files")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("envelope", "all"):
        envelope(args.workers)
    if args.stage == "summary":
        summarize_envelope()
    if args.stage in ("lattice", "all"):
        lattice(args.workers)
    if args.stage in ("bins", "all"):
        cands = list(candidates(args.kinds, args.widths, args.centers, args.floors, args.walls))
        bins(args.workers, cands, args.tag)


if __name__ == "__main__":
    main()
