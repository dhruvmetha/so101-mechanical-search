"""The demonstrator's tools for the bin task. Claude plays an episode by calling these from the shell.

    python demonstrator/tools.py new EP SEED buried|partial
    python demonstrator/tools.py observe EP            park the arm, look with both cameras, list what's seen
    python demonstrator/tools.py check EP KIND X Y Z DEG [STROKE]
    python demonstrator/tools.py act EP KIND X Y Z DEG [STROKE] "why"
    python demonstrator/tools.py status EP
    python demonstrator/tools.py undo EP "why"       take back the last act if it spilled; nothing else
    python demonstrator/tools.py list                every episode and how it ended
    python demonstrator/tools.py places EP [X Y]     checked spots to set a held cube down, away from X Y or the red

An episode is a JSON file in demonstrator/episodes/ listing every operation since reset. Each command rebuilds the
scene by replaying it, which is exact (checked 2026-09-29, bit-identical across processes). KIND is pick, place,
push or sweep. A push is a 5 cm stroke with the jaws closed; a sweep is the same motion with a STROKE in metres you
choose, 0.03 to 0.18, and may start up to 5 mm from a wall and up to 3 cm above the rim. X Y Z are metres in the world
frame. DEG is jaw yaw for pick and place, heading for push and sweep. Only act counts as a move. Parking costs robot
time but no move.

Perception (perception.py): the front camera gives colour, depth and masks, the wrist camera colour and masks only.
check reports, for a pick, the cubes each finger would pass through and the target's clearance; for a push or sweep,
what it would move and the risks. act refuses the picks that calibration showed almost always miss (see SHIFT_MAX_MM
and HIT_REFUSE_MM) and the paths where the arm hits itself; other contact is the demonstrator's call.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import so101sim  # noqa: E402,F401  before mujoco
import mujoco  # noqa: E402
from scipy import ndimage  # noqa: E402
from so101sim import primitives, rest, units  # noqa: E402
from so101sim.mechsearch import TARGET, MechSearchPrimitiveEnv  # noqa: E402

from perception import CUBE, HALF, Camera, box_pose, clean_mask, top_face_pose, yaw_and_tilt  # noqa: E402
from reach import Clearance  # noqa: E402

EPISODES = HERE / "episodes"
# Hidden state the players must not read: each episode's friction and density, kept out of the episode file (b5000's
# player saw them there by accident, 2026-09-30). Replays read them from here.
HIDDEN = EPISODES / ".hidden"
CELL = 0.002
MAX_MOVES = 15
MAX_UNDOS = 8
# The chance that a missed pick gets reset (dm1487, 2026-09-30). It sets how often the demonstrations miss: reset every
# miss and the arm never misses, reset none and every miss stays and demonstrations run longer. dm1487 chose 0.1; each
# record stores the value it was played with.
MISS_RESET_EPS = 0.1
TRUST = 0.6  # top-face coverage below this: the pose is only a rough guess
MARGIN = 0.003  # a finger point this far below a seen surface counts as a hit
# Empty-bin picks near the base, 2026-09-30, 182 runs. With the jaws 3 mm or more from every wall, a planned arm-to-itself
# gap of 10 mm or more tracked to 6 mm in 54 of 55 runs, 0 to 10 mm in 14 of 16, an overlap up to 10 mm in 7 of 14, and
# a deeper overlap in none of 17. Jaws within 3 mm of a wall tracked in 8 of 80, whatever the wrist did.
SELF_JAM_MM = -10.0
SELF_TIGHT_MM = 0.0
WALL_GAP_MM = 3.0
# Miss study, 2026-09-30 (outputs/miss_study_2026-09-30): all 10 picks aimed more than 4 mm off a cube's centre toward
# the moving finger missed, because the fixed finger came down on the cube's own top; no hit was shifted more than
# 3.4 mm. act refuses a shift over SHIFT_MAX_MM when the target's pose is trusted.
SHIFT_MAX_MM = 3.0
# Pick calibration under the stiff contacts, 2026-09-30 (342 picks on 24 fresh piles, outputs/pick_calibration_2026-09-30):
# with a finger 10 mm or more into a neighbour the target lifted in 33 of 209 picks, since a stiff neighbour stops the
# finger and the jaws close above the target (the old soft contacts let the finger press it down). Picks that pass
# every refusal lifted 50 of 60 (83%), 49 of 57 with unseen space under 40 mm. act refuses HIT_REFUSE_MM and SELF_JAM_MM.
HIT_REFUSE_MM = 10.0
# Push calibration, 2026-09-30 (384 strokes on 24 fresh piles, outputs/push_calibration_2026-09-30/round2): a stroke that
# carries a cube riding at layer 2 or higher to within PUSH_WALL_MM of the wall ahead spilled 25 times in 149 (17%);
# the rest spilled 10 in 235 (4.3%). act refuses the first kind. Held out on 24 new piles (round3): refused 30 in 164
# (18%), allowed 10 in 220 (4.5%), and among allowed strokes unseen space 30 mm or deeper in the path spilled 9 in 165
# (5.5%), under 30 mm 1 in 55. That stays a warning, since refusing it would remove most pushes.
PUSH_WALL_MM = 15.0
PUSH_UNSEEN_MM = 30.0


def short(name):
    return "RED" if name == TARGET else "g" + name.rsplit("_", 1)[1]


class Episode:
    def __init__(self, name, size=64, on_advance=None, rec=None):
        """Rebuild episode `name` by replaying it. `size` is the camera image size, which doesn't touch the physics.
        `on_advance(episode, row)` runs before every 30 Hz frame after reset; export.py records demos with it.
        `rec` supplies the record directly instead of reading the episode file (the planner keeps records in memory
        until it saves them)."""
        self.path = EPISODES / f"{name}.json"
        self.name = name
        self.rec = json.loads(self.path.read_text()) if rec is None else rec
        self.prim = MechSearchPrimitiveEnv(height=size, width=size)
        self.env = self.prim.env
        self.m, self.d = self.env.model, self.env.data
        self.bin = self.env.bin
        self.front = Camera(self.m, "front")
        # The wrist camera sees the pile from the base side, the opposite way to the front camera, so the two views
        # cover each other's blind spots. Parked, it costs no motion (2026-09-30).
        self.wrist = Camera(self.m, "wrist_cam")
        self.geom_name = {self.m.geom(n).id: n for n in self.env.cubes}
        self.adr = self.m.jnt_qposadr[[self.m.joint(n).id for n in units.JOINT_NAMES]]
        self.grip_range = tuple(self.m.jnt_range[self.m.joint("gripper").id])
        self.finger_bodies = {self.m.body(b).id for b in ("gripper", "moving_jaw_so101_v1")}
        # The wrist camera mount must not touch the scene either (dm1487, 2026-09-30). Its two collision boxes are
        # sampled at corners, edge midpoints and face centres and tested like the fingers.
        mount = self.m.body("camera_mount").id
        self.mount_boxes = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == mount and self.m.geom_contype[g]]
        k = np.array([-1.0, 0.0, 1.0])
        self.box_unit = np.array([[a, b, c] for a in k for b in k for c in k])
        self.probe = mujoco.MjData(self.m)
        # Near the base, one of the two wrist rolls for a grasp folds the arm until the wrist camera mount hits the
        # shoulder (e401). check measures how close the arm comes to itself along the whole path, for both rolls.
        self.clearance = Clearance(self.m)
        self.robot_s = 0.0
        # The episode's friction and density (hidden state, drawn at reset from the seed) are stored the first time, in
        # HIDDEN rather than the episode file, and replays use the stored values, so a change to the drawing code can't
        # break old episodes.
        hidden = HIDDEN / f"{name}.json"
        physics = json.loads(hidden.read_text()) if hidden.exists() else self.rec.pop("physics", None)
        options = {"buried": self.rec["buried"]}
        if physics is not None:
            options["physics"] = physics
        self.prim.reset(seed=self.rec["seed"], options=options)
        self.physics = dict(self.env.physics)
        if on_advance is not None:
            advance = self.env.advance

            def recorded(row):
                on_advance(self, row)
                return advance(row)

            self.env.advance = recorded
        self.last = None
        self.op_index = -1
        for k, op in enumerate(self.rec["ops"]):
            self.op_index = k
            self._apply(op)

    def close(self):
        self.front.close()
        self.wrist.close()
        self.env.close()

    def save(self):
        self.path.write_text(json.dumps(self.rec, indent=1))
        HIDDEN.mkdir(exist_ok=True)
        (HIDDEN / f"{self.name}.json").write_text(json.dumps(self.physics))

    # ---- operations -----------------------------------------------------------------------------

    def _settle(self):
        hold = self.env.joint_state.copy()

        def frame():
            self.env.advance(hold)
            self.robot_s += 1 / 30

        return rest.come_to_rest(frame, self.d, rest.free_dofs(self.m, self.env.cubes))

    def _apply(self, op):
        if op["op"] == "park":
            home = np.asarray(self.env.layout.home, dtype=np.float32).copy()
            # Home opens the gripper. Parks recorded before 2026-09-29 did that even while holding and dropped the
            # red in e400; newer parks carry keep_grip and keep squeezing a held cube.
            if op.get("keep_grip") and 14 < float(self.env.joint_state[5]) < 30:
                home[5] = primitives.GRIP_CLOSE
            for row in primitives.lead_in(self.env.joint_state, home[None, :]):
                self.env.advance(row)
                self.robot_s += 1 / 30
            self._settle()
            self.last = None
            return None
        if op.get("unpark") and self.last is None:
            self._unpark()
        # Acts recorded before 2026-09-30 carry no flip and let IK choose the wrist roll, as they did then.
        action = {"kind": op["kind"], "xyz": op["xyz"], "angle": float(np.radians(op["deg"])), "flip": op.get("flip"),
                  "lift_z": op.get("lift_z"), "stroke": op.get("stroke")}
        _, reward, term, trunc, info = self.prim.step(action)
        self.robot_s += info.get("frames", 0) / 30
        self._settle()
        self.last = {"grip": float(self.env.joint_state[5]), "success": bool(info.get("is_success")),
                     "failure": info.get("failure"), "refused": info.get("refused"), "terminated": bool(term)}
        return self.last

    def _unpark_plan(self):
        c = self.bin["center"]
        grip = float(self.env.joint_state[5])
        wp = primitives._Waypoint([c[0], c[1], self.prim.travel_z], 0.0, grip, hold=5)
        arms, grips = primitives._trace(self.m, [wp], primitives.GRIP_PLAN)
        return primitives._retime(arms, grips)

    def _unpark(self):
        """From the park pose, go to a straight-down hover over the bin's middle with the wrist unrotated.

        At home the wrist camera mount (camera_box2) rests on the shoulder. A move straight from home that needs a
        big wrist roll grinds the mount on the shoulder, the roll lags up to 93 deg, and with the jaws open that
        throws the grasp point about 4.5 cm (e401 move 2). From this hover, the primitive's own crossing turns the
        wrist with the arm stretched out, where nothing collides. Acts recorded before 2026-09-30 lack the unpark
        flag and replay without this.
        """
        for row in primitives.lead_in(self.env.joint_state, self._unpark_plan()):
            self.env.advance(row)
            self.robot_s += 1 / 30

    def moves(self):
        return sum(1 for op in self.rec["ops"] if op["op"] == "act")

    def outcome(self):
        """'solved', 'failed: why', 'out of moves' or None while still going."""
        acts = [op for op in self.rec["ops"] if op["op"] == "act"]
        if acts and acts[-1]["result"]["success"]:
            return "solved"
        if acts and acts[-1]["result"]["terminated"]:
            return f"failed: {acts[-1]['result']['failure']}"
        if len(acts) >= MAX_MOVES:
            return "out of moves"
        return None

    def parked(self):
        return bool(self.rec["ops"]) and self.rec["ops"][-1]["op"] == "park"

    # ---- seeing ---------------------------------------------------------------------------------

    def see(self):
        """Poses of seen cubes, and a height map with the cube that owns each cell's top surface.

        The front camera gives colour, depth and exact masks. The wrist camera gives colour and exact masks only, as
        the real one has no depth (dm1487, 2026-09-30), so its pixels reach 3D only through something known. Floor
        pixels land on the bin floor's plane and mark that floor seen and empty. A cube or red pixel the front camera
        doesn't place is followed along its ray to where it first meets the front camera's surface; those cubes are
        listed as rough ("ray"). Until 2026-09-30 observe back-projected wrist depth, which the real arm can't."""
        rgb, depth, geom, pts = self.front.capture(self.d)
        w_rgb, w_geom = self.wrist.capture_colour(self.d)
        origin, dirs = self.wrist.rays(self.d)
        c, h = self.bin["center"], self.bin["half"]
        floor_z = self.bin["floor_z"]
        n = int(round(2 * h / CELL))
        floor_id = self.m.geom("bin_floor").id
        front_cam = self.d.cam_xpos[self.front.cid].copy()

        cubes, masks = {}, {}
        for g, name in self.geom_name.items():
            mask = geom == g
            if mask.sum() >= 40:
                mask = clean_mask(mask, pts)
            if mask.sum() < 40:
                continue
            p = pts[mask]
            pos, yaw, cov, _ = top_face_pose(p)
            yaw, tilt, how = float(np.degrees(yaw)), None, "top"
            if cov < TRUST:
                # Less than 60% of the top seen: fit the whole box instead. Measured 2026-09-29 over 186 cube views,
                # the box fit put partly seen cubes within about 1 mm median against 7 to 9 mm from the top face,
                # but about 1 in 10 is off by up to 16 mm, so these stay "rough".
                pos, rot, _ = box_pose(p, front_cam, np.radians(yaw))
                yaw, tilt = yaw_and_tilt(rot)
                how = "fit"
            layer = int(round((pos[2] - floor_z - HALF) / CUBE)) + 1
            cubes[name] = {"pos": pos, "yaw": yaw, "cov": cov, "px": int(mask.sum()), "layer": layer, "tilt": tilt,
                           "how": how, "views": "F"}
            masks[name] = mask
        w_masks = {}
        for g, name in self.geom_name.items():
            mask = ndimage.binary_erosion(w_geom == g, iterations=1)
            if mask.sum() >= 40:
                w_masks[name] = mask
                if name in cubes:
                    cubes[name]["views"] += "W"

        # The height map: the front camera's points, plus the wrist camera's floor pixels on the floor's plane.
        names = list(masks)
        labels = np.full(pts.shape[:2], -1)
        for k, name in enumerate(names):
            labels[masks[name]] = k
        labels[ndimage.binary_erosion(geom == floor_id, iterations=1)] = len(names)
        keep = labels >= 0
        w_floor = ndimage.binary_erosion(w_geom == floor_id, iterations=1)
        d_floor = dirs[w_floor]
        on_floor = origin + ((floor_z - origin[2]) / d_floor[:, 2])[:, None] * d_floor
        p = np.concatenate([pts[keep], on_floor])
        lab = np.concatenate([labels[keep], np.full(len(on_floor), len(names))])
        i = np.floor((p[:, 0] - (c[0] - h)) / CELL).astype(int)
        j = np.floor((p[:, 1] - (c[1] - h)) / CELL).astype(int)
        ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        p, lab, cell = p[ok], lab[ok], i[ok] * n + j[ok]
        order = np.argsort(p[:, 2])  # highest last, so it wins the cell
        height = np.full(n * n, -np.inf)
        owner = np.full(n * n, -1)
        height[cell[order]] = p[order, 2]
        owner[cell[order]] = lab[order]
        height, owner = height.reshape(n, n), owner.reshape(n, n)
        unseen = ~np.isfinite(height)
        # A cell no camera can see is at most as high as what hides it, so estimate it by the highest seen height
        # within 2 cm.
        est = ndimage.maximum_filter(np.where(unseen, -np.inf, height), size=int(0.02 / CELL))
        est[~np.isfinite(est)] = self.bin["rim_z"]
        surface = np.where(unseen, est, height)

        # Cubes only the wrist camera sees: follow the mask's rays onto the surface. Rough by construction, since an
        # unseen cell's height is only an upper bound.
        for name, mask in w_masks.items():
            if name in cubes:
                continue
            hit = self._march(origin, dirs[mask], surface)
            if len(hit) < 20:
                continue
            top = float(np.percentile(hit[:, 2], 90))
            layer = int(np.clip(round((top - floor_z) / CUBE), 1, 3))
            xy = np.median(hit[:, :2], 0)
            _, yaw, _, _ = top_face_pose(hit)
            cubes[name] = {"pos": np.array([xy[0], xy[1], floor_z + HALF + (layer - 1) * CUBE]),
                           "yaw": float(np.degrees(yaw)), "cov": 0.0, "px": int(mask.sum()), "layer": layer,
                           "tilt": None, "how": "ray", "views": "W"}

        # Red pixels, as a region: a thin red line at a cube's edge marked the hidden red's slot in e400 before observe
        # could list it. The front camera places them by depth. The wrist camera's go along their rays onto the
        # surface, only when the front camera sees none: on 12 piles (2026-09-30) mixing them in moved the region's
        # median from 14 to 18 mm off the red's centre, since unseen cells are only an upper bound.
        red = np.zeros((0, 3))
        for img, place in ((rgb, lambda m: pts[m]), (w_rgb, lambda m: self._march(origin, dirs[m], surface))):
            rr, gg, bb = (img[..., k].astype(int) for k in range(3))
            r = place((rr > 90) & (rr > 2 * gg) & (rr > 2 * bb))
            red = r[(np.abs(r[:, :2] - c) < h).all(1)] if len(r) else red
            if len(red):
                break
        self.view = {"cubes": cubes, "rgb": rgb, "wrist_rgb": w_rgb, "names": names + ["floor"],
                     "height": surface, "owner": owner, "unseen": unseen, "n": n, "red_px": red}
        return self.view

    def _march(self, origin, dirs, surface, step=0.001, most=1500):
        """Where rays from `origin` along `dirs` first drop to the surface height map, one point per ray that
        meets it inside the bin. At most `most` rays, evenly spaced, to bound the cost."""
        if len(dirs) > most:
            dirs = dirs[np.linspace(0, len(dirs) - 1, most).astype(int)]
        if not len(dirs):
            return np.zeros((0, 3))
        c, h, n = self.bin["center"], self.bin["half"], surface.shape[0]
        t = np.arange(0.05, 0.9, step)
        pts = origin + t[None, :, None] * dirs[:, None, :]
        i = np.floor((pts[..., 0] - (c[0] - h)) / CELL).astype(int)
        j = np.floor((pts[..., 1] - (c[1] - h)) / CELL).astype(int)
        inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        below = np.zeros(inside.shape, bool)
        below[inside] = pts[..., 2][inside] <= surface[i[inside], j[inside]]
        first = below.argmax(1)
        met = below[np.arange(len(dirs)), first]
        return pts[np.arange(len(dirs)), first][met]

    def cells(self, xy):
        c, h, n = self.bin["center"], self.bin["half"], self.view["n"]
        i = np.clip(((xy[:, 0] - (c[0] - h)) / CELL).astype(int), 0, n - 1)
        j = np.clip(((xy[:, 1] - (c[1] - h)) / CELL).astype(int), 0, n - 1)
        return i, j

    def spots(self, k=4):
        """Flat patches a held cube can be set down on: the floor, or a level top at least a cube wide."""
        v = self.view
        hgt = v["height"]
        seen = ~v["unseen"]
        w = int(round(0.030 / CELL))  # a cube footprint
        # Flat means the seen cells in a cube-sized patch agree within 5 mm and at least 60% of it is seen. From the oblique
        # front camera, requiring every cell seen left no spot at all (seed 400, 61% of the bin unseen).
        seen_h = np.where(seen, hgt, np.nan)
        lo = ndimage.minimum_filter(np.where(seen, hgt, np.inf), size=w)
        hi = ndimage.maximum_filter(np.where(seen, hgt, -np.inf), size=w)
        frac = ndimage.uniform_filter(seen.astype(float), size=w)
        flat = (hi - lo < 0.005) & (frac >= 0.6) & seen
        hgt = np.where(np.isnan(seen_h), hgt, seen_h)
        c, h = self.bin["center"], self.bin["half"]
        ii, jj = np.nonzero(flat)
        xy = np.stack([c[0] - h + (ii + 0.5) * CELL, c[1] - h + (jj + 0.5) * CELL], 1)
        inside = (np.abs(xy - c) < self.prim._inset - 0.004).all(1)
        xy, z = xy[inside], hgt[ii, jj][inside]
        if len(xy) == 0:
            return []
        out = []
        for idx in np.argsort(z):  # lowest first, spread out
            if all(np.linalg.norm(xy[idx] - o[:2]) > 0.04 for o in out):
                out.append(np.array([xy[idx][0], xy[idx][1], z[idx] + HALF + 0.003]))
            if len(out) >= k:
                break
        return out

    # ---- checking -------------------------------------------------------------------------------

    def _finger_local(self):
        """Points on each finger mesh in its own frame: the vertices, plus a 2 mm grid on every face longer than 3 mm.
        The meshes have faces up to 60 mm across with vertices only at the corners, and a neighbour's top poking into
        the middle of one went unseen: d1005's clean check across a 2 mm gap missed."""
        out = {}
        for gi in range(self.m.ngeom):
            if self.m.geom_bodyid[gi] not in self.finger_bodies or self.m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if self.m.geom_contype[gi] == 0 and self.m.geom_conaffinity[gi] == 0:
                continue
            mid = self.m.geom_dataid[gi]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            f = self.m.mesh_face[self.m.mesh_faceadr[mid]:self.m.mesh_faceadr[mid] + self.m.mesh_facenum[mid]]
            pts = [v.astype(float)]
            for a, b, c in v[f].astype(float):
                longest = max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a))
                if longest <= 0.003:
                    continue
                k = int(np.ceil(longest / 0.002))
                ij = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], float) / k
                pts.append(a + ij[:, :1] * (b - a) + ij[:, 1:] * (c - a))
            out[gi] = np.unique(np.round(np.concatenate(pts), 4), axis=0)
        return out

    def finger_points(self, row):
        dd = self.probe
        dd.qpos[self.adr] = units.to_rad(np.asarray(row, dtype=float), self.grip_range)
        mujoco.mj_kinematics(self.m, dd)
        if not hasattr(self, "_fingers"):
            self._fingers = self._finger_local()
        return np.concatenate([v @ dd.geom_xmat[gi].reshape(3, 3).T + dd.geom_xpos[gi]
                               for gi, v in self._fingers.items()])

    def mount_points(self):
        """Sample points on the camera mount's collision boxes, at the pose finger_points last set."""
        dd = self.probe
        out = []
        for g in self.mount_boxes:
            local = self.box_unit * self.m.geom_size[g]
            out.append(local @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g])
        return np.concatenate(out)

    def plan_path(self, kind, xyz, deg, flip, stroke=None):
        """Every frame act would command, from the unpark hover (if parked) through the bridge and the primitive,
        and the primitive's own frames."""
        traj = self.prim._plan(kind, np.asarray(xyz, float), float(np.radians(deg)), flip, stroke=stroke)
        start = self._unpark_plan()[-1] if self.last is None else self.env.joint_state
        return primitives.bridge(self.m, start, traj, hover_z=self.prim.travel_z), traj

    def yaw_error(self, kind, traj, deg):
        """How far the jaws end up from the asked-for yaw at the primitive's lowest frame, in degrees mod 180. IK
        gives up yaw before position, so a wrist roll pinned at its limit shows up here."""
        want = deg + (90.0 if kind in ("push", "sweep") else 0.0)
        row = min(traj, key=lambda r: primitives.tool_pose(self.m, r, self.probe)[0][2])
        got = np.degrees(primitives.tool_pose(self.m, row, self.probe)[1])
        return abs((got - want + 90.0) % 180.0 - 90.0)

    def fixed_side(self, traj, xyz):
        """Heading in degrees from the target to the fixed jaw tip at the primitive's lowest frame. The fixed finger
        lands 4 to 14 mm outside that face and the moving one 20 to 28 mm outside the opposite face (d1009)."""
        row = min(traj, key=lambda r: primitives.tool_pose(self.m, r, self.probe)[0][2])
        self.finger_points(row)  # sets the probe to this frame
        tip = self.probe.geom_xpos[self.m.geom("fixed_jaw_sph_tip1").id]
        d = tip[:2] - np.asarray(xyz[:2], float)
        return float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0

    def choose(self, kind, xyz, deg, stroke=None):
        """Check both wrist rolls. Returns (the flip to use, {flip: check result})."""
        results = {f: self.check(kind, xyz, deg, f, stroke) for f in (0, 1)}
        ok = [f for f, r in results.items() if r["runs"] and r["yaw_err"] < 5.0]
        if not ok:
            ok = [f for f, r in results.items() if r["runs"]]
        if not ok:
            return None, results
        return max(ok, key=lambda f: (results[f]["self_mm"] > 0, results[f]["self_mm"] > 10,
                                      -sum(results[f]["hits"].values()), results[f]["self_mm"])), results

    def held_cube(self):
        """The seen cube in the jaws, if the gripper reads holding: the one nearest the jaw midpoint."""
        if not 14 < float(self.env.joint_state[5]) < 30:
            return None
        grasp = primitives.tool_pose(self.m, self.env.joint_state, self.probe)[0]
        near = [(np.linalg.norm(cb["pos"] - grasp), n) for n, cb in self.view["cubes"].items()]
        near = [t for t in near if t[0] < 0.03]
        return min(near)[1] if near else None

    def check(self, kind, xyz, deg, flip, stroke=None):
        try:
            full, traj = self.plan_path(kind, xyz, deg, flip, stroke)
        except Exception as exc:
            return {"runs": False, "why": str(exc)}
        v = self.view
        held = self.held_cube()  # the cameras see it in the jaws; the fingers can't hit it
        hits, unseen, wall, wall_gap = {}, 0.0, 0.0, np.inf
        self_mm, pair = np.inf, ""
        mount_hits, mount_wall = {}, 0.0
        c, h = self.bin["center"], self.bin["half"]
        top_seen = max(self.bin["rim_z"], float(v["height"].max())) + 0.005
        for row in full[::2]:
            fp = self.finger_points(row)  # also leaves the probe at this arm pose
            s_, _, p_ = self.clearance.measure(self.probe)
            if s_ * 1000 < self_mm:
                self_mm, pair = s_ * 1000, p_
            mp_ = self.mount_points()
            mp_ = mp_[mp_[:, 2] < self.bin["rim_z"] + 0.005]
            if len(mp_):
                out_m = (np.abs(mp_[:, :2] - self.bin["center"]) > self.bin["half"]).any(1)
                if out_m.any():
                    mount_wall = max(mount_wall, float(self.bin["rim_z"] - mp_[out_m, 2].min()))
                mp_ = mp_[~out_m]
                if len(mp_):
                    mi, mj = self.cells(mp_[:, :2])
                    md = v["height"][mi, mj] - mp_[:, 2]
                    for q in np.nonzero(md > MARGIN)[0]:
                        who = "unseen" if v["unseen"][mi[q], mj[q]] else v["names"][v["owner"][mi[q], mj[q]]]
                        if who != held:
                            mount_hits[who] = max(mount_hits.get(who, 0.0), float(md[q]))
            # Points above everything seen can't touch anything. Until 2026-09-30 the cut was the rim, which dropped every
            # contact with a third-layer cube (they ride above the rim): pushes along the top saw nothing, and a pick
            # could put a finger into a top-layer neighbour unflagged.
            fp = fp[fp[:, 2] < top_seen]
            if len(fp) == 0:
                continue
            low = fp[fp[:, 2] < self.bin["rim_z"]]
            if len(low):
                wall_gap = min(wall_gap, float((h - np.abs(low[:, :2] - c).max(1)).min()) * 1000)
            outside = (np.abs(fp[:, :2] - c) > h).any(1)
            if outside.any():
                wall = max(wall, float(self.bin["rim_z"] - fp[outside, 2].min()))
            fp = fp[~outside]
            if len(fp) == 0:
                continue
            i, j = self.cells(fp[:, :2])
            depth = v["height"][i, j] - fp[:, 2]
            hit = depth > MARGIN
            for k in np.nonzero(hit)[0]:
                if v["unseen"][i[k], j[k]]:
                    unseen = max(unseen, float(depth[k]))
                else:
                    who = v["names"][v["owner"][i[k], j[k]]]
                    if who != held:
                        hits[who] = max(hits.get(who, 0.0), float(depth[k]))
        out = {"runs": True, "frames": len(full), "hits": hits, "unseen_mm": unseen * 1000, "wall_mm": wall * 1000,
               "wall_gap_mm": wall_gap, "self_mm": self_mm, "self_pair": pair,
               "yaw_err": self.yaw_error(kind, traj, deg), "fixed_deg": self.fixed_side(traj, xyz),
               "mount_hits": mount_hits, "mount_wall_mm": mount_wall * 1000}
        if kind == "pick":
            out.update(self.pick_target(full, xyz, out["fixed_deg"]))
        if kind in ("push", "sweep"):
            out.update(self.push_report(xyz, deg, stroke, hits))
        return out

    def push_report(self, xyz, deg, stroke, hits):
        """What a push or sweep should move, judged from the cameras' view only: the cubes its fingers meet, the ones
        lined up in front of those, how far each is carried, and the risks: a cube driven into a wall while riding
        high enough to go over it, the red moving. Straight-line geometry, not a simulation: hidden cubes move too.

        Before 2026-09-30 check reported every cube a push touched as a hit, as for a pick, so every push looked
        dangerous and the players almost never pushed (1 push in 260 moves)."""
        v = self.view
        cubes = v["cubes"]
        d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        side = np.array([-d[1], d[0]])
        stroke = self.prim.push_stroke if stroke is None else float(stroke)
        start = np.asarray(xyz[:2], float)
        carried = {n: None for n, x in hits.items() if n in cubes and x > MARGIN}
        # Cubes lined up in front of a carried cube, on about its layer, get shoved along with it.
        frontier = list(carried)
        while frontier:
            c = frontier.pop()
            pc = cubes[c]["pos"]
            for o, cb in cubes.items():
                if o in carried or abs(cb["pos"][2] - pc[2]) > 0.02:
                    continue
                rel = cb["pos"][:2] - pc[:2]
                if 0.0 < rel @ d < CUBE + 0.008 and abs(rel @ side) < CUBE:
                    carried[o] = None
                    frontier.append(o)
        c, h = self.bin["center"], self.bin["half"]
        report, risky = {}, []
        for n in carried:
            cb = cubes[n]
            travel = max(0.0, stroke - (max(0.0, (cb["pos"][:2] - start) @ d - HALF)))
            # Room before the wall in the push's direction, from the cube's near side of the wall.
            room = min(((h - HALF) - s_ * (cb["pos"][k] - c[k])) / abs(d[k]) if abs(d[k]) > 1e-6 else np.inf
                       for k, s_ in ((0, np.sign(d[0])), (1, np.sign(d[1]))))
            jam = travel > room
            top = cb["pos"][2] + HALF
            high = cb["layer"] >= 2 or top > self.bin["rim_z"] - 0.005
            report[n] = {"travel_mm": round(min(travel, max(room, 0.0)) * 1000, 1), "to_wall_mm": round(room * 1000, 1),
                         "jam": bool(jam), "high": bool(high)}
            if high and room < travel + PUSH_WALL_MM / 1000:
                risky.append(n)
        out = {"moves": report, "spill_risk": risky, "moves_red": TARGET in carried, "stroke": stroke}
        if TARGET in cubes:
            # What the stroke does for the red's access: covers on its top and blocked ends of its two jaw axes, now
            # and with every predicted mover shifted by its predicted travel.
            after = {n: dict(cb) for n, cb in cubes.items()}
            for n, m in report.items():
                pos = np.array(after[n]["pos"], float)
                pos[:2] += d * m["travel_mm"] / 1000
                after[n]["pos"] = pos
            out["red_access"] = {"before": red_access(cubes), "after": red_access(after)}
        return {"push": out}
    def pick_target(self, full, xyz, fixed_deg):
        """For a pick: the cube it is aimed at, how far the aim sits off that cube's centre toward the moving finger,
        and how deep any finger comes down inside the cube's own top outline. The height-map hits leave the target
        out, since the fingers close on it, so without this a finger landing on the target's top went unflagged:
        7 of the 22 misses studied on 2026-09-30."""
        near = [(np.linalg.norm(cb["pos"][:2] - np.asarray(xyz[:2], float)), n) for n, cb in self.view["cubes"].items()]
        near = [t for t in near if t[0] < 0.012]
        if not near:
            return {"target": None}
        name = min(near)[1]
        cb = self.view["cubes"][name]
        toward_moving = -np.array([np.cos(np.radians(fixed_deg)), np.sin(np.radians(fixed_deg))])
        shift = float(np.dot(np.asarray(xyz[:2], float) - cb["pos"][:2], toward_moving)) * 1000
        yaw = np.radians(cb["yaw"])
        rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        top = cb["pos"][2] + HALF
        on_top = 0.0
        for row in full[::2]:
            # Only while the jaws are open. The plan then closes them to GRIP_CLOSE, through the cube, since the
            # cube is what stops them.
            if row[5] <= primitives.GRIP_PLAN:
                continue
            fp = self.finger_points(row)
            local = (fp[:, :2] - cb["pos"][:2]) @ rot.T
            inside = (np.abs(local) < HALF - 0.0015).all(1) & (fp[:, 2] < top)
            if inside.any():
                on_top = max(on_top, float(top - fp[inside, 2].min()))
        return {"target": name, "trusted": cb["cov"] >= TRUST, "shift_mm": shift, "on_top_mm": on_top * 1000}

    # ---- reporting ------------------------------------------------------------------------------

    def _labelled(self, rgb, cam, size=11):
        img = Image.fromarray(rgb).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=size)
        pos, rot = self.d.cam_xpos[cam.cid], self.d.cam_xmat[cam.cid].reshape(3, 3)
        for name, cb in self.view["cubes"].items():
            cp = rot.T @ (cb["pos"] + np.array([0, 0, HALF]) - pos)
            if cp[2] >= 0:
                continue  # behind the camera
            u = cam.w / 2 + cam.fx * cp[0] / -cp[2]
            w = cam.h / 2 - cam.fy * cp[1] / -cp[2]
            colour = (255, 255, 0) if cb["cov"] >= TRUST else (0, 255, 255)
            dr.text((u - 6, w - 6), short(name), fill=colour, font=font)
        return img

    def picture(self, path):
        """Two panels. Left, the front camera cropped to the bin: top is the base side (low x), left is -y. Right,
        the wrist camera on the parked arm, looking down from above the base: left is the base side (low x), right
        the far wall (high x), top +y and bottom -y."""
        v = self.view
        front = self._labelled(v["rgb"], self.front, 7).crop((225, 150, 425, 300)).resize((800, 600), Image.NEAREST)
        wrist = self._labelled(v["wrist_rgb"], self.wrist, 18).resize((800, 600), Image.BILINEAR)
        both = Image.new("RGB", (1600, 600))
        both.paste(front, (0, 0))
        both.paste(wrist, (800, 0))
        both.save(path)


def axis_ends(view, name):
    """For each of a seen cube's two jaw axes, the seen cubes at each end, and how much of each end the camera can't see.

    An end is the strip beside one face, 0 to 30 mm out and as wide as the face plus 10 mm, from the floor up to the
    cube's own top plus a layer. The jaws need 4 to 14 mm free on one end and 20 to 28 mm on the other, so a cube
    listed at either end blocks that axis until it is lifted (d1012). Gap is the distance from this cube's face to
    the nearest listed cube's face along the axis, roughly, from centres."""
    cubes = view["cubes"]
    me = cubes[name]
    c = np.asarray(me["pos"])
    out = []
    for yaw in (me["yaw"], me["yaw"] + 90.0):
        a = np.radians(yaw)
        u = np.array([np.cos(a), np.sin(a)])
        v = np.array([-u[1], u[0]])
        ends = {}
        for sign in (1, -1):
            found = []
            for other, cb in cubes.items():
                if other == name:
                    continue
                d = np.asarray(cb["pos"]) - c
                along, across = sign * (d[:2] @ u), abs(d[:2] @ v)
                if 0.0 < along < CUBE + 0.030 and across < CUBE and -HALF < d[2] < CUBE + 0.010:
                    found.append((along - CUBE, other, cb["layer"]))
            ends[sign] = sorted(found)
        out.append((yaw % 180.0, ends))
    return out


def red_access(cubes):
    """Covers on the red's top, and blocked ends of each of its two jaw axes, from listed cubes."""
    red = cubes[TARGET]
    rc = np.asarray(red["pos"])
    covers = [n for n, cb in cubes.items() if n != TARGET and np.linalg.norm(np.asarray(cb["pos"][:2]) - rc[:2]) < CUBE * 0.8
              and cb["pos"][2] > rc[2] + HALF]
    axes = axis_ends({"cubes": cubes}, TARGET)
    return {"covers": covers, "axes": [(round(yaw), sum(bool(v) for v in ends.values())) for yaw, ends in axes]}


def print_push(p):
    moves = ", ".join(f"{short(n)} ~{m['travel_mm']:.0f} mm" + (" INTO THE WALL" if m["jam"] else "")
                      for n, m in sorted(p["moves"].items(), key=lambda kv: -kv[1]["travel_mm"])) or "nothing seen"
    print(f"  stroke {p['stroke'] * 1000:.0f} mm moves: {moves}")
    if p["spill_risk"]:
        print(f"  SPILL RISK, act refuses: {', '.join(short(n) for n in p['spill_risk'])} rides at layer 2 or higher and "
              f"would end within {PUSH_WALL_MM:.0f} mm of the wall ahead (17% of such strokes spilled in calibration)")
    if p["moves_red"]:
        print("  moves the red (check where it ends up: against a wall or under a cube is worse)")
    if "red_access" in p:
        b, a = p["red_access"]["before"], p["red_access"]["after"]
        axes = "; ".join(f"axis {yb} deg: blocked ends {nb} -> {na}" + (" (OPENS)" if nb and not na else "")
                         for (yb, nb), (_, na) in zip(b["axes"], a["axes"]))
        print(f"  red's access, predicted: covers {len(b['covers'])} -> {len(a['covers'])}; {axes}")


def rough_note(cb):
    """How an untrusted pose was found. Wrist-only cubes ("ray") come from pixel rays, with no tilt estimate."""
    if cb["how"] == "ray":
        return "rough, wrist camera only, placed along its pixels' rays (about 15 mm off; up to 33)"
    return f"rough, box fit, tilt {cb['tilt']:.0f} deg"


def holding(grip):
    """Closed on a cube reads 14 to 30. An empty close reads about 10, and open reads 35 or more."""
    if 14 < grip < 30:
        return "HOLDING"
    return "empty, closed" if grip <= 14 else "open"


def fmt(xyz):
    return "(" + ", ".join(f"{v:.3f}" for v in xyz) + ")"


def cmd_new(name, seed, kind):
    EPISODES.mkdir(exist_ok=True)
    path = EPISODES / f"{name}.json"
    if path.exists():
        sys.exit(f"{path} exists")
    path.write_text(json.dumps({"seed": int(seed), "buried": kind == "buried", "miss_reset_eps": MISS_RESET_EPS,
                                "ops": []}, indent=1))
    print(f"episode {name}: seed {seed}, {kind}")


def cmd_observe(ep):
    # Parking while holding dropped the red twice in e400, once with the gripper still closed: the swing home shook
    # it out. So a held cube stays where it is and the camera looks past the arm, which hides about 3% of the bin.
    held = 14 < float(ep.env.joint_state[5]) < 30
    # A finished episode is a finished demonstration: looking at it must not add a park after the last frame.
    if not ep.parked() and not held and not ep.outcome():
        op = {"op": "park", "keep_grip": True}
        ep._apply(op)
        ep.rec["ops"].append(op)
        ep.save()
    if held:
        print("HOLDING a cube: looking without parking, so the arm hides part of the bin.")
    v = ep.see()
    n_obs = sum(1 for op in ep.rec["ops"] if op["op"] == "park")
    img = EPISODES / f"{ep.name}_obs{n_obs:02d}.png"
    ep.picture(img)
    cubes = v["cubes"]
    grip = float(ep.env.joint_state[5])
    print(f"episode {ep.name}: {ep.moves()} of {MAX_MOVES} moves used, robot time {ep.robot_s:.1f} s, "
          f"gripper {grip:.1f} ({holding(grip)})")
    red = cubes.get(TARGET)
    glimpse = v["red_px"]
    if len(glimpse):
        lo, hi = glimpse.min(0), glimpse.max(0)
        print(f"red pixels in the image: {len(glimpse)}, around {fmt(np.median(glimpse, 0))}, x {lo[0]:.3f} to "
              f"{hi[0]:.3f}, y {lo[1]:.3f} to {hi[1]:.3f}, z {lo[2]:.3f} to {hi[2]:.3f}")
    else:
        print("red pixels in the image: none")
    if red is None:
        print("RED: not seen")
    else:
        state = "top face seen" if red["cov"] >= TRUST else rough_note(red)
        print(f"RED: {state}: {red['px']} px, top face {red['cov']:.0%}, at {fmt(red['pos'])}, layer {red['layer']}, "
              f"yaw {red['yaw']:.0f}")
    if red is not None:
        for yaw, ends in axis_ends(v, TARGET):
            parts = []
            for sign in (1, -1):
                cubes_at = ", ".join(f"{short(n)} (layer {layer}, gap {max(gap, 0) * 1000:.0f} mm)"
                                     for gap, n, layer in ends[sign]) or "clear of seen cubes"
                parts.append(f"{'+' if sign > 0 else '-'} end: {cubes_at}")
            print(f"  RED jaw axis {yaw:3.0f} deg: " + "; ".join(parts))
    print(f"seen cubes (pose trusted when top face >= {TRUST:.0%}; F front camera, W wrist camera), highest first:")
    for name in sorted(cubes, key=lambda n: -cubes[n]["pos"][2]):
        cb = cubes[name]
        flag = "" if cb["cov"] >= TRUST else "  " + rough_note(cb)
        print(f"  {short(name):>4} layer {cb['layer']} at {fmt(cb['pos'])} yaw {cb['yaw']:4.0f}  top {cb['cov']:4.0%}"
              f"  {cb['views']:>2}{flag}")
    print(f"unseen: {v['unseen'].mean():.0%} of the bin floor area")
    print("flat spots to set a cube on (place xyz):", ", ".join(fmt(s) for s in ep.spots()) or "none")
    print(f"image: {img}")


def cmd_check(ep, kind, x, y, z, deg, stroke=None):
    ep.see()
    flip, results = ep.choose(kind, [x, y, z], deg, stroke)
    if flip is None:
        why = results[0].get("why") or results[1].get("why")
        print(f"{kind} {fmt([x, y, z])} {deg:.0f} deg: REFUSED or unreachable: {why}")
        return
    r, other = results[flip], results[1 - flip]
    # The cube a pick is aimed at always shows up, since the fingers close on it. Name it and leave it out.
    target = None
    if kind == "pick":
        near = [(np.linalg.norm(cb["pos"][:2] - np.array([x, y])), n) for n, cb in ep.view["cubes"].items()]
        near = [t for t in near if t[0] < 0.012]
        target = min(near)[1] if near else None
    hits = ", ".join(f"{short(k) if k != 'floor' else 'floor'} {v * 1000:.0f} mm" for k, v in
                     sorted(r["hits"].items(), key=lambda kv: -kv[1]) if k != target) or "none"
    print(f"{kind} {fmt([x, y, z])} {deg:.0f} deg: runs, {r['frames'] / 30:.1f} s, wrist roll {flip}")
    if kind == "pick":
        print(f"  fixed finger toward {r['fixed_deg']:.0f} deg, moving finger toward {(r['fixed_deg'] + 180) % 360:.0f} "
              f"deg. Shift the aim at most {SHIFT_MAX_MM:.0f} mm away from the fixed finger, or it lands on the cube's "
              f"own top.")
        if r.get("target"):
            flag = " TOO FAR: act refuses this" if r["trusted"] and r["shift_mm"] > SHIFT_MAX_MM else ""
            print(f"  aim is {r['shift_mm']:+.1f} mm off {short(r['target'])}'s centre toward the moving finger"
                  f" ({'trusted' if r['trusted'] else 'rough'} pose){flag}")
            if r["on_top_mm"] > MARGIN * 1000:
                print(f"  A FINGER COMES DOWN ON {short(r['target'])}'S OWN TOP, {r['on_top_mm']:.0f} mm below it: act "
                      f"refuses this. Misses studied on 2026-09-30 all closed above the cube this way.")
    alt = f"{other['self_mm']:.0f} mm" if other["runs"] else "unreachable"
    print(f"  arm to itself: {r['self_mm']:.0f} mm at the closest ({r['self_pair']}); the other roll: {alt}")
    if r["self_mm"] < SELF_JAM_MM:
        print(f"  HITS ITSELF: {-r['self_mm']:.0f} mm of overlap. The arm won't follow this path; it lags and sweeps "
              f"sideways (e401 knocked a cube out that way). None of 17 such picks tracked.")
    elif r["self_mm"] < SELF_TIGHT_MM:
        print(f"  tight: {-r['self_mm']:.0f} mm of overlap with itself. About half of these tracked on an empty bin.")
    if r["yaw_err"] >= 5.0:
        print(f"  jaws end {r['yaw_err']:.0f} deg off the asked yaw: the wrist roll is at its limit")
    aside = f" (target {short(target)} left out)" if target else ""
    if kind in ("push", "sweep"):
        print_push(r["push"])
        if r["unseen_mm"] >= PUSH_UNSEEN_MM:
            print(f"  pushes through unseen space {r['unseen_mm']:.0f} mm deep: hidden cubes move too; 5.5% of such "
                  f"allowed strokes spilled one in calibration, against 2% with less. Prefer strokes that end away "
                  f"from the walls.")
    else:
        print(f"  fingers pass through seen cubes: {hits}{aside}")
    if kind == "pick":
        worst = max((x for k, x in r["hits"].items() if k != target), default=0.0) * 1000
        if worst >= HIT_REFUSE_MM:
            print(f"  REFUSED by act: a finger goes {worst:.0f} mm into a neighbour; 33 of 209 such picks lifted the "
                  f"target under the stiff contacts.")
    print(f"  into unseen space: {r['unseen_mm']:.0f} mm deep" if r["unseen_mm"] else "  into unseen space: none")
    if r["wall_mm"]:
        print(f"  bin wall: fingers go {r['wall_mm']:.0f} mm below the rim outside the bin")
    elif r["wall_gap_mm"] < WALL_GAP_MM:
        print(f"  bin wall: fingers pass {r['wall_gap_mm']:.0f} mm from a wall. Under the stiff contacts 21 of 40 otherwise "
              f"clean picks this close lifted the target (pick calibration, 2026-09-30); the open finger catches the wall.")
    if r["mount_hits"] or r["mount_wall_mm"]:
        mh = ", ".join(f"{short(k) if k not in ('floor', 'unseen') else k} {v * 1000:.0f} mm"
                       for k, v in sorted(r["mount_hits"].items(), key=lambda kv: -kv[1]))
        print(f"  CAMERA MOUNT hits the scene: {mh or ''}{' bin wall ' + format(r['mount_wall_mm'], '.0f') + ' mm' if r['mount_wall_mm'] else ''}")


def place_options(ep, v, avoid=None, k=6):
    """Where to set a held cube down: tops of well-seen layer-1 cubes and open floor, away from the dig, each checked
    at two yaws. Replaces checking spots one by one (agents found the flat-spot list often next to the dig or failing
    check; the tops of trusted layer-1 cubes worked, d1003, d1006, d1015). Returns (avoid, best first list of
    (badness, xy, z, deg, what, check result))."""
    held = ep.held_cube()
    if avoid is None:
        red = v["cubes"].get(TARGET)
        if red is not None and red is not v["cubes"].get(held):
            avoid = np.asarray(red["pos"][:2])
        elif len(v["red_px"]):
            avoid = np.median(v["red_px"][:, :2], 0)
    c, h = ep.bin["center"], ep.bin["half"]
    cands = []
    for name, cb in v["cubes"].items():
        if name in (held, TARGET) or cb["cov"] < TRUST or cb["layer"] != 1 or (cb["tilt"] or 0) > 5:
            continue
        if (h - np.abs(np.asarray(cb["pos"][:2]) - c)).min() < 0.02:
            continue  # by a wall: a finger would catch it
        cands.append((np.asarray(cb["pos"][:2]), cb["pos"][2] + CUBE + 0.003, cb["yaw"], f"on {short(name)}"))
    for sp in ep.spots():
        cands.append((np.asarray(sp[:2]), sp[2], 0.0, "flat spot"))
    if avoid is not None:
        cands = [t for t in cands if np.linalg.norm(t[0] - avoid) > 0.06]
        cands.sort(key=lambda t: -np.linalg.norm(t[0] - avoid))
    good = []
    for xy, z, yaw, what in cands[:2 * k]:
        for deg in (yaw % 180.0, (yaw + 90.0) % 180.0):
            flip, res = ep.choose("place", [xy[0], xy[1], z], deg)
            if flip is None:
                continue
            r = res[flip]
            # While holding, the arm hangs in front of the cameras, so unseen space is mostly the arm's shadow
            # (d1016: 78 mm at every spot). Rank by it only when the arm is parked.
            bad = sum(r["hits"].values()) * 1000 + (r["wall_mm"] > 0) * 50 + (r["wall_gap_mm"] < WALL_GAP_MM) * 20 \
                + (r["self_mm"] < SELF_TIGHT_MM) * 30 + (0 if held else r["unseen_mm"] / 10)
            good.append((bad, xy, z, deg, what, r))
    good.sort(key=lambda t: t[0])
    return avoid, good[:k]


def cmd_places(ep, avoid=None, k=6):
    v = ep.see()
    held = ep.held_cube()
    avoid, good = place_options(ep, v, avoid, k)
    where = f", at least 6 cm from {fmt(avoid)}" if avoid is not None else ""
    if held:
        print("holding: the arm blocks part of both views, so unseen space here is mostly the arm's shadow")
    print(f"places{where}, best first (place X Y Z DEG):")
    for bad, xy, z, deg, what, r in good:
        hits = ", ".join(f"{short(n) if n not in ('floor',) else n} {d * 1000:.0f} mm" for n, d in r["hits"].items())
        notes = [f"hits {hits}" if hits else "no hits", f"unseen {r['unseen_mm']:.0f} mm",
                 f"arm to itself {r['self_mm']:.0f} mm"]
        if r["wall_mm"] or r["wall_gap_mm"] < WALL_GAP_MM:
            notes.append("near a wall")
        print(f"  place {xy[0]:.3f} {xy[1]:.3f} {z:.3f} {deg:.0f}   {what}: " + ", ".join(notes))
    if not good:
        print("  none found; check spots by hand")


def refusal(kind, r):
    """Why act refuses a move, given check's result r for the wrist roll choose kept, or None. The calibrated
    refusals live here only; cmd_act and the driver both call it."""
    if not r:
        return None
    if kind == "pick" and r.get("target"):
        if r["trusted"] and r["shift_mm"] > SHIFT_MAX_MM:
            return (f"the aim is {r['shift_mm']:.1f} mm off {short(r['target'])}'s centre toward the moving finger "
                    f"(limit {SHIFT_MAX_MM:.0f} mm), so the fixed finger lands on its top.")
        if r["on_top_mm"] > MARGIN * 1000:
            return f"a finger comes down on {short(r['target'])}'s own top ({r['on_top_mm']:.0f} mm)."
    if kind == "pick":
        worst = max(((k, x) for k, x in r["hits"].items() if k != r.get("target")), key=lambda kx: kx[1],
                    default=(None, 0.0))
        if worst[1] * 1000 >= HIT_REFUSE_MM:
            return (f"a finger goes {worst[1] * 1000:.0f} mm into "
                    f"{short(worst[0]) if worst[0] != 'floor' else 'the floor'} (limit {HIT_REFUSE_MM:.0f} mm). Under "
                    f"stiff contacts that closes the jaws above the target: 33 of 209 such picks lifted it. Clear that "
                    f"side first.")
    if kind in ("push", "sweep") and r["push"]["spill_risk"]:
        return (f"{', '.join(short(n) for n in r['push']['spill_risk'])} rides at layer 2 or higher and would end "
                f"within {PUSH_WALL_MM:.0f} mm of the wall ahead; 17% of such strokes spilled. Push away from the "
                f"wall, shorten the stroke, or pick it off first.")
    if r.get("self_mm", 0) < SELF_JAM_MM:
        return (f"the arm overlaps itself by {-r['self_mm']:.0f} mm on this path; none of 23 such picks tracked. Try "
                f"the other axis or a different spot.")
    return None


def cmd_act(ep, kind, x, y, z, deg, why, stroke=None):
    if ep.outcome():
        sys.exit(f"episode {ep.name} is over: {ep.outcome()}. Undo the last move to keep playing.")
    ep.see()
    flip, results = ep.choose(kind, [x, y, z], deg, stroke)
    r = results.get(flip, {}) if flip is not None else {}
    why_not = refusal(kind, r)
    if why_not:
        sys.exit(f"refused: {why_not} Nothing moved.")
    op = {"op": "act", "kind": kind, "xyz": [x, y, z], "deg": deg, "why": why, "unpark": True, "flip": flip}
    if stroke is not None:
        op["stroke"] = stroke
    res = ep._apply(op)
    op["result"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}
    ep.rec["ops"].append(op)
    ep.save()
    grip = res["grip"]
    print(f"move {ep.moves()}: {kind} {fmt([x, y, z])} {deg:.0f} deg, wrist roll {flip} -> gripper {grip:.1f} "
          f"({holding(grip)}), success {res['success']}, failure {res['failure']}, refused {res['refused']}, "
          f"robot time {ep.robot_s:.1f} s")
    if missed(op):
        reset, u, eps = reset_miss(ep)
        if reset:
            print(f"missed pick: reset by the draw ({u:.3f} < eps {eps}). The scene is back as it was before this "
                  f"move. Observe before the next move.")
        else:
            print(f"missed pick: kept (draw {u:.3f} >= eps {eps}). Observe and recover from it.")


def spilled(act):
    """Whether a recorded act knocked an object out of the container, which ends the episode."""
    return "left the bin" in str(act["result"].get("failure") or "")


def draw(seed, move, attempt):
    """A fixed number in [0, 1) for an episode's move and attempt, so a replay or a rerun draws the same."""
    h = hashlib.sha256(f"{seed}:{move}:{attempt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def take_back(ep, why, granted):
    """Drop the last act and whatever came after it. The dropped ops go into the record's "undone" list with the
    reason and what allowed it, so the attempt stays on file. A demonstration keeps only what was kept."""
    ops = ep.rec["ops"]
    last = max(k for k, op in enumerate(ops) if op["op"] == "act")
    # Take back the park that led into the act too, so the scene is as it was when that decision was made.
    cut = last - 1 if last > 0 and ops[last - 1]["op"] == "park" else last
    undone = ep.rec.setdefault("undone", [])
    undone.append({"at_move": sum(1 for op in ops[:last] if op["op"] == "act") + 1, "why": why, "granted": granted,
                   "ops": ops[cut:]})
    ep.rec["ops"] = ops[:cut]
    ep.save()
    return ops[last]


def cmd_undo(ep, why):
    """Take back the last act if it spilled."""
    if len(ep.rec.get("undone", [])) >= MAX_UNDOS:
        sys.exit(f"undo limit ({MAX_UNDOS}) reached")
    acts = [op for op in ep.rec["ops"] if op["op"] == "act"]
    if not acts:
        sys.exit("nothing to undo")
    # Undo is privileged: the next move gets chosen after seeing how the hidden pile answered, which no real robot
    # can do. dm1487 allows it to reset a spill, and discourages even that (2026-09-30). A missed pick gets reset
    # only by the draw in act, never by choice.
    if not spilled(acts[-1]):
        sys.exit("undo only takes back a spill, and the last move spilled nothing. Keep it and recover from it.")
    a = take_back(ep, why, "spill")
    print(f"undid move {ep.rec['undone'][-1]['at_move']}: {a['kind']} {fmt(a['xyz'])} {a['deg']:.0f} deg -> "
          f"{a['result']}. {MAX_UNDOS - len(ep.rec['undone'])} undos left. Observe before the next move.")


def missed(act):
    """A pick whose gripper closed on nothing. The robot reads that from its own gripper joint."""
    return act["kind"] == "pick" and not act["result"]["terminated"] and act["result"]["grip"] <= 14


def reset_miss(ep):
    """After a missed pick, reset it with probability eps. The tool draws, not the player: if the player chose, it
    would reset the misses that hurt and keep the ones that helped, and that choice uses what the cameras never saw.
    Returns the draw and eps, and whether it reset."""
    eps = ep.rec.setdefault("miss_reset_eps", MISS_RESET_EPS)
    undone = ep.rec.get("undone", [])
    u = draw(ep.rec["seed"], sum(1 for op in ep.rec["ops"] if op["op"] == "act"), len(undone))
    reset = u < eps and len(undone) < MAX_UNDOS
    if reset:
        take_back(ep, f"missed pick, reset by the draw ({u:.3f} < {eps})", "miss draw")
    else:
        ep.save()
    return reset, u, eps


def cmd_list():
    for path in sorted(EPISODES.glob("*.json")):
        rec = json.loads(path.read_text())
        acts = [op for op in rec["ops"] if op["op"] == "act"]
        end = "going"
        if acts and acts[-1]["result"]["success"]:
            end = "solved"
        elif acts and acts[-1]["result"]["terminated"]:
            end = "failed"
        elif len(acts) >= MAX_MOVES:
            end = "out of moves"
        print(f"{path.stem:>8}  seed {rec['seed']:5d}  {'buried ' if rec['buried'] else 'partial'}  "
              f"{len(acts):2d} moves  {len(rec.get('undone', [])):d} undone  {end}")


def cmd_status(ep):
    for k, op in enumerate(ep.rec["ops"]):
        if op["op"] == "act":
            print(k, op["kind"], op["xyz"], op["deg"], op["result"], "|", op["why"])
        else:
            print(k, "park")
    for u in ep.rec.get("undone", []):
        a = next(op for op in u["ops"] if op["op"] == "act")
        print(f"undone at move {u['at_move']}: {a['kind']} {a['xyz']} {a['deg']} -> {a['result']} | {u['why']}")
    print(f"moves {ep.moves()}, robot time {ep.robot_s:.1f} s, {ep.outcome() or 'still going'}")


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "new":
        return cmd_new(*args)
    if cmd == "list":
        return cmd_list()
    ep = Episode(args[0])
    try:
        if cmd == "observe":
            cmd_observe(ep)
        elif cmd == "check":
            n = 7 if args[1] == "sweep" else 6
            cmd_check(ep, args[1], *map(float, args[2:6]), *(map(float, args[6:n])))
        elif cmd == "act":
            n = 7 if args[1] == "sweep" else 6
            cmd_act(ep, args[1], *map(float, args[2:6]), " ".join(args[n:]), *(map(float, args[6:n])))
        elif cmd == "status":
            cmd_status(ep)
        elif cmd == "places":
            cmd_places(ep, np.array([float(args[1]), float(args[2])]) if len(args) >= 3 else None)
        elif cmd == "undo":
            cmd_undo(ep, " ".join(args[1:]))
        else:
            sys.exit(f"unknown command {cmd}")
    finally:
        ep.close()


if __name__ == "__main__":
    main()
