"""Pick the shelf's board height, opening, front edge, depth and width, then the start pose and front camera.

Runs the sizing rule in docs/superpowers/specs/2026-09-26-shelf-search-design.md:

    python scripts/size_shelf.py            # report what it would choose
    python scripts/size_shelf.py --write    # and rewrite models/shelf_objects.xml

Candidates go best first: the largest area of cylinder centres, then the smallest opening, then the front edge
nearest the base, then a board 6 cm up before 9 and 3. The first that passes every check wins. Level IK doesn't
depend on the shelf, so it runs over worker processes and is cached across candidates. Collisions need a compiled
model per candidate, because a compiled model's world geoms don't move when geom_pos changes. The cylinder count
needs resets, so scripts/shelf_smoke.py --tune-count picks it afterwards.
"""

import argparse
import dataclasses
import datetime
import itertools
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import so101sim  # noqa: F401  before mujoco, which reads MUJOCO_GL at import
from so101sim import SCENE_PATH, SHELF_SCENE_PATH, grasp, ik, primitives, shelf_checks as sc, shelf_layout as sl
from so101sim import side_primitives as sp, units

import mujoco

BOARD_ZS = (0.06, 0.09, 0.03)
OPENINGS = (0.13, 0.15, 0.18)
FRONTS = (0.27, 0.285, 0.30)
DEPTHS = (0.16, 0.18, 0.20, 0.22)
WIDTHS = (0.20, 0.24, 0.28, 0.32)
CAM_POSITIONS = tuple(itertools.product((-0.10, 0.05), (-0.30, 0.0, 0.30), (0.20, 0.25, 0.30, 0.40)))
HOME_YS = (0.0, 0.08, -0.08)
HOME_RISES = (0.0, 0.03)
BATCH_PER_WORKER = 32

_MODEL = None


def _init_worker():
    global _MODEL
    _MODEL = mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def _solve(key):
    x, y, z, grip = key
    return key, sc.level_pose(_MODEL, (x, y, z), grip)


def candidates():
    layouts = [sl.ShelfLayout(board_z=b, opening=h, front_x=f, depth=d, width=w)
               for b, h, f, d, w in itertools.product(BOARD_ZS, OPENINGS, FRONTS, DEPTHS, WIDTHS)]
    return sorted(layouts, key=lambda L: (-round(sc.region_area(L), 6), L.opening, L.front_x,
                                          BOARD_ZS.index(L.board_z)))


def describe(L):
    x_lo, x_hi, y_lo, y_hi = L.region()
    return (f"board {L.board_z * 100:+.0f} cm, opening {L.opening * 100:.0f} cm, front edge {L.front_x:.3f} m, "
            f"{L.depth * 100:.0f} deep x {L.width * 100:.0f} wide, centres over {(x_hi - x_lo) * 100:.1f} x "
            f"{(y_hi - y_lo) * 100:.1f} cm")


class Sizer:
    """Level IK with a cache shared across candidates, and the collision checks per candidate."""

    def __init__(self, pool, workers, pan):
        self.pool = pool
        self.batch = BATCH_PER_WORKER * workers
        self.pan = pan
        self.q = {}

    def solve(self, points, z, grip):
        """Poses for `points` at height z, and the first point with no level solution or None. Stops solving at
        the first batch with a miss."""
        keys = [(round(float(x), 4), round(float(y), 4), round(float(z), 4), float(grip)) for x, y in points]
        todo = [k for k in dict.fromkeys(keys) if k not in self.q]
        for i in range(0, len(todo), self.batch):
            batch = todo[i:i + self.batch]
            for key, q in self.pool.imap_unordered(_solve, batch, chunksize=2):
                self.q[key] = q
            if any(self.q[k] is None for k in batch):
                break
        missed = [k[:2] for k in keys if k in self.q and self.q[k] is None]
        return [self.q.get(k) for k in keys], (missed[0] if missed else None)

    def check(self, L, open_value):
        geo = sp.Geometry(self.pan, L.front_x, L.aisle_x, L.standoff_x, L.grasp_z, L.carry_z, L.push_z)
        checks = sc.sizing_checks(L, geo, open_value)
        for name, points, z, ik_grip, _, _ in checks:
            _, miss = self.solve(points, z, ik_grip)
            if miss is not None:
                return f"{name}: no level solution at z={z:+.3f}, first at {miss}"
        model = sc.add_clearance(sl.load(L))
        data = mujoco.MjData(model)
        arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
        for name, points, z, ik_grip, grip, exempt in checks:
            poses, _ = self.solve(points, z, ik_grip)
            for p, q in zip(points, poses):
                if exempt is not None and exempt(p):
                    continue
                if not sc.clear(model, data, q, grip, arm, scene):
                    return (f"{name}: the arm comes within {sc.CLEARANCE * 1000:.0f} mm of the shelf or table at "
                            f"{tuple(np.round(p, 4))}")
        return None


def coverage(L, camera, home):
    model = sl.load(L, probes=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *sc.IMAGE_SIZE)
    try:
        return sc.marker_coverage(model, data, renderer, camera, home)
    finally:
        renderer.close()


def home_candidates(L, model):
    """Level, mount-up poses with the jaw midpoint on the standoff line, in lerobot units."""
    grip_rad = primitives._grip_rad(model, primitives.GRIP_PLAN)
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    for y, rise in itertools.product(HOME_YS, HOME_RISES):
        q, info = ik.solve_side(model, (L.standoff_x, y, L.grasp_z + rise), gripper=grip_rad)
        if info["pos_err"] < 1e-3 and info["tilt"] < 1.0:
            yield rise, tuple(round(float(v), 2) for v in units.to_lerobot(np.r_[q, grip_rad], grip_range))


def pick_home_and_camera(L):
    """The start pose and front camera. Of the poses whose wrist camera sees WRIST_MIN of the board and that have a
    front camera spot by `shelf_checks.lowest_camera`, the lowest pose, then the lowest camera, then the most board.
    Lower is better for both, because cameras near the cylinder tops see sides, so a red can hide (spec, third
    amendment). Returns (home, cam_pos, front share, wrist share), or None."""
    model = sc.add_clearance(sl.load(L))
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
    best = None
    for rise, home in home_candidates(L, model):
        sc.set_pose(model, data, home)
        if grasp.touching(data, arm, scene):
            print(f"  start pose {list(home)}: touches the shelf")
            continue
        wrist = coverage(dataclasses.replace(L, home=home), "wrist_cam", home)
        print(f"  start pose {list(home)}: the wrist camera sees {wrist:.0%} of the board", flush=True)
        if wrist < sc.WRIST_MIN:
            continue
        options = []
        for cam in CAM_POSITIONS:
            front = coverage(dataclasses.replace(L, home=home, cam_pos=cam), "front", home)
            print(f"    front camera at {cam}: sees {front:.0%}", flush=True)
            options.append((cam, front))
        pick = sc.lowest_camera(options)
        if pick is not None and (best is None or (rise, pick[0][2], -pick[1]) < best[0]):
            best = ((rise, pick[0][2], -pick[1]), (home, pick[0], pick[1], wrist))
    return None if best is None else best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite models/shelf_objects.xml")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    started = time.time()
    arm_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    pan, open_value = ik.pan_axis_xy(arm_model), sp.side_open_value(arm_model)

    chosen, tried = None, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        sizer = Sizer(pool, args.workers, pan)
        for L in candidates():
            if sc.region_area(L) < sc.MIN_AREA:
                break
            why = sizer.check(L, open_value)
            print(f"{describe(L)}: {why or 'passes'}", flush=True)
            if why is None:
                chosen = L
                break
            tried += 1
    if chosen is None:
        sys.exit(f"STOP: no shelf reaches {sc.MIN_AREA * 1e4:.0f} cm^2 of graspable centres. "
                 "Bring this back to dm1487 before building more.")

    best = pick_home_and_camera(chosen)
    if best is None:
        sys.exit(f"STOP: no start pose has its wrist camera see {sc.WRIST_MIN:.0%} of the board with a front camera "
                 f"spot that sees {sc.FRONT_MIN:.0%}. Bring this back to dm1487.")
    home, cam, front, wrist = best
    committed = sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))
    chosen = dataclasses.replace(chosen, home=home, cam_pos=cam, n_cyls=committed.n_cyls)
    notes = [
        f"Chosen {datetime.date.today()} by scripts/size_shelf.py in {time.time() - started:.0f} s, after {tried} "
        "better-ranked candidates failed.",
        describe(chosen) + ".",
        "Every centre on a 5 mm grid, the aisle, the standoff line, every pull path and every push point passes "
        f"level reach, and clears the shelf and table by {sc.CLEARANCE * 1000:.0f} mm more than {sc.WALL_STRIP * 100:.0f} cm "
        "from a side wall; the "
        f"open jaw at {open_value:.0f} clears everywhere more than {sc.RIGHT_STRIP * 100:.0f} cm from the right wall.",
        f"Front camera at {list(cam)}, the lowest spot that sees {sc.FRONT_MIN:.0%}, sees {front:.0%} of the empty "
        f"board with the arm at start pose {list(home)}, "
        f"whose wrist camera sees {wrist:.0%}.",
    ]
    notes += [n for n in sl.read_notes() if n.startswith("Cylinder count")]
    print("\n".join(notes))
    if args.write:
        sl.write(chosen, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


if __name__ == "__main__":
    main()
