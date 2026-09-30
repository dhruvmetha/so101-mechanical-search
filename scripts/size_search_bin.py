"""Pick the mechanical-search bin, riser, walls, start pose and front camera height.

Runs the sizing rule in docs/superpowers/specs/2026-09-25-mechanical-search-design.md:

    python scripts/size_search_bin.py            # report what it would choose
    python scripts/size_search_bin.py --write    # and rewrite models/search_objects.xml

Candidates go best first: widest bin, then lowest riser, then tallest wall, then the centre nearest
0.20 m. The first one that passes every check wins. Reach is top-down IK on a 5 mm grid of cube centres,
cached across candidates and spread over worker processes, and a check stops at the first batch with a
miss. The cube count needs resets, so scripts/mechsearch_smoke.py --tune-count picks it afterwards.
"""

import argparse
import dataclasses
import datetime
import multiprocessing as mp
import os
import sys
import time

import so101sim  # noqa: F401  before mujoco, which reads MUJOCO_GL at import
from so101sim import SCENE_PATH, SEARCH_SCENE_PATH, grasp, primitives, search_checks as sc, search_layout as sl
from so101sim.env import HOME

import mujoco

WIDTHS = (0.24, 0.22, 0.20, 0.18, 0.15, 0.12)  # 20 to 24 cm added in the spec's second amendment
DEPTHS = (0.06, 0.08, 0.10, 0.12)
WALLS = (0.06, 0.05, 0.04)  # capped at 6 cm by the spec's amendment
# Nearest 0.20 first. Of the two 1 cm offsets, farther out goes first, because on the flat table the edge
# nearest the base was where reach ran short.
CENTERS = (0.20, 0.21, 0.19)
CAM_ZS = (0.45, 0.55, 0.65, 0.75)
# HOME first, as the spec says. The others raise the wrist, in case HOME's gripper reaches into the drop
# column or its wrist camera sees too little of the floor.
HOME_CANDIDATES = (
    tuple(float(v) for v in HOME),
    (0.0, -45.0, 30.0, 90.0, 0.0, 50.0),
    (0.0, -30.0, 15.0, 90.0, 0.0, 50.0),
    (0.0, -45.0, 15.0, 75.0, 0.0, 50.0),
)
BATCH_PER_WORKER = 64

_MODEL = None


def _init_worker():
    global _MODEL
    _MODEL = mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def _reach(key):
    x, y, z, grip = key
    return key, sc.reachable(_MODEL, (x, y, z), grip)


def _poses(key):
    x, y, z = key
    return key, sc.grasp_poses(_MODEL, (x, y, z))


def candidates():
    for width in WIDTHS:
        for depth in DEPTHS:
            for wall in WALLS:
                for cx in CENTERS:
                    yield sl.Layout(depth=depth, width=width, center_x=cx, wall=wall)


def describe(layout):
    return (f"bin {layout.width * 100:.0f} cm at x={layout.center_x:.2f}, riser {layout.depth * 100:.0f} cm, "
            f"walls {layout.wall * 100:.0f} cm, travel z {layout.travel_z:+.3f}")


class Sizer:
    """Runs the reach and clearance checks, caching IK across candidates that share a height."""

    def __init__(self, pool, workers):
        self.pool = pool
        self.batch = BATCH_PER_WORKER * workers
        self.reach = {}
        self.poses = {}

    def misses(self, points, z, grip):
        keys = [(x, y, round(z, 4), grip) for x, y in points]
        todo = [k for k in keys if k not in self.reach]
        for i in range(0, len(todo), self.batch):
            batch = todo[i:i + self.batch]
            for key, ok in self.pool.imap_unordered(_reach, batch, chunksize=4):
                self.reach[key] = ok
            if not all(self.reach[k] for k in batch):
                break
        return [k[:2] for k in keys if k in self.reach and not self.reach[k]]

    def check_reach(self, layout):
        points = sc.grid_points(layout)
        f = layout.floor_z
        checks = (  # likeliest to fail first
            ("travel holding a cube", layout.travel_z, primitives.GRIP_HOLDING),
            ("travel with empty jaws", layout.travel_z, primitives.GRIP_PLAN),
            ("layer-3 grasp", f + 5 * primitives.CUBE_HALF, primitives.GRIP_PLAN),
            ("layer-1 grasp", f + primitives.CUBE_HALF, primitives.GRIP_PLAN),
        )
        for name, z, grip in checks:
            missed = self.misses(points, z, grip)
            if missed:
                return f"{name} at z={z:+.3f} misses {len(missed)} or more points, first {missed[0]}"
        return None

    def check_clearance(self, layout):
        model = sl.load(layout)
        data = mujoco.MjData(model)
        arm, scene = grasp.arm_geom_ids(model), sc.scene_geom_ids(model)
        z = round(layout.floor_z + primitives.CUBE_HALF, 4)
        keys = [(x, y, z) for x, y in sc.grid_points(layout) if not sc.near_wall(layout, (x, y))]
        todo = [k for k in keys if k not in self.poses]
        for key, poses in self.pool.imap_unordered(_poses, todo, chunksize=4):
            self.poses[key] = poses
        blocked = [k[:2] for k in keys if not sc.clearance_ok(model, data, self.poses[k], arm, scene)]
        if blocked:
            return f"open jaw or arm hits the bin, riser or table at {len(blocked)} points, first {blocked[0]}"
        return None


def coverage(layout, camera, home):
    model = sl.load(layout, probes=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *sc.IMAGE_SIZE)
    try:
        return sc.marker_coverage(model, data, renderer, camera, home)
    finally:
        renderer.close()


def pick_home(layout):
    model = sl.load(layout, probes=True)
    data = mujoco.MjData(model)
    for home in HOME_CANDIDATES:
        if not sc.drop_column_clear(model, data, home):
            print(f"  start pose {list(home)}: the arm reaches into the drop column")
            continue
        seen = coverage(layout, "wrist_cam", home)
        print(f"  start pose {list(home)}: the wrist camera sees {seen:.0%} of the floor")
        if seen >= sc.WRIST_MIN:
            return home, seen
    return None, None


def pick_camera(layout):
    for cam_z in CAM_ZS:
        trial = dataclasses.replace(layout, cam_z=cam_z)
        seen = coverage(trial, "front", trial.home)
        print(f"  front camera at z={cam_z:.2f}: sees {seen:.0%} of the floor")
        if seen >= sc.FRONT_MIN:
            return trial, seen
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite models/search_objects.xml")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    started = time.time()

    chosen, tried = None, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        sizer = Sizer(pool, args.workers)
        for layout in candidates():
            why = sizer.check_reach(layout) or sizer.check_clearance(layout)
            print(f"{describe(layout)}: {why or 'passes'}", flush=True)
            if why is None:
                chosen = layout
                break
            tried += 1
    if chosen is None:
        sys.exit("STOP: no bin passes, down to 12 cm wide. Bring this back to dm1487 before building more.")

    home, wrist = pick_home(chosen)
    if home is None:
        sys.exit("STOP: no start pose keeps the arm out of the drop column with the wrist camera seeing half "
                 "the floor. Bring this back to dm1487.")
    chosen, front = pick_camera(dataclasses.replace(chosen, home=home))
    if chosen is None:
        sys.exit("STOP: no front camera height sees 70% of the floor. Bring this back to dm1487.")

    # Keep a cube count that scripts/mechsearch_smoke.py --tune-count already picked.
    committed = sl.Layout.from_model(mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH)))
    chosen = dataclasses.replace(chosen, n_cubes=committed.n_cubes)
    notes = [
        f"Chosen {datetime.date.today()} by scripts/size_search_bin.py in {time.time() - started:.0f} s, "
        f"after {tried} better-ranked candidates failed.",
        describe(chosen) + ".",
        "Every cube centre on a 5 mm grid passes top-down reach at layer 1, layer 3 and travel height, and "
        "the open jaw clears the bin, riser and table everywhere more than 2 cm from a wall.",
        f"Front camera at z={chosen.cam_z:.2f} sees {front:.0%} of the empty floor. Start pose "
        f"{list(chosen.home)} stays out of the drop column, and its wrist camera sees {wrist:.0%}.",
    ]
    notes += [n for n in sl.read_notes() if n.startswith("Cube count")]
    print("\n".join(notes))
    if args.write:
        sl.write(chosen, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


if __name__ == "__main__":
    main()
