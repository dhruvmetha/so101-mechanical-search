"""Shelf scenes: upright cylinders (the red and one or more grays) at given board positions.

A scene is a small JSON file in experiments/sahal/scenes/ (or a subfolder, e.g. scenes/random4/):

    {"name": "L1_front", "level": "L1", "description": "one gray right in front of the red",
     "friction": 0.55, "positions": {"cyl_red": [0.43, 0.0], "cyl_gray_0": [0.37, 0.0], "cyl_gray_1": [0.44, 0.11]}}

    scene = load_scene(SCENES_DIR / "L1_front.json")
    report = place_scene(env, scene)              # env: RedAndGraysShelf; raises SceneError if it doesn't stand
    report = place_scene(env, scene, friction=0.4)
    in_graspable_zone(xy)                         # study 01's empty-shelf zone where pick + pull work
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import numpy as np

from common.shelf_env import RedAndGraysShelf
from so101sim.shelf import TARGET
from so101sim import shelf_layout

SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"
GRAY_PREFIX = "cyl_gray_"
GRASP_ZONE_CSV = Path(__file__).resolve().parents[1] / "studies" / "01_grasp_pull" / "outputs" / "heatmap_runs.csv"
GRASP_ZONE_GRID_M = 0.01  # study 01's cell spacing; its cell centres sit at (k + 0.5) * GRASP_ZONE_GRID_M
DENSITY_KG_M3 = 900.0
MIN_SURFACE_GAP_M = 0.005  # cylinders at least this far apart, surface to surface (the env's own spacing)
MAX_SETTLE_SHIFT_M = 0.002  # a cylinder may move at most this far while settling
MAX_SETTLE_TILT_DEG = 2.0
RESET_SEED = 0


class SceneError(ValueError):
    """A scene that can't be placed: overlapping, off the board, or not standing still and upright."""


@dataclass(frozen=True)
class Scene:
    """Where the red and the grays stand (world metres) and the board friction."""

    name: str
    level: str
    description: str
    friction: float
    positions: dict[str, tuple[float, float]]


def save_scene(scene: Scene, directory: Path = SCENES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene.name}.json"
    data = asdict(scene)
    data["positions"] = {k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in scene.positions.items()}
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path


def load_scene(path: Path) -> Scene:
    data = json.loads(Path(path).read_text())
    return Scene(name=data["name"], level=data["level"], description=data["description"],
                 friction=float(data["friction"]),
                 positions={k: (float(v[0]), float(v[1])) for k, v in data["positions"].items()})


def load_scenes(directory: Path = SCENES_DIR) -> list[Scene]:
    return [load_scene(p) for p in sorted(directory.glob("*.json"))]


def layout_problems(scene: Scene, layout) -> list[str]:
    """Why the scene's positions are not a valid start, before any physics: no red, no gray or an unknown name,
    overlaps, off the board (the whole footprint must be on it and inside the walls). Empty if fine."""
    problems = []
    radius = shelf_layout.CYL_RADIUS  # read at call time: common.cylinder_size may change it
    grays = [n for n in scene.positions if n != TARGET]
    if TARGET not in scene.positions or not grays or any(not n.startswith(GRAY_PREFIX) for n in grays):
        problems.append(f"needs {TARGET} and one or more {GRAY_PREFIX}N, got {sorted(scene.positions)}")
    names = sorted(scene.positions)
    for i, a in enumerate(names):
        xy = np.asarray(scene.positions[a])
        if not (layout.front_x + radius <= xy[0] <= layout.back_x - radius
                and abs(xy[1]) <= layout.half_width - radius):
            problems.append(f"{a} at {xy.tolist()} is not fully on the board")
        for b in names[i + 1:]:
            gap = float(np.linalg.norm(xy - np.asarray(scene.positions[b]))) - 2 * radius
            if gap < MIN_SURFACE_GAP_M:
                problems.append(f"{a} and {b} are {gap * 1000:.1f} mm apart (minimum {MIN_SURFACE_GAP_M * 1000:g})")
    return problems


def place_scene(env: RedAndGraysShelf, scene: Scene, friction: float | None = None) -> dict:
    """Reset the env with the scene's cylinders, let them settle, and check they stand where they were put.
    `friction` overrides the scene's. Returns reset_scene's report. Raises SceneError."""
    problems = layout_problems(scene, env.env.layout)
    if problems:
        raise SceneError(f"{scene.name}: " + "; ".join(problems))
    report = env.reset_scene(seed=RESET_SEED, red_xy=scene.positions[TARGET],
                             gray_xys={g: xy for g, xy in scene.positions.items() if g != TARGET},
                             physics={"friction": scene.friction if friction is None else friction,
                                      "density": DENSITY_KG_M3})
    for name, xy in scene.positions.items():
        shift = float(np.linalg.norm(report["xy"][name] - np.asarray(xy)))
        if shift > MAX_SETTLE_SHIFT_M:
            problems.append(f"{name} moved {shift * 1000:.1f} mm while settling")
        if report["tilt"][name] > MAX_SETTLE_TILT_DEG:
            problems.append(f"{name} settled tilted {report['tilt'][name]:.1f} deg")
    if not report["resting"]:
        problems.append("did not come to rest")
    if report["touched_arm"]:
        problems.append("a cylinder touched the arm while settling")
    if problems:
        raise SceneError(f"{scene.name}: " + "; ".join(problems))
    return report


def graspable_cells(path: Path = GRASP_ZONE_CSV, action_box_only: bool = True,
                    grid_m: float = GRASP_ZONE_GRID_M) -> set[tuple[int, int]]:
    """Grid cells (integer indices) where every run grasped the lone red and pulled it out, from a CSV with columns
    x, y, in_action_box, pull_success and cell centres at (k + 0.5) * grid_m. Default: study 01's 1 cm cells.
    Study 01 widened the action box; with `action_box_only` (default) only cells the unmodified env accepts count,
    since its pick refuses a red outside the box (296 of study 01's 396 graspable cells)."""
    runs: dict[tuple[int, int], list[bool]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if action_box_only and row["in_action_box"] != "True":
                continue
            cell = (_cell_index(float(row["x"]), grid_m), _cell_index(float(row["y"]), grid_m))
            runs.setdefault(cell, []).append(row["pull_success"] == "True")
    return {cell for cell, ok in runs.items() if all(ok)}


def in_graspable_zone(xy, cells: set[tuple[int, int]], grid_m: float = GRASP_ZONE_GRID_M) -> bool:
    """True if all four cells around `xy` (the corners of its grid square) are graspable."""
    i, j = (float(xy[0]) / grid_m - 0.5, float(xy[1]) / grid_m - 0.5)
    return all((a, b) in cells for a in (int(np.floor(i)), int(np.ceil(i))) for b in (int(np.floor(j)), int(np.ceil(j))))


def _cell_index(coordinate_m: float, grid_m: float = GRASP_ZONE_GRID_M) -> int:
    """Grid index k of a cell centre at (k + 0.5) * grid_m."""
    return round(coordinate_m / grid_m - 0.5)
