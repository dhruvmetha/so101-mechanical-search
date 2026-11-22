"""Hand-made shelf scenes: exactly 3 upright cylinders (the red and 2 grays) at given board positions.

A scene is a small JSON file in experiments/sahal/scenes/:

    {"name": "L1_front", "level": "L1", "description": "one gray right in front of the red",
     "friction": 0.55, "positions": {"cyl_red": [0.43, 0.0], "cyl_gray_0": [0.37, 0.0], "cyl_gray_1": [0.44, 0.11]}}

    scene = load_scene(SCENES_DIR / "L1_front.json")
    report = place_scene(env, scene)              # env: RedAndGraysShelf; raises SceneError if it doesn't stand
    report = place_scene(env, scene, friction=0.4)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import numpy as np

from common.shelf_env import RedAndGraysShelf
from so101sim.shelf import TARGET
from so101sim.shelf_layout import CYL_RADIUS

SCENES_DIR = Path(__file__).resolve().parents[1] / "scenes"
GRAYS = ("cyl_gray_0", "cyl_gray_1")
DENSITY_KG_M3 = 900.0
MIN_SURFACE_GAP_M = 0.005  # cylinders at least this far apart, surface to surface (the env's own spacing)
MAX_SETTLE_SHIFT_M = 0.002  # a cylinder may move at most this far while settling
MAX_SETTLE_TILT_DEG = 2.0
RESET_SEED = 0


class SceneError(ValueError):
    """A scene that can't be placed: overlapping, off the board, or not standing still and upright."""


@dataclass(frozen=True)
class Scene:
    """Where the red and the two grays stand (world metres) and the board friction."""

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
    """Why the scene's positions are not a valid start, before any physics: wrong cylinders, overlaps, off the
    board (the whole footprint must be on it and inside the walls). Empty if fine."""
    problems = []
    if set(scene.positions) != {TARGET, *GRAYS}:
        problems.append(f"needs exactly {TARGET} and {', '.join(GRAYS)}, got {sorted(scene.positions)}")
    names = sorted(scene.positions)
    for i, a in enumerate(names):
        xy = np.asarray(scene.positions[a])
        if not (layout.front_x + CYL_RADIUS <= xy[0] <= layout.back_x - CYL_RADIUS
                and abs(xy[1]) <= layout.half_width - CYL_RADIUS):
            problems.append(f"{a} at {xy.tolist()} is not fully on the board")
        for b in names[i + 1:]:
            gap = float(np.linalg.norm(xy - np.asarray(scene.positions[b]))) - 2 * CYL_RADIUS
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
                             gray_xys={g: scene.positions[g] for g in GRAYS},
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
