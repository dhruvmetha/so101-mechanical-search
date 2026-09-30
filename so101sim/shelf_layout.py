"""The shelf scene, generated from a few numbers.

models/shelf_objects.xml and models/shelf_scene.xml are written by `write`, never by hand. scripts/size_shelf.py
picks the layout and scripts/shelf_smoke.py picks the cylinder count. The numbers sit in <custom> fields, so
`ShelfLayout.from_model` reads them back out of any compiled model.

World z = 0 is the plane the arm's base stands on, which is also the table top. The shelf unit stands on the table
in front of the arm, facing it: a board at `board_z`, two side walls, a back wall and a top board `opening` above
the board. The gripper goes in level along rays from the pan axis, so everything the task needs sits above the
base, where level reach is wide. Measurements: docs/superpowers/specs/2026-09-26-shelf-search-design.md.
"""

import contextlib
import dataclasses
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from so101sim import SHELF_SCENE_PATH
from so101sim.env import HOME

MODELS_DIR = SHELF_SCENE_PATH.parent
OBJECTS_PATH = MODELS_DIR / "shelf_objects.xml"

PANEL = 0.015  # every shelf panel is 1.5 cm thick
LEG_HALF = 0.008
TABLE_HALF = (0.45, 0.45, 0.02)
TABLE_CENTER_X = 0.25
CAM_FOVY = 55
SHELF_RGBA = "0.72 0.58 0.40 1"

TARGET_NAME = "cyl_red"
CYL_RADIUS = 0.015
CYL_HALF_HEIGHT = 0.04
# The cube's density and friction, inline on each geom rather than in a default class. condim 6, not the cube's 4:
# condim 4 ignores the rolling coefficient, and a cylinder pushed off-centre wobbled on its rim indefinitely, its
# tilt growing from 7.8 to 9.0 degrees over 10 s. With condim 6 it stood still before the push ended (2026-09-26).
CYL_ATTRS = (f'type="cylinder" size="{CYL_RADIUS} {CYL_HALF_HEIGHT}" density="1000" condim="6" '
             'friction="1 0.005 0.0001"')
RED_RGBA = "0.85 0.15 0.15 1"
GRAY_RGBA = "0.6 0.6 0.62 1"
GRAY_GROUP = 1  # grays go in geom group 1 so a render can leave them out, as the bin does

GRASP_UP = 0.04  # jaw midpoint over the board for a grasp: halfway up a cylinder
CARRY_UP = 0.05  # a held cylinder rides 1 cm off the board
# Measured 2026-09-26 with a 5 cm sideways push on a lone cylinder: at 2 cm it ended leaning 9 degrees, at 2.5 cm it
# slid 4 cm and ended upright, at 3 cm it toppled.
PUSH_UP = 0.025
AISLE = 0.02  # a held cylinder only moves sideways with its centre this far behind the front edge
STANDOFF = 0.03  # every primitive starts and ends with the jaw midpoint this far outside the front edge
ROW_START = 0.055  # reset keeps centres this far behind the front edge, 5 mm clear of a cylinder in the aisle
WALL_GAP = 0.005  # and r + this from the side and back walls

# Cylinders wait on the ground plane, behind the shelf, until reset places them.
PARK_X = 1.2
PARK_SPACING = 0.05
GROUND_Z = -0.75

# Probe scenes only: 6 mm tiles on the board, 1 cm apart, that the camera checks count in a segmentation render.
MARKER_PREFIX = "marker_"
MARKER_STEP = 0.01
MARKER_HALF = 0.003

NOTES_HEADER = "       Notes:"
NOTE_PREFIX = "       * "


@dataclasses.dataclass(frozen=True)
class ShelfLayout:
    board_z: float = 0.06  # board top above the base plane
    opening: float = 0.13  # board top to the top board's underside
    front_x: float = 0.30  # the board's front edge
    depth: float = 0.18  # front edge to the back wall's inner face
    width: float = 0.24  # side wall to side wall, inside
    cam_pos: tuple = (0.05, -0.30, 0.40)  # front camera
    n_cyls: int = 18  # one red, the rest gray
    home: tuple = tuple(float(v) for v in HOME)  # start pose, lerobot units

    @property
    def back_x(self):
        return self.front_x + self.depth

    @property
    def half_width(self):
        return self.width / 2

    @property
    def grasp_z(self):
        return self.board_z + GRASP_UP

    @property
    def carry_z(self):
        return self.board_z + CARRY_UP

    @property
    def push_z(self):
        return self.board_z + PUSH_UP

    @property
    def aisle_x(self):
        return self.front_x + AISLE

    @property
    def standoff_x(self):
        return self.front_x - STANDOFF

    def region(self):
        """(x_lo, x_hi, y_lo, y_hi) for cylinder centres: where reset puts them and where the primitive env acts."""
        inset = CYL_RADIUS + WALL_GAP
        return (self.front_x + ROW_START, self.back_x - inset, -self.half_width + inset, self.half_width - inset)

    def on_board(self, xy):
        return bool(self.front_x <= xy[0] <= self.back_x and abs(xy[1]) <= self.half_width)

    @classmethod
    def from_model(cls, model):
        d = model.numeric("shelf_layout").data
        return cls(board_z=float(d[0]), opening=float(d[1]), front_x=float(d[2]), depth=float(d[3]),
                   width=float(d[4]), n_cyls=int(round(d[5])),
                   cam_pos=tuple(float(v) for v in model.numeric("shelf_camera").data),
                   home=tuple(float(v) for v in model.numeric("shelf_home").data))


def _f(*values):
    return " ".join(f"{float(v):.6g}" for v in values)


def _box(name, lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return (f'      <geom name="{name}" type="box" size="{_f(*((hi - lo) / 2))}" pos="{_f(*((hi + lo) / 2))}" '
            f'rgba="{SHELF_RGBA}"/>')


def cylinder_names(n):
    return [TARGET_NAME] + [f"cyl_gray_{i}" for i in range(n - 1)]


def objects_xml(layout, probes=False, notes=()):
    """The objects file for `layout`. `probes` adds the board markers."""
    for note in notes:
        if "--" in note:
            raise ValueError(f"an XML comment can't hold '--': {note!r}")
    L, t = layout, PANEL
    f, b, hw, back, top = L.front_x, L.board_z, L.half_width, L.back_x, L.board_z + L.opening
    legs = ((f + LEG_HALF, -hw - t + LEG_HALF), (f + LEG_HALF, hw + t - LEG_HALF),
            (back + t - LEG_HALF, -hw - t + LEG_HALF), (back + t - LEG_HALF, hw + t - LEG_HALF))
    out = [
        "<mujoco>",
        "  <!-- Generated by so101sim/shelf_layout.py, never edited by hand. scripts/size_shelf.py picks the layout",
        "       and scripts/shelf_smoke.py picks the cylinder count, each with its write flag.",
        NOTES_HEADER,
        *[NOTE_PREFIX + note for note in notes],
        "  -->",
        "  <custom>",
        f'    <numeric name="shelf_layout" data="{_f(L.board_z, L.opening, L.front_x, L.depth, L.width, L.n_cyls)}"/>',
        f'    <numeric name="shelf_camera" data="{_f(*L.cam_pos)}"/>',
        f'    <numeric name="shelf_home" data="{_f(*L.home)}"/>',
        "  </custom>",
        "  <worldbody>",
        f'    <geom name="table" type="box" size="{_f(*TABLE_HALF)}" pos="{_f(TABLE_CENTER_X, 0, -TABLE_HALF[2])}" '
        'material="table"/>',
        '    <body name="shelf">',
        _box("shelf_board", [f, -hw - t, b - t], [back + t, hw + t, b]),
        _box("shelf_top", [f, -hw - t, top], [back + t, hw + t, top + t]),
        _box("shelf_back", [back, -hw, b], [back + t, hw, top]),
        _box("shelf_left", [f, hw, b], [back, hw + t, top]),
        _box("shelf_right", [f, -hw - t, b], [back, -hw, top]),
        *[_box(f"shelf_leg_{i}", [x - LEG_HALF, y - LEG_HALF, 0.0], [x + LEG_HALF, y + LEG_HALF, b - t])
          for i, (x, y) in enumerate(legs)],
    ]
    if probes:
        for i, x in enumerate(np.arange(f + 0.005, back, MARKER_STEP)):
            for j, y in enumerate(np.arange(-hw + 0.005, hw, MARKER_STEP)):
                out.append(f'      <geom name="{MARKER_PREFIX}{i}_{j}" type="box" '
                           f'size="{_f(MARKER_HALF, MARKER_HALF, 0.0005)}" pos="{_f(x, y, b + 0.0005)}" '
                           'contype="0" conaffinity="0" rgba="0.9 0.9 0.2 1"/>')
    out += [
        "    </body>",
        f'    <body name="front_cam_target" pos="{_f(f + L.depth / 2, 0, L.grasp_z)}"/>',
        f'    <camera name="front" mode="targetbody" target="front_cam_target" pos="{_f(*L.cam_pos)}" '
        f'fovy="{CAM_FOVY}"/>',
    ]
    names = cylinder_names(L.n_cyls)
    for i, name in enumerate(names):
        red = name == TARGET_NAME
        y = (i - (len(names) - 1) / 2) * PARK_SPACING
        out += [
            f'    <body name="{name}" pos="{_f(PARK_X, y, GROUND_Z + CYL_HALF_HEIGHT)}">',
            f'      <freejoint name="{name}"/>',
            f'      <geom name="{name}" {CYL_ATTRS} rgba="{RED_RGBA if red else GRAY_RGBA}"'
            + ("" if red else f' group="{GRAY_GROUP}"') + "/>",
            "    </body>",
        ]
    out += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(out)


def scene_xml(objects_file):
    return ('<mujoco model="so101_shelf">\n'
            "  <!-- Generated by so101sim/shelf_layout.py: the shared stage, then the shelf objects. -->\n"
            '  <include file="stage.xml"/>\n'
            f'  <include file="{objects_file}"/>\n'
            "</mujoco>\n")


def write(layout, notes=()):
    """Rewrite the committed shelf scene for `layout`."""
    OBJECTS_PATH.write_text(objects_xml(layout, notes=notes))
    SHELF_SCENE_PATH.write_text(scene_xml(OBJECTS_PATH.name))


@contextlib.contextmanager
def candidate_scene(layout, probes=False):
    """Path to a compilable scene for `layout`, deleted on exit.

    The files go in models/ because MuJoCo resolves `meshdir` against the top-level file. .gitignore covers the
    names in case a crash leaves them. A fresh compile per candidate is also the only way to move the shelf:
    editing geom_pos on a compiled model doesn't move a world-body geom.
    """
    fd, objects = tempfile.mkstemp(prefix=".candidate_shelf_", suffix="_objects.xml", dir=MODELS_DIR)
    os.close(fd)
    scene = objects[: -len("_objects.xml")] + "_scene.xml"
    try:
        Path(objects).write_text(objects_xml(layout, probes=probes))
        Path(scene).write_text(scene_xml(Path(objects).name))
        yield Path(scene)
    finally:
        for path in (objects, scene):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)


def load(layout, probes=False):
    """Compile `layout` without touching the committed scene."""
    with candidate_scene(layout, probes) as path:
        return mujoco.MjModel.from_xml_path(str(path))


def parse_notes(xml):
    lines = xml.splitlines()
    if NOTES_HEADER not in lines:
        return []
    notes = []
    for line in lines[lines.index(NOTES_HEADER) + 1:]:
        if not line.startswith(NOTE_PREFIX):
            break
        notes.append(line[len(NOTE_PREFIX):])
    return notes


def read_notes(path=None):
    """The notes in the committed objects file, or [] if there isn't one yet."""
    path = OBJECTS_PATH if path is None else Path(path)
    return parse_notes(path.read_text()) if path.exists() else []
