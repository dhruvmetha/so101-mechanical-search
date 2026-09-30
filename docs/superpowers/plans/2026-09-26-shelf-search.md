# Shelf search implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A shelf bay in front of the SO-101 holds rows of upright 3 by 8 cm cylinders. A 30 Hz env and a
pick/place/push/pull/look primitive env ask a policy to find the one red cylinder and pull it out, with nothing
toppled and every gray still on the board. Then Claude plays the primitive env from images as the planner.

**Architecture:** A `ShelfLayout` dataclass generates the shelf scene, and a sizing script picks the layout by
level-gripper IK and collision checks. A new `ik.solve_side` keeps the gripper level with the wrist camera mount
on top, and `side_primitives.py` builds every move as straight lines along rays from the pan axis. `ShelfEnv`
subclasses `SO101Env` and places cylinders at reset and checks topple, off-board and success every frame.
`ShelfPrimitiveEnv` wraps it the way `MechSearchPrimitiveEnv` wraps `MechSearchEnv`. `scripts/shelf_play.py`
turns pixel choices into actions for the planner run.

**Tech stack:** Python 3.10, mujoco 3.8.1 on CPU with EGL rendering, gymnasium, numpy, pytest, imageio, Pillow.

**Spec:** `docs/superpowers/specs/2026-09-26-shelf-search-design.md`. Read it first. This plan argues from it.

## Global constraints

- Run Python as `PY=/common/users/dm1487/envs/so101sim/bin/python`. `conda activate` fails in this non-interactive
  shell. Tests: `$PY -m pytest -q -m "not mjx"`.
- `import so101sim` before `mujoco` in every script and test module.
- Call `close()` on every env and renderer you create.
- Never edit `models/robotstudio_so101/`, `models/stage.xml`, `so101sim/mechsearch.py`, `so101sim/search_*.py`
  or any bin file. Another session owns them. Import from them, don't change them.
- Lerobot units outside MuJoCo: arm joints in degrees, gripper 0 closed to 100 open, `units.JOINT_NAMES` order.
- Cylinders: radius 0.015 m, height 0.08 m, density 1000, condim 4, friction "1 0.005 0.0001".
- Topple: any cylinder tilted more than 20 degrees from vertical, held or not, ends the episode.
- Off the board: a gray whose centre leaves the board's footprint ends the episode, and so does the red when not
  held by both jaws.
- Success: red held by both jaws, centre at least 0.02 m in front of the front edge, tilt under 20 degrees, 5
  frames in a row.
- `ShelfEnv` defaults: `max_steps=2700`, `hidden_prob=0.5`. `ShelfPrimitiveEnv` defaults: `max_primitives=20`,
  push stroke 0.05 m, 16 push headings.
- Heights: grasp at board + 0.04, carry at board + 0.05. Lines: aisle at front + 0.02, standoff at front − 0.03.
  Reset keeps every centre at x ≥ front + 0.055, at least 0.035 m apart, r + 0.005 from side and back walls.
- Reset settles up to 0.5 s or until max|qvel| ≤ 0.2. A start needs every cylinder upright within 2 degrees, a free
  spot 0.045 m from every centre, and no arm contact. Hidden: zero face pixels in both cameras at 640 by 480 with
  `mechsearch.face_pixels`. Partial: at least 100 in one camera and at most half of the unoccluded count in each.
- Sizing: board 3, 6 or 9 cm; opening 13, 15 or 18 cm; front edge 0.27, 0.285 or 0.30 m; depth 16 to 22 cm; width
  20 to 32 cm. Front camera at least 60% of the board, wrist at least 40% at `SHELF_HOME`.
- Cylinder count starts at 18 and floors at 8. At least half of resets accepted for hidden and partial separately.
  Reset averaging over 3 s gets reported, not fixed by cutting cylinders.
- Stop points. Stop and bring it back to dm1487 if:
  - no layout reaches 216 cm² of graspable centres (Task 4),
  - no start pose or front camera position passes (Task 4),
  - the count falls below 8 (Task 7),
  - the dig-out gate fails on all of seeds 0 to 19 (Task 8).

  Never relax a threshold to get past one.
- Git: don't stage and don't commit. `mechanical-search` has no commits yet and another session's work is staged.
  dm1487 decides how this lands.
- Scratch: `SCRATCH=/tmp/claude-89862/-common-home-dm1487-robotics-research-rearrange/11e13158-8a7b-4ccb-ae12-5b7ed70d2564/scratchpad`.
- Rendering: check `nvidia-smi`, then `export MUJOCO_EGL_DEVICE_ID=2` or another free card.
- Videos: dm1487 asked for a video after every run that makes one. Send each mp4 with SendUserFile as soon as it's
  written, with a one-line caption saying what it shows and how the run ended.

## Where this departs from the spec's wording

A scratch spike on 2026-09-26 ran a level side pick, aisle carry, pull, place and low pushes on a mock shelf with
free cylinders. It measured these and they change details, not requirements.

- **Rays start at the pan axis**, `ik.pan_axis_xy`, which sits at (0.0388, 0), not at the origin.
- **Level reach's inner limit** is 0.220 m from the pan axis at grasp height with `GRIP_PLAN`, 0.225 m holding at
  carry height, and 0.240 m closed at push height. The outer limit is 0.440 m. So the standoff line at front − 0.03
  only works with the front edge at 0.30 m. The spec's candidates stay; the checks reject 0.27 and 0.285.
- **Push enters at grasp height and drops to push height past the front edge**, because level reach can't get as
  close to the base at push height. So sizing check 4 covers the standoff line at grasp and carry height only, and
  check 6 covers push height on the front-edge line, the aisle line and the centres.
- **Push height is fixed at board + 2.5 cm.** The spike pushed a lone cylinder 5 cm sideways: at 2 cm it ended
  leaning 9 degrees, at 2.5 cm it slid 4 cm with 5 degrees of peak tilt and ended upright, and at 3 cm it toppled.
  A test pins that behaviour instead of choosing the lowest height that clears the board.
- **A push pixel lands on the board plane**, like place and look. A push starts beside a cylinder, not on its axis.
  Only pick deprojects to the grasp plane and steps one radius in.
- **Cylinder geoms carry their attributes inline**, not through a `cylinder` default class. A second top-level
  `<default>` in an included file is untested here and nothing needs it.
- **`SHELF_HOME` and the front camera position live in the XML**, as `<numeric name="shelf_home">` and
  `<numeric name="shelf_camera">`, read back by `ShelfLayout.from_model`, the same pattern as the bin.
- **The stop rule's "12 by 18 cm" is an area**, 216 cm². The reach band is only about 11 cm deep past the aisle,
  so the region comes out wide and shallow.
- **Holding** means the gripper was last commanded at most `GRIP_CLOSE` + 1 and the joint reads at least 14. An
  empty close reaches 10. A 3 cm cylinder stops it near 18.
- **The side-open gripper value** is the smallest whole lerobot value with a jaw gap of at least 5 cm, 33 by the
  spike's gap curve. IK plans at `GRIP_PLAN`, so the open jaws aren't centred: the fixed jaw slides in about 3 mm off
  the cylinder, and the moving jaw has the rest.
- **`info["cylinders"]`** carries each cylinder's position and tilt, for tests and scripted baselines.

## Review focus

The five inputs most likely to bite someone using this, which the spec doesn't pin down:

1. A pick at an empty spot closes on nothing and leaves the arm in the aisle. The next primitive has to back out
   along the ray, not ramp joints through the bay. Test: Task 8, `test_a_missed_pick_backs_out_cleanly`.
2. A planner pixel on the top board, the sky or the table deprojects somewhere off the board. It has to be refused,
   not crash. Test: Task 9, `test_a_pixel_off_the_board_is_refused`.
3. A wrist pixel chosen after the arm moved has to use the wrist pose at that moment. Test: Task 9,
   `test_wrist_pixels_deproject_from_the_current_pose`.
4. A push heading toward the robot ends outside the region. Refused. Test: Task 8,
   `test_a_push_out_of_the_shelf_is_refused`.
5. `reset()` with no seed and no options. Test: Task 5, `test_reset_needs_no_seed_or_options`.

## File map

| File | Task | Responsibility |
|---|---|---|
| `so101sim/ik.py` | 1 | gains `pan_axis_xy`, `solve_side`, `SIDE_UP_AXIS`, `UP` |
| `tests/test_side_ik.py` | 1 | level IK |
| `so101sim/__init__.py` | 2 | gains `SHELF_SCENE_PATH` |
| `so101sim/shelf_layout.py` | 2 | `ShelfLayout`, XML generation, candidate scenes |
| `models/shelf_objects.xml`, `models/shelf_scene.xml` | 2, 4, 7 | generated, never hand-edited |
| `tests/test_shelf_layout.py` | 2 | layout and XML |
| `so101sim/side_primitives.py` | 3 | `Geometry`, pick, place, pull, push, look, bridge |
| `tests/test_side_primitives.py` | 3 | trajectories, no physics |
| `so101sim/shelf_checks.py` | 4 | sizing checks |
| `scripts/size_shelf.py` | 4 | picks the layout, start pose and camera |
| `tests/test_shelf_checks.py` | 4 | checks and the committed layout |
| `so101sim/shelf.py` | 5, 6, 8 | `ShelfEnv`, `ShelfPrimitiveEnv` |
| `tests/test_shelf.py` | 5, 6, 8 | env behaviour and the dig-out gate |
| `scripts/shelf_smoke.py` | 7 | reset video, acceptance, cylinder count |
| `scripts/shelf_play.py` | 9 | pixel actions and replay for the planner |
| `tests/test_shelf_play.py` | 9 | lens model and deprojection |
| `outputs/shelf_play_300*`, `outputs/shelf_play_301*` | 10 | planner runs, video and log |
| `CLAUDE.md` | 11 | new files, commands, traps, hardware requirement |

---

### Task 1: Level side IK

**Files:**
- Modify: `so101sim/ik.py` (append after `_topdown_from`)
- Test: `tests/test_side_ik.py`

**Interfaces:**
- Produces: `ik.pan_axis_xy(model) -> np.ndarray (2,)`; `ik.solve_side(model, target, gripper=0.0, q_init=None,
  restarts=6, pos_tol=1e-3, ...) -> (q (5,) radians, {"pos_err": m, "tilt": deg, "heading": deg})`;
  `ik.SIDE_UP_AXIS = 1`; `ik.UP`.

- [ ] **Step 1: Write the failing tests**

`tests/test_side_ik.py`:

```python
import numpy as np
import pytest

from so101sim import SCENE_PATH, ik, primitives

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def plan_grip(model):
    return primitives._grip_rad(model, primitives.GRIP_PLAN)


def test_the_pan_axis_sits_in_front_of_the_origin(model):
    np.testing.assert_allclose(ik.pan_axis_xy(model), [0.0388, 0.0], atol=0.001)


@pytest.mark.parametrize("target", [(0.36, 0.0, 0.10), (0.42, 0.10, 0.10), (0.30, -0.12, 0.11)])
def test_solve_side_reaches_level_with_the_mount_up(model, target):
    q, info = ik.solve_side(model, target, gripper=plan_grip(model))
    assert info["pos_err"] < 1e-3 and info["tilt"] < 1.0, info
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[[model.joint(n).id for n in ik.ARM_JOINTS]]] = q
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = plan_grip(model)
    mujoco.mj_kinematics(model, data)
    R = data.site_xmat[model.site("gripperframe").id].reshape(3, 3).copy()
    assert R[2, ik.SIDE_UP_AXIS] > 0.999  # the camera mount points up
    assert abs(R[2, ik.TOOL_AXIS]) < 0.02 and abs(R[2, ik.JAW_AXIS]) < 0.02  # tool and jaws level
    np.testing.assert_allclose(ik.grasp_point(model, data), target, atol=1e-3)


def test_heading_follows_the_ray_from_the_pan_axis(model):
    pan = ik.pan_axis_xy(model)
    for y in (-0.10, 0.0, 0.10):
        _, info = ik.solve_side(model, (0.38, y, 0.10), gripper=plan_grip(model))
        ray = np.degrees(np.arctan2(y - pan[1], 0.38 - pan[0]))
        assert abs(info["heading"] - ray) < 8.0, (y, info)


def test_solve_side_reports_what_it_cannot_reach(model):
    # 0.11 m from the pan axis, well inside level reach's 0.22 m inner limit
    _, info = ik.solve_side(model, (0.15, 0.0, 0.10), gripper=plan_grip(model), restarts=2)
    assert info["pos_err"] > 5e-3 or info["tilt"] > 2.0, info


def test_a_seed_keeps_the_solution_on_its_branch(model):
    q0, _ = ik.solve_side(model, (0.36, 0.0, 0.10), gripper=plan_grip(model))
    q1, info = ik.solve_side(model, (0.364, 0.0, 0.10), gripper=plan_grip(model), q_init=q0, restarts=0)
    assert info["pos_err"] < 1e-3
    assert np.abs(q1 - q0).max() < np.radians(5)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_side_ik.py -q`
Expected: FAIL with `AttributeError: module 'so101sim.ik' has no attribute 'pan_axis_xy'`.

- [ ] **Step 3: Implement**

Append to `so101sim/ik.py`, and add one line to the module docstring after the `solve_topdown` paragraph:
"`solve_side` is the shelf's: the gripper level, jaws horizontal, camera mount on top."

```python
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
```

- [ ] **Step 4: Run them to see them pass**

Run: `$PY -m pytest tests/test_side_ik.py tests/test_ik.py -q`
Expected: all pass. If `test_the_pan_axis_sits_in_front_of_the_origin` fails with zeros, `mj_kinematics` isn't
filling `xanchor` in this build; call `mujoco.mj_forward` instead and rerun.

---

### Task 2: Generate the shelf scene from a `ShelfLayout`

**Files:**
- Modify: `so101sim/__init__.py`
- Create: `so101sim/shelf_layout.py`, `tests/test_shelf_layout.py`
- Generated: `models/shelf_objects.xml`, `models/shelf_scene.xml`

**Interfaces:**
- Consumes: `so101sim.env.HOME`.
- Produces: `SHELF_SCENE_PATH`; `ShelfLayout(board_z, opening, front_x, depth, width, cam_pos, n_cyls, home)` with
  properties `back_x, half_width, grasp_z, carry_z, push_z, aisle_x, standoff_x`, methods `region() -> (x_lo, x_hi,
  y_lo, y_hi)`, `on_board(xy) -> bool`, `ShelfLayout.from_model(model)`; module functions `objects_xml(layout,
  probes=False, notes=())`, `scene_xml(objects_file)`, `write(layout, notes=())`, `candidate_scene(layout,
  probes=False)` (context manager yielding a path), `load(layout, probes=False) -> MjModel`, `read_notes()`;
  constants `CYL_RADIUS, CYL_HALF_HEIGHT, GRAY_GROUP, MARKER_PREFIX, TARGET_NAME = "cyl_red"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shelf_layout.py`:

```python
import dataclasses

import numpy as np
import pytest

from so101sim import SHELF_SCENE_PATH, shelf_layout as sl
from so101sim.env import SO101Env

import mujoco  # after so101sim, which sets MUJOCO_GL

LAYOUT = sl.ShelfLayout(board_z=0.06, opening=0.13, front_x=0.30, depth=0.18, width=0.24,
                        cam_pos=(0.05, -0.3, 0.4), n_cyls=6)


@pytest.fixture(scope="module")
def model():
    return sl.load(LAYOUT)


def geom_box(model, name):
    g = model.geom(name)
    return g.pos - g.size, g.pos + g.size


def test_layout_round_trips_through_the_xml(model):
    back = sl.ShelfLayout.from_model(model)
    for field in dataclasses.fields(sl.ShelfLayout):
        np.testing.assert_allclose(getattr(back, field.name), getattr(LAYOUT, field.name), atol=1e-6)


def test_the_shelf_sits_where_the_layout_says(model):
    lo, hi = geom_box(model, "shelf_board")
    assert hi[2] == pytest.approx(0.06) and lo[0] == pytest.approx(0.30)
    lo, _ = geom_box(model, "shelf_top")
    assert lo[2] == pytest.approx(0.19)
    lo, _ = geom_box(model, "shelf_back")
    assert lo[0] == pytest.approx(0.48)
    assert geom_box(model, "shelf_left")[0][1] == pytest.approx(0.12)
    assert geom_box(model, "shelf_right")[1][1] == pytest.approx(-0.12)


def test_heights_lines_and_region():
    assert LAYOUT.grasp_z == pytest.approx(0.10) and LAYOUT.carry_z == pytest.approx(0.11)
    assert LAYOUT.push_z == pytest.approx(0.085)
    assert LAYOUT.aisle_x == pytest.approx(0.32) and LAYOUT.standoff_x == pytest.approx(0.27)
    np.testing.assert_allclose(LAYOUT.region(), [0.355, 0.46, -0.10, 0.10], atol=1e-9)
    assert LAYOUT.on_board((0.31, 0.11)) and not LAYOUT.on_board((0.29, 0.0))


def test_cylinders_wait_off_stage_and_only_grays_are_in_the_gray_group(model):
    names = [model.body(b).name for b in range(model.nbody) if model.body(b).name.startswith("cyl_")]
    assert names[0] == sl.TARGET_NAME and len(names) == 6
    for name in names:
        g = model.geom(name)
        assert g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        np.testing.assert_allclose(g.size[:2], [0.015, 0.04])
        assert g.group == (0 if name == sl.TARGET_NAME else sl.GRAY_GROUP)
        assert model.body(name).pos[0] == pytest.approx(sl.PARK_X)


def test_probes_add_markers_on_the_board_that_collide_with_nothing():
    probed = sl.load(LAYOUT, probes=True)
    markers = [g for g in range(probed.ngeom) if probed.geom(g).name.startswith(sl.MARKER_PREFIX)]
    assert len(markers) > 100
    for g in markers:
        assert probed.geom_contype[g] == 0 and probed.geom_conaffinity[g] == 0
        assert LAYOUT.on_board(probed.geom_pos[g][:2])


def test_notes_with_a_double_dash_are_refused():
    with pytest.raises(ValueError):
        sl.objects_xml(LAYOUT, notes=("bad -- note",))


def test_the_committed_scene_loads_in_the_base_env():
    env = SO101Env(scene_path=SHELF_SCENE_PATH, height=64, width=64)
    try:
        env.reset(seed=0)
    finally:
        env.close()
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_shelf_layout.py -q`
Expected: FAIL with `ImportError: cannot import name 'SHELF_SCENE_PATH'`.

- [ ] **Step 3: Implement**

`so101sim/__init__.py`, add after `SEARCH_SCENE_PATH`:

```python
SHELF_SCENE_PATH = Path(__file__).resolve().parent.parent / "models" / "shelf_scene.xml"
```

`so101sim/shelf_layout.py`:

```python
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
# The cube's density and friction, inline on each geom rather than in a default class.
CYL_ATTRS = (f'type="cylinder" size="{CYL_RADIUS} {CYL_HALF_HEIGHT}" density="1000" condim="4" '
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
```

- [ ] **Step 4: Write the default scene, then run the tests**

```bash
$PY -c "import so101sim; from so101sim import shelf_layout as sl; sl.write(sl.ShelfLayout(), notes=['Default layout, not sized yet.'])"
$PY -m pytest tests/test_shelf_layout.py -q
```

Expected: all pass. `test_the_committed_scene_loads_in_the_base_env` proves the timestep still divides 1/30 s.

---

### Task 3: Side primitives

**Files:**
- Create: `so101sim/side_primitives.py`, `tests/test_side_primitives.py`

**Interfaces:**
- Consumes: `ik.solve_side`, `ik.pan_axis_xy`, `ik.SIDE_UP_AXIS`, `ik.grasp_point`; from `primitives`:
  `_Waypoint, _retime, _grip_rad, lead_in, Unreachable, GRIP_CLOSE, GRIP_HOLDING, GRIP_PLAN, MAX_JOINT_STEP_DEG`;
  `ShelfLayout`.
- Produces: `Geometry(pan_xy, front_x, aisle_x, standoff_x, grasp_z, carry_z, push_z)` with
  `Geometry.from_layout(model, layout)` and `geo.on_ray(through_xy, x) -> (2,)`; `jaw_gap(model, value) -> m`;
  `side_open_value(model) -> float`; `pick(model, geo, xy, open_value)`; `place(model, geo, here_xy, xy,
  open_value)`; `pull(model, geo, here_xy)`; `push(model, geo, xy, heading, stroke=PUSH_STROKE)`; `look(model, geo,
  xy)`; `bridge(model, geo, current, traj)`; `tool_point(model, frame) -> (xyz, level)`; `push_directions(n=16)`;
  constants `PUSH_STROKE = 0.05`, `PUSH_DIRECTIONS = 16`. Every primitive returns an (N, 6) float32 array in
  lerobot units.

- [ ] **Step 1: Write the failing tests**

`tests/test_side_primitives.py`:

```python
import numpy as np
import pytest

from so101sim import SHELF_SCENE_PATH, ik, primitives, side_primitives as sp, units
from so101sim.shelf_layout import ShelfLayout

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def setup():
    model = mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH))
    layout = ShelfLayout.from_model(model)
    return model, layout, sp.Geometry.from_layout(model, layout), sp.side_open_value(model)


def mids(model, traj, ik_grip):
    """Jaw midpoint of every frame, with the gripper set to the value IK planned at."""
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    out = []
    for row in traj:
        frame = np.array(row, dtype=np.float64)
        frame[5] = ik_grip
        data.qpos[adr] = units.to_rad(frame, grip_range)
        mujoco.mj_kinematics(model, data)
        out.append(ik.grasp_point(model, data).copy())
    return np.array(out)


def lateral(geo, through_xy, points):
    """Distance of each point from the ray through `through_xy`, in the horizontal plane."""
    u = np.asarray(through_xy) - geo.pan_xy
    u /= np.linalg.norm(u)
    d = points[:, :2] - geo.pan_xy
    return np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])


def centre(layout):
    x_lo, x_hi, y_lo, y_hi = layout.region()
    return np.array([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2 + 0.03])


def max_step(traj):
    return np.abs(np.diff(traj, axis=0)).max()


def test_side_open_value_opens_to_five_centimetres(setup):
    model, _, _, open_value = setup
    assert sp.jaw_gap(model, open_value) >= 0.05 > sp.jaw_gap(model, open_value - 1)
    assert 25 <= open_value <= 40


def test_on_ray_stays_on_the_ray(setup):
    _, layout, geo, _ = setup
    p = geo.on_ray([0.40, 0.10], layout.standoff_x)
    assert p[0] == pytest.approx(layout.standoff_x)
    assert lateral(geo, [0.40, 0.10], np.array([[*p, 0.0]]))[0] < 1e-9


def test_pick_goes_in_along_the_ray_and_stops_in_the_aisle(setup):
    model, layout, geo, open_value = setup
    xy = centre(layout)
    traj = sp.pick(model, geo, xy, open_value)
    assert max_step(traj) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, traj, primitives.GRIP_PLAN)
    np.testing.assert_allclose(m[0], [*geo.on_ray(xy, layout.standoff_x), layout.grasp_z], atol=2e-3)
    np.testing.assert_allclose(m[-1], [*geo.on_ray(xy, layout.aisle_x), layout.carry_z], atol=2e-3)
    assert np.linalg.norm(m - [*xy, layout.grasp_z], axis=1).min() < 2e-3
    assert lateral(geo, xy, m).max() < 3e-3


def test_place_keeps_the_held_cylinder_over_the_board(setup):
    model, layout, geo, open_value = setup
    here = geo.on_ray(centre(layout), layout.aisle_x)
    xy = centre(layout) + [0.03, -0.08]
    traj = sp.place(model, geo, here, xy, open_value)
    assert max_step(traj) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, traj, primitives.GRIP_HOLDING)
    closed = traj[:, 5] <= primitives.GRIP_CLOSE + 0.5
    assert (m[closed, 0] >= layout.aisle_x - 3e-3).all()
    assert np.linalg.norm(m - [*xy, layout.grasp_z], axis=1).min() < 2e-3
    np.testing.assert_allclose(m[-1], [*geo.on_ray(xy, layout.standoff_x), layout.grasp_z], atol=2e-3)


def test_pull_takes_the_cylinder_off_the_board(setup):
    model, layout, geo, _ = setup
    here = geo.on_ray(centre(layout), layout.aisle_x)
    m = mids(model, sp.pull(model, geo, here), primitives.GRIP_HOLDING)
    np.testing.assert_allclose(m[-1], [*geo.on_ray(here, layout.standoff_x), layout.carry_z], atol=2e-3)
    assert layout.front_x - m[-1][0] >= 0.02


def test_push_drops_only_past_the_front_edge(setup):
    model, layout, geo, _ = setup
    xy = centre(layout)
    traj = sp.push(model, geo, xy, -np.pi / 2)
    assert max_step(traj) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, traj, primitives.GRIP_CLOSE)
    low = m[:, 2] < layout.grasp_z - 0.005
    assert (m[low, 0] >= layout.front_x - 3e-3).all()
    end = xy + sp.PUSH_STROKE * np.array([0.0, -1.0])
    assert np.linalg.norm(m - [*end, layout.push_z], axis=1).min() < 2e-3
    for p in (m[0], m[-1]):
        assert p[0] == pytest.approx(layout.standoff_x, abs=2e-3)


def test_look_stops_on_the_standoff_line(setup):
    model, layout, geo, _ = setup
    xy = centre(layout)
    m = mids(model, sp.look(model, geo, xy), primitives.GRIP_PLAN)
    np.testing.assert_allclose(m[-1], [*geo.on_ray(xy, layout.standoff_x), layout.grasp_z], atol=2e-3)


def test_bridge_between_standoff_poses_stays_outside_the_shelf(setup):
    model, layout, geo, _ = setup
    x_lo, x_hi, y_lo, y_hi = layout.region()
    here = sp.look(model, geo, [x_hi, y_hi])[-1]
    traj = sp.look(model, geo, [x_hi, y_lo])
    bridged = sp.bridge(model, geo, here, traj)
    assert max_step(bridged) <= primitives.MAX_JOINT_STEP_DEG + 1e-4
    m = mids(model, bridged, primitives.GRIP_PLAN)
    assert (m[:, 0] <= layout.standoff_x + 5e-3).all()


def test_bridge_backs_out_of_the_aisle_along_the_ray(setup):
    model, layout, geo, open_value = setup
    xy = centre(layout)
    here = sp.pick(model, geo, xy, open_value)[-1]
    x_lo, _, y_lo, _ = layout.region()
    bridged = sp.bridge(model, geo, here, sp.look(model, geo, [x_lo, y_lo]))
    m = mids(model, bridged, primitives.GRIP_PLAN)
    inside = m[:, 0] > layout.standoff_x + 5e-3
    assert inside.any()
    assert lateral(geo, xy, m[inside]).max() < 3e-3


def test_a_point_out_of_reach_raises(setup):
    model, _, geo, open_value = setup
    with pytest.raises(primitives.Unreachable):
        sp.pick(model, geo, [0.62, 0.0], open_value)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_side_primitives.py -q`
Expected: FAIL with `ImportError: cannot import name 'side_primitives'`.

- [ ] **Step 3: Implement**

`so101sim/side_primitives.py`:

```python
"""Pick, place, pull, push and look on a shelf, with the gripper level and the camera mount on top.

Each primitive returns an (N, 6) array of joint targets in lerobot units at 30 Hz, like `primitives`, and touches
no physics. With the tool level, this arm can only point along a ray from the pan axis (see `ik.solve_side`), so
every move goes in and out along rays. Primitives start and end on the standoff line outside the shelf, except
that pick stops with the held cylinder on the aisle line, still over the board, and place and pull start there.
Held cylinders only move sideways along the aisle line, so a gray never leaves the board unless a policy pulls it.
Spec: docs/superpowers/specs/2026-09-26-shelf-search-design.md.
"""

import dataclasses

import mujoco
import numpy as np

from so101sim import ik, units
from so101sim.primitives import (GRIP_CLOSE, GRIP_HOLDING, GRIP_PLAN, MAX_JOINT_STEP_DEG, Unreachable, _Waypoint,
                                 _grip_rad, _retime, lead_in)

# A 3 cm cylinder plus 1 cm a side. GRIP_OPEN swings the moving jaw 55 mm out, which knocked over the right-hand
# neighbour 2 cm away in the 2026-09-26 spike.
SIDE_OPEN_GAP = 0.05
PUSH_STROKE = 0.05
PUSH_DIRECTIONS = 16
LIN_STEP = 0.004
POS_TOL = 2e-3
TILT_TOL = 2.0
# A seeded solve that fails falls back to a fresh one. If that lands this far from the seed on any joint, it found
# another branch, and ramping between branches in joint space would sweep the gripper through the shelf.
BRANCH_JUMP = np.radians(20.0)
LEVEL_DEG = 5.0  # bridge treats a gripper this close to level as level
ON_LINE = 0.005  # and a jaw midpoint this close to the standoff line as on it


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The lines and heights the primitives move on."""

    pan_xy: np.ndarray
    front_x: float
    aisle_x: float
    standoff_x: float
    grasp_z: float
    carry_z: float
    push_z: float

    @classmethod
    def from_layout(cls, model, layout):
        return cls(ik.pan_axis_xy(model), layout.front_x, layout.aisle_x, layout.standoff_x,
                   layout.grasp_z, layout.carry_z, layout.push_z)

    def on_ray(self, through_xy, x):
        """Where the ray from the pan axis through `through_xy` crosses the line at `x`."""
        d = np.asarray(through_xy, dtype=np.float64)[:2] - self.pan_xy
        return self.pan_xy + d * (x - self.pan_xy[0]) / d[0]


def jaw_gap(model, lerobot_value):
    """Clear gap between the jaw tips at a gripper value, in metres."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = _grip_rad(model, lerobot_value)
    mujoco.mj_kinematics(model, data)
    ids = [model.geom(g).id for g in ik.JAW_TIP_GEOMS]
    return float(np.linalg.norm(data.geom_xpos[ids[0]] - data.geom_xpos[ids[1]]) - model.geom_size[ids, 0].sum())


def side_open_value(model, gap=SIDE_OPEN_GAP):
    """The smallest whole lerobot gripper value with a jaw gap of at least `gap`."""
    for value in range(101):
        if jaw_gap(model, value) >= gap:
            return float(value)
    raise ValueError(f"the jaws never open {gap * 1000:.0f} mm")


def push_directions(n=PUSH_DIRECTIONS):
    return np.arange(n) * (2 * np.pi / n)


def _ok(info):
    return info["pos_err"] < POS_TOL and info["tilt"] < TILT_TOL


def _wp(xy, z, grip, hold=0):
    return _Waypoint([float(xy[0]), float(xy[1]), float(z)], 0.0, grip, hold)


def _trace(model, waypoints, ik_grip, q_start=None, lin_step=LIN_STEP):
    """Solve level IK along the straight tool path through `waypoints`, each solve seeded by the last.

    `ik_grip` is the gripper value in lerobot units that IK plans against, held fixed and independent of the
    commanded gripper column. `q_start` seeds the first solve, so a bridge stays on the branch the arm is on.
    """
    grip_rad = _grip_rad(model, ik_grip)
    arms, grips = [], []

    def solve(pos, seed):
        q, info = ik.solve_side(model, pos, gripper=grip_rad, q_init=seed, restarts=0 if seed is not None else 6)
        if not _ok(info):
            q, info = ik.solve_side(model, pos, gripper=grip_rad, restarts=6)
            if _ok(info) and seed is not None and np.abs(q - seed).max() > BRANCH_JUMP:
                raise Unreachable(f"level IK jumped branches at {np.round(pos, 4).tolist()}")
        if not _ok(info):
            raise Unreachable(f"no level solution at {np.round(pos, 4).tolist()}: "
                              f"{info['pos_err'] * 1000:.1f} mm off, {info['tilt']:.1f} deg of tilt")
        return q

    def add(q, grip, n=1):
        for _ in range(n):
            arms.append(q)
            grips.append(grip)

    seed = solve(waypoints[0].pos, q_start)
    add(seed, waypoints[0].grip, 1 + waypoints[0].hold)
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(b.pos - a.pos) / lin_step)))
        for i in range(1, steps + 1):
            t = i / steps
            seed = solve(a.pos + t * (b.pos - a.pos), seed)
            add(seed, a.grip + t * (b.grip - a.grip))
        add(seed, b.grip, b.hold)
    return np.array(arms), np.array(grips)


def pick(model, geo, xy, open_value):
    """Grasp the cylinder standing at `xy` and draw it back to the aisle, still over the board.

    In along the ray at grasp height with the jaws open, close and hold while the moving jaw sweeps the cylinder
    against the fixed one, rise 1 cm, then back out along the same ray until the jaw midpoint is on the aisle line.
    IK plans at GRIP_PLAN so the closed jaws land centred.
    """
    so, aisle = geo.on_ray(xy, geo.standoff_x), geo.on_ray(xy, geo.aisle_x)
    path = [
        _wp(so, geo.grasp_z, open_value),
        _wp(xy, geo.grasp_z, open_value, hold=3),
        _wp(xy, geo.grasp_z, GRIP_CLOSE, hold=8),
        _wp(xy, geo.carry_z, GRIP_CLOSE),
        _wp(aisle, geo.carry_z, GRIP_CLOSE, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_PLAN))


def place(model, geo, here_xy, xy, open_value):
    """Carry the held cylinder from the aisle point `here_xy` along the aisle line to the ray through `xy`, in to
    `xy`, set it down, open, and back out to the standoff line."""
    path = [
        _wp(geo.on_ray(here_xy, geo.aisle_x), geo.carry_z, GRIP_CLOSE),
        _wp(geo.on_ray(xy, geo.aisle_x), geo.carry_z, GRIP_CLOSE),
        _wp(xy, geo.carry_z, GRIP_CLOSE, hold=2),
        _wp(xy, geo.grasp_z, GRIP_CLOSE, hold=3),
        _wp(xy, geo.grasp_z, open_value, hold=8),
        _wp(geo.on_ray(xy, geo.standoff_x), geo.grasp_z, open_value, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_HOLDING))


def pull(model, geo, here_xy):
    """Take the held cylinder straight out along its ray from the aisle to the standoff line, off the board.

    That puts its centre 3 cm past the front edge. It's the success move for the red cylinder and a failure for
    any gray.
    """
    path = [
        _wp(geo.on_ray(here_xy, geo.aisle_x), geo.carry_z, GRIP_CLOSE),
        _wp(geo.on_ray(here_xy, geo.standoff_x), geo.carry_z, GRIP_CLOSE, hold=3),
    ]
    return _retime(*_trace(model, path, GRIP_HOLDING))


def push(model, geo, xy, heading, stroke=PUSH_STROKE):
    """Jaws closed, in along the ray to `xy`, `stroke` metres along `heading`, then back the way it came.

    Goes in at grasp height and drops to push height only past the front edge, because level reach can't get as
    close to the base at push height: 0.279 m straight ahead against 0.259 m at grasp height (2026-09-26).
    """
    xy = np.asarray(xy, dtype=np.float64)
    end = xy + stroke * np.array([np.cos(heading), np.sin(heading)])
    so, edge = geo.on_ray(xy, geo.standoff_x), geo.on_ray(xy, geo.front_x)
    path = [
        _wp(so, geo.grasp_z, GRIP_CLOSE),
        _wp(edge, geo.grasp_z, GRIP_CLOSE),
        _wp(edge, geo.push_z, GRIP_CLOSE),
        _wp(xy, geo.push_z, GRIP_CLOSE),
        _wp(end, geo.push_z, GRIP_CLOSE, hold=2),
        _wp(xy, geo.push_z, GRIP_CLOSE),
        _wp(edge, geo.push_z, GRIP_CLOSE),
        _wp(edge, geo.grasp_z, GRIP_CLOSE),
        _wp(so, geo.grasp_z, GRIP_CLOSE, hold=2),
    ]
    return _retime(*_trace(model, path, GRIP_CLOSE))


def look(model, geo, xy, hold=5):
    """Stop on the standoff line on the ray through `xy`, so the wrist camera looks down that ray."""
    return _retime(*_trace(model, [_wp(geo.on_ray(xy, geo.standoff_x), geo.grasp_z, GRIP_PLAN, hold=hold)],
                           GRIP_PLAN))


def tool_point(model, frame, data=None):
    """Jaw midpoint for a lerobot frame with the gripper taken at GRIP_PLAN, and whether the gripper is level.

    The midpoint moves 65 mm across the gripper's range, so both ends of a bridge are measured at one value.
    """
    data = mujoco.MjData(model) if data is None else data
    frame = np.array(frame, dtype=np.float64)
    frame[5] = GRIP_PLAN
    adr = model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]
    data.qpos[adr] = units.to_rad(frame, tuple(model.jnt_range[model.joint("gripper").id]))
    mujoco.mj_kinematics(model, data)
    up = data.site_xmat[model.site("gripperframe").id].reshape(3, 3)[2, ik.SIDE_UP_AXIS]
    return ik.grasp_point(model, data).copy(), bool(up >= np.cos(np.radians(LEVEL_DEG)))


def bridge(model, geo, current, traj, max_step=MAX_JOINT_STEP_DEG):
    """Get from wherever the arm is to the start of `traj` without sweeping through the shelf.

    A level arm inside the bay first backs straight out along its own ray to the standoff line, then slides along
    that line and onto the start. A joint ramp is an arbitrary curve in tool space, which inside the bay knocks
    cylinders over, so it only covers small final corrections, a start that isn't on the standoff line (place and
    pull start where pick left the arm), and poses that aren't level.
    """
    current = np.asarray(current, dtype=np.float64)
    here, here_level = tool_point(model, current)
    there, there_level = tool_point(model, traj[0])
    if (np.linalg.norm(here - there) < 0.003 or not (here_level and there_level)
            or there[0] > geo.standoff_x + ON_LINE):
        return lead_in(current, traj, max_step)
    path = [here]
    if here[0] > geo.standoff_x + ON_LINE:
        path.append(np.r_[geo.on_ray(here, geo.standoff_x), here[2]])
    path += [np.r_[there[:2], path[-1][2]], there]
    waypoints = [_Waypoint(p, 0.0, float(current[5])) for p in path]
    try:
        cross = _retime(*_trace(model, waypoints, GRIP_PLAN, q_start=np.radians(current[:5])), max_step)
    except Unreachable:
        return lead_in(current, traj, max_step)
    return np.vstack([lead_in(current, cross, max_step), lead_in(cross[-1], traj, max_step)])
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_side_primitives.py -q`
Expected: all pass. If `test_push_drops_only_past_the_front_edge` raises `Unreachable` at the front-edge line at
push height, the default layout's front edge is too close; that's the reach limit Task 4 sizes against, so check
the layout is the default `front_x=0.30` before changing any code.

---

### Task 4: Size the shelf, start pose and front camera

**Files:**
- Create: `so101sim/shelf_checks.py`, `scripts/size_shelf.py`, `tests/test_shelf_checks.py`
- Regenerated: `models/shelf_objects.xml`

**Interfaces:**
- Consumes: `shelf_layout`, `side_primitives.Geometry`, `side_primitives.side_open_value`, `ik.solve_side`,
  `grasp.arm_geom_ids`, `grasp.touching`, `search_checks.set_arm`, `search_checks.set_pose`,
  `search_checks.marker_coverage`.
- Produces: `shelf_checks.GRID, RIGHT_STRIP, FRONT_MIN, WRIST_MIN, MIN_AREA, IMAGE_SIZE`, `grid(lo, hi)`,
  `centre_points(layout)`, `aisle_points(layout)`, `ray_points(layout, geo)`, `near_right_wall(layout, xy)`,
  `region_area(layout)`, `level_pose(model, xyz, grip) -> q | None`, `shelf_geom_ids(model)`, `clear(model, data, q,
  grip, arm_ids, scene_ids) -> bool`. A committed, sized `models/shelf_objects.xml`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shelf_checks.py`:

```python
import numpy as np
import pytest

from so101sim import SCENE_PATH, SHELF_SCENE_PATH, grasp, primitives, shelf_checks as sc, shelf_layout as sl
from so101sim import side_primitives as sp

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def arm_model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


@pytest.fixture(scope="module")
def committed():
    return sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))


def test_centre_points_cover_the_region_on_a_5_mm_grid():
    layout = sl.ShelfLayout()
    pts = np.array(sc.centre_points(layout))
    x_lo, x_hi, y_lo, y_hi = layout.region()
    np.testing.assert_allclose([pts[:, 0].min(), pts[:, 0].max()], [x_lo, x_hi], atol=0.0026)
    np.testing.assert_allclose([pts[:, 1].min(), pts[:, 1].max()], [y_lo, y_hi], atol=0.0026)
    assert sc.region_area(layout) == pytest.approx((x_hi - x_lo) * (y_hi - y_lo))


def test_the_right_wall_strip_is_on_the_right():
    layout = sl.ShelfLayout()
    assert sc.near_right_wall(layout, (0.40, -layout.half_width + 0.05))
    assert not sc.near_right_wall(layout, (0.40, layout.half_width - 0.05))


def test_level_pose_agrees_with_the_measured_band(arm_model):
    assert sc.level_pose(arm_model, (0.36, 0.0, 0.10), primitives.GRIP_PLAN) is not None
    assert sc.level_pose(arm_model, (0.15, 0.0, 0.10), primitives.GRIP_PLAN) is None


def test_clear_catches_a_top_board_at_grasp_height(arm_model):
    low_top = sl.ShelfLayout(opening=0.035)  # the top board's underside 5 mm under the jaw midpoint
    model = sl.load(low_top)
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
    q = sc.level_pose(arm_model, (0.40, 0.0, low_top.grasp_z), primitives.GRIP_PLAN)
    assert not sc.clear(model, data, q, primitives.GRIP_PLAN, arm, scene)
    fine = sl.ShelfLayout()
    model = sl.load(fine)
    q = sc.level_pose(arm_model, (0.40, 0.0, fine.grasp_z), primitives.GRIP_PLAN)
    assert sc.clear(model, mujoco.MjData(model), q, primitives.GRIP_PLAN, grasp.arm_geom_ids(model),
                    sc.shelf_geom_ids(model))


def test_committed_layout_came_from_the_sizing_script():
    assert any(n.startswith("Chosen") for n in sl.read_notes())


def test_committed_layout_reaches_its_corners_level(arm_model, committed):
    x_lo, x_hi, y_lo, y_hi = committed.region()
    for x in (x_lo, x_hi):
        for y in (y_lo, y_hi):
            assert sc.level_pose(arm_model, (x, y, committed.grasp_z), primitives.GRIP_PLAN) is not None, (x, y)
    for y in (y_lo, y_hi):
        assert sc.level_pose(arm_model, (committed.aisle_x, y, committed.carry_z), primitives.GRIP_HOLDING) is not None


def test_committed_cameras_see_enough_of_the_board(committed):
    model = sl.load(committed, probes=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *sc.IMAGE_SIZE)
    try:
        assert sc.marker_coverage(model, data, renderer, "front", committed.home) >= sc.FRONT_MIN
        assert sc.marker_coverage(model, data, renderer, "wrist_cam", committed.home) >= sc.WRIST_MIN
    finally:
        renderer.close()


def test_committed_home_is_level_on_the_standoff_line(committed):
    model = mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH))
    point, level = sp.tool_point(model, committed.home)
    assert level and point[0] == pytest.approx(committed.standoff_x, abs=2e-3)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_shelf_checks.py -q`
Expected: FAIL with `ImportError: cannot import name 'shelf_checks'`.

- [ ] **Step 3: Implement the checks**

`so101sim/shelf_checks.py`:

```python
"""The checks that choose the shelf layout.

scripts/size_shelf.py runs all of them. tests/test_shelf_checks.py reruns the cheap ones against the committed
scene. Rules and thresholds: the sizing rule in docs/superpowers/specs/2026-09-26-shelf-search-design.md.
"""

import numpy as np

from so101sim import grasp, ik, primitives
from so101sim.search_checks import marker_coverage, set_arm, set_pose  # noqa: F401  the bin's probe helpers
from so101sim.shelf_layout import MARKER_PREFIX

GRID = 0.005
# The moving jaw opens toward the right wall, so centres this close to it need a push before a pick.
RIGHT_STRIP = 0.06
FRONT_MIN = 0.60
WRIST_MIN = 0.40
MIN_AREA = 0.12 * 0.18  # the spec's stop rule, read as an area
IMAGE_SIZE = (480, 640)  # height, width: the env's default observation size


def grid(lo, hi, step=GRID):
    n = int(np.floor((hi - lo) / step + 1e-9))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def centre_points(layout):
    x_lo, x_hi, y_lo, y_hi = layout.region()
    return [(x, y) for x in grid(x_lo, x_hi) for y in grid(y_lo, y_hi)]


def aisle_points(layout):
    _, _, y_lo, y_hi = layout.region()
    return [(round(layout.aisle_x, 4), y) for y in grid(y_lo, y_hi)]


def ray_points(layout, geo, step=GRID):
    """Points on every pull path: from each aisle point out along its ray to the standoff line."""
    out = []
    for p in aisle_points(layout):
        a, b = geo.on_ray(p, layout.aisle_x), geo.on_ray(p, layout.standoff_x)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [tuple(np.round(a + (i / n) * (b - a), 4)) for i in range(n + 1)]
    return out


def near_right_wall(layout, xy, strip=RIGHT_STRIP):
    return xy[1] + layout.half_width <= strip + 1e-9


def region_area(layout):
    x_lo, x_hi, y_lo, y_hi = layout.region()
    return max(0.0, x_hi - x_lo) * max(0.0, y_hi - y_lo)


def level_pose(model, xyz, grip):
    """Arm angles in radians for a level, mount-up gripper with its jaw midpoint within 1 mm of `xyz`, or None.
    `grip` is the lerobot gripper value IK plans against."""
    q, info = ik.solve_side(model, xyz, gripper=primitives._grip_rad(model, grip), restarts=4)
    return q if info["pos_err"] < 1e-3 and info["tilt"] < 2.0 else None


def shelf_geom_ids(model):
    """Every shelf panel and leg, plus the table: everything fixed the arm could hit. Probe markers excluded."""
    ids = {g for g in range(model.ngeom) if model.geom(g).name.startswith("shelf_")}
    return frozenset(ids | {model.geom("table").id})


def clear(model, data, q, grip, arm_ids, scene_ids):
    """True if the arm at `q`, gripper at `grip` in lerobot units, touches nothing in `scene_ids`."""
    set_arm(model, data, q, grip)
    return not grasp.touching(data, arm_ids, scene_ids)


__all__ = ["GRID", "RIGHT_STRIP", "FRONT_MIN", "WRIST_MIN", "MIN_AREA", "IMAGE_SIZE", "MARKER_PREFIX", "grid",
           "centre_points", "aisle_points", "ray_points", "near_right_wall", "region_area", "level_pose",
           "shelf_geom_ids", "clear", "marker_coverage", "set_arm", "set_pose"]
```

`search_checks.marker_coverage` counts geoms whose names start with `search_layout.MARKER_PREFIX`, which is the
same string, `"marker_"`, as `shelf_layout.MARKER_PREFIX`. Confirm with
`grep -n MARKER_PREFIX so101sim/search_layout.py` before relying on it.

- [ ] **Step 4: Implement the sizing script**

`scripts/size_shelf.py`:

```python
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
CAM_POSITIONS = tuple(itertools.product((-0.10, 0.05), (-0.30, 0.0, 0.30), (0.30, 0.40)))
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
        # Farthest first, where reach runs out.
        centres = sorted(sc.centre_points(L), key=lambda p: -np.hypot(p[0] - self.pan[0], p[1] - self.pan[1]))
        aisle = sc.aisle_points(L)
        standoff = [tuple(np.round(geo.on_ray(p, L.standoff_x), 4)) for p in aisle]
        edge = [tuple(np.round(geo.on_ray(p, L.front_x), 4)) for p in aisle]
        G = primitives
        checks = (  # name, points, height, IK gripper, gripper for the collision check, exempt
            ("grasp", centres, L.grasp_z, G.GRIP_PLAN, G.GRIP_PLAN, None),
            ("open jaw", centres, L.grasp_z, G.GRIP_PLAN, open_value, lambda p: sc.near_right_wall(L, p)),
            ("aisle", aisle, L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, None),
            ("standoff at grasp height", standoff, L.grasp_z, G.GRIP_PLAN, G.GRIP_PLAN, None),
            ("standoff at carry height", standoff, L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, None),
            ("pull path", sc.ray_points(L, geo), L.carry_z, G.GRIP_HOLDING, G.GRIP_HOLDING, None),
            ("push", edge + aisle + centres, L.push_z, G.GRIP_CLOSE, G.GRIP_CLOSE, None),
        )
        for name, points, z, ik_grip, _, _ in checks:
            _, miss = self.solve(points, z, ik_grip)
            if miss is not None:
                return f"{name}: no level solution at z={z:+.3f}, first at {miss}"
        model = sl.load(L)
        data = mujoco.MjData(model)
        arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
        for name, points, z, ik_grip, grip, exempt in checks:
            poses, _ = self.solve(points, z, ik_grip)
            for p, q in zip(points, poses):
                if exempt is not None and exempt(p):
                    continue
                if not sc.clear(model, data, q, grip, arm, scene):
                    return f"{name}: the arm hits the shelf or table at {tuple(np.round(p, 4))}"
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
            yield tuple(round(float(v), 2) for v in units.to_lerobot(np.r_[q, grip_rad], grip_range))


def pick_home_and_camera(L):
    """The start pose and front camera: of the poses whose wrist camera sees WRIST_MIN of the board, the one that
    leaves the best front camera spot the most board. Returns (home, cam_pos, front share, wrist share) or None."""
    model = sl.load(L)
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.shelf_geom_ids(model)
    best = None
    for home in home_candidates(L, model):
        sc.set_pose(model, data, home)
        if grasp.touching(data, arm, scene):
            print(f"  start pose {list(home)}: touches the shelf")
            continue
        wrist = coverage(dataclasses.replace(L, home=home), "wrist_cam", home)
        print(f"  start pose {list(home)}: the wrist camera sees {wrist:.0%} of the board", flush=True)
        if wrist < sc.WRIST_MIN:
            continue
        for cam in CAM_POSITIONS:
            front = coverage(dataclasses.replace(L, home=home, cam_pos=cam), "front", home)
            print(f"    front camera at {cam}: sees {front:.0%}", flush=True)
            if best is None or front > best[2]:
                best = (home, cam, front, wrist)
    return best


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
    if best is None or best[2] < sc.FRONT_MIN:
        seen = "no start pose passes the wrist check" if best is None else (
            f"the best front camera spot {best[1]} sees {best[2]:.0%}")
        sys.exit(f"STOP: {seen}; the spec needs {sc.FRONT_MIN:.0%} front and {sc.WRIST_MIN:.0%} wrist. "
                 "Bring this back to dm1487.")
    home, cam, front, wrist = best
    committed = sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))
    chosen = dataclasses.replace(chosen, home=home, cam_pos=cam, n_cyls=committed.n_cyls)
    notes = [
        f"Chosen {datetime.date.today()} by scripts/size_shelf.py in {time.time() - started:.0f} s, after {tried} "
        "better-ranked candidates failed.",
        describe(chosen) + ".",
        "Every centre on a 5 mm grid, the aisle, the standoff line, every pull path and every push point passes "
        f"level reach and clears the shelf and table; the open jaw at {open_value:.0f} clears everywhere more than "
        f"{sc.RIGHT_STRIP * 100:.0f} cm from the right wall.",
        f"Front camera at {list(cam)} sees {front:.0%} of the empty board with the arm at start pose {list(home)}, "
        f"whose wrist camera sees {wrist:.0%}.",
    ]
    notes += [n for n in sl.read_notes() if n.startswith("Cylinder count")]
    print("\n".join(notes))
    if args.write:
        sl.write(chosen, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the sizing script**

```bash
export MUJOCO_EGL_DEVICE_ID=2
$PY scripts/size_shelf.py 2>&1 | tee $SCRATCH/size_shelf.log
```

Expected: most candidates fail on reach at the back corners or on the standoff line, and one with
`front edge 0.300 m` passes. If it stops with `STOP`, stop here and report the log to dm1487. Don't widen a
candidate list or lower a threshold.

- [ ] **Step 6: Write it and run the tests**

```bash
$PY scripts/size_shelf.py --write
$PY -m pytest tests/test_shelf_checks.py tests/test_shelf_layout.py tests/test_side_primitives.py -q
```

Expected: all pass. Task 3's tests rerun because the committed layout changed.

---

### Task 5: `ShelfEnv` reset

**Files:**
- Create: `so101sim/shelf.py`, `tests/test_shelf.py`

**Interfaces:**
- Consumes: `SO101Env` (`_apply`, `_obs`, `_render`, `_qpos_adr`, `_gripper_range`, `_n_substeps`, `joint_state`),
  `env.CAMERAS`, `env.CONTROL_HZ`; `grasp.jaw_geom_ids`, `grasp.arm_geom_ids`, `grasp.touching`,
  `grasp.is_grasped`; from `mechsearch`: `face_pixels`, `VISIBILITY_SIZE`, `VISIBLE_MIN_PIXELS`,
  `PARTIAL_MAX_SHARE`; `ShelfLayout`, `CYL_HALF_HEIGHT`, `CYL_RADIUS`, `GRAY_GROUP`, `TARGET_NAME`.
- Produces: `shelf.TARGET`, `shelf.TASK`, `shelf.TOPPLE_DEG`, `shelf.tilt_deg(xmat)`, `ShelfEnv(scene_path,
  max_steps=2700, hidden_prob=0.5, height, width)` with attributes `layout, region, cyls, grays` and methods
  `reset`, `cylinder_pose(name) -> {"pos", "tilt"}`, `tilt(name)`, `settled()`, `free_spot() -> (2,) | None`,
  `target_pixels(hide_grays=False) -> {"front": int, "wrist": int}`, `task_info()`, `close()`. `info` keys:
  `target, cylinders, hidden, target_pixels, target_visible_share, reset_attempts, n_on_board, is_success, failure,
  steps, task`.

- [ ] **Step 1: Write the failing tests**

`tests/test_shelf.py`:

```python
import numpy as np
import pytest

from so101sim import grasp
from so101sim.mechsearch import PARTIAL_MAX_SHARE, VISIBLE_MIN_PIXELS
from so101sim.shelf import TARGET, TASK, ShelfEnv

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def env():
    e = ShelfEnv(height=64, width=64)
    yield e
    e.close()


def positions(env):
    return {n: env.cylinder_pose(n)["pos"] for n in env.cyls}


def set_cylinder(env, name, pos, tilt_deg=0.0):
    adr = env.model.jnt_qposadr[env.model.body(name).jntadr[0]]
    half = np.radians(tilt_deg) / 2
    env.data.qpos[adr:adr + 3] = pos
    env.data.qpos[adr + 3:adr + 7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)


# ---- reset --------------------------------------------------------------------------------------


def test_reset_is_deterministic(env):
    env.reset(seed=3)
    a = positions(env)
    env.reset(seed=3)
    b = positions(env)
    for name in env.cyls:
        np.testing.assert_allclose(a[name], b[name], atol=1e-9)


def test_a_hidden_reset_hides_the_red_cylinder_from_both_cameras(env):
    _, info = env.reset(seed=0, options={"hidden": True})
    assert info["hidden"] is True
    assert info["target_pixels"] == {"front": 0, "wrist": 0}


def test_a_partial_reset_shows_part_of_the_red_cylinder_but_not_most(env):
    _, info = env.reset(seed=0, options={"hidden": False})
    assert info["hidden"] is False
    assert max(info["target_pixels"].values()) >= VISIBLE_MIN_PIXELS
    assert all(s <= PARTIAL_MAX_SHARE for s in info["target_visible_share"].values()), info["target_visible_share"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reset_leaves_upright_spaced_cylinders_with_room(env, seed):
    env.reset(seed=seed)
    x_lo, x_hi, y_lo, y_hi = env.region
    pts = []
    for name in env.cyls:
        pose = env.cylinder_pose(name)
        assert pose["tilt"] < 2.0, name
        assert x_lo - 0.002 <= pose["pos"][0] <= x_hi + 0.002 and y_lo - 0.002 <= pose["pos"][1] <= y_hi + 0.002
        pts.append(pose["pos"][:2])
    pts = np.array(pts)
    gaps = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts))
    assert gaps.min() >= 0.035 - 0.002
    assert env.settled()
    assert env.free_spot() is not None


def test_reset_needs_no_seed_or_options(env):
    obs, info = env.reset()
    assert info["n_on_board"] == len(env.cyls)
    assert obs["observation.state"].shape == (6,)


def test_reset_starts_at_the_layout_home(env):
    env.reset(seed=0)
    np.testing.assert_allclose(env.joint_state, env.layout.home, atol=1.0)


def test_the_hidden_label_does_not_depend_on_the_observation_size(env):
    big = ShelfEnv()
    try:
        _, small = env.reset(seed=4)
        _, large = big.reset(seed=4)
        assert small["hidden"] == large["hidden"]
        assert small["target_pixels"] == large["target_pixels"]
    finally:
        big.close()
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_shelf.py -q`
Expected: FAIL with `ImportError: cannot import name 'TARGET' from 'so101sim.shelf'` (no module yet).

- [ ] **Step 3: Implement reset**

`so101sim/shelf.py`:

```python
"""Mechanical search on a shelf: find the one red cylinder among gray ones and pull it out, toppling nothing.

Spec: docs/superpowers/specs/2026-09-26-shelf-search-design.md.

`ShelfEnv` is the 30 Hz env, with `SO101Env`'s actions and observations, so a lerobot policy runs on it unchanged.
`ShelfPrimitiveEnv` wraps it so one step is a whole pick, place, push, pull or look, with the 30 Hz frames still
running underneath and every check on each one.

Three rules shape this file. A cylinder tilted past 20 degrees has toppled, held or not, and ends the episode. Every
gray stays on the board the whole time, held or not, so the only way off the shelf is `pull`, and only the red
cylinder should take it. And the observation holds images and joint angles only. Cylinder poses and the hidden label
go in `info`, because a policy on hardware gets neither.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101sim import SHELF_SCENE_PATH, grasp, primitives, side_primitives as sp, units
from so101sim.env import CAMERAS, CONTROL_HZ, SO101Env
from so101sim.mechsearch import PARTIAL_MAX_SHARE, VISIBILITY_SIZE, VISIBLE_MIN_PIXELS, face_pixels
from so101sim.shelf_layout import CYL_HALF_HEIGHT, CYL_RADIUS, GRAY_GROUP, TARGET_NAME, ShelfLayout

TARGET = TARGET_NAME
TASK = "retrieve the red cylinder from the shelf"

MIN_SPACING = 2 * CYL_RADIUS + 0.005
# A hidden red cylinder goes in the back two thirds of the region, with a gray on each camera's sight line to it,
# this far in front of it.
HIDDEN_BACK = 1 / 3
SIGHT_GAP = (0.035, 0.05)
PLACE_LIFT = 0.0005  # cylinders are placed this far over the board, so they settle onto it
SETTLE_QVEL = 0.2
SETTLE_SECONDS = 0.5
UPRIGHT_DEG = 2.0
REGION_SLACK = 0.002
FREE_SPOT_CLEARANCE = 0.045
FREE_SPOT_GRID = 0.005
MAX_RESET_ATTEMPTS = 50
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

TOPPLE_DEG = 20.0  # just under atan(3 / 8) = 20.6, past which a 3 x 8 cm cylinder can't stand back up
SUCCESS_PAST_EDGE = 0.02
SUCCESS_FRAMES = 5


def tilt_deg(xmat):
    """Degrees between a body's z axis, which is a cylinder's axis, and vertical."""
    return float(np.degrees(np.arccos(np.clip(np.asarray(xmat).reshape(3, 3)[2, 2], -1.0, 1.0))))


class ShelfEnv(SO101Env):
    def __init__(self, scene_path=SHELF_SCENE_PATH, max_steps=2700, hidden_prob=0.5, **env_kwargs):
        super().__init__(scene_path=scene_path, **env_kwargs)
        self.max_steps = max_steps
        self.hidden_prob = float(hidden_prob)
        self.layout = ShelfLayout.from_model(self.model)
        self.region = self.layout.region()
        names = [self.model.body(b).name for b in range(self.model.nbody)]
        self.cyls = tuple(sorted((n for n in names if n.startswith("cyl_")), key=lambda n: (n != TARGET, len(n), n)))
        self.grays = tuple(n for n in self.cyls if n != TARGET)
        joints = {n: self.model.body(n).jntadr[0] for n in self.cyls}
        self._qadr = {n: self.model.jnt_qposadr[j] for n, j in joints.items()}
        self._dofadr = {n: self.model.jnt_dofadr[j] for n, j in joints.items()}
        self._body = {n: self.model.body(n).id for n in self.cyls}
        self._geom = {n: self.model.geom(n).id for n in self.cyls}
        self._cyl_geoms = frozenset(self._geom.values())
        self._jaw_ids = grasp.jaw_geom_ids(self.model)
        self._arm_ids = grasp.arm_geom_ids(self.model)
        self._home = np.asarray(self.layout.home, dtype=np.float64)
        self._seg = mujoco.Renderer(self.model, *VISIBILITY_SIZE)
        self._seg.enable_segmentation_rendering()
        self._hide_grays = mujoco.MjvOption()
        self._hide_grays.geomgroup[GRAY_GROUP] = 0
        self._steps = 0
        self._streak = 0
        self._reset_info = {}

    # ---- gym api -------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Seed np_random only. SO101Env.reset would render an observation nobody uses.
        gym.Env.reset(self, seed=seed)
        options = options or {}
        hidden = options.get("hidden")
        if hidden is None:
            hidden = bool(self.np_random.random() < self.hidden_prob)
        reason = None
        for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
            self._start_pose()
            reason, labels = self._build(bool(hidden))
            if reason is None:
                break
        else:
            raise RuntimeError(f"no usable shelf in {MAX_RESET_ATTEMPTS} attempts, the last because {reason}")
        self._steps, self._streak = 0, 0
        self._reset_info = {"hidden": bool(hidden), **labels, "reset_attempts": attempt}
        return self._obs(), self._info(False, None)

    def close(self):
        self._seg.close()
        super().close()

    # ---- task state ----------------------------------------------------------------------------

    def cylinder_pose(self, name):
        adr = self._qadr[name]
        return {"pos": self.data.qpos[adr:adr + 3].copy(), "tilt": self.tilt(name)}

    def tilt(self, name):
        return tilt_deg(self.data.xmat[self._body[name]])

    def settled(self):
        return bool(np.abs(self.data.qvel).max() <= SETTLE_QVEL)

    def free_spot(self):
        """The first point on a 5 mm grid over the region at least FREE_SPOT_CLEARANCE from every cylinder
        standing on the board, or None."""
        x_lo, x_hi, y_lo, y_hi = self.region
        xs = np.arange(x_lo, x_hi + 1e-9, FREE_SPOT_GRID)
        ys = np.arange(y_lo, y_hi + 1e-9, FREE_SPOT_GRID)
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        centres = np.array([self._xy(n) for n in self.cyls if self.layout.on_board(self._xy(n))]).reshape(-1, 2)
        if len(centres) == 0:
            return grid[len(grid) // 2].copy()
        gaps = np.linalg.norm(grid[:, None, :] - centres[None, :, :], axis=-1).min(axis=1)
        free = np.flatnonzero(gaps >= FREE_SPOT_CLEARANCE)
        return grid[free[0]].copy() if len(free) else None

    def target_pixels(self, hide_grays=False):
        """Face pixels of the red cylinder in each camera, at VISIBILITY_SIZE. With `hide_grays`, what each camera
        would see of it if nothing covered it."""
        counts = {}
        for name, camera in CAMERAS.items():
            self._seg.update_scene(self.data, camera=camera, scene_option=self._hide_grays if hide_grays else None)
            seg = self._seg.render()
            counts[name] = face_pixels((seg[..., 0] == self._geom[TARGET]) & (seg[..., 1] == GEOM))
        return counts

    def task_info(self):
        """`info` as it stands, for a primitive that was refused before any frame ran."""
        return self._info(False, None)

    # ---- internals -----------------------------------------------------------------------------

    def _xy(self, name):
        adr = self._qadr[name]
        return self.data.qpos[adr:adr + 2].copy()

    def _start_pose(self):
        mujoco.mj_resetData(self.model, self.data)  # also parks every cylinder back off stage
        q = units.to_rad(self._home, self._gripper_range)
        self.data.qpos[self._qpos_adr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _in_region(self, xy, slack=0.0):
        x_lo, x_hi, y_lo, y_hi = self.region
        return bool(x_lo - slack <= xy[0] <= x_hi + slack and y_lo - slack <= xy[1] <= y_hi + slack)

    def _free(self, xy, placed):
        return self._in_region(xy) and all(np.linalg.norm(xy - p) >= MIN_SPACING for p in placed)

    def _sample_xy(self, placed, x_lo=None, tries=200):
        lo_x, hi_x, lo_y, hi_y = self.region
        lo_x = lo_x if x_lo is None else x_lo
        for _ in range(tries):
            xy = np.array([self.np_random.uniform(lo_x, hi_x), self.np_random.uniform(lo_y, hi_y)])
            if self._free(xy, placed):
                return xy
        return None

    def _place(self, name, xy):
        adr, dof = self._qadr[name], self._dofadr[name]
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], self.layout.board_z + CYL_HALF_HEIGHT + PLACE_LIFT]
        self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dof:dof + 6] = 0.0

    def _settle(self):
        """Step until still or SETTLE_SECONDS pass. Returns True if a cylinder touched the arm on the way."""
        mujoco.mj_forward(self.model, self.data)
        touched = False
        for _ in range(int(round(SETTLE_SECONDS * CONTROL_HZ))):
            mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
            touched = touched or grasp.touching(self.data, self._cyl_geoms, self._arm_ids)
            if self.settled():
                break
        return touched

    def _build(self, hidden):
        """Stand every cylinder on the board. Returns (why the shelf is unusable or None, labels for info)."""
        grays = list(self.grays)
        self.np_random.shuffle(grays)
        placed = []

        def put(name, xy):
            self._place(name, xy)
            placed.append(np.asarray(xy, dtype=np.float64))

        if hidden:
            x_lo, x_hi, _, _ = self.region
            red = self._sample_xy(placed, x_lo=x_lo + HIDDEN_BACK * (x_hi - x_lo))
            if red is None:
                return "no room for the red cylinder", {}
            put(TARGET, red)
            for camera in CAMERAS.values():
                eye = self.data.cam_xpos[self.model.camera(camera).id][:2]
                toward = (eye - red) / np.linalg.norm(eye - red)
                xy = red + self.np_random.uniform(*SIGHT_GAP) * toward
                if grays and self._free(xy, placed):
                    put(grays.pop(), xy)
            queue = grays
        else:
            queue = grays + [TARGET]
        for name in queue:
            xy = self._sample_xy(placed)
            if xy is None:
                return "no room left on the shelf", {}
            put(name, xy)
        touched = self._settle()
        return self._rejection(hidden, touched)

    def _rejection(self, hidden, touched):
        if touched:
            return "a cylinder touched the arm while settling", {}
        for name in self.cyls:
            if self.tilt(name) > UPRIGHT_DEG:
                return f"{name} tilted {self.tilt(name):.1f} deg while settling", {}
            if not self._in_region(self._xy(name), REGION_SLACK):
                return f"{name} slid out of the region", {}
        if not self.settled():
            return "the shelf was still moving", {}
        if self.free_spot() is None:
            return "no free spot left on the board", {}
        pixels = self.target_pixels()
        if hidden:
            if any(pixels.values()):
                return f"the hidden red cylinder shows {pixels}", {}
            return None, {"target_pixels": pixels, "target_visible_share": {c: 0.0 for c in pixels}}
        if max(pixels.values()) < VISIBLE_MIN_PIXELS:
            return f"the red cylinder shows only {pixels}", {}
        unoccluded = self.target_pixels(hide_grays=True)
        shares = {c: pixels[c] / unoccluded[c] if unoccluded[c] else 0.0 for c in pixels}
        if max(shares.values()) > PARTIAL_MAX_SHARE:
            return f"the red cylinder shows too much of itself {shares}", {}
        return None, {"target_pixels": pixels, "target_visible_share": shares}

    def _info(self, success, failure):
        adr = self._qadr[TARGET]
        return {
            "target": {"pos": self.data.qpos[adr:adr + 3].copy(), "quat": self.data.qpos[adr + 3:adr + 7].copy()},
            "cylinders": {n: self.cylinder_pose(n) for n in self.cyls},
            **self._reset_info,
            "n_on_board": sum(self.layout.on_board(self._xy(n)) for n in self.cyls),
            "is_success": bool(success),
            "failure": failure,
            "steps": self._steps,
            "task": TASK,
        }
```

The imports of `spaces`, `primitives` and `sp` are for Task 8, which adds `ShelfPrimitiveEnv` to this file.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_shelf.py -q`
Expected: all pass. If a reset raises `RuntimeError: no usable shelf in 50 attempts`, the message's last reason
says which rule rejects. The count isn't tuned yet (Task 7 does that), so run
`$PY -c "import so101sim; from so101sim import shelf_layout as sl; import mujoco; from so101sim import SHELF_SCENE_PATH as P; L = sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(P))); import dataclasses; sl.write(dataclasses.replace(L, n_cyls=12), sl.read_notes())"`
to drop to 12 for now, then rerun.

---

### Task 6: `ShelfEnv` step, with topple, off-board, success and time-limit checks

**Files:**
- Modify: `so101sim/shelf.py` (inside `ShelfEnv`)
- Test: `tests/test_shelf.py` (append)

**Interfaces:**
- Produces: `ShelfEnv.step(action)`, `ShelfEnv.advance(action) -> (reward, terminated, truncated, info)` without
  rendering, `ShelfEnv.held(name) -> bool`, `ShelfEnv._target_retrieved()`. Failure strings start with
  `"toppled: "` or `"left the shelf: "` and end with the cylinder's name.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_shelf.py`:

```python
# ---- step ---------------------------------------------------------------------------------------


def test_holding_still_keeps_everything_standing(env):
    env.reset(seed=0)
    hold = env.joint_state.copy()
    for _ in range(15):
        _, reward, terminated, truncated, info = env.step(hold)
        assert not terminated and not truncated and reward == 0.0
    assert info["failure"] is None and info["task"] == TASK
    assert info["n_on_board"] == len(env.cyls)


def test_a_tipped_gray_ends_the_episode(env):
    env.reset(seed=0)
    gray = env.grays[0]
    pos = env.cylinder_pose(gray)["pos"]
    set_cylinder(env, gray, pos, tilt_deg=25.0)
    _, reward, terminated, _, info = env.step(env.joint_state)
    assert terminated and reward == 0.0
    assert info["failure"] == f"toppled: {gray}"


def test_a_gray_off_the_board_ends_the_episode(env):
    env.reset(seed=0)
    gray = env.grays[0]
    set_cylinder(env, gray, [env.layout.front_x - 0.05, 0.0, env.layout.board_z + 0.04])
    _, _, terminated, _, info = env.step(env.joint_state)
    assert terminated and info["failure"] == f"left the shelf: {gray}"


def test_the_red_cylinder_may_leave_only_while_held(env, monkeypatch):
    env.reset(seed=0, options={"hidden": False})
    out = [env.layout.front_x - 0.03, 0.0, env.layout.carry_z]
    for held, failure in ((True, None), (False, f"left the shelf: {TARGET}")):
        set_cylinder(env, TARGET, out)
        monkeypatch.setattr(env, "held", lambda name, held=held: held)
        assert env._failure() == failure


def test_success_needs_the_red_held_out_past_the_edge_and_upright(env, monkeypatch):
    env.reset(seed=0, options={"hidden": False})
    monkeypatch.setattr(grasp, "is_grasped", lambda *args: True)
    f, z = env.layout.front_x, env.layout.carry_z
    for x, tilt, retrieved in ((f - 0.03, 0.0, True), (f - 0.01, 0.0, False), (f - 0.03, 25.0, False)):
        set_cylinder(env, TARGET, [x, 0.0, z], tilt_deg=tilt)
        assert env._target_retrieved() is retrieved, (x, tilt)


def test_success_needs_five_frames_in_a_row(env, monkeypatch):
    env.reset(seed=0)
    hold = env.joint_state.copy()
    frames = iter([True] * 4 + [False] + [True] * 5)
    monkeypatch.setattr(env, "_target_retrieved", lambda: next(frames))
    rewards = []
    for _ in range(10):
        _, reward, terminated, _, info = env.step(hold)
        rewards.append(reward)
        if terminated:
            break
    assert rewards == [0.0] * 9 + [1.0]
    assert info["is_success"]


def test_step_limit_truncates(env):
    env.reset(seed=0)
    old, env.max_steps = env.max_steps, 3
    try:
        flags = [env.step(env.joint_state)[3] for _ in range(3)]
    finally:
        env.max_steps = old
    assert flags == [False, False, True]


def test_object_poses_stay_out_of_the_observation(env):
    obs, info = env.reset(seed=0)
    assert set(obs) == {"observation.state", "observation.images.front", "observation.images.wrist"}
    assert "target" in info and "cylinders" in info
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_shelf.py -q -k "step or tipped or board or held or success or limit or poses"`
Expected: FAIL. `SO101Env.step` returns reward 0 forever, and `ShelfEnv` has no `_failure`.

- [ ] **Step 3: Implement**

Add to `ShelfEnv`, in the gym api section after `reset`:

```python
    def step(self, action):
        reward, terminated, truncated, info = self.advance(action)
        return self._obs(), reward, terminated, truncated, info

    def advance(self, action):
        """One 30 Hz frame of physics and checks, without rendering.

        The primitive env calls this every frame and renders once at the end. Checks run in the spec's order: a
        topple, a cylinder off the board, then success, then the time limit.
        """
        self._apply(action)
        self._steps += 1
        failure = self._failure()
        success = False
        if failure is None:
            self._streak = self._streak + 1 if self._target_retrieved() else 0
            success = self._streak >= SUCCESS_FRAMES
        terminated = success or failure is not None
        truncated = not terminated and self.max_steps is not None and self._steps >= self.max_steps
        return float(success), terminated, truncated, self._info(success, failure)
```

And in the task state section:

```python
    def held(self, name):
        """Both jaws on the cylinder."""
        return grasp.is_grasped(self.model, self.data, self._geom[name], self._jaw_ids)
```

And in internals:

```python
    def _failure(self):
        for name in self.cyls:
            if self.tilt(name) > TOPPLE_DEG:
                return f"toppled: {name}"
        for name in self.cyls:
            if not self.layout.on_board(self._xy(name)) and (name != TARGET or not self.held(name)):
                return f"left the shelf: {name}"
        return None

    def _target_retrieved(self):
        """Both jaws on the red cylinder, its centre SUCCESS_PAST_EDGE in front of the front edge, and upright."""
        if not self.held(TARGET):
            return False
        return (self._xy(TARGET)[0] <= self.layout.front_x - SUCCESS_PAST_EDGE
                and self.tilt(TARGET) < TOPPLE_DEG)
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_shelf.py -q`
Expected: all pass.

---

### Task 7: Smoke script, reset cost and the cylinder count

**Files:**
- Create: `scripts/shelf_smoke.py`
- Regenerated: `models/shelf_objects.xml` (the count and a note)

**Interfaces:**
- Consumes: `ShelfEnv`, `shelf.MAX_RESET_ATTEMPTS`, `shelf_layout.candidate_scene`, `write`, `read_notes`.
- Produces: `outputs/shelf_smoke.mp4`, a `Cylinder count` note in the committed XML.

- [ ] **Step 1: Write the script**

`scripts/shelf_smoke.py`:

```python
"""Reset the shelf env many times: starting shelves on video, reset cost, and the cylinder count.

    python scripts/shelf_smoke.py --episodes 8 --out outputs/shelf_smoke.mp4
    python scripts/shelf_smoke.py --episodes 100 --no-video
    python scripts/shelf_smoke.py --tune-count --write

Acceptance is the share of attempted shelves that reset keeps, 1 / mean attempts. The spec wants at least half kept
for hidden and for partial starts, and a reset under 3 s on average. --tune-count starts at 18 cylinders and takes
grays away until both pass, and --write records the count in models/shelf_objects.xml.
"""

import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np

import so101sim  # noqa: F401  before mujoco, which reads MUJOCO_GL at import
from so101sim import SHELF_SCENE_PATH, shelf_layout as sl
from so101sim.env import CONTROL_HZ
from so101sim.shelf import MAX_RESET_ATTEMPTS, ShelfEnv

import mujoco

START_COUNT = 18
MIN_COUNT = 8
MIN_ACCEPTANCE = 0.5
RESET_BUDGET_S = 3.0
HOLD_S = 1.0

_ENVS = {}


def _measure(job):
    """One reset in a worker process, reusing that worker's env for the same scene."""
    scene, seed, hidden = job
    if scene not in _ENVS:
        for old in _ENVS.values():
            old.close()
        _ENVS.clear()
        _ENVS[scene] = ShelfEnv(scene_path=scene, height=64, width=64)
    started = time.perf_counter()
    try:
        _, info = _ENVS[scene].reset(seed=seed, options={"hidden": hidden})
        attempts = info["reset_attempts"]
    except RuntimeError:
        attempts = MAX_RESET_ATTEMPTS
    return hidden, time.perf_counter() - started, attempts


def measure(scene, seeds, workers):
    jobs = [(str(scene), seed, hidden) for hidden in (True, False) for seed in seeds]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_measure, jobs, chunksize=2)
    stats = {}
    for hidden, label in ((True, "hidden"), (False, "partial")):
        mine = [r for r in rows if r[0] == hidden]
        stats[label] = {"acceptance": len(mine) / sum(r[2] for r in mine),
                        "reset_s": float(np.mean([r[1] for r in mine]))}
    return stats


def report(stats):
    return "; ".join(f"{label} kept {s['acceptance']:.0%}, {s['reset_s']:.1f} s per reset"
                     for label, s in stats.items())


def video(args):
    env = ShelfEnv()
    frames = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=episode, options={"hidden": episode % 2 == 0})
            print(f"episode {episode}: {'hidden' if info['hidden'] else 'partial'}, red pixels "
                  f"{info['target_pixels']}, {info['reset_attempts']} attempts")
            frame = np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1)
            frames += [frame] * int(HOLD_S * CONTROL_HZ)
    finally:
        env.close()
    imageio.mimwrite(args.out, frames, fps=CONTROL_HZ, macro_block_size=1)
    print(f"wrote {len(frames)} frames to {args.out}")


def tune(args):
    committed = sl.ShelfLayout.from_model(mujoco.MjModel.from_xml_path(str(SHELF_SCENE_PATH)))
    for count in range(START_COUNT, MIN_COUNT - 1, -1):
        layout = dataclasses.replace(committed, n_cyls=count)
        with sl.candidate_scene(layout) as scene:
            stats = measure(scene, range(args.seeds), args.workers)
        print(f"{count} cylinders: {report(stats)}", flush=True)
        if all(s["acceptance"] >= MIN_ACCEPTANCE for s in stats.values()):
            break
    else:
        sys.exit(f"STOP: no count down to {MIN_COUNT} keeps half the hidden and half the partial shelves. "
                 "Bring this back to dm1487.")
    slow = [label for label, s in stats.items() if s["reset_s"] > RESET_BUDGET_S]
    if slow:
        print(f"REPORT TO dm1487: reset averages over {RESET_BUDGET_S:.0f} s for {', '.join(slow)} starts. "
              "The spec says report it, not cut cylinders to meet it.")
    if args.write:
        notes = [n for n in sl.read_notes() if not n.startswith("Cylinder count")]
        notes.append(f"Cylinder count {count}, the most with at least half of shelves kept over {args.seeds} "
                     f"seeds each: {report(stats)}.")
        sl.write(layout, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/shelf_smoke.mp4"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--tune-count", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.tune_count:
        return tune(args)
    if not args.no_video:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        video(args)
    print(f"over {args.episodes} seeds: {report(measure(SHELF_SCENE_PATH, range(args.episodes), args.workers))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Tune the count**

```bash
$PY scripts/shelf_smoke.py --tune-count --write 2>&1 | tee $SCRATCH/shelf_count.log
```

Expected: a count between 8 and 18 with both acceptances at least 50%. If it prints `STOP`, stop and report the
log. The likeliest reason is hidden starts, because a camera above the board sees over a front gray to the red's
top. Don't change the hidden recipe or the pixel rule without dm1487.

- [ ] **Step 3: Record the starting shelves and look at them**

```bash
$PY scripts/shelf_smoke.py --episodes 8 --out outputs/shelf_smoke.mp4
$PY -c "import imageio; r = imageio.get_reader('outputs/shelf_smoke.mp4'); [imageio.imwrite(f'$SCRATCH/start_{i}.png', r.get_data(i * 30 + 15)) for i in range(4)]"
```

Open `$SCRATCH/start_0.png` to `start_3.png` with the Read tool. Check that cylinders stand on the board in the
region and nothing floats or clips the walls, that the hidden starts (0 and 2) show no red, and that the partial
ones (1 and 3) show some. Send `outputs/shelf_smoke.mp4` to dm1487 with SendUserFile.

- [ ] **Step 4: Run the suite**

Run: `$PY -m pytest tests/test_shelf.py tests/test_shelf_layout.py tests/test_shelf_checks.py -q`
Expected: all pass with the tuned count.

---

### Task 8: `ShelfPrimitiveEnv`, and proof the task can be done

**Files:**
- Modify: `so101sim/shelf.py` (append)
- Test: `tests/test_shelf.py` (append)

**Interfaces:**
- Consumes: `ShelfEnv.advance`, `joint_state`, `task_info`, `settled`, `_obs`, `render`; all of `side_primitives`.
- Produces: `ShelfPrimitiveEnv(max_primitives=20, push_stroke=0.05, on_frame=None, **env_kwargs)`, attributes
  `env, model, data, geo, open_value, action_space, observation_space`, `PRIMITIVES = ("pick", "place", "push",
  "pull", "look")`, methods `reset`, `step(action)` with action a dict `{"kind", "xy", "angle"}` or a tuple
  `(kind, xy, angle)`, `holding() -> bool`, `render`, `close`. Step `info` adds `primitive`, `frames`, `refused`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_shelf.py`:

```python
# ---- primitive env ------------------------------------------------------------------------------

from so101sim import side_primitives as sp  # noqa: E402
from so101sim.shelf import ShelfPrimitiveEnv  # noqa: E402

SOLVABLE_SEED = None  # set in Task 8 step 6


@pytest.fixture(scope="module")
def prim():
    e = ShelfPrimitiveEnv(height=64, width=64)
    yield e
    e.close()


def xy_of(prim, name):
    return prim.env.cylinder_pose(name)["pos"][:2]


def ray_frame(prim, target_xy):
    """Unit vector along the ray through `target_xy`, the unit vector to its left, and its distance along."""
    d = np.asarray(target_xy) - prim.geo.pan_xy
    u = d / np.linalg.norm(d)
    return u, np.array([-u[1], u[0]]), d @ u


def blockers(prim, target_xy, skip=()):
    """Cylinders in the way of a side grasp at `target_xy`, nearest the front first: on its ray between the aisle
    and it, or beside it where the jaws go. The moving jaw opens to the right, negative lateral."""
    u, n, along = ray_frame(prim, target_xy)
    aisle_along = (prim.geo.on_ray(target_xy, prim.geo.aisle_x) - prim.geo.pan_xy) @ u - along
    hits = []
    for name in prim.env.cyls:
        p = xy_of(prim, name)
        if name in skip or np.allclose(p, target_xy, atol=1e-6):
            continue
        a, lat = (p - prim.geo.pan_xy) @ u - along, (p - prim.geo.pan_xy) @ n
        in_ray = aisle_along - 0.02 <= a <= -0.02 and abs(lat) < 0.045
        beside = abs(a) < 0.04 and -0.075 < lat < 0.05
        if in_ray or beside:
            hits.append((a, name))
    return [name for _, name in sorted(hits)]


def place_spot(prim, keep_clear_of):
    """A board spot for a cleared gray: FREE_SPOT_CLEARANCE from every centre, away from the red's ray and sides,
    with its own way in clear. Nearest the front first."""
    x_lo, x_hi, y_lo, y_hi = prim.env.region
    u, n, along = ray_frame(prim, keep_clear_of)
    centres = np.array([xy_of(prim, c) for c in prim.env.cyls])
    best = None
    for x in np.arange(x_lo, x_hi + 1e-9, 0.005):
        for y in np.arange(y_lo, y_hi + 1e-9, 0.005):
            xy = np.array([x, y])
            if np.linalg.norm(centres - xy, axis=1).min() < 0.045:
                continue
            a, lat = (xy - prim.geo.pan_xy) @ u - along, (xy - prim.geo.pan_xy) @ n
            if abs(lat) < 0.09 and a < 0.05:
                continue
            if blockers(prim, xy):
                continue
            if best is None or x < best[0]:
                best = (x, xy)
    return None if best is None else best[1]


def run(prim, *actions):
    for action in actions:
        _, _, terminated, _, info = prim.step(action)
        if terminated or info["refused"]:
            return info
    return info


def dig_out(prim, seed, max_moves=6):
    """Clear what blocks the red cylinder's ray and jaw sides, placing each gray at a free spot, then pull it out.
    Chosen from info poses, not images: this proves the task can be done, it isn't a policy."""
    prim.reset(seed=seed, options={"hidden": True})
    for _ in range(max_moves):
        red = xy_of(prim, TARGET)
        block = blockers(prim, red)
        if not block:
            break
        name = block[0]
        if blockers(prim, xy_of(prim, name)):
            return {"is_success": False, "failure": f"{name} is blocked itself", "refused": None}
        spot = place_spot(prim, red)
        if spot is None:
            return {"is_success": False, "failure": "no place to put a gray", "refused": None}
        info = run(prim, ("pick", xy_of(prim, name), 0.0), ("place", spot, 0.0))
        if info["failure"] or info["refused"]:
            return info
    return run(prim, ("pick", xy_of(prim, TARGET), 0.0), ("pull", xy_of(prim, TARGET), 0.0))


def a_clear_gray(prim, seeds=range(10)):
    """Reset to the first partial seed with a gray nothing blocks, and return its name."""
    for seed in seeds:
        prim.reset(seed=seed, options={"hidden": False})
        for name in prim.env.grays:
            if not blockers(prim, xy_of(prim, name)) and not sp_near_right(prim, name):
                return name
    pytest.skip("no clear gray in the first seeds")


def sp_near_right(prim, name):
    return xy_of(prim, name)[1] + prim.env.layout.half_width <= 0.06


def test_action_bounds_are_the_region(prim):
    box = prim.action_space["xy"]
    x_lo, x_hi, y_lo, y_hi = prim.env.region
    np.testing.assert_allclose(box.low, [x_lo, y_lo], atol=1e-6)
    np.testing.assert_allclose(box.high, [x_hi, y_hi], atol=1e-6)


def refused(prim, action):
    before = prim.env.joint_state.copy()
    _, reward, terminated, _, info = prim.step(action)
    assert info["refused"] and info["frames"] == 0 and not terminated and reward == 0.0
    np.testing.assert_array_equal(prim.env.joint_state, before)
    return info["refused"]


def test_place_and_pull_need_a_held_cylinder(prim):
    prim.reset(seed=0)
    centre = prim.action_space["xy"].low + 0.02
    assert "holding" in refused(prim, ("place", centre, 0.0))
    assert "holding" in refused(prim, ("pull", centre, 0.0))


def test_actions_outside_the_box_are_refused(prim):
    prim.reset(seed=0)
    assert "outside" in refused(prim, ("pick", prim.action_space["xy"].high + [0.05, 0.0], 0.0))


def test_a_push_out_of_the_shelf_is_refused(prim):
    prim.reset(seed=0)
    low = prim.action_space["xy"].low
    assert "outside" in refused(prim, ("push", [low[0], 0.0], np.pi))  # heading back toward the robot


def test_a_look_ends_level_on_the_standoff_line(prim):
    prim.reset(seed=0)
    _, _, terminated, _, info = prim.step(("look", prim.action_space["xy"].high, 0.0))
    assert info["refused"] is None and info["frames"] > 0 and not terminated
    point, level = sp.tool_point(prim.model, prim.env.joint_state)
    assert level and point[0] == pytest.approx(prim.geo.standoff_x, abs=3e-3)


def test_a_clear_gray_can_be_picked_and_placed_upright(prim):
    name = a_clear_gray(prim)
    spot = place_spot(prim, xy_of(prim, TARGET))
    _, _, terminated, _, info = prim.step(("pick", xy_of(prim, name), 0.0))
    assert not terminated and info["refused"] is None, info["failure"]
    assert prim.holding()
    assert "holding" in refused(prim, ("pick", spot, 0.0))
    _, _, terminated, _, info = prim.step(("place", spot, 0.0))
    assert not terminated, info["failure"]
    assert not prim.holding()
    pose = prim.env.cylinder_pose(name)
    assert pose["tilt"] < 5.0 and np.linalg.norm(pose["pos"][:2] - spot) < 0.01


def test_pulling_a_gray_off_the_shelf_fails(prim):
    name = a_clear_gray(prim)
    prim.step(("pick", xy_of(prim, name), 0.0))
    _, reward, terminated, _, info = prim.step(("pull", xy_of(prim, name), 0.0))
    assert terminated and reward == 0.0
    assert info["failure"] == f"left the shelf: {name}"


def test_a_missed_pick_backs_out_cleanly(prim):
    prim.reset(seed=0, options={"hidden": False})
    spot = prim.env.free_spot()
    _, _, terminated, _, info = prim.step(("pick", spot, 0.0))
    assert not terminated and not prim.holding(), info["failure"]
    _, _, terminated, _, info = prim.step(("look", prim.action_space["xy"].low, 0.0))
    assert not terminated and info["refused"] is None, info["failure"]


def test_a_low_push_slides_a_cylinder_without_tipping_it(prim):
    name = a_clear_gray(prim)
    start = xy_of(prim, name) + [0.0, 0.035]  # on its left, pushing right
    _, _, terminated, _, info = prim.step(("push", start, -np.pi / 2))
    assert not terminated, info["failure"]
    moved = np.linalg.norm(xy_of(prim, name) - (start - [0.0, 0.035]))
    assert moved > 0.02 and prim.env.tilt(name) < 5.0


def test_the_primitive_limit_truncates(prim):
    prim.reset(seed=0)
    old, prim.max_primitives = prim.max_primitives, 2
    try:
        flags = [prim.step(("pull", [0.4, 0.0], 0.0))[3] for _ in range(2)]
    finally:
        prim.max_primitives = old
    assert flags == [False, True]


def test_a_hidden_red_cylinder_can_be_dug_out(prim):
    assert SOLVABLE_SEED is not None, "run the seed search in Task 8 step 6"
    info = dig_out(prim, SOLVABLE_SEED)
    assert info["is_success"], (info["failure"], info.get("refused"))
    assert info["n_on_board"] == len(prim.env.cyls) - 1  # everything but the red, which is in the jaws
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_shelf.py -q -k "prim or refused or push or pick or look or dug or limit or bounds"`
Expected: FAIL with `ImportError: cannot import name 'ShelfPrimitiveEnv'`.

- [ ] **Step 3: Implement**

Append to `so101sim/shelf.py`, and add these constants near the top of the file with the others:

```python
MAX_PRIMITIVES = 20
PUSH_STROKE = sp.PUSH_STROKE
SETTLE_MAX_FRAMES = 60
# Positions this close outside the action box get clipped into it instead of refused.
ACTION_TOLERANCE = 0.005
# Holding means the gripper was last told to close and the joint stopped well short of it: an empty close reaches
# 10, a 3 cm cylinder stops it near 18. The real arm reads the same joint.
HOLDING_MIN = 14.0
```

```python
class _Refused(Exception):
    """A primitive the env won't run."""


class ShelfPrimitiveEnv(gym.Env):
    """One step is one whole pick, place, push, pull or look on the shelf.

    The action is {"kind", "xy", "angle"}, or the tuple (kind, xy, angle). `xy` is a point on the board: the
    cylinder to pick, the spot to place on, where a push starts, or the ray to look down. `angle` is the push
    heading and is ignored otherwise. `pull` ignores `xy` and takes the held cylinder out along the ray it came in
    on. There's no z: every cylinder stands on the same board.

    Refused, with the arm left where it is and the step still counted: `xy` outside the box, a push ending outside
    it, IK failing, pick, push or look while holding, and place or pull while not holding.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    PRIMITIVES = ("pick", "place", "push", "pull", "look")

    def __init__(self, max_primitives=MAX_PRIMITIVES, push_stroke=PUSH_STROKE, on_frame=None, **env_kwargs):
        # A primitive runs to completion inside one step, so a recorder needs a hook to see the frames under it.
        self.on_frame = on_frame
        self.env = ShelfEnv(max_steps=None, **env_kwargs)
        self.model, self.data = self.env.model, self.env.data
        self.geo = sp.Geometry.from_layout(self.model, self.env.layout)
        self.open_value = sp.side_open_value(self.model)
        self.max_primitives = int(max_primitives)
        self.push_stroke = float(push_stroke)
        x_lo, x_hi, y_lo, y_hi = self.env.region
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Dict({
            "kind": spaces.Discrete(len(self.PRIMITIVES)),
            "xy": spaces.Box(np.array([x_lo, y_lo], dtype=np.float32), np.array([x_hi, y_hi], dtype=np.float32)),
            "angle": spaces.Box(-np.pi, np.pi, (), dtype=np.float32),
        })
        self._n = 0
        self._last_grip = float(self.env.layout.home[5])
        self._aisle_xy = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        self._aisle_xy = None
        out = self.env.reset(seed=seed, options=options)
        self._last_grip = float(self.env.layout.home[5])
        return out

    def holding(self):
        return self._last_grip <= primitives.GRIP_CLOSE + 1.0 and float(self.env.joint_state[5]) >= HOLDING_MIN

    def step(self, action):
        kind, xy, angle = self._parse(action)
        self._n += 1
        refused, terminated, frames = None, False, 0
        info = self.env.task_info()
        try:
            traj = self._plan(kind, xy, angle)
        except _Refused as exc:
            refused, traj = str(exc), None
        except primitives.Unreachable as exc:
            refused, traj = f"unreachable: {exc}", None

        if traj is not None:
            traj = sp.bridge(self.model, self.geo, self.env.joint_state, traj)
            for row in traj:
                terminated, info = self._frame(row)
                frames += 1
                if terminated:
                    break
            else:
                for _ in range(SETTLE_MAX_FRAMES):
                    terminated, info = self._frame(traj[-1])
                    frames += 1
                    if terminated or self.env.settled():
                        break

        reward = 1.0 if info["is_success"] else 0.0
        truncated = not terminated and self._n >= self.max_primitives
        info = {**info, "primitive": kind, "frames": frames, "refused": refused}
        return self.env._obs(), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _frame(self, row):
        self._last_grip = float(row[5])
        _, terminated, _, info = self.env.advance(row)
        if self.on_frame is not None:
            self.on_frame(self)
        return terminated, info

    def _parse(self, action):
        if isinstance(action, dict):
            kind, xy, angle = action["kind"], action.get("xy", (0.0, 0.0)), action.get("angle", 0.0)
        else:
            kind, xy, angle = action
        if isinstance(kind, (int, np.integer)):
            kind = self.PRIMITIVES[int(kind)]
        if kind not in self.PRIMITIVES:
            raise ValueError(f"unknown primitive {kind!r}, expected one of {self.PRIMITIVES}")
        return kind, np.asarray(xy, dtype=np.float64).reshape(2), float(angle)

    def _inside(self, xy, what):
        box = self.action_space["xy"]
        clipped = np.clip(xy, box.low, box.high)
        if np.abs(xy - clipped).max() > ACTION_TOLERANCE:
            raise _Refused(f"{what} {np.round(xy, 4).tolist()} is outside the action box")
        return clipped

    def _plan(self, kind, xy, angle):
        holding = self.holding()
        if kind in ("place", "pull") and not holding:
            raise _Refused(f"{kind} needs a held cylinder and the gripper isn't holding one")
        if kind in ("pick", "push", "look") and holding:
            raise _Refused(f"{kind} with a cylinder still in the gripper; place or pull it first (holding)")
        if kind == "pull":
            return sp.pull(self.model, self.geo, self._aisle_xy)
        xy = self._inside(xy, kind)
        if kind == "pick":
            traj = sp.pick(self.model, self.geo, xy, self.open_value)
            self._aisle_xy = self.geo.on_ray(xy, self.geo.aisle_x)
            return traj
        if kind == "place":
            return sp.place(self.model, self.geo, self._aisle_xy, xy, self.open_value)
        if kind == "look":
            return sp.look(self.model, self.geo, xy)
        end = xy + self.push_stroke * np.array([np.cos(angle), np.sin(angle)])
        self._inside(end, f"a push heading {np.degrees(angle):.0f} deg ends at")
        return sp.push(self.model, self.geo, xy, angle, stroke=self.push_stroke)
```

- [ ] **Step 4: Run the tests, except the gate**

Run: `$PY -m pytest tests/test_shelf.py -q -k "not dug_out"`
Expected: all pass. The ones most likely to need work are `test_a_low_push_slides_a_cylinder_without_tipping_it`
and `test_a_clear_gray_can_be_picked_and_placed_upright`, since they're the first real physics. A failure there is
information, not something to paper over. Use superpowers:systematic-debugging, render the run with the
`on_frame` hook, and look before changing a constant. A changed constant needs a comment with what was measured.

- [ ] **Step 5: Look at one pick and place**

```bash
$PY - <<'EOF'
import imageio, numpy as np, so101sim
from so101sim.shelf import ShelfPrimitiveEnv
frames = []
env = ShelfPrimitiveEnv(on_frame=lambda e: frames.append(np.concatenate([e.env._render("front"), e.env._render("wrist")], 1)))
_, info = env.reset(seed=1, options={"hidden": False})
name = next(iter(info["cylinders"]))
for action in (("pick", info["cylinders"]["cyl_gray_0"]["pos"][:2], 0.0), ("place", env.env.free_spot(), 0.0)):
    _, _, term, _, info = env.step(action)
    print(action[0], info["refused"], info["failure"])
env.close()
imageio.mimwrite("outputs/shelf_pick_place.mp4", frames, fps=30, macro_block_size=1)
EOF
```

Pull a few frames out of `outputs/shelf_pick_place.mp4` into `$SCRATCH` and read them. The gripper should go in
level along a ray, with the mount on top, and back out. Whether it knocks something over depends on the gray's
neighbours; `cyl_gray_0` isn't chosen for clearance. Send `outputs/shelf_pick_place.mp4` to dm1487.

- [ ] **Step 6: Find a solvable seed**

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0, "tests")
import so101sim
from test_shelf import dig_out
from so101sim.shelf import ShelfPrimitiveEnv
env = ShelfPrimitiveEnv(height=64, width=64)
wins = []
for seed in range(20):
    info = dig_out(env, seed)
    print(seed, info["is_success"], info["failure"], info.get("refused"), flush=True)
    if info["is_success"]:
        wins.append(seed)
env.close()
print("solvable:", wins)
EOF
```

Set `SOLVABLE_SEED` in `tests/test_shelf.py` to the first seed in `solvable`, with a comment giving the date and
how many of 0 to 19 succeeded. Then record that dig-out and send it to dm1487:

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0, "tests")
import imageio, numpy as np, so101sim
from test_shelf import SOLVABLE_SEED, dig_out
from so101sim.shelf import ShelfPrimitiveEnv
frames = []
env = ShelfPrimitiveEnv(on_frame=lambda e: frames.append(np.concatenate([e.env._render("front"), e.env._render("wrist")], 1)))
info = dig_out(env, SOLVABLE_SEED)
env.close()
print(info["is_success"], info["failure"])
imageio.mimwrite(f"outputs/shelf_digout_seed{SOLVABLE_SEED}.mp4", frames, fps=30, macro_block_size=1)
EOF
```

The scripted dig-out works from privileged poses, so caption it as a solvability check, not a policy. If none do, stop and bring the printed failures back to dm1487. That's the spec's
gate. Don't tune `blockers` or `place_spot` to rescue one seed without saying so in the report.

- [ ] **Step 7: Run the whole shelf suite**

Run: `$PY -m pytest tests/test_shelf.py -q`
Expected: all pass.

---

### Task 9: `scripts/shelf_play.py`, pixel actions for the planner

**Files:**
- Create: `scripts/shelf_play.py`, `tests/test_shelf_play.py`

**Interfaces:**
- Consumes: `ShelfPrimitiveEnv`, `env.CAMERAS`, `shelf_layout.CYL_RADIUS`, `side_primitives.push_directions`.
- Produces: `pixel_ray(model, data, camera, u, v, width, height) -> (origin (3,), direction (3,))`,
  `project(model, data, camera, point, width, height) -> (u, v)`, `deproject(prim, action, width, height) -> (2,) |
  None`, `to_env_action(prim, action) -> dict`, `grid_image(img) -> np.ndarray`, `run(seed, start, actions, out,
  video)`. The CLI in the docstring.

- [ ] **Step 1: Write the failing tests**

`tests/test_shelf_play.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

import so101sim  # noqa: F401
from so101sim.env import CAMERAS
from so101sim.shelf import TARGET, ShelfPrimitiveEnv

import mujoco  # after so101sim, which sets MUJOCO_GL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import shelf_play  # noqa: E402

W, H = 640, 480


@pytest.fixture(scope="module")
def prim():
    e = ShelfPrimitiveEnv()  # 640 x 480, the size the planner sees
    yield e
    e.close()


def red_centroid(prim, camera):
    """Mean pixel of the red cylinder in a segmentation render with the grays hidden."""
    seg = prim.env._seg
    seg.update_scene(prim.data, camera=camera, scene_option=prim.env._hide_grays)
    ids = seg.render()
    v, u = np.nonzero((ids[..., 0] == prim.model.geom(TARGET).id) & (ids[..., 1] == 5))
    assert len(u) > 50, f"the red cylinder is out of view of {camera}"
    return u.mean(), v.mean()


@pytest.mark.parametrize("camera", list(CAMERAS.values()))
def test_project_and_pixel_ray_are_inverses(prim, camera):
    prim.reset(seed=0)
    point = np.array([0.40, 0.03, prim.env.layout.grasp_z])
    u, v = shelf_play.project(prim.model, prim.data, camera, point, W, H)
    origin, d = shelf_play.pixel_ray(prim.model, prim.data, camera, u, v, W, H)
    t = (point - origin) @ d
    assert np.linalg.norm(origin + t * d - point) < 1e-3


@pytest.mark.parametrize("camera", list(CAMERAS.values()))
def test_the_lens_model_matches_the_renderer(prim, camera):
    prim.reset(seed=0, options={"hidden": False})
    if camera == CAMERAS["wrist"]:
        prim.step(("look", prim.env.cylinder_pose(TARGET)["pos"][:2], 0.0))
    red = prim.env.cylinder_pose(TARGET)["pos"]
    u, v = shelf_play.project(prim.model, prim.data, camera, red, W, H)
    cu, cv = red_centroid(prim, camera)
    assert np.hypot(u - cu, v - cv) < 12, ((u, v), (cu, cv))


def test_a_pick_pixel_lands_on_the_cylinder_axis(prim):
    prim.reset(seed=0, options={"hidden": False})
    red = prim.env.cylinder_pose(TARGET)["pos"]
    face = red.copy()
    face[2] = prim.env.layout.grasp_z
    u, v = shelf_play.project(prim.model, prim.data, "front", face, W, H)
    xy = shelf_play.deproject(prim, {"kind": "pick", "cam": "front", "px": [u, v]}, W, H)
    assert np.linalg.norm(xy - red[:2]) < 0.012


def test_wrist_pixels_deproject_from_the_current_pose(prim):
    prim.reset(seed=0, options={"hidden": False})
    red = prim.env.cylinder_pose(TARGET)["pos"]
    prim.step(("look", red[:2] + [0.0, 0.06], 0.0))  # the wrist has moved since reset
    spot = np.array([red[0], red[1], prim.env.layout.board_z])
    u, v = shelf_play.project(prim.model, prim.data, "wrist_cam", spot, W, H)
    xy = shelf_play.deproject(prim, {"kind": "place", "cam": "wrist", "px": [u, v]}, W, H)
    assert np.linalg.norm(xy - red[:2]) < 0.005


def test_a_pixel_off_the_board_is_refused(prim):
    prim.reset(seed=0)
    action = shelf_play.to_env_action(prim, {"kind": "pick", "cam": "front", "px": [320, 5]})
    _, _, terminated, _, info = prim.step(action)
    assert info["refused"] and not terminated


def test_grid_image_keeps_the_size():
    img = np.zeros((H, W, 3), np.uint8)
    assert shelf_play.grid_image(img).shape == (H, W, 3)
```

The segmentation test compares `ids[..., 1] == 5` because `mjOBJ_GEOM` is 5; `shelf.GEOM` holds the same value.

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_shelf_play.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf_play'`.

- [ ] **Step 3: Implement**

`scripts/shelf_play.py`:

```python
"""Play ShelfPrimitiveEnv from camera images, as a planner that sees only what a policy sees.

    python scripts/shelf_play.py --seed 300 --start hidden --actions outputs/shelf_play_300/actions.json \
        --out outputs/shelf_play_300 [--video]

Replays every action in the JSON list from reset, since the sim is deterministic, and writes front.png and
wrist.png of the latest observation with a labelled pixel grid. Prints what a policy gets back per step and nothing
else: refused and why, success, the failure message. Never a pose.

An action names a pixel, not a position:

    {"kind": "pick" | "place" | "push" | "pull" | "look", "cam": "front" | "wrist", "px": [u, v], "heading": k}

The ray through that pixel comes from the camera's pose and lens. The wrist camera's pose follows from the joint
angles, which the real arm knows too. Pick intersects the ray with the grasp-height plane, which lands on the
cylinder's visible face, then steps one radius further along the ray's horizontal direction to its axis. Place,
push and look intersect the board's plane. `heading` is one of 16 push directions, k * 22.5 degrees anticlockwise
from +x, which points away from the robot. `pull` needs no pixel.
"""

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

import so101sim  # noqa: F401  before mujoco, which reads MUJOCO_GL at import
from so101sim import side_primitives as sp
from so101sim.env import CAMERAS, CONTROL_HZ
from so101sim.shelf import ShelfPrimitiveEnv
from so101sim.shelf_layout import CYL_RADIUS

GRID_STEP = 40
LABEL_STEP = 80


def _lens(model, cam, width, height):
    """Pixel focal lengths and principal point.

    A camera with a sensor size, like the wrist camera, renders its sensor stretched over the image, so its pixels
    aren't square at 640 x 480. Otherwise fovy is the vertical field of view and pixels are square.
    """
    if model.cam_sensorsize[cam].any():
        fx, fy, px, py = model.cam_intrinsic[cam]
        sw, sh = model.cam_sensorsize[cam]
        return fx / sw * width, fy / sh * height, width / 2 + px / sw * width, height / 2 - py / sh * height
    f = 0.5 * height / np.tan(np.radians(model.cam_fovy[cam]) / 2)
    return f, f, width / 2, height / 2


def pixel_ray(model, data, camera, u, v, width, height):
    """World origin and unit direction of the ray through pixel (u, v), u across and v down from the top left."""
    cam = model.camera(camera).id
    fx, fy, cx, cy = _lens(model, cam, width, height)
    local = np.array([(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -1.0])
    d = data.cam_xmat[cam].reshape(3, 3) @ local
    return data.cam_xpos[cam].copy(), d / np.linalg.norm(d)


def project(model, data, camera, point, width, height):
    """Pixel (u, v) where a world point lands."""
    cam = model.camera(camera).id
    fx, fy, cx, cy = _lens(model, cam, width, height)
    p = data.cam_xmat[cam].reshape(3, 3).T @ (np.asarray(point, dtype=np.float64) - data.cam_xpos[cam])
    return cx + fx * p[0] / -p[2] - 0.5, cy - fy * p[1] / -p[2] - 0.5


def deproject(prim, action, width=640, height=480):
    """Board xy for a pixel action, or None for pull or a ray that never comes down to the plane."""
    if action["kind"] == "pull":
        return None
    origin, d = pixel_ray(prim.model, prim.data, CAMERAS[action.get("cam", "front")], *action["px"], width, height)
    layout = prim.env.layout
    z = layout.grasp_z if action["kind"] == "pick" else layout.board_z
    if d[2] > -1e-6 or origin[2] <= z:
        return None
    hit = origin + ((z - origin[2]) / d[2]) * d
    if action["kind"] == "pick":
        return hit[:2] + CYL_RADIUS * d[:2] / np.linalg.norm(d[:2])
    return hit[:2]


def to_env_action(prim, action):
    """The env action for a pixel action. A pixel that misses the board becomes a point the env will refuse."""
    xy = deproject(prim, action)
    if xy is None and action["kind"] != "pull":
        xy = prim.action_space["xy"].high + 1.0
    heading = sp.push_directions()[int(action.get("heading", 0)) % sp.PUSH_DIRECTIONS]
    return {"kind": action["kind"], "xy": np.zeros(2) if xy is None else xy, "angle": heading}


def grid_image(img):
    """A copy with a line every GRID_STEP pixels and u,v labels every LABEL_STEP."""
    out = Image.fromarray(np.ascontiguousarray(img))
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for u in range(0, w, GRID_STEP):
        draw.line([(u, 0), (u, h)], fill=(255, 255, 0) if u % LABEL_STEP == 0 else (120, 120, 0))
    for v in range(0, h, GRID_STEP):
        draw.line([(0, v), (w, v)], fill=(255, 255, 0) if v % LABEL_STEP == 0 else (120, 120, 0))
    for u in range(0, w, LABEL_STEP):
        for v in range(0, h, LABEL_STEP):
            draw.text((u + 2, v + 1), f"{u},{v}", fill=(255, 255, 255))
    return np.asarray(out)


def run(seed, start, actions, out, video=False):
    out.mkdir(parents=True, exist_ok=True)
    frames = []

    def record(p):
        frames.append(np.concatenate([p.env._render("front"), p.env._render("wrist")], axis=1))

    prim = ShelfPrimitiveEnv(on_frame=record if video else None)
    try:
        obs, _ = prim.reset(seed=seed, options={"hidden": start == "hidden"})
        print(f"reset: seed {seed}, {start} start, {len(actions)} actions to replay")
        for i, action in enumerate(actions):
            obs, _, terminated, truncated, info = prim.step(to_env_action(prim, action))
            where = "" if action["kind"] == "pull" else f" {action.get('cam', 'front')} px {action['px']}"
            result = f"refused ({info['refused']})" if info["refused"] else f"ran {info['frames']} frames"
            if info["is_success"]:
                result += ", SUCCESS"
            if info["failure"]:
                result += f", FAILURE: {info['failure']}"
            print(f"{i}: {action['kind']}{where}"
                  f"{' heading ' + str(action['heading']) if 'heading' in action else ''} -> {result}")
            if terminated or truncated:
                print("episode over" + (" (primitive limit)" if truncated else ""))
                break
        Image.fromarray(grid_image(obs["observation.images.front"])).save(out / "front.png")
        Image.fromarray(grid_image(obs["observation.images.wrist"])).save(out / "wrist.png")
        print(f"wrote {out / 'front.png'} and {out / 'wrist.png'}")
    finally:
        prim.close()
    if video and frames:
        path = out.parent / f"{out.name}.mp4"
        imageio.mimwrite(path, frames, fps=CONTROL_HZ, macro_block_size=1)
        print(f"wrote {len(frames)} frames to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", choices=("hidden", "partial"), required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    actions = json.loads(args.actions.read_text()) if args.actions.exists() else []
    if not isinstance(actions, list):
        sys.exit(f"{args.actions} must hold a JSON list of actions")
    run(args.seed, args.start, actions, args.out, args.video)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_shelf_play.py -q`
Expected: all pass. If `test_the_lens_model_matches_the_renderer` fails only for `wrist_cam`, MuJoCo isn't
stretching the sensor over the image the way `_lens` assumes. Print both centroids, try square pixels (`fy / sh *
height` for both focal lengths), and keep whichever the renderer agrees with, with a comment saying which was
measured.

---

### Task 10: Claude plays seeds 300 hidden and 301 partial

**Files:**
- Create: `outputs/shelf_play_300/actions.json`, `outputs/shelf_play_301/actions.json`, `outputs/shelf_play_300.md`,
  `outputs/shelf_play_301.md`, `outputs/shelf_play_300.mp4`, `outputs/shelf_play_301.mp4`

**Rules for the planner:**
- Decide only from `front.png`, `wrist.png` and the printed step results. Don't read `info`, poses, the scene XML,
  a segmentation render, or anything that knows where the red cylinder is.
- Look before acting: a `look` costs a primitive and nothing else. Use it when the red isn't in view.
- No undo. An action stays once it has run. A refused action still counts. A failure ends the attempt and is
  reported as a failure.
- 20 primitives at most.

- [ ] **Step 1: See the start of seed 300**

```bash
mkdir -p outputs/shelf_play_300 && echo '[]' > outputs/shelf_play_300/actions.json
$PY scripts/shelf_play.py --seed 300 --start hidden --actions outputs/shelf_play_300/actions.json --out outputs/shelf_play_300
```

Read `outputs/shelf_play_300/front.png` and `wrist.png`.

- [ ] **Step 2: Act, one primitive at a time**

Append one action to `actions.json`, rerun the same command, and read both images again. Before each action, write
a line in `outputs/shelf_play_300.md` with what the images show, the action, and why. Afterwards, add what came
back. Repeat until success, failure, or 20 primitives. Tell dm1487 in one line what's happening every few actions.

- [ ] **Step 3: Record it**

```bash
$PY scripts/shelf_play.py --seed 300 --start hidden --actions outputs/shelf_play_300/actions.json --out outputs/shelf_play_300 --video
```

- [ ] **Step 4: Seed 301, partial**

Repeat steps 1 to 3 with `--seed 301 --start partial` and the `shelf_play_301` paths.

- [ ] **Step 5: Report**

Send each run's mp4 with SendUserFile as soon as step 3 writes it, before starting the next seed. Report each
run's outcome as it happened, success or not, the number of primitives, and where the planning went wrong if it
did.

---

### Task 11: Docs, and the full suite

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with small, targeted edits**

Another session edits this file too, so re-read it right before each edit and change only these places:

- "What this is": one sentence saying a second search task exists, a shelf of upright cylinders reached sideways,
  spec `docs/superpowers/specs/2026-09-26-shelf-search-design.md`.
- Commands: `python scripts/size_shelf.py --write`, `python scripts/shelf_smoke.py --episodes 8`,
  `python scripts/shelf_smoke.py --tune-count --write`, and the `shelf_play.py` line from its docstring.
- Layout and conventions: `ik.solve_side`, `side_primitives.py`, `shelf_layout.py` and `shelf_checks.py`, and
  `shelf.py`, one or two sentences each, in the style of the existing bullets.
- Traps:
  - The wrist camera mount decides the side-grasp roll. Mount down, it hits a board 6 cm above the base.
  - In that roll the moving jaw always opens to the robot's right, so a side grasp needs room on the right.
  - A level gripper only points along rays from the pan axis at (0.0388, 0), not from the origin.
  - Level reach runs 0.22 to 0.44 m from the pan axis, and its inner limit moves out to 0.24 m at push height.
  - Editing `geom_pos` on a compiled model doesn't move a world-body geom. Recompile.
  - A push 2.5 cm above the board slides a cylinder. 2 cm leaves it leaning and 3 cm tips it.
- The real arm: the shelf board sits `board_z` above the base's mounting plane with its front edge `front_x` from
  the base origin, both from the committed layout, as a hardware requirement like the bin's riser.

- [ ] **Step 2: Run everything**

```bash
$PY -m pytest -q -m "not mjx"
```

Expected: every test passes, the bin's and binpick's included, since nothing of theirs changed. Paste the summary
line into the final report.

- [ ] **Step 3: Report**

Tell dm1487 what was built, the chosen layout and count from the XML notes, the dig-out gate's seed and hit rate,
both planner runs' outcomes, and that nothing is staged or committed.
