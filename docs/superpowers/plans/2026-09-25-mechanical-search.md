# Mechanical search implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bin sunk below the SO-101's base holds a pile of 30 mm cubes up to three layers deep. A 30 Hz env and a pick/place/push primitive env ask a policy to find the one red cube and lift it out without any cube leaving the bin.

**Architecture:** The scene splits into a shared `stage.xml` plus one objects file per task. The mechanical-search objects file is generated from a small `Layout` dataclass, and a sizing script picks that layout by IK and collision checks. `MechSearchEnv` subclasses `SO101Env`, building the pile at reset and checking success and failure every frame. `MechSearchPrimitiveEnv` wraps it the way `BinPickEnv` wraps `SO101Env`.

**Tech stack:** Python 3.10, mujoco 3.8.1 on CPU with EGL rendering, gymnasium, numpy, pytest, imageio. Conda prefix env at `/common/users/dm1487/envs/so101sim`.

**Spec:** `docs/superpowers/specs/2026-09-25-mechanical-search-design.md`. Read it before starting. This plan argues from it.

## Global constraints

- Activate the env first: `conda activate /common/users/dm1487/envs/so101sim`.
- `import so101sim` before `mujoco` in every script and test module. The package sets `MUJOCO_GL=egl`, and MuJoCo reads it at import.
- Call `close()` on every env and renderer you create.
- Never edit `models/robotstudio_so101/`.
- Everything outside MuJoCo is in lerobot units. Arm joints are in degrees, and the gripper runs from 0 closed to 100 open, in `units.JOINT_NAMES` order.
- Cubes are 30 mm, `primitives.CUBE_HALF = 0.015`.
- Success needs both jaws on the red cube and its lowest corner at least 0.01 m over the rim, for 5 frames in a row.
- `MechSearchEnv` defaults: `max_steps=1800`, `buried_prob=0.5`. `MechSearchPrimitiveEnv` defaults: `max_primitives=15`, push stroke 0.05 m.
- Reset drops cubes in waves of at most 4, spaced at least 0.045 m apart, each wave's bottoms 0.01 m over the highest cube in the bin. A wave settles until max|qvel| ≤ 0.2, capped at 1 s of sim time.
- A pile is kept only if all of these hold:
  - Every cube's center is inside the bin.
  - No cube's center is above floor + 0.085 m.
  - The pile is settled.
  - The floor has a free spot at least 0.018 m from the walls and 0.045 m from every bottom-layer cube center.
  - No cube touched the arm.
  - A buried pile shows 0 red pixels in both cameras. A visible pile shows at least 100 in one camera, at 640×480.
- `TRAVEL_Z` = max(floor + 0.11, rim + 0.03).
- Front camera must see ≥ 70% of the empty floor, wrist camera ≥ 50% at the start pose. The drop column runs from the floor to floor + 0.13.
- Cube count starts at 30 and floors at 10. At least half of resets must be accepted, for buried and visible starts separately. Reset should average under 3 s. If it doesn't, report the number rather than cutting cubes.
- **Stop points.** Stop and bring it back to dm1487 in any of these cases:
  - No layout passes, down to 12 cm wide (Task 5).
  - No start pose or no front camera height passes (Task 5).
  - The cube count falls below 10 (Task 8).
  - The dig-out test can't pass on any of 20 seeds (Task 9).

  Never relax a threshold to get past one.
- The repo has no commits yet. Each task ends by staging its files with `git add`. Run `git commit` only if dm1487 has said to commit.
- Scratch files go in `SCRATCH=/tmp/claude-89862/-common-home-dm1487-robotics-research-rearrange/3f0a8add-81a1-4f27-b2ac-ccd25a8cff24/scratchpad`.
- For rendering, check `nvidia-smi` and set `MUJOCO_EGL_DEVICE_ID` to a free card.
- CPU suite: `python -m pytest -q -m "not mjx"`. The full suite adds the MJX test, about 10 s warm.

## Where this departs from the spec's wording

Each change keeps what the spec requires but changes where it lives or how it's measured.

- **`SEARCH_HOME` and `TRAVEL_Z`** don't live as constants in `mechsearch.py`. The start pose is stored in the generated XML as `<numeric name="search_home">`, and `Layout.from_model` reads it back. `TRAVEL_Z` is computed from the bin with `search_layout.travel_z`. That way the sizing script's choice can't drift from the code.
- **Camera coverage** is measured with 6 mm marker tiles, 1 cm apart, on the floor of a probe copy of the scene, counted in a segmentation render. The definition is the same as the spec's, a floor point the camera actually sees, but the renderer does the projection and occlusion itself instead of `mj_ray` plus hand-written camera math. The wrist camera uses `sensorsize` intrinsics, which that math would get wrong.
- **The drop column** is an invisible collision box in the probe scene, and the arm is tested against it with MuJoCo's own contacts.
- **The cube and bin-wall default classes** move into `stage.xml`, so both scenes' cubes stay identical.
- **`SO101Env.step`** gains an `_apply` helper, so `MechSearchEnv` can run a frame without rendering. The primitive env renders once per primitive, not once per frame.
- **`info["cubes"]`** carries every cube's pose, the same as `BinPickEnv`'s info. The dig-out test and any scripted planner need it.
- **`primitives.bridge`** used to fall back to a joint ramp whenever the gripper was below z = 0.03. That threshold now sits 0.025 below `hover_z`, which is still 0.03 at the default hover height. Without the change, every crossing in a sunken bin, where travel height is about 0.01 to 0.04, would ramp in joint space and sweep the gripper through the pile. CLAUDE.md forbids that.

## Review focus

These are the five inputs most likely to break for someone using this, none of which the spec pins down:

1. **Observation size.** Asking for small observation images must not change the buried label. The pixel count comes from its own 640×480 renderer. Task 6 tests this.
2. **Bare reset.** `reset()` with no seed and no options must build a valid pile. Task 6 tests this.
3. **Box corners.** A primitive aimed at a corner of the action box must either run or be refused, never raise. Task 9 tests this.
4. **Pick while holding.** Picking again while a cube is still in the jaws must not raise. The held cube drops inside the bin. Task 9 tests this.
5. **Failure mid-primitive.** If a failure happens partway through a primitive, the arm must stop on that frame, and the step returns immediately. Task 9 tests this.

## File map

| File | What it does |
|---|---|
| `models/stage.xml` (new) | Arm include, physics, visuals, assets, default classes, lights, ground, base collision. |
| `models/objects.xml` (modify) | Gains the bin-pick table and front camera, loses the default classes. |
| `models/scene.xml` (modify) | Becomes `stage.xml` + `objects.xml`. |
| `models/search_objects.xml` (generated) | The search task's table, riser, bin, cubes, camera and layout numbers. |
| `models/search_scene.xml` (generated) | `stage.xml` + `search_objects.xml`. |
| `so101sim/__init__.py` (modify) | Adds `SEARCH_SCENE_PATH`. |
| `so101sim/grasp.py` (new) | Jaw and arm geom sets, the two-jaw grasp check, and the contact test between two geom sets. |
| `so101sim/binpick.py` (modify) | Uses `grasp`. |
| `so101sim/primitives.py` (modify) | `pick` takes a grasp height. `bridge`'s low threshold becomes relative to `hover_z`. |
| `so101sim/env.py` (modify) | `step` split into `_apply` and the observation. |
| `so101sim/search_layout.py` (new) | `Layout`, the XML generator, `travel_z`, candidate scenes, notes. |
| `so101sim/search_checks.py` (new) | Reach, clearance, drop-column and camera-coverage checks. |
| `so101sim/mechsearch.py` (new) | `MechSearchEnv` and `MechSearchPrimitiveEnv`. |
| `scripts/size_search_bin.py` (new) | Picks the layout, start pose and camera height, then writes the scene. |
| `scripts/mechsearch_smoke.py` (new) | Video of starting piles, reset time, acceptance, cube-count tuning. |
| `tests/test_grasp.py`, `tests/test_search_layout.py`, `tests/test_search_checks.py`, `tests/test_mechsearch.py` (new) | Tests. |
| `tests/test_binpick.py` (modify) | Tests for the `pick` and `bridge` changes. |
| `.gitignore`, `models/README.md`, `README.md`, `CLAUDE.md` (modify) | Docs and ignores. |

---

### Task 1: Split the shared stage out of `scene.xml`

**Files:**
- Create: `models/stage.xml`
- Modify: `models/scene.xml`, `models/objects.xml`, `models/README.md`

**Interfaces:**
- Produces: `models/stage.xml`, which any top-level scene in `models/` includes first. It defines the default classes `cube` and `bin_wall` and the material `table`. The bin-pick scene compiles to the same model as before.

This task is a pure refactor, so its test is a before/after fingerprint of the compiled model plus the existing suite.

- [ ] **Step 1: Save a fingerprint of the current model**

Write `$SCRATCH/fingerprint.py`:

```python
import sys

import so101sim
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path(str(so101sim.SCENE_PATH))
fields = ["geom_type", "geom_size", "geom_pos", "geom_quat", "geom_rgba", "geom_friction", "geom_contype",
          "geom_conaffinity", "geom_group", "geom_bodyid", "geom_condim", "body_pos", "body_quat", "body_mass",
          "body_parentid", "cam_pos", "cam_quat", "cam_fovy", "cam_mode", "cam_targetbodyid", "light_pos",
          "light_dir", "jnt_range", "actuator_gainprm", "mat_rgba", "tex_type"]
fp = {f: np.asarray(getattr(m, f)) for f in fields}
fp["timestep"] = np.array([m.opt.timestep])
fp["names"] = np.array([m.geom(i).name for i in range(m.ngeom)] + [m.body(i).name for i in range(m.nbody)]
                       + [m.camera(i).name for i in range(m.ncam)])
if sys.argv[1] == "save":
    np.savez(sys.argv[2], **fp)
    print(f"saved {len(fp)} fields")
else:
    ref = np.load(sys.argv[2])
    for k in fp:
        np.testing.assert_array_equal(fp[k], ref[k], err_msg=k)
    print(f"identical across {len(fp)} fields")
```

Run: `python $SCRATCH/fingerprint.py save $SCRATCH/scene_before.npz`
Expected: `saved 28 fields`

- [ ] **Step 2: Create `models/stage.xml`**

```xml
<mujoco>
  <!-- What every task scene shares: the arm, physics settings, visuals, default classes, lights, the
       ground and the base's stand-in collision box. A task's top-level scene includes this file, then its
       own objects file with the table, the front camera and the task's objects.

       The vendored arm, untouched. Its compiler sets meshdir="assets", which MuJoCo resolves against the
       top-level file's directory, so the compiler line after the include points it back at the vendored
       assets. That's also why every top-level scene has to live in models/. -->
  <include file="robotstudio_so101/so101.xml"/>
  <compiler meshdir="robotstudio_so101/assets"/>

  <!-- 10 physics steps per 30 Hz control step. Menagerie ships 0.005. -->
  <option timestep="0.0033333333333333335"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="160" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="table" rgba="0.55 0.42 0.3 1"/>
  </asset>

  <!-- Shared so a cube behaves the same in every task. -->
  <default>
    <default class="cube">
      <geom type="box" size="0.015 0.015 0.015" density="1000" condim="4" friction="1 0.005 0.0001"/>
    </default>
    <default class="bin_wall">
      <geom type="box" rgba="0.3 0.3 0.35 1"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <light pos="0.6 0.4 1.0" dir="-0.5 -0.3 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.75" material="floor"/>

    <!-- The MJCF base has no collision geometry, so planners and contacts pass through it. Stand-in box
         from the so101 session's planning_scene.xml. The shoulder is bolted to the base, hence the exclude. -->
    <body name="base_collision" pos="0.010 0 0.040">
      <geom name="base_box" type="box" size="0.045 0.045 0.040" group="3"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="base_collision" body2="shoulder"/>
  </contact>
</mujoco>
```

- [ ] **Step 3: Replace `models/scene.xml`**

```xml
<mujoco model="so101_table">
  <!-- The bin-picking scene: the shared stage, then this task's table, front camera and objects. -->
  <include file="stage.xml"/>
  <include file="objects.xml"/>
</mujoco>
```

- [ ] **Step 4: Move the table and front camera into `models/objects.xml`, and drop its `<default>` block**

Replace everything from the opening `<mujoco>` through the line `<worldbody>` with the block below. Leave the cube bodies and the bin body that follow exactly as they are.

```xml
<mujoco>
  <!-- The bin-picking task's table, front camera and objects. The cube and bin_wall classes come from
       stage.xml. Edit this file to change what's on the table.

       Every position here is measured against what the arm can reach with the gripper pointing straight
       down, which is the only grasp family a 5-DOF arm has. Top-down grasping reaches x 0.12..0.34,
       |y| <= 0.24. Releasing a held cube over a 2.5 cm rim reaches x 0.14..0.28, |y| <= 0.22. Jaw yaw is
       fully controllable inside x <= 0.26, and starts dropping out beyond that at negative y. The cube
       spawn region in binpick.py sits inside all three limits. -->
  <worldbody>
    <!-- Table top surface is z=0, where the arm base sits. -->
    <geom name="table" type="box" size="0.35 0.45 0.02" pos="0.2 0 -0.02" material="table"/>

    <body name="front_cam_target" pos="0.18 0 0.02"/>
    <camera name="front" mode="targetbody" target="front_cam_target" pos="0.7 0 0.45" fovy="50"/>
```

- [ ] **Step 5: Check that the compiled model is unchanged**

Run: `python $SCRATCH/fingerprint.py check $SCRATCH/scene_before.npz`
Expected: `identical across 28 fields`. If a field differs, the element order moved. Fix the XML, don't loosen the check.

- [ ] **Step 6: Run the suites**

Run: `python -m pytest -q`
Expected: every test passes, MJX included, because `so101sim.mjx` loads the same scene.

- [ ] **Step 7: Update `models/README.md`**

Change "Put changes in `scene.xml`, so a menagerie update is a clean re-copy." to "Put changes in `stage.xml` or a task's objects file, so a menagerie update is a clean re-copy."

Then replace the block from "`scene.xml` includes the arm and adds:" through "`objects.xml` holds the cubes and the bin. Edit it to change the task." with:

```markdown
`stage.xml` includes the arm and adds what every scene shares:

- A compiler `meshdir` override. MuJoCo resolves `meshdir` against the top-level file, so the vendored
  `meshdir="assets"` would look in `models/assets/` without it.
- Timestep 1/300 s, so a 30 Hz control step is 10 physics steps. Menagerie ships 0.005 s.
- The `cube` and `bin_wall` default classes, so a cube behaves the same in every task.
- `base_collision`, a box standing in for the base. The MJCF base has visual meshes only, and menagerie
  removed the collision meshes on purpose. The box size came from the `so101` session, whose first try
  at 0.052 clipped the upper arm by 3.7 mm at rest.
- Lights and the ground plane.

Each task has a top-level scene in `models/` that includes `stage.xml` and then its own objects file. The
top-level file has to live here, because this is the directory `meshdir` resolves against.

- `scene.xml` is the bin-picking scene. `objects.xml` holds its table, with the top at z=0 where the arm
  base sits, a `front` camera aimed at `front_cam_target`, the cubes and the bin. The arm's own
  `wrist_cam` comes from menagerie.
```

- [ ] **Step 8: Stage**

```bash
git add models/stage.xml models/scene.xml models/objects.xml models/README.md
```

---

### Task 2: Move the grasp and contact checks into `so101sim/grasp.py`

**Files:**
- Create: `so101sim/grasp.py`, `tests/test_grasp.py`
- Modify: `so101sim/binpick.py`

**Interfaces:**
- Produces:
  - `grasp.JAW_PREFIXES: tuple[str, str]`
  - `grasp.jaw_geom_ids(model) -> frozenset[int]`
  - `grasp.arm_geom_ids(model) -> frozenset[int]`, every geom on a body below `base`
  - `grasp.is_grasped(model, data, geom_id, jaw_ids) -> bool`
  - `grasp.touching(data, group_a, group_b) -> bool`, true if any current contact joins the two sets

- [ ] **Step 1: Write the failing tests** in `tests/test_grasp.py`

```python
import pytest

from so101sim import SCENE_PATH, grasp, ik

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def test_jaw_geoms_cover_both_jaws_and_nothing_else(model):
    names = {model.geom(i).name for i in grasp.jaw_geom_ids(model)}
    assert set(ik.JAW_TIP_GEOMS) <= names
    assert all(n.startswith(grasp.JAW_PREFIXES) for n in names)
    assert any(n.startswith("fixed_jaw") for n in names)
    assert any(n.startswith("moving_jaw") for n in names)


def test_arm_geoms_are_the_moving_links_only(model):
    bodies = {model.body(model.geom_bodyid[g]).name for g in grasp.arm_geom_ids(model)}
    assert {"shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"} <= bodies
    assert not bodies & {"world", "base", "base_collision", "bin", "cube_red"}


def test_a_resting_scene_has_no_grasp_and_no_arm_contact(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    red = model.geom("cube_red").id
    assert not grasp.is_grasped(model, data, red, grasp.jaw_geom_ids(model))
    assert not grasp.touching(data, grasp.arm_geom_ids(model), {red})


def test_touching_sees_a_cube_resting_on_the_table(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    assert grasp.touching(data, {model.geom("cube_red").id}, {model.geom("table").id})
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_grasp.py -q`
Expected: FAIL, `ImportError: cannot import name 'grasp'`.

- [ ] **Step 3: Write `so101sim/grasp.py`**

```python
"""Contact checks against the arm, shared by the task envs."""

from so101sim import ik

JAW_PREFIXES = ("fixed_jaw", "moving_jaw")


def jaw_geom_ids(model):
    """Every geom on either jaw: the tip spheres plus the boxes named fixed_jaw_* and moving_jaw_*."""
    ids = {model.geom(name).id for name in ik.JAW_TIP_GEOMS}
    ids |= {i for i in range(model.ngeom) if model.geom(i).name.startswith(JAW_PREFIXES)}
    return frozenset(ids)


def arm_geom_ids(model):
    """Every geom on a body below the arm's `base`, which is every link that moves."""
    base = model.body("base").id
    moving = set()
    for b in range(model.nbody):
        a = b
        while a not in (0, base):
            a = model.body_parentid[a]
        if a == base and b != base:
            moving.add(b)
    return frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] in moving)


def is_grasped(model, data, geom_id, jaw_ids):
    """True when geoms from both jaws are touching `geom_id`."""
    touched = set()
    for c in data.contact[: data.ncon]:
        if geom_id in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == geom_id else c.geom1
            if other in jaw_ids:
                touched.add(model.geom(other).name.split("_jaw")[0])
    return {"fixed", "moving"} <= touched


def touching(data, group_a, group_b):
    """True if any current contact joins a geom in `group_a` to a geom in `group_b`."""
    for c in data.contact[: data.ncon]:
        if (c.geom1 in group_a and c.geom2 in group_b) or (c.geom2 in group_a and c.geom1 in group_b):
            return True
    return False
```

- [ ] **Step 4: Point `BinPickEnv` at it**

In `so101sim/binpick.py`, change the import line to `from so101sim import SCENE_PATH, grasp, primitives`. In `__init__`, replace:

```python
        self._jaw_geoms = {self.model.geom(g).id for g in primitives.ik.JAW_TIP_GEOMS}
        for prefix in ("fixed_jaw", "moving_jaw"):
            self._jaw_geoms |= {
                self.model.geom(i).id for i in range(self.model.ngeom)
                if self.model.geom(i).name.startswith(prefix)
            }
```

with:

```python
        self._jaw_geoms = grasp.jaw_geom_ids(self.model)
```

Then replace the body of `is_grasped`:

```python
    def is_grasped(self, name):
        """True when both jaws are touching this cube."""
        return grasp.is_grasped(self.model, self.data, self._cube_geom[name], self._jaw_geoms)
```

- [ ] **Step 5: Run the new tests and the bin-pick suite**

Run: `python -m pytest tests/test_grasp.py tests/test_binpick.py -q`
Expected: all pass. `test_pick_then_place_puts_a_cube_in_the_bin` exercises `is_grasped` through `is_lifted`.

- [ ] **Step 6: Stage**

```bash
git add so101sim/grasp.py so101sim/binpick.py tests/test_grasp.py
```

---

### Task 3: Let `pick` grasp at any height, and make `bridge`'s low threshold follow `hover_z`

**Files:**
- Modify: `so101sim/primitives.py`, `tests/test_binpick.py`

**Interfaces:**
- Produces:
  - `primitives.pick(model, xy, yaw, hover_z=HOVER_Z, z=CUBE_HALF)`. `z` is the jaw midpoint's grasp height.
  - `primitives.BRIDGE_LOW_MARGIN = 0.025`. `bridge` ramps in joint space only when the gripper is tilted more than 20° or sits below `hover_z - BRIDGE_LOW_MARGIN`.

- [ ] **Step 1: Write the failing tests** by appending to the "primitives" section of `tests/test_binpick.py`

```python
def test_pick_grasps_at_the_requested_height(model):
    traj = primitives.pick(model, (0.22, -0.05), 0.0, hover_z=0.08, z=0.045)
    planned = traj.copy()
    planned[:, 5] = primitives.GRIP_PLAN  # IK solved the arm at GRIP_PLAN, so measure it there
    lowest = min(primitives.tool_pose(model, f)[0][2] for f in planned)
    assert abs(lowest - 0.045) < 2e-3


def test_pick_defaults_to_a_cube_on_the_table(model):
    a = primitives.pick(model, (0.22, -0.05), 0.3)
    b = primitives.pick(model, (0.22, -0.05), 0.3, z=primitives.CUBE_HALF)
    np.testing.assert_array_equal(a, b)


def test_bridge_crosses_in_tool_space_at_a_low_hover(model):
    """A sunken bin puts travel height near z=0.02. Bridge must still cross in a straight tool-space line
    there. A joint ramp between these two poses bows about 2 cm off the line."""
    hover = 0.02
    start = primitives.pick(model, (0.20, 0.10), np.pi / 2, hover_z=hover)[-1]
    target = primitives.pick(model, (0.20, -0.10), np.pi / 2, hover_z=hover)
    crossing = primitives.bridge(model, start, target, hover_z=hover)
    xs = [primitives.tool_pose(model, f)[0][0] for f in crossing[: len(crossing) - len(target)]]
    assert max(abs(x - 0.20) for x in xs) < 0.005
```

The yaw is π/2 so the jaws close along y. Gripper changes then shift the jaw midpoint along y, not x, and x stays a clean test of the path.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_binpick.py -q -k "requested_height or defaults_to or low_hover"`
Expected: FAIL. The first two raise `TypeError: pick() got an unexpected keyword argument 'z'`. The third fails its assertion, with the crossing bowed well past 5 mm.

- [ ] **Step 3: Change `pick`**

In `so101sim/primitives.py`:

```python
def pick(model, xy, yaw, hover_z=HOVER_Z, z=CUBE_HALF):
    """Grasp a cube at `xy` with its faces aligned to `yaw` and its centre at height `z`.

    The default `z` is a cube standing on the table. A cube in a pile sits higher, and a cube in a sunken
    bin sits lower. Hovers, descends with the jaws open, closes on the cube, lifts. The arm holds still
    while the jaws close, which is what the real servo does, so the moving jaw sweeps the cube against the
    fixed one.
    """
    x, y = float(xy[0]), float(xy[1])
    top = np.array([x, y, hover_z])
    grasp = np.array([x, y, z])
```

The rest of the function stays as it is.

- [ ] **Step 4: Change `bridge`**

Add next to `MAX_JOINT_STEP_DEG`:

```python
# bridge treats a gripper this far below the hover height as not yet hovering, and ramps in joint space
# instead. Relative to hover_z because a sunken bin puts travel height well under the table-level 0.055,
# and a fixed threshold there would send every crossing through the joint ramp and the pile.
BRIDGE_LOW_MARGIN = 0.025
```

In `bridge`, replace `if tilt > np.radians(20.0) or here[2] < 0.03:` with:

```python
    if tilt > np.radians(20.0) or here[2] < hover_z - BRIDGE_LOW_MARGIN:
```

At the default `hover_z=0.055` the threshold is still 0.03, so existing behavior doesn't change.

- [ ] **Step 5: Run the bin-pick suite**

Run: `python -m pytest tests/test_binpick.py -q`
Expected: all pass, including `test_bridge_keeps_the_gripper_high_when_crossing`.

- [ ] **Step 6: Stage**

```bash
git add so101sim/primitives.py tests/test_binpick.py
```

---

### Task 4: Generate the search scene from a `Layout`

**Files:**
- Create: `so101sim/search_layout.py`, `tests/test_search_layout.py`, and the generated `models/search_objects.xml` and `models/search_scene.xml`
- Modify: `so101sim/__init__.py`, `.gitignore`, `models/README.md`

**Interfaces:**
- Consumes: `env.HOME`, `primitives.CUBE_HALF`, `binpick.bin_geometry`, `binpick.cube_names`.
- Produces:
  - `so101sim.SEARCH_SCENE_PATH: Path`
  - `search_layout.Layout(depth=0.10, width=0.15, center_x=0.20, wall=0.08, cam_z=0.55, n_cubes=30, home=HOME)`, frozen. Properties: `floor_z`, `rim_z`, `travel_z`. Classmethod: `from_model(model)`.
  - `search_layout.travel_z(floor_z, rim_z) -> float`
  - `search_layout.objects_xml(layout, probes=False, notes=()) -> str`
  - `search_layout.scene_xml(objects_file) -> str`
  - `search_layout.write(layout, notes=())`, which writes both generated files
  - `search_layout.candidate_scene(layout, probes=False)`, a context manager yielding a scene `Path`
  - `search_layout.load(layout, probes=False) -> MjModel`
  - `search_layout.parse_notes(xml) -> list[str]` and `search_layout.read_notes(path=None) -> list[str]`
  - Constants: `LAYER = 0.03`, `PILE_LAYERS = 3`, `MARKER_PREFIX = "marker_"`, `DROP_CEILING = 0.13`, `MODELS_DIR`, `OBJECTS_PATH`
  - Geom names in the scene: `table`, `riser`, `bin_floor`, `bin_wall_{px,nx,py,ny}`, `cube_red`, `cube_gray_{i}`, plus `marker_{i}_{j}` and `drop_column` in probe scenes. The camera is `front`. Custom numerics are `search_layout`, holding `[depth, width, center_x, wall, cam_z, n_cubes]`, and `search_home`, holding six lerobot values.

- [ ] **Step 1: Write the failing tests** in `tests/test_search_layout.py`

```python
import numpy as np
import pytest

from so101sim import SEARCH_SCENE_PATH, search_layout as sl
from so101sim.binpick import bin_geometry, cube_names
from so101sim.env import SO101Env

import mujoco  # after so101sim, which sets MUJOCO_GL


def test_default_layout_compiles_and_reads_back():
    layout = sl.Layout()
    assert sl.Layout.from_model(sl.load(layout)) == layout


def test_bin_geometry_matches_the_layout():
    layout = sl.Layout(depth=0.08, width=0.12, center_x=0.21, wall=0.06)
    b = bin_geometry(sl.load(layout))
    np.testing.assert_allclose(b["center"], [0.21, 0.0], atol=1e-9)
    assert b["half"] == pytest.approx(0.06)
    assert b["floor_z"] == pytest.approx(layout.floor_z)
    assert b["rim_z"] == pytest.approx(layout.rim_z)


def test_table_and_riser_put_the_base_on_a_step():
    model = sl.load(sl.Layout(depth=0.10))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table, riser = model.geom("table").id, model.geom("riser").id
    assert data.geom_xpos[table][2] + model.geom_size[table][2] == pytest.approx(-0.10)
    assert data.geom_xpos[riser][2] + model.geom_size[riser][2] == pytest.approx(0.0)
    assert data.geom_xpos[riser][2] - model.geom_size[riser][2] == pytest.approx(-0.10)


def test_cubes_are_one_red_then_grays_parked_off_the_table():
    model = sl.load(sl.Layout(n_cubes=7))
    names = cube_names(model)
    assert names[0] == "cube_red" and len(names) == 7
    assert all(n.startswith("cube_gray_") for n in names[1:])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert all(data.body(n).xpos[0] > 1.0 for n in names)


def test_probe_scenes_add_markers_and_a_hidden_drop_column():
    model = sl.load(sl.Layout(), probes=True)
    markers = [g for g in range(model.ngeom) if model.geom(g).name.startswith(sl.MARKER_PREFIX)]
    assert len(markers) > 100
    assert all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in markers)
    assert model.geom_group[model.geom("drop_column").id] > 2  # the renderer draws groups 0 to 2
    plain = sl.load(sl.Layout())
    assert not any(plain.geom(g).name.startswith(sl.MARKER_PREFIX) for g in range(plain.ngeom))


def test_candidate_files_are_cleaned_up():
    before = set(sl.MODELS_DIR.glob(".candidate_*"))
    sl.load(sl.Layout())
    assert set(sl.MODELS_DIR.glob(".candidate_*")) == before


def test_notes_round_trip_and_reject_double_hyphens():
    xml = sl.objects_xml(sl.Layout(), notes=["first line", "second line"])
    assert sl.parse_notes(xml) == ["first line", "second line"]
    with pytest.raises(ValueError):
        sl.objects_xml(sl.Layout(), notes=["a -- b"])


def test_travel_height_clears_a_full_pile_and_the_rim():
    for wall in (0.06, 0.08, 0.10):
        layout = sl.Layout(depth=0.10, wall=wall)
        cube_bottom = layout.travel_z - 0.015
        assert cube_bottom >= layout.floor_z + 0.09 + 0.005 - 1e-9
        assert cube_bottom >= layout.rim_z + 0.015 - 1e-9


def test_committed_search_scene_loads_in_the_base_env():
    env = SO101Env(scene_path=SEARCH_SCENE_PATH, height=64, width=64)
    try:
        assert sl.Layout.from_model(env.model).n_cubes == len(cube_names(env.model))
        assert SEARCH_SCENE_PATH.read_text() == sl.scene_xml(sl.OBJECTS_PATH.name)
    finally:
        env.close()
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_search_layout.py -q`
Expected: FAIL, `ImportError: cannot import name 'SEARCH_SCENE_PATH'`.

- [ ] **Step 3: Add the path** at the end of `so101sim/__init__.py`

```python
SEARCH_SCENE_PATH = Path(__file__).resolve().parent.parent / "models" / "search_scene.xml"
```

- [ ] **Step 4: Write `so101sim/search_layout.py`**

```python
"""The mechanical-search scene, generated from a few numbers.

models/search_objects.xml and models/search_scene.xml are written by `write`, never by hand.
scripts/size_search_bin.py picks the layout and scripts/mechsearch_smoke.py picks the cube count. The XML
keeps the numbers in <custom> fields, so `Layout.from_model` reads them back out of any compiled model.

World z = 0 is the plane the arm's base stands on. The table top sits at z = -depth, a riser fills the gap
under the base, and the bin stands on the table. Every height the task uses is near or below the base,
where top-down reach is wide. The measurements behind that are in
docs/superpowers/specs/2026-09-25-mechanical-search-design.md.
"""

import contextlib
import dataclasses
import os
import tempfile
from pathlib import Path

import mujoco

from so101sim import SEARCH_SCENE_PATH
from so101sim.env import HOME
from so101sim.primitives import CUBE_HALF

MODELS_DIR = SEARCH_SCENE_PATH.parent
OBJECTS_PATH = MODELS_DIR / "search_objects.xml"

FLOOR_THICKNESS = 0.004
WALL_THICKNESS = 0.004
TABLE_HALF = (0.35, 0.45, 0.02)
TABLE_CENTER_X = 0.2
RISER_HALF = 0.06
RISER_X = 0.010  # under base_collision
CAM_X = 0.70
CAM_FOVY = 50

LAYER = 2 * CUBE_HALF
PILE_LAYERS = 3
CARRY_GAP = 0.005  # a carried cube's bottom over a full pile
RIM_GAP = 0.015  # a lifted cube's bottom over the rim, past the 0.01 success threshold

# Cubes wait on the ground plane, behind the front camera, until reset drops them into the bin.
PARK_X = 1.2
PARK_SPACING = 0.05
GROUND_Z = -0.75

# Probe scenes only. Markers are 6 mm tiles on the bin floor, 1 cm apart, that the camera checks count in a
# segmentation render. The drop column is an invisible box over the bin, as high as reset drops cubes from,
# that the start-pose check tests the arm against.
MARKER_PREFIX = "marker_"
MARKER_STEP = 0.01
MARKER_HALF = 0.003
DROP_CEILING = 0.13

NOTES_HEADER = "       Notes:"
NOTE_PREFIX = "       * "


def travel_z(floor_z, rim_z):
    """Jaw-midpoint height for every move across the bin.

    High enough that a carried cube's bottom clears a full three-layer pile by CARRY_GAP, and that a lifted
    cube's bottom clears the rim by RIM_GAP.
    """
    return max(floor_z + PILE_LAYERS * LAYER + CARRY_GAP + CUBE_HALF, rim_z + RIM_GAP + CUBE_HALF)


@dataclasses.dataclass(frozen=True)
class Layout:
    depth: float = 0.10  # riser height, and how far the table sits below the base
    width: float = 0.15  # bin inside width
    center_x: float = 0.20
    wall: float = 0.08  # wall height above the floor top
    cam_z: float = 0.55  # front camera height
    n_cubes: int = 30  # one red, the rest gray
    home: tuple = tuple(float(v) for v in HOME)  # start pose, lerobot units

    @property
    def floor_z(self):
        return -self.depth + FLOOR_THICKNESS

    @property
    def rim_z(self):
        return self.floor_z + self.wall

    @property
    def travel_z(self):
        return travel_z(self.floor_z, self.rim_z)

    @classmethod
    def from_model(cls, model):
        d = model.numeric("search_layout").data
        return cls(depth=float(d[0]), width=float(d[1]), center_x=float(d[2]), wall=float(d[3]),
                   cam_z=float(d[4]), n_cubes=int(round(d[5])),
                   home=tuple(float(v) for v in model.numeric("search_home").data))


def _f(*values):
    return " ".join(f"{float(v):.6g}" for v in values)


def objects_xml(layout, probes=False, notes=()):
    """The objects file for `layout`. `probes` adds the floor markers and the drop column."""
    for note in notes:
        if "--" in note:
            raise ValueError(f"an XML comment can't hold '--': {note!r}")
    L = layout
    half_in = L.width / 2
    half_out = half_in + WALL_THICKNESS
    t = WALL_THICKNESS / 2
    wall_z = FLOOR_THICKNESS + L.wall / 2
    out = [
        "<mujoco>",
        "  <!-- Generated by so101sim/search_layout.py, never edited by hand. scripts/size_search_bin.py picks",
        "       the layout and scripts/mechsearch_smoke.py picks the cube count, each with its write flag.",
        NOTES_HEADER,
        *[NOTE_PREFIX + note for note in notes],
        "  -->",
        "  <custom>",
        f'    <numeric name="search_layout" data="{_f(L.depth, L.width, L.center_x, L.wall, L.cam_z, L.n_cubes)}"/>',
        f'    <numeric name="search_home" data="{_f(*L.home)}"/>',
        "  </custom>",
        "  <worldbody>",
        f'    <geom name="table" type="box" size="{_f(*TABLE_HALF)}" '
        f'pos="{_f(TABLE_CENTER_X, 0, -L.depth - TABLE_HALF[2])}" material="table"/>',
        f'    <geom name="riser" type="box" size="{_f(RISER_HALF, RISER_HALF, L.depth / 2)}" '
        f'pos="{_f(RISER_X, 0, -L.depth / 2)}" rgba="0.45 0.35 0.25 1"/>',
        f'    <body name="front_cam_target" pos="{_f(L.center_x, 0, L.floor_z)}"/>',
        f'    <camera name="front" mode="targetbody" target="front_cam_target" pos="{_f(CAM_X, 0, L.cam_z)}" '
        f'fovy="{CAM_FOVY}"/>',
        f'    <body name="bin" pos="{_f(L.center_x, 0, -L.depth)}">',
        f'      <geom name="bin_floor" class="bin_wall" size="{_f(half_out, half_out, FLOOR_THICKNESS / 2)}" '
        f'pos="{_f(0, 0, FLOOR_THICKNESS / 2)}"/>',
        f'      <geom name="bin_wall_px" class="bin_wall" size="{_f(t, half_out, L.wall / 2)}" '
        f'pos="{_f(half_in + t, 0, wall_z)}"/>',
        f'      <geom name="bin_wall_nx" class="bin_wall" size="{_f(t, half_out, L.wall / 2)}" '
        f'pos="{_f(-half_in - t, 0, wall_z)}"/>',
        f'      <geom name="bin_wall_py" class="bin_wall" size="{_f(half_out, t, L.wall / 2)}" '
        f'pos="{_f(0, half_in + t, wall_z)}"/>',
        f'      <geom name="bin_wall_ny" class="bin_wall" size="{_f(half_out, t, L.wall / 2)}" '
        f'pos="{_f(0, -half_in - t, wall_z)}"/>',
    ]
    if probes:
        n = int((half_in - 0.005) / MARKER_STEP + 1e-9)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                out.append(
                    f'      <geom name="{MARKER_PREFIX}{i + n}_{j + n}" type="box" '
                    f'size="{_f(MARKER_HALF, MARKER_HALF, 0.0005)}" '
                    f'pos="{_f(i * MARKER_STEP, j * MARKER_STEP, FLOOR_THICKNESS + 0.0005)}" '
                    f'contype="0" conaffinity="0" rgba="0.9 0.9 0.2 1"/>')
        out.append(f'      <geom name="drop_column" type="box" group="5" rgba="0 0 0 0" '
                   f'size="{_f(half_in, half_in, DROP_CEILING / 2)}" '
                   f'pos="{_f(0, 0, FLOOR_THICKNESS + DROP_CEILING / 2)}"/>')
    out.append("    </body>")
    names = ["cube_red"] + [f"cube_gray_{i}" for i in range(L.n_cubes - 1)]
    for i, name in enumerate(names):
        rgba = "0.85 0.15 0.15 1" if name == "cube_red" else "0.55 0.55 0.55 1"
        y = (i - (len(names) - 1) / 2) * PARK_SPACING
        out += [
            f'    <body name="{name}" pos="{_f(PARK_X, y, GROUND_Z + CUBE_HALF)}">',
            f'      <freejoint name="{name}"/>',
            f'      <geom name="{name}" class="cube" rgba="{rgba}"/>',
            "    </body>",
        ]
    out += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(out)


def scene_xml(objects_file):
    return ('<mujoco model="so101_search">\n'
            "  <!-- Generated by so101sim/search_layout.py: the shared stage, then the search objects. -->\n"
            '  <include file="stage.xml"/>\n'
            f'  <include file="{objects_file}"/>\n'
            "</mujoco>\n")


def write(layout, notes=()):
    """Rewrite the committed search scene for `layout`."""
    OBJECTS_PATH.write_text(objects_xml(layout, notes=notes))
    SEARCH_SCENE_PATH.write_text(scene_xml(OBJECTS_PATH.name))


@contextlib.contextmanager
def candidate_scene(layout, probes=False):
    """Path to a compilable scene for `layout`, deleted on exit.

    The files go in models/ because MuJoCo resolves `meshdir` against the top-level file, so a scene
    anywhere else can't find the arm's meshes. .gitignore covers the names in case a crash leaves them.
    """
    fd, objects = tempfile.mkstemp(prefix=".candidate_", suffix="_objects.xml", dir=MODELS_DIR)
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

- [ ] **Step 5: Ignore leftover candidate files**

Append to `.gitignore`:

```
models/.candidate_*
```

- [ ] **Step 6: Write the provisional scene**

This is the spec's starting point, a 10 cm riser, a 15 cm bin and 30 cubes. Task 5 replaces it.

Run:
```bash
python -c "import so101sim; from so101sim import search_layout as sl; sl.write(sl.Layout(), ['Provisional starting layout from the spec. scripts/size_search_bin.py replaces it.'])"
```
Expected: `models/search_objects.xml` and `models/search_scene.xml` exist.

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_search_layout.py -q`
Expected: all pass.

- [ ] **Step 8: Mention the search scene in `models/README.md`**

Append to the list that ends with the `scene.xml` bullet from Task 1:

```markdown
- `search_scene.xml` is the mechanical-search scene: a table 10 cm or so below the base, a riser under
  the base, a deep bin and a pile of cubes. It and `search_objects.xml` are generated by
  `so101sim/search_layout.py`. Don't edit them by hand. `scripts/size_search_bin.py --write` rewrites
  them.
```

- [ ] **Step 9: Stage**

```bash
git add so101sim/__init__.py so101sim/search_layout.py tests/test_search_layout.py .gitignore \
        models/search_objects.xml models/search_scene.xml models/README.md
```

---

### Task 5: Size the bin, riser, walls, start pose and front camera

> Amended 2026-09-25. The first sizing run found no layout, which is a spec stop point. dm1487 chose walls of at most 6 cm and a clearance check that skips any cube within 2 cm of a wall, not only corners. The spec's amendment under "Sizing rule" has the numbers. The code below is the amended version: `near_wall` replaces `in_corner`, and `WALLS` is `(0.06, 0.05, 0.04)`.

**Files:**
- Create: `so101sim/search_checks.py`, `scripts/size_search_bin.py`, `tests/test_search_checks.py`
- Regenerate: `models/search_objects.xml`, `models/search_scene.xml`

**Interfaces:**
- Consumes: everything from Task 4, `grasp.arm_geom_ids`, `grasp.touching`, `ik.solve_topdown`, `primitives._grip_rad`, `primitives.GRIP_PLAN`, `primitives.GRIP_HOLDING`, `primitives.GRIP_OPEN`.
- Produces:
  - `search_checks.GRID = 0.005`, `YAWS`, `WALL_STRIP = 0.02`, `FRONT_MIN = 0.70`, `WRIST_MIN = 0.50`, `IMAGE_SIZE = (480, 640)`
  - `grid_points(layout, step=GRID) -> list[tuple[float, float]]`
  - `near_wall(layout, xy, strip=WALL_STRIP) -> bool`, true when the cube's face is within `strip` of any wall
  - `reachable(model, xyz, grip) -> bool`, with `grip` in lerobot units
  - `grasp_poses(model, xyz) -> list[np.ndarray]`
  - `scene_geom_ids(model) -> frozenset[int]`
  - `clearance_ok(model, data, poses, arm_ids, scene_ids) -> bool`
  - `drop_column_clear(model, data, home) -> bool`
  - `marker_coverage(model, data, renderer, camera, home) -> float`
  - A committed layout whose notes start with `Chosen`.

- [ ] **Step 1: Write the failing tests** in `tests/test_search_checks.py`

```python
import numpy as np
import pytest

from so101sim import SCENE_PATH, SEARCH_SCENE_PATH, grasp, primitives, search_checks as sc, search_layout as sl

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def arm_model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


@pytest.fixture(scope="module")
def committed():
    return sl.Layout.from_model(mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH)))


def test_grid_covers_every_cube_centre_that_fits():
    pts = np.array(sc.grid_points(sl.Layout(width=0.12, center_x=0.20)))
    assert len(pts) == 19 * 19
    np.testing.assert_allclose([pts[:, 0].min(), pts[:, 0].max()], [0.155, 0.245], atol=1e-9)
    np.testing.assert_allclose([pts[:, 1].min(), pts[:, 1].max()], [-0.045, 0.045], atol=1e-9)


def test_wall_strip_covers_cubes_near_any_wall_not_just_corners():
    layout = sl.Layout(width=0.12, center_x=0.20)  # cube centres run 0.155 to 0.245 in x, -0.045 to 0.045 in y
    assert sc.near_wall(layout, (0.245, 0.045))  # corner, faces flush
    assert sc.near_wall(layout, (0.245, 0.0))  # flush against the far wall, mid-edge
    assert sc.near_wall(layout, (0.20, -0.025))  # face exactly 2 cm from a side wall counts
    assert not sc.near_wall(layout, (0.20, -0.02))  # face 2.5 cm away
    assert not sc.near_wall(layout, (0.20, 0.0))


def test_reach_check_agrees_with_the_measured_band(arm_model):
    assert sc.reachable(arm_model, (0.20, 0.0, -0.05), primitives.GRIP_PLAN)
    assert not sc.reachable(arm_model, (0.20, 0.0, 0.12), primitives.GRIP_PLAN)


def test_clearance_catches_jaws_driven_into_the_floor():
    layout = sl.Layout()
    model = sl.load(layout)
    data = mujoco.MjData(model)
    arm, scene = grasp.arm_geom_ids(model), sc.scene_geom_ids(model)
    below = sc.grasp_poses(model, (layout.center_x, 0.0, layout.floor_z - 0.01))
    assert below, "IK reaches 1 cm under the floor; the measured band runs down to z = -0.20"
    assert not sc.clearance_ok(model, data, below, arm, scene)
    above = sc.grasp_poses(model, (layout.center_x, 0.0, layout.floor_z + 0.05))
    assert sc.clearance_ok(model, data, above, arm, scene)


def test_committed_layout_came_from_the_sizing_script():
    assert any(n.startswith("Chosen") for n in sl.read_notes())


def test_committed_layout_reaches_its_grid_corners(arm_model, committed):
    reach = committed.width / 2 - primitives.CUBE_HALF
    heights = ((committed.floor_z + primitives.CUBE_HALF, primitives.GRIP_PLAN),
               (committed.floor_z + 5 * primitives.CUBE_HALF, primitives.GRIP_PLAN),
               (committed.travel_z, primitives.GRIP_HOLDING))
    for dx in (-reach, reach):
        for dy in (-reach, reach):
            for z, grip in heights:
                assert sc.reachable(arm_model, (committed.center_x + dx, dy, z), grip), (dx, dy, z)


def test_committed_start_pose_stays_out_of_the_drop_column(committed):
    model = sl.load(committed, probes=True)
    assert sc.drop_column_clear(model, mujoco.MjData(model), committed.home)


def test_committed_cameras_see_enough_of_the_floor(committed):
    model = sl.load(committed, probes=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *sc.IMAGE_SIZE)
    try:
        assert sc.marker_coverage(model, data, renderer, "front", committed.home) >= sc.FRONT_MIN
        assert sc.marker_coverage(model, data, renderer, "wrist_cam", committed.home) >= sc.WRIST_MIN
    finally:
        renderer.close()
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_search_checks.py -q`
Expected: FAIL, `ImportError: cannot import name 'search_checks'`.

- [ ] **Step 3: Write `so101sim/search_checks.py`**

```python
"""The checks that choose the mechanical-search layout.

scripts/size_search_bin.py runs all of them to pick a layout. tests/test_search_checks.py reruns the cheap
ones against the committed scene. The rules and thresholds are the spec's sizing rule and its cameras and
start pose section: docs/superpowers/specs/2026-09-25-mechanical-search-design.md.
"""

import mujoco
import numpy as np

from so101sim import grasp, ik, primitives, units
from so101sim.search_layout import MARKER_PREFIX

GRID = 0.005
YAWS = tuple(np.radians([0.0, 45.0, 90.0, 135.0]))
# A cube whose face is this close to a wall needs a push before a straight-down grasp fits: the gripper is
# wider than the cube. Measured in the spec's amendment.
WALL_STRIP = 0.02
FRONT_MIN = 0.70
WRIST_MIN = 0.50
IMAGE_SIZE = (480, 640)  # height, width: the env's default observation size
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def grid_points(layout, step=GRID):
    """Cube-centre positions on a `step` grid, out to where a cube touches the walls."""
    reach = layout.width / 2 - primitives.CUBE_HALF
    n = int(np.floor(reach / step + 1e-9))
    offsets = np.arange(-n, n + 1) * step
    return [(round(layout.center_x + dx, 4), round(float(dy), 4)) for dx in offsets for dy in offsets]


def near_wall(layout, xy, strip=WALL_STRIP):
    """True when the cube's face is within `strip` of any wall, the edge included."""
    face_x = layout.width / 2 - abs(xy[0] - layout.center_x) - primitives.CUBE_HALF
    face_y = layout.width / 2 - abs(xy[1]) - primitives.CUBE_HALF
    return min(face_x, face_y) <= strip + 1e-9


def _ok(info):
    return info["pos_err"] < 1e-3 and info["tilt"] < 1.0


def reachable(model, xyz, grip):
    """True if top-down IK puts the jaw midpoint within 1 mm of `xyz`, pointing down to 1 degree, for at
    least one of YAWS. `grip` is the gripper value in lerobot units that IK plans against."""
    grip_rad = primitives._grip_rad(model, grip)
    return any(_ok(ik.solve_topdown(model, xyz, yaw=yaw, gripper=grip_rad, restarts=2)[1]) for yaw in YAWS)


def grasp_poses(model, xyz):
    """Arm angles in radians, one per yaw that IK reaches at `xyz`, planned at GRIP_PLAN like `pick`."""
    grip_rad = primitives._grip_rad(model, primitives.GRIP_PLAN)
    poses = []
    for yaw in YAWS:
        q, info = ik.solve_topdown(model, xyz, yaw=yaw, gripper=grip_rad, restarts=2)
        if _ok(info):
            poses.append(q)
    return poses


def scene_geom_ids(model):
    """The bin, the riser and the table: everything fixed the arm could hit. Probe geoms excluded."""
    bin_id = model.body("bin").id
    ids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bin_id
           and not model.geom(g).name.startswith(MARKER_PREFIX) and model.geom(g).name != "drop_column"}
    return frozenset(ids | {model.geom("riser").id, model.geom("table").id})


def _joint_adr(model):
    return model.jnt_qposadr[[model.joint(n).id for n in units.JOINT_NAMES]]


def set_arm(model, data, arm_rad, gripper):
    """Five arm joints in radians plus the gripper in lerobot units, then forward, which finds contacts."""
    adr = _joint_adr(model)
    data.qpos[adr[:5]] = arm_rad
    data.qpos[adr[5]] = primitives._grip_rad(model, gripper)
    mujoco.mj_forward(model, data)


def set_pose(model, data, pose):
    """All six joints from lerobot units, then forward."""
    grip_range = tuple(model.jnt_range[model.joint("gripper").id])
    data.qpos[_joint_adr(model)] = units.to_rad(np.asarray(pose, dtype=np.float64), grip_range)
    mujoco.mj_forward(model, data)


def clearance_ok(model, data, poses, arm_ids, scene_ids):
    """True if at least one of `poses`, with the jaws opened to GRIP_OPEN, touches nothing in the scene."""
    for q in poses:
        set_arm(model, data, q, primitives.GRIP_OPEN)
        if not grasp.touching(data, arm_ids, scene_ids):
            return True
    return False


def drop_column_clear(model, data, home):
    """True if no arm geom touches the probe scene's drop column with the arm at `home`."""
    set_pose(model, data, home)
    return not grasp.touching(data, grasp.arm_geom_ids(model), {model.geom("drop_column").id})


def marker_coverage(model, data, renderer, camera, home):
    """Share of the probe scene's floor markers that `camera` sees, with the arm at `home`."""
    set_pose(model, data, home)
    markers = {g for g in range(model.ngeom) if model.geom(g).name.startswith(MARKER_PREFIX)}
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera=camera)
        seg = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    seen = set(np.unique(seg[..., 0][seg[..., 1] == GEOM]).tolist())
    return len(markers & seen) / len(markers)
```

- [ ] **Step 4: Run the unit tests for the checks**

Run: `python -m pytest tests/test_search_checks.py -q -k "grid_covers or wall_strip or reach_check or clearance"`
Expected: 4 pass. The four committed-layout tests aren't selected yet.

- [ ] **Step 5: Write `scripts/size_search_bin.py`**

```python
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
```

- [ ] **Step 6: Run the sizing**

Run: `nvidia-smi` first, and pick a free card for `N`. Then:
```bash
MUJOCO_EGL_DEVICE_ID=N python scripts/size_search_bin.py --write 2>&1 | tee $SCRATCH/sizing.log
```
Expected: one line per candidate tried, then the start-pose and camera lines, then the notes, then `wrote .../search_objects.xml`. This takes several minutes. If it prints `STOP:`, stop here and report the log to dm1487. That's a spec stop point.

- [ ] **Step 7: Run all the check tests**

Run: `python -m pytest tests/test_search_checks.py tests/test_search_layout.py -q`
Expected: all pass. If `test_committed_cameras_see_enough_of_the_floor` fails, the script and the test disagree about coverage. Find out why before going on.

- [ ] **Step 8: Stage**

```bash
git add so101sim/search_checks.py scripts/size_search_bin.py tests/test_search_checks.py \
        models/search_objects.xml models/search_scene.xml
```

---

### Task 6: `MechSearchEnv` reset, building the pile

> Amended 2026-09-25 by the spec's second amendment. Buried starts box the red cube in: four grays flush against its sides and one on top, placed as one wave. `target_pixels` counts face pixels through `face_pixels`, a 5x5 erosion, so seams between touching cubes don't count. Two tests were added: `test_face_pixels_drop_seams_and_keep_faces` and `test_a_buried_red_cube_sits_on_the_floor_boxed_in`. `BURY_JITTER` is gone. `BOX_GAP`, `BOX_STEP`, `BOX_INSET` and `SEAM_RADIUS` are new. The code in `so101sim/mechsearch.py` is the reference where it differs from the blocks below.

**Files:**
- Create: `so101sim/mechsearch.py`, `tests/test_mechsearch.py`

**Interfaces:**
- Consumes: `SO101Env` internals `_qpos_adr`, `_gripper_range`, `_n_substeps`, `_renderer`, `_obs()`, `joint_state`. Also `grasp.jaw_geom_ids`, `grasp.arm_geom_ids`, `grasp.touching`, `binpick.bin_geometry`, `binpick.cube_names`, `search_layout.Layout`, `LAYER`, `PILE_LAYERS`, and `env.CAMERAS`, `env.CONTROL_HZ`.
- Produces:
  - `mechsearch.TARGET = "cube_red"`, `TASK = "retrieve the red cube from the bin"`, `VISIBLE_MIN_PIXELS = 100`, `MAX_RESET_ATTEMPTS = 50`
  - `MechSearchEnv(scene_path=SEARCH_SCENE_PATH, max_steps=1800, buried_prob=0.5, **SO101Env kwargs)`, with attributes `layout`, `bin`, `cubes`, `grays`, `max_steps`, `buried_prob`
  - `reset(seed=None, options=None) -> (obs, info)`, where `options` may hold `{"buried": bool}`
  - `info` keys: `target` (`pos`, `quat`), `cubes` (per-cube `pos`, `yaw`), `buried`, `target_pixels` (`front`, `wrist`), `reset_attempts`, `n_in_bin`, `is_success`, `failure`, `steps`, `task`
  - Methods: `cube_pose(name) -> {"pos", "yaw"}`, `settled() -> bool`, `free_spot() -> np.ndarray | None`, `target_pixels() -> dict`, `task_info() -> dict`, `close()`
  - Task 7 adds `step`, `advance` and `_target_retrieved`. Until then `step` is `SO101Env`'s.

- [ ] **Step 1: Write the failing tests** in `tests/test_mechsearch.py`

```python
import numpy as np
import pytest

from so101sim import primitives
from so101sim.mechsearch import TARGET, TASK, VISIBLE_MIN_PIXELS, MechSearchEnv, MechSearchPrimitiveEnv

import mujoco  # after so101sim, which sets MUJOCO_GL


@pytest.fixture(scope="module")
def env():
    e = MechSearchEnv(height=64, width=64)
    yield e
    e.close()


def positions(env):
    return {n: env.cube_pose(n)["pos"] for n in env.cubes}


def move_cube(env, name, xy):
    adr = env.model.jnt_qposadr[env.model.body(name).jntadr[0]]
    env.data.qpos[adr:adr + 2] = xy
    env.data.qpos[adr + 2] = env.bin["floor_z"] + primitives.CUBE_HALF
    mujoco.mj_forward(env.model, env.data)


def outside_the_bin(env):
    return env.bin["center"] + np.array([0.0, env.bin["half"] + 0.05])


# ---- reset --------------------------------------------------------------------------------------


def test_reset_is_deterministic(env):
    env.reset(seed=3)
    a = positions(env)
    env.reset(seed=3)
    b = positions(env)
    for name in env.cubes:
        np.testing.assert_allclose(a[name], b[name], atol=1e-9)


def test_buried_reset_hides_the_red_cube_from_both_cameras(env):
    _, info = env.reset(seed=0, options={"buried": True})
    assert info["buried"] is True
    assert info["target_pixels"] == {"front": 0, "wrist": 0}


def test_visible_reset_shows_the_red_cube(env):
    _, info = env.reset(seed=0, options={"buried": False})
    assert info["buried"] is False
    assert max(info["target_pixels"].values()) >= VISIBLE_MIN_PIXELS


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reset_leaves_a_settled_pile_of_at_most_three_layers_with_room(env, seed):
    env.reset(seed=seed)
    ceiling = env.bin["floor_z"] + 5 * primitives.CUBE_HALF + 0.01
    for name in env.cubes:
        p = env.cube_pose(name)["pos"]
        assert np.abs(p[:2] - env.bin["center"]).max() < env.bin["half"], name
        assert p[2] <= ceiling, name
    assert env.settled()
    assert env.free_spot() is not None


def test_reset_needs_no_seed_or_options(env):
    obs, info = env.reset()
    assert info["n_in_bin"] == len(env.cubes)
    assert obs["observation.state"].shape == (6,)


def test_reset_starts_at_the_layout_home(env):
    env.reset(seed=0)
    np.testing.assert_allclose(env.joint_state, env.layout.home, atol=1.0)


def test_buried_label_does_not_depend_on_the_observation_size(env):
    big = MechSearchEnv()  # 640 x 480, the default
    try:
        _, small = env.reset(seed=4)
        _, large = big.reset(seed=4)
        assert small["buried"] == large["buried"]
        assert small["target_pixels"] == large["target_pixels"]
    finally:
        big.close()
```

`MechSearchPrimitiveEnv` and `TASK` get imported now so later tasks only append tests. Until Task 9 defines the class, add a one-line stub at the bottom of `mechsearch.py`: `class MechSearchPrimitiveEnv: pass`. Task 9 replaces it.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_mechsearch.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'so101sim.mechsearch'`.

- [ ] **Step 3: Write `so101sim/mechsearch.py`, with reset only**

```python
"""Mechanical search: find the one red cube in a bin of gray ones and lift it out.

Spec: docs/superpowers/specs/2026-09-25-mechanical-search-design.md.

`MechSearchEnv` is the 30 Hz env. Actions and observations are `SO101Env`'s, so a lerobot policy runs on
it unchanged. `MechSearchPrimitiveEnv` wraps it so one step is a whole pick, place or push, with the 30 Hz
frames still happening underneath and every task check running on each one.

Two rules shape this file. Every cube stays in the bin: once any cube's centre leaves the bin's footprint
the episode ends as a failure, so all rearranging happens inside the bin. And the observation holds images
and joint angles only. Cube poses, the buried label and the red pixel counts go in `info`, because a
policy on hardware gets none of them.
"""

import gymnasium as gym
import mujoco
import numpy as np

from so101sim import SEARCH_SCENE_PATH, grasp, primitives, search_layout as sl, units
from so101sim.binpick import bin_geometry, cube_names
from so101sim.env import CAMERAS, CONTROL_HZ, SO101Env

TARGET = "cube_red"
TASK = "retrieve the red cube from the bin"
CUBE_HALF = primitives.CUBE_HALF

# Reset drops cubes in waves of up to WAVE, at least WAVE_SPACING apart so none land on each other in
# mid-air, with each wave's bottoms DROP_GAP over the highest cube already in the bin.
WAVE = 4
WAVE_SPACING = 0.045
DROP_GAP = 0.01
DROP_INSET = CUBE_HALF + 0.003
# The buried red cube and the free floor spot keep this far from every wall, centre to wall.
WALL_INSET = 0.018
BURY_JITTER = 0.005
# Two cubes' half-diagonals plus 2.6 mm, so a cube fits at the spot whatever its neighbours' yaw.
FREE_SPOT_CLEARANCE = 0.045
FREE_SPOT_GRID = 0.005
SETTLE_QVEL = 0.2
SETTLE_SECONDS = 1.0
MAX_RESET_ATTEMPTS = 50

# Visibility is counted at the default observation size whatever size the env renders, so asking for
# small images doesn't change which piles count as buried.
VISIBILITY_SIZE = (480, 640)
VISIBLE_MIN_PIXELS = 100
GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

SUCCESS_CLEARANCE = 0.01
SUCCESS_FRAMES = 5


class MechSearchEnv(SO101Env):
    def __init__(self, scene_path=SEARCH_SCENE_PATH, max_steps=1800, buried_prob=0.5, **env_kwargs):
        super().__init__(scene_path=scene_path, **env_kwargs)
        self.max_steps = max_steps
        self.buried_prob = float(buried_prob)
        self.layout = sl.Layout.from_model(self.model)
        self.bin = bin_geometry(self.model)
        self.cubes = cube_names(self.model)
        self.grays = tuple(n for n in self.cubes if n != TARGET)
        self._qadr = {n: self.model.jnt_qposadr[self.model.body(n).jntadr[0]] for n in self.cubes}
        self._dofadr = {n: self.model.jnt_dofadr[self.model.body(n).jntadr[0]] for n in self.cubes}
        self._body = {n: self.model.body(n).id for n in self.cubes}
        self._geom = {n: self.model.geom(n).id for n in self.cubes}
        self._cube_geoms = frozenset(self._geom.values())
        self._jaw_ids = grasp.jaw_geom_ids(self.model)
        self._arm_ids = grasp.arm_geom_ids(self.model)
        self._home = np.asarray(self.layout.home, dtype=np.float64)
        self._seg = mujoco.Renderer(self.model, *VISIBILITY_SIZE)
        self._seg.enable_segmentation_rendering()
        self._dropped = []
        self._steps = 0
        self._streak = 0
        self._reset_info = {}

    # ---- gym api -------------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Seed np_random only. SO101Env.reset would also render an observation nobody uses.
        gym.Env.reset(self, seed=seed)
        options = options or {}
        buried = options.get("buried")
        if buried is None:
            buried = bool(self.np_random.random() < self.buried_prob)
        reason = None
        for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
            self._start_pose()
            reason, pixels = self._build_pile(bool(buried))
            if reason is None:
                break
        else:
            raise RuntimeError(f"no usable pile in {MAX_RESET_ATTEMPTS} attempts, the last because {reason}")
        self._steps, self._streak = 0, 0
        self._reset_info = {"buried": bool(buried), "target_pixels": pixels, "reset_attempts": attempt}
        return self._obs(), self._info(False, None)

    def close(self):
        self._seg.close()
        super().close()

    # ---- task state ----------------------------------------------------------------------------

    def cube_pose(self, name):
        """Centre position, and yaw of the cube's x axis about world z."""
        adr = self._qadr[name]
        mat = self.data.xmat[self._body[name]].reshape(3, 3)
        return {"pos": self.data.qpos[adr:adr + 3].copy(), "yaw": float(np.arctan2(mat[1, 0], mat[0, 0]))}

    def settled(self):
        return bool(np.abs(self.data.qvel).max() <= SETTLE_QVEL)

    def free_spot(self):
        """The first grid point on the floor where a cube fits, or None.

        Conservative: WALL_INSET from every wall and FREE_SPOT_CLEARANCE from the centre of every
        bottom-layer cube. A cube on a higher layer can still overhang it.
        """
        reach = self.bin["half"] - WALL_INSET
        n = int(np.floor(reach / FREE_SPOT_GRID + 1e-9))
        offsets = np.arange(-n, n + 1) * FREE_SPOT_GRID
        grid = self.bin["center"] + np.stack(np.meshgrid(offsets, offsets, indexing="ij"), -1).reshape(-1, 2)
        layer_top = self.bin["floor_z"] + sl.LAYER
        bottom = np.array([self.data.qpos[self._qadr[c]:self._qadr[c] + 2] for c in self.cubes
                           if self.data.qpos[self._qadr[c] + 2] < layer_top]).reshape(-1, 2)
        if len(bottom) == 0:
            return grid[len(grid) // 2].copy()
        gaps = np.linalg.norm(grid[:, None, :] - bottom[None, :, :], axis=-1).min(axis=1)
        free = np.flatnonzero(gaps >= FREE_SPOT_CLEARANCE)
        return grid[free[0]].copy() if len(free) else None

    def target_pixels(self):
        """Pixels of the red cube in each camera, from a segmentation render at VISIBILITY_SIZE."""
        counts = {}
        for name, camera in CAMERAS.items():
            self._seg.update_scene(self.data, camera=camera)
            seg = self._seg.render()
            counts[name] = int(np.count_nonzero((seg[..., 0] == self._geom[TARGET]) & (seg[..., 1] == GEOM)))
        return counts

    def task_info(self):
        """`info` as it stands, for a primitive that was refused before any frame ran."""
        return self._info(False, None)

    # ---- internals -----------------------------------------------------------------------------

    def _start_pose(self):
        mujoco.mj_resetData(self.model, self.data)
        q = units.to_rad(self._home, self._gripper_range)
        self.data.qpos[self._qpos_adr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _outside(self, name):
        adr = self._qadr[name]
        return bool(np.abs(self.data.qpos[adr:adr + 2] - self.bin["center"]).max() > self.bin["half"])

    def _sample_xy(self, taken, inset, tries=200):
        """A random xy inside the bin, `inset` from the walls and WAVE_SPACING from every point in `taken`."""
        reach = self.bin["half"] - inset
        for _ in range(tries):
            xy = self.bin["center"] + self.np_random.uniform(-reach, reach, 2)
            if all(np.linalg.norm(xy - t) >= WAVE_SPACING for t in taken):
                return xy
        return None

    def _place(self, name, xy, z):
        yaw = self.np_random.uniform(-np.pi, np.pi)
        adr, dof = self._qadr[name], self._dofadr[name]
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[dof:dof + 6] = 0.0

    def _drop_z(self):
        """Centre height for the next wave: bottoms DROP_GAP over the highest cube in the bin so far."""
        tops = [self.data.qpos[self._qadr[n] + 2] + CUBE_HALF for n in self._dropped]
        return max(tops, default=self.bin["floor_z"]) + DROP_GAP + CUBE_HALF

    def _settle(self):
        """Step until still or SETTLE_SECONDS pass. Returns True if a cube touched the arm on the way."""
        touched = False
        for _ in range(int(round(SETTLE_SECONDS * CONTROL_HZ))):
            mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
            touched = touched or grasp.touching(self.data, self._cube_geoms, self._arm_ids)
            if self.settled():
                break
        return touched

    def _build_pile(self, buried):
        """Drop every cube into the bin in waves. Returns (why the pile is unusable or None, red pixels)."""
        grays = list(self.grays)
        self.np_random.shuffle(grays)
        self._dropped = []
        fixed = []
        if buried:
            red_xy = self._sample_xy([], WALL_INSET)
            self._place(TARGET, red_xy, self.bin["floor_z"] + CUBE_HALF)
            self._dropped.append(TARGET)
            fixed = [(grays.pop(0), red_xy + self.np_random.uniform(-BURY_JITTER, BURY_JITTER, 2))]
            queue = grays
        else:
            queue = grays + [TARGET]
        touched = False
        while fixed or queue:
            names, queue = queue[:WAVE - len(fixed)], queue[WAVE - len(fixed):]
            wave, spots = list(fixed), [xy for _, xy in fixed]
            fixed = []
            for name in names:
                xy = self._sample_xy(spots, DROP_INSET)
                if xy is None:
                    return "no room left to drop a wave", None
                spots.append(xy)
                wave.append((name, xy))
            z = self._drop_z()
            for name, xy in wave:
                self._place(name, xy, z)
                self._dropped.append(name)
            touched = self._settle() or touched
        mujoco.mj_forward(self.model, self.data)
        return self._rejection(buried, touched)

    def _rejection(self, buried, touched):
        if touched:
            return "a cube touched the arm while settling", None
        outside = [n for n in self.cubes if self._outside(n)]
        if outside:
            return f"{outside[0]} landed outside the bin", None
        ceiling = self.bin["floor_z"] + (sl.PILE_LAYERS - 1) * sl.LAYER + CUBE_HALF + 0.01
        high = [n for n in self.cubes if self.data.qpos[self._qadr[n] + 2] > ceiling]
        if high:
            return f"{high[0]} sits above the third layer", None
        if not self.settled():
            return "the pile was still moving after the last wave", None
        if self.free_spot() is None:
            return "no free spot left on the floor", None
        pixels = self.target_pixels()
        if buried and any(pixels.values()):
            return f"the buried red cube shows {pixels}", None
        if not buried and max(pixels.values()) < VISIBLE_MIN_PIXELS:
            return f"the red cube shows only {pixels}", None
        return None, pixels

    def _info(self, success, failure):
        adr = self._qadr[TARGET]
        return {
            "target": {"pos": self.data.qpos[adr:adr + 3].copy(), "quat": self.data.qpos[adr + 3:adr + 7].copy()},
            "cubes": {n: self.cube_pose(n) for n in self.cubes},
            **self._reset_info,
            "n_in_bin": sum(not self._outside(n) for n in self.cubes),
            "is_success": bool(success),
            "failure": failure,
            "steps": self._steps,
            "task": TASK,
        }


class MechSearchPrimitiveEnv:  # replaced in Task 9
    pass
```

- [ ] **Step 4: Run the tests**

Run: `MUJOCO_EGL_DEVICE_ID=N python -m pytest tests/test_mechsearch.py -q`
Expected: all pass. With 30 cubes each reset takes a second or more, so the file takes a minute or so. If `reset` raises `RuntimeError`, print the rejection reasons for seeds 0 to 4 before touching any constant. The spec's thresholds are fixed.

- [ ] **Step 5: Stage**

```bash
git add so101sim/mechsearch.py tests/test_mechsearch.py
```

---

### Task 7: `MechSearchEnv` step, with the success, failure and time-limit checks

**Files:**
- Modify: `so101sim/env.py`, `so101sim/mechsearch.py`, `tests/test_mechsearch.py`

**Interfaces:**
- Produces:
  - `SO101Env._apply(action)`, which clips the action and runs one control step of physics
  - `MechSearchEnv.advance(action) -> (reward, terminated, truncated, info)`, a frame with no rendering
  - `MechSearchEnv.step(action) -> (obs, reward, terminated, truncated, info)`
  - `MechSearchEnv._target_retrieved() -> bool`, which the success test stubs

- [ ] **Step 1: Write the failing tests** by appending to `tests/test_mechsearch.py`

```python
# ---- step ---------------------------------------------------------------------------------------


def test_holding_still_keeps_every_cube_in_the_bin(env):
    env.reset(seed=0)
    hold = env.joint_state.copy()
    for _ in range(15):
        _, reward, terminated, truncated, info = env.step(hold)
        assert not terminated and not truncated and reward == 0.0
    assert info["n_in_bin"] == len(env.cubes)
    assert info["failure"] is None and info["task"] == TASK


def test_a_cube_outside_the_bin_ends_the_episode(env):
    env.reset(seed=0)
    gray = env.grays[0]
    move_cube(env, gray, outside_the_bin(env))
    _, reward, terminated, _, info = env.step(env.joint_state)
    assert terminated and reward == 0.0
    assert gray in info["failure"]


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
    assert "target" in info and "cubes" in info
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_mechsearch.py -q -k "holding_still or outside_the_bin or five_frames or step_limit"`
Expected: FAIL. `SO101Env.step` never terminates and returns an empty `info`, so there's a `KeyError` on `n_in_bin`, and the other asserts fail.

- [ ] **Step 3: Split `SO101Env.step`**

In `so101sim/env.py`, replace `step` with:

```python
    def step(self, action):
        self._apply(action)
        return self._obs(), 0.0, False, False, {}

    def _apply(self, action):
        """Clip `action` to the joint ranges and run one 30 Hz control step of physics."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = units.to_rad(action, self._gripper_range)
        mujoco.mj_step(self.model, self.data, nstep=self._n_substeps)
```

- [ ] **Step 4: Add the step methods to `MechSearchEnv`**

Put these in the "gym api" section of `so101sim/mechsearch.py`, after `reset`:

```python
    def step(self, action):
        reward, terminated, truncated, info = self.advance(action)
        return self._obs(), reward, terminated, truncated, info

    def advance(self, action):
        """One 30 Hz frame of physics and task checks, without rendering.

        The primitive env calls this for every frame and renders once at the end. Checks run in the spec's
        order: a cube out of the bin, then success, then the time limit.
        """
        self._apply(action)
        self._steps += 1
        failure = self._left_bin()
        success = False
        if failure is None:
            self._streak = self._streak + 1 if self._target_retrieved() else 0
            success = self._streak >= SUCCESS_FRAMES
        terminated = success or failure is not None
        truncated = not terminated and self.max_steps is not None and self._steps >= self.max_steps
        return float(success), terminated, truncated, self._info(success, failure)
```

And these in "internals":

```python
    def _left_bin(self):
        for name in self.cubes:
            if self._outside(name):
                return f"cube left the bin: {name}"
        return None

    def _lowest_corner(self, name):
        """The z of a cube's lowest corner, whatever way up it sits."""
        mat = self.data.xmat[self._body[name]].reshape(3, 3)
        return float(self.data.qpos[self._qadr[name] + 2] - CUBE_HALF * np.abs(mat[2]).sum())

    def _target_retrieved(self):
        """Both jaws on the red cube, with its lowest corner SUCCESS_CLEARANCE over the rim."""
        if not grasp.is_grasped(self.model, self.data, self._geom[TARGET], self._jaw_ids):
            return False
        return self._lowest_corner(TARGET) >= self.bin["rim_z"] + SUCCESS_CLEARANCE
```

- [ ] **Step 5: Run the env tests and the base env tests**

Run: `python -m pytest tests/test_mechsearch.py tests/test_env.py tests/test_binpick.py -q`
Expected: all pass.

- [ ] **Step 6: Stage**

```bash
git add so101sim/env.py so101sim/mechsearch.py tests/test_mechsearch.py
```

---

### Task 8: Smoke script, reset cost, and the cube count

**Files:**
- Create: `scripts/mechsearch_smoke.py`
- Regenerate: `models/search_objects.xml`, `models/search_scene.xml`

**Interfaces:**
- Consumes: `MechSearchEnv`, `MAX_RESET_ATTEMPTS`, `info["reset_attempts"]`, `search_layout.candidate_scene`, `read_notes`, `write`.
- Produces: the committed `n_cubes`, and a notes line starting with `Cube count`.

- [ ] **Step 1: Write `scripts/mechsearch_smoke.py`**

```python
"""Reset the mechanical-search env many times: starting piles on video, reset cost, and the cube count.

    python scripts/mechsearch_smoke.py --episodes 8 --out outputs/mechsearch.mp4
    python scripts/mechsearch_smoke.py --episodes 100 --no-video
    python scripts/mechsearch_smoke.py --tune-count --write

Acceptance is the share of dropped piles that reset keeps, 1 / mean attempts. The spec wants at least half
kept for buried and for visible starts, and a reset under 3 s on average. --tune-count starts at 30 cubes
and takes grays away until acceptance passes, and --write records the count in models/search_objects.xml.
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
from so101sim import SEARCH_SCENE_PATH, search_layout as sl
from so101sim.env import CONTROL_HZ
from so101sim.mechsearch import MAX_RESET_ATTEMPTS, MechSearchEnv

import mujoco

START_COUNT = 30
MIN_COUNT = 10
MIN_ACCEPTANCE = 0.5
RESET_BUDGET_S = 3.0
HOLD_S = 1.0

_ENVS = {}


def _measure(job):
    """One reset in a worker process, reusing that worker's env for the same scene."""
    scene, seed, buried = job
    if scene not in _ENVS:
        for old in _ENVS.values():
            old.close()
        _ENVS.clear()
        _ENVS[scene] = MechSearchEnv(scene_path=scene)
    started = time.perf_counter()
    try:
        _, info = _ENVS[scene].reset(seed=seed, options={"buried": buried})
        attempts = info["reset_attempts"]
    except RuntimeError:
        attempts = MAX_RESET_ATTEMPTS
    return buried, time.perf_counter() - started, attempts


def measure(scene, seeds, workers):
    jobs = [(str(scene), seed, buried) for buried in (True, False) for seed in seeds]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_measure, jobs, chunksize=2)
    stats = {}
    for buried, label in ((True, "buried"), (False, "visible")):
        mine = [r for r in rows if r[0] == buried]
        stats[label] = {"acceptance": len(mine) / sum(r[2] for r in mine),
                        "reset_s": float(np.mean([r[1] for r in mine]))}
    return stats


def report(stats):
    return "; ".join(f"{label} kept {s['acceptance']:.0%}, {s['reset_s']:.1f} s per reset"
                     for label, s in stats.items())


def video(args):
    env = MechSearchEnv()
    frames = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=episode, options={"buried": episode % 2 == 0})
            print(f"episode {episode}: {'buried' if info['buried'] else 'visible'}, red pixels "
                  f"{info['target_pixels']}, {info['reset_attempts']} attempts")
            frame = np.concatenate([obs["observation.images.front"], obs["observation.images.wrist"]], axis=1)
            frames += [frame] * int(HOLD_S * CONTROL_HZ)
    finally:
        env.close()
    imageio.mimwrite(args.out, frames, fps=CONTROL_HZ, macro_block_size=1)
    print(f"wrote {len(frames)} frames to {args.out}")


def tune(args):
    committed = sl.Layout.from_model(mujoco.MjModel.from_xml_path(str(SEARCH_SCENE_PATH)))
    for count in range(START_COUNT, MIN_COUNT - 1, -1):
        layout = dataclasses.replace(committed, n_cubes=count)
        with sl.candidate_scene(layout) as scene:
            stats = measure(scene, range(args.seeds), args.workers)
        print(f"{count} cubes: {report(stats)}", flush=True)
        if all(s["acceptance"] >= MIN_ACCEPTANCE for s in stats.values()):
            break
    else:
        sys.exit(f"STOP: fewer than half of piles are kept even at {MIN_COUNT} cubes. Bring this back to dm1487.")
    slow = [label for label, s in stats.items() if s["reset_s"] > RESET_BUDGET_S]
    if slow:
        print(f"REPORT TO dm1487: reset averages over {RESET_BUDGET_S:.0f} s for {', '.join(slow)} starts. "
              "The spec says report it, not cut cubes to meet it.")
    if args.write:
        notes = [n for n in sl.read_notes() if not n.startswith("Cube count")]
        notes.append(f"Cube count {count}, the most with at least half of piles kept over {args.seeds} "
                     f"seeds each: {report(stats)}.")
        sl.write(layout, notes)
        print(f"wrote {sl.OBJECTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/mechsearch.mp4"))
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
    print(f"over {args.episodes} seeds: {report(measure(SEARCH_SCENE_PATH, range(args.episodes), args.workers))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Look at a few starting piles**

Run: `MUJOCO_EGL_DEVICE_ID=N python scripts/mechsearch_smoke.py --episodes 6 --out outputs/mechsearch.mp4`
Expected: six lines alternating buried and visible, with zero red pixels on the buried ones, then `wrote 180 frames`. Open `outputs/mechsearch.mp4` or pull a frame out and look at it. The cubes should be in a pile inside the bin, with the arm clear above it.

- [ ] **Step 3: Tune the cube count**

Run: `MUJOCO_EGL_DEVICE_ID=N python scripts/mechsearch_smoke.py --tune-count --write 2>&1 | tee $SCRATCH/count.log`
Expected: one line per count tried, then `wrote .../search_objects.xml`. On `STOP:`, stop and report the log. If it prints `REPORT TO dm1487`, keep going but carry that line into the final summary.

- [ ] **Step 4: Rerun every search test against the tuned scene**

Run: `python -m pytest tests/test_search_layout.py tests/test_search_checks.py tests/test_mechsearch.py -q`
Expected: all pass.

- [ ] **Step 5: Stage**

```bash
git add scripts/mechsearch_smoke.py models/search_objects.xml models/search_scene.xml
```

---

### Task 9: `MechSearchPrimitiveEnv`, and proof the task can be done

**Files:**
- Modify: `so101sim/mechsearch.py`, `tests/test_mechsearch.py`

**Interfaces:**
- Consumes: `MechSearchEnv.advance`, `task_info`, `settled`, `joint_state`, `_obs`, `free_spot`, `cube_pose`, `grays`, `bin`. Also `primitives.pick/place/push/bridge/Unreachable` and `search_layout.travel_z`.
- Produces:
  - `MechSearchPrimitiveEnv(max_primitives=15, push_stroke=0.05, on_frame=None, **MechSearchEnv kwargs except max_steps)`, with attributes `env`, `model`, `data`, `bin`, `travel_z`, `max_primitives`, `push_stroke`, `on_frame`
  - `action_space`: a `Dict` of `kind` (`Discrete(3)`), `xyz` (a `Box` over the bin interior, floor + 0.015 to floor + 0.075), and `angle` (a `Box` from −π to π)
  - `step(action)`, where `action` is a dict or a `(kind, xyz, angle)` tuple and `kind` is a name or an index. `info` adds `primitive`, `frames` and `refused`.
  - `mechsearch.MAX_PRIMITIVES = 15`, `PUSH_STROKE = 0.05`, `SETTLE_MAX_FRAMES = 60`

- [ ] **Step 1: Write the failing tests** by appending to `tests/test_mechsearch.py`

```python
# ---- primitive env ------------------------------------------------------------------------------

SOLVABLE_SEED = None  # set in step 6, after running dig_out over seeds 0 to 19


@pytest.fixture(scope="module")
def prim():
    e = MechSearchPrimitiveEnv(height=64, width=64)
    yield e
    e.close()


def grab(prim, name):
    """Where to pick `name`, clipped into the action box, and its yaw."""
    box = prim.action_space["xyz"]
    p = prim.env.cube_pose(name)
    return np.clip(p["pos"], box.low, box.high), p["yaw"]


def covers(pos, red):
    """True for a cube resting over the red one, near enough in xy to overlap it at any yaw."""
    return (np.linalg.norm(pos[:2] - red[:2]) < 2 * primitives.CUBE_HALF * np.sqrt(2)
            and pos[2] > red[2] + primitives.CUBE_HALF)


def dig_out(prim, seed):
    """Lift every gray resting over the red cube onto a free floor spot, then pick the red cube."""
    prim.reset(seed=seed, options={"buried": True})
    env = prim.env
    low_z = prim.action_space["xyz"].low[2]
    for _ in range(prim.max_primitives // 2):
        red = env.cube_pose(TARGET)["pos"]
        above = [n for n in env.grays if covers(env.cube_pose(n)["pos"], red)]
        if not above:
            break
        xyz, yaw = grab(prim, max(above, key=lambda n: env.cube_pose(n)["pos"][2]))
        _, _, terminated, _, info = prim.step(("pick", xyz, yaw))
        spot = env.free_spot()
        if terminated or spot is None:
            return info
        _, _, terminated, _, info = prim.step(("place", [*spot, low_z + 0.005], yaw))
        if terminated:
            return info
    xyz, yaw = grab(prim, TARGET)
    return prim.step(("pick", xyz, yaw))[4]


def test_action_bounds_are_the_bin_interior_from_layer_one_to_three(prim):
    box, b = prim.action_space["xyz"], prim.bin
    inset = b["half"] - primitives.CUBE_HALF
    np.testing.assert_allclose(box.low, [b["center"][0] - inset, -inset, b["floor_z"] + 0.015], atol=1e-6)
    np.testing.assert_allclose(box.high, [b["center"][0] + inset, inset, b["floor_z"] + 0.075], atol=1e-6)


def test_a_push_that_would_leave_the_bin_is_refused(prim):
    prim.reset(seed=0)
    before = prim.env.joint_state.copy()
    _, reward, terminated, _, info = prim.step(("push", prim.action_space["xyz"].high, 0.0))
    assert info["refused"] and info["frames"] == 0
    assert not terminated and reward == 0.0
    np.testing.assert_array_equal(prim.env.joint_state, before)


def test_actions_outside_the_box_are_refused(prim):
    prim.reset(seed=0)
    _, _, _, _, info = prim.step(("pick", prim.action_space["xyz"].high + [0.05, 0.0, 0.0], 0.0))
    assert info["refused"] and info["frames"] == 0


def test_picks_at_the_corners_of_the_box_run_or_are_refused(prim):
    """Corners are where IK is likeliest to give out. Running and being refused are both fine. Raising isn't."""
    low, high = prim.action_space["xyz"].low, prim.action_space["xyz"].high
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            prim.reset(seed=0)
            _, _, _, _, info = prim.step(("pick", [x, y, low[2]], 0.0))
            assert info["refused"] or info["frames"] > 0


def test_picking_again_while_holding_does_not_raise(prim):
    prim.reset(seed=1, options={"buried": True})
    grays = sorted(prim.env.grays, key=lambda n: -prim.env.cube_pose(n)["pos"][2])
    for name in grays[:2]:
        xyz, yaw = grab(prim, name)
        _, _, terminated, _, info = prim.step(("pick", xyz, yaw))
        assert {"primitive", "frames", "refused", "failure"} <= set(info)
        if terminated:
            break


def test_the_episode_stops_on_the_frame_a_cube_leaves(prim):
    prim.reset(seed=0)
    gray = prim.env.grays[0]
    move_cube(prim.env, gray, outside_the_bin(prim.env))
    top = prim.action_space["xyz"].high[2]
    _, reward, terminated, _, info = prim.step(("pick", [prim.bin["center"][0], 0.0, top], 0.0))
    assert terminated and reward == 0.0
    assert info["frames"] == 1 and gray in info["failure"]


def test_the_primitive_limit_truncates(prim):
    prim.reset(seed=0)
    old, prim.max_primitives = prim.max_primitives, 2
    try:
        flags = [prim.step(("push", prim.action_space["xyz"].high, 0.0))[3] for _ in range(2)]
    finally:
        prim.max_primitives = old
    assert flags == [False, True]


def test_a_buried_red_cube_can_be_dug_out(prim):
    assert SOLVABLE_SEED is not None, "run the seed search in Task 9 step 6"
    info = dig_out(prim, SOLVABLE_SEED)
    assert info["is_success"], (info["failure"], info.get("refused"))
    assert info["n_in_bin"] == len(prim.env.cubes)
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_mechsearch.py -q -k "prim or push or refused or corners or holding or stops_on or primitive_limit or dug_out"`
Expected: FAIL. The `prim` fixture raises `TypeError`, because the stub takes no arguments.

- [ ] **Step 3: Replace the stub with the real class**

In `so101sim/mechsearch.py`, add `from gymnasium import spaces` to the imports. Add these constants below `SUCCESS_FRAMES`:

```python
MAX_PRIMITIVES = 15
PUSH_STROKE = 0.05
SETTLE_MAX_FRAMES = 60
```

Then replace `class MechSearchPrimitiveEnv:  # replaced in Task 9` and its `pass` with:

```python
class _Refused(Exception):
    """A primitive the env won't run: outside the action box, or a push that would leave the bin."""


class MechSearchPrimitiveEnv(gym.Env):
    """One step is one whole pick, place or push inside the bin.

    The action is {"kind", "xyz", "angle"}, or the tuple (kind, xyz, angle). `xyz` is the jaw midpoint:
    the grasp point for pick, the held cube's centre at release for place, and the closed jaw tips' height
    along a push. z is in the action because one xy can hold a cube on any of three layers. The angle is
    jaw yaw for pick, the held cube's yaw for place, and the heading for push. Every move across the bin
    happens at `travel_z`.

    Anything outside the action box, and any push that would end outside the bin, is refused. So is
    anything IK can't reach. The arm doesn't move, the step still counts, and `info["refused"]` says why.
    `info["failure"]` is only for things that end the episode.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    PRIMITIVES = ("pick", "place", "push")

    def __init__(self, max_primitives=MAX_PRIMITIVES, push_stroke=PUSH_STROKE, on_frame=None, **env_kwargs):
        # A primitive runs to completion inside one step, so a recorder needs a hook to see the 30 Hz
        # frames underneath it.
        self.on_frame = on_frame
        self.env = MechSearchEnv(max_steps=None, **env_kwargs)
        self.model, self.data = self.env.model, self.env.data
        self.bin = self.env.bin
        self.max_primitives = int(max_primitives)
        self.push_stroke = float(push_stroke)
        self.travel_z = sl.travel_z(self.bin["floor_z"], self.bin["rim_z"])

        self._inset = self.bin["half"] - CUBE_HALF
        c, f = self.bin["center"], self.bin["floor_z"]
        low = np.array([c[0] - self._inset, c[1] - self._inset, f + CUBE_HALF], dtype=np.float32)
        high = np.array([c[0] + self._inset, c[1] + self._inset,
                         f + CUBE_HALF + (sl.PILE_LAYERS - 1) * sl.LAYER], dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Dict({
            "kind": spaces.Discrete(len(self.PRIMITIVES)),
            "xyz": spaces.Box(low, high, dtype=np.float32),
            "angle": spaces.Box(-np.pi, np.pi, (), dtype=np.float32),
        })
        self._n = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        kind, xyz, angle = self._parse(action)
        self._n += 1
        refused, terminated, frames = None, False, 0
        info = self.env.task_info()
        try:
            traj = self._plan(kind, xyz, angle)
        except _Refused as exc:
            refused, traj = str(exc), None
        except primitives.Unreachable as exc:
            refused, traj = f"unreachable: {exc}", None

        if traj is not None:
            traj = primitives.bridge(self.model, self.env.joint_state, traj, hover_z=self.travel_z)
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
        _, terminated, _, info = self.env.advance(row)
        if self.on_frame is not None:
            self.on_frame(self)
        return terminated, info

    def _parse(self, action):
        if isinstance(action, dict):
            kind, xyz, angle = action["kind"], action["xyz"], action.get("angle", 0.0)
        else:
            kind, xyz, angle = action
        if isinstance(kind, (int, np.integer)):
            kind = self.PRIMITIVES[int(kind)]
        if kind not in self.PRIMITIVES:
            raise ValueError(f"unknown primitive {kind!r}, expected one of {self.PRIMITIVES}")
        return kind, np.asarray(xyz, dtype=np.float64).reshape(3), float(angle)

    def _plan(self, kind, xyz, angle):
        if not self.action_space["xyz"].contains(xyz.astype(np.float32)):
            raise _Refused(f"{np.round(xyz, 4).tolist()} is outside the action box")
        xy, z = xyz[:2], float(xyz[2])
        if kind == "pick":
            return primitives.pick(self.model, xy, angle, hover_z=self.travel_z, z=z)
        if kind == "place":
            return primitives.place(self.model, xy, angle, hover_z=self.travel_z, release_z=z)
        end = xy + self.push_stroke * np.array([np.cos(angle), np.sin(angle)])
        if np.abs(end - self.bin["center"]).max() > self._inset + 1e-9:
            raise _Refused(f"a push from {np.round(xy, 4).tolist()} heading {np.degrees(angle):.0f} deg "
                           "would end outside the bin")
        return primitives.push(self.model, xy, angle, stroke=self.push_stroke, hover_z=self.travel_z, push_z=z)
```

- [ ] **Step 4: Run everything except the dig-out test**

Run: `MUJOCO_EGL_DEVICE_ID=N python -m pytest tests/test_mechsearch.py -q -k "not dug_out"`
Expected: all pass.

- [ ] **Step 5: Find a seed where a buried red cube comes out**

Run:
```bash
MUJOCO_EGL_DEVICE_ID=N python - <<'EOF'
import sys
sys.path.insert(0, "tests")
import so101sim  # noqa: F401
from so101sim.mechsearch import MechSearchPrimitiveEnv
from test_mechsearch import dig_out

prim = MechSearchPrimitiveEnv(height=64, width=64)
try:
    for seed in range(20):
        info = dig_out(prim, seed)
        print(seed, "success" if info["is_success"] else f"no: {info['failure'] or info.get('refused')}",
              f"in bin {info['n_in_bin']}/{len(prim.env.cubes)}", flush=True)
finally:
    prim.close()
EOF
```
Expected: 20 lines. Take the first seed marked `success`. If none of the 20 succeed, that's a spec stop point. Report the 20 lines to dm1487 and stop.

- [ ] **Step 6: Pin the seed and run the whole file**

In `tests/test_mechsearch.py`, set `SOLVABLE_SEED` to the seed from step 5, and change its comment to `# first of seeds 0 to 19 that dig_out clears, found <date>`. Also note how many of the 20 succeeded. That count is the only success rate this plan measures, so it goes in the final summary.

Run: `MUJOCO_EGL_DEVICE_ID=N python -m pytest tests/test_mechsearch.py -q`
Expected: all pass.

- [ ] **Step 7: Stage**

```bash
git add so101sim/mechsearch.py tests/test_mechsearch.py
```

---

### Task 10: Docs, and the full suite

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

In "What this is", replace the sentences from "The end use is undecided" through "don't build task logic on that assumption." with:

```markdown
The end use is still undecided: VLA fine-tuning or RL. The task was decided on 2026-09-25: mechanical
search. A bin sunk below the base holds a pile of gray cubes up to three layers deep, and the robot has to
find the one red cube and lift it out without any cube leaving the bin. Spec:
`docs/superpowers/specs/2026-09-25-mechanical-search-design.md`. `BinPickEnv`, which moves loose cubes
into a small bin, came first and still works.
```

In "Commands", add after the `binpick_demo.py` lines:

```bash
python scripts/size_search_bin.py --write          # re-pick the search bin, riser, walls, start pose, camera
python scripts/mechsearch_smoke.py --episodes 8    # starting piles from both cameras, reset cost, acceptance
python scripts/mechsearch_smoke.py --tune-count --write   # re-pick the cube count
```

In "Layout and conventions", change the first bullet's "Scene changes go in `models/scene.xml`, and task objects in `models/objects.xml`." to "Shared scene pieces go in `models/stage.xml`. Each task's table, camera and objects go in its own objects file." Then add these bullets after the `binpick.py` bullet:

```markdown
- `so101sim/search_layout.py` generates `models/search_objects.xml` and `models/search_scene.xml` from a
  `Layout`: riser height, bin width and centre, wall height, front camera height, cube count, start pose.
  The numbers live in the XML's `<custom>` fields and `Layout.from_model` reads them back, so never edit
  the generated files. `scripts/size_search_bin.py` picks the layout with the checks in
  `so101sim/search_checks.py`, and its notes at the top of the XML say what it measured.
- `so101sim/mechsearch.py` holds `MechSearchEnv`, a 30 Hz env with `SO101Env`'s actions and observations,
  and `MechSearchPrimitiveEnv`, where one step is a pick, place or push inside the bin. Every cube leaving
  the bin ends the episode as a failure. Success is the red cube held with its lowest corner 1 cm over the
  rim for 5 frames. The buried label and red pixel counts come from a 640x480 segmentation render at
  reset, whatever the observation size.
- `so101sim/grasp.py` holds the contact checks both task envs share.
```

In "Traps", change "`scene.xml` overrides `meshdir` after including the vendored arm." to "`stage.xml` overrides `meshdir` after including the vendored arm." Then add:

```markdown
- Every top-level scene has to sit in `models/`, because `meshdir` resolves against the top-level file.
  `search_layout.candidate_scene` writes its temporary scenes there for that reason, named `.candidate_*`
  and ignored by git.
- In the search scene the table is below the base, so grasp heights are negative. `primitives.bridge`
  decides whether the arm is hovering relative to `hover_z`. Before 2026-09-25 it used a fixed z = 0.03,
  which would have sent every crossing in the sunken bin through a joint ramp and the pile.
- Search-env cubes wait on the ground plane at x = 1.2 until reset drops them, so a cube there in a debug
  view hasn't fallen off anything.
- A search reset with 20 to 30 cubes takes a second or more, so `tests/test_mechsearch.py` takes minutes,
  not seconds.
```

In "The real arm", add to the list of unknowns:

```markdown
- The search task puts the table `Layout.depth` below the base, the riser height `scripts/size_search_bin.py`
  chose. The real setup needs the arm on a riser that high, or a bin set that far into the table.
```

- [ ] **Step 2: Update `README.md`**

Replace the paragraph "Nothing task-specific exists yet. Rewards, demos and training wait on two open questions: RL or VLA fine-tuning, and what the task is." with:

```markdown
The task is mechanical search: find the one red cube in a bin of gray ones and lift it out, rearranging
only inside the bin. `so101sim.mechsearch` has a 30 Hz env and a pick/place/push primitive env for it.
Whether this trains a VLA or an RL policy is still open. The older `BinPickEnv`, which moves loose cubes
into a small bin, still works.
```

Replace the test-count comment `# 19 tests, about 10 s` with `# a few minutes, most of it search resets`. After the `SO101Env` example block, add:

```python
from so101sim.mechsearch import MechSearchPrimitiveEnv

env = MechSearchPrimitiveEnv()
obs, info = env.reset(seed=0, options={"buried": True})
p = info["cubes"]["cube_gray_0"]
obs, reward, terminated, truncated, info = env.step(("pick", p["pos"], p["yaw"]))
env.close()
```

In "Where things are", change the `models/` bullet to "`models/` holds the menagerie SO-101, pinned and untouched, the shared `stage.xml`, and one scene per task. See `models/README.md`." Change the `so101sim/` bullet to "`so101sim/` has the unit conversion, IK, primitives, both task envs, the search layout generator and the MJX loader." Add a bullet: "`docs/superpowers/specs/2026-09-25-mechanical-search-design.md` is the current task's spec."

- [ ] **Step 3: Run the whole suite**

Run: `MUJOCO_EGL_DEVICE_ID=N python -m pytest -q`
Expected: every test passes, MJX included.

- [ ] **Step 4: Stage, then summarize for dm1487**

```bash
git add CLAUDE.md README.md
```

The summary should give:

- The chosen layout, from the notes in `models/search_objects.xml`.
- The cube count and its acceptance rates.
- The reset time, including any line that reported it over 3 s.
- How many of the 20 dig-out seeds succeeded.
- Any stop point that was hit.
- The staged but uncommitted files, with a question about whether to commit them.
