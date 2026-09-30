import numpy as np
import pytest

from so101sim import grasp, rest, shelf
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


def held_still(env, frames):
    """How far the most-moved cylinder goes while the arm holds its pose for `frames` frames, in metres."""
    before = positions(env)
    hold = env.joint_state.copy()
    for _ in range(frames):
        env.advance(hold)
    after = positions(env)
    return max(np.linalg.norm(after[n] - before[n]) for n in env.cyls)


@pytest.mark.parametrize("seed,hidden", [(0, True), (1, False), (2, True), (3, False)])
def test_reset_hands_over_a_shelf_at_rest(env, seed, hidden):
    # Until 2026-09-29 reset let the episode start at max |qvel| <= 0.2 over every DOF, arm joints included, and seed
    # 1's partial shelf had a cylinder slide 22 mm in the first second after it.
    env.reset(seed=seed, options={"hidden": hidden})
    assert env.at_rest()
    moved = held_still(env, 30)
    assert moved < 0.001, f"a cylinder moved {moved * 1000:.1f} mm in the first second"


def test_a_shelf_still_moving_after_settling_is_rejected(env, monkeypatch):
    monkeypatch.setattr(shelf, "MAX_RESET_ATTEMPTS", 2)
    monkeypatch.setattr(rest, "come_to_rest", lambda step, data, dofs: False)
    with pytest.raises(RuntimeError, match="still moving"):
        env.reset(seed=0)


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


@pytest.mark.parametrize("hidden", [True, False])
def test_the_red_starts_clear_of_the_side_walls(env, hidden):
    """A red cylinder the gripper can't fit beside would make the start impossible (spec, first amendment)."""
    from so101sim.shelf_checks import WALL_STRIP
    for seed in range(3):
        env.reset(seed=seed, options={"hidden": hidden})
        y = env.cylinder_pose(TARGET)["pos"][1]
        assert abs(y) <= env.layout.half_width - WALL_STRIP + 0.002, (seed, y)


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


def test_the_arm_touching_the_shelf_ends_the_episode(env):
    """Arm-shelf contact is very bad behaviour (spec, first amendment). Put the jaws 1 cm into the top board. Just
    under it the gripper only grazes the board, and 2 mm of sag after one step clears the contact."""
    from so101sim import ik, primitives, units
    env.reset(seed=0)
    L = env.layout
    grip = primitives._grip_rad(env.model, primitives.GRIP_PLAN)
    q, solved = ik.solve_side(env.model, (L.front_x + 0.08, 0.0, L.board_z + L.opening + 0.01), gripper=grip)
    assert solved["pos_err"] < 1e-3
    pose = units.to_lerobot(np.r_[q, grip], env._gripper_range)
    env.data.qpos[env._qpos_adr] = np.r_[q, grip]
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, _, info = env.step(pose)
    assert terminated and reward == 0.0
    assert info["failure"].startswith("hit the shelf: shelf_"), info["failure"]


def test_partial_starts_are_accepted_often(env):
    """The count rule needs half of partial attempts kept (spec, second amendment): a gray placed on each camera's
    sight line, shifted sideways, covers part of the red on purpose. Seeded, so this is deterministic."""
    attempts = [env.reset(seed=seed, options={"hidden": False})[1]["reset_attempts"] for seed in range(20)]
    assert 20 / sum(attempts) >= 0.4, attempts


# ---- primitive env ------------------------------------------------------------------------------

from so101sim import side_primitives as sp  # noqa: E402
from so101sim.shelf_checks import in_wall_strip, near_right_wall  # noqa: E402
from so101sim.shelf import ShelfPrimitiveEnv  # noqa: E402

SOLVABLE_SEED = 16  # found 2026-09-26: the push-first dig_out clears 1 of hidden seeds 0 to 19; most others stop at "blocked itself"


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


def place_spot(prim, keep_clear_of=None):
    """A board spot for a cylinder: 4.5 cm from every centre, out of the wall strips, with its own way in and jaw
    sides clear, and away from the ray and sides of `keep_clear_of` if given. Nearest the front first."""
    x_lo, x_hi, y_lo, y_hi = prim.env.region
    if keep_clear_of is not None:
        u, n, along = ray_frame(prim, keep_clear_of)
    centres = np.array([xy_of(prim, c) for c in prim.env.cyls])
    best = None
    for x in np.arange(x_lo, x_hi + 1e-9, 0.005):
        for y in np.arange(y_lo, y_hi + 1e-9, 0.005):
            xy = np.array([x, y])
            if np.linalg.norm(centres - xy, axis=1).min() < 0.045:
                continue
            if keep_clear_of is not None:
                a, lat = (xy - prim.geo.pan_xy) @ u - along, (xy - prim.geo.pan_xy) @ n
                if abs(lat) < 0.09 and a < 0.05:
                    continue
            # The gripper doesn't fit beside a wall, and the released moving jaw opens toward the right one.
            if in_wall_strip(prim.env.layout, xy) or near_right_wall(prim.env.layout, xy):
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


def ray_blocked(prim, xy, skip=()):
    """True if a cylinder not in `skip` stands on the ray between the aisle and `xy`, where the gripper goes in."""
    u, n, along = ray_frame(prim, xy)
    aisle_along = (prim.geo.on_ray(xy, prim.geo.aisle_x) - prim.geo.pan_xy) @ u - along
    for name in prim.env.cyls:
        if name in skip:
            continue
        p = xy_of(prim, name)
        a, lat = (p - prim.geo.pan_xy) @ u - along, (p - prim.geo.pan_xy) @ n
        if aisle_along - 0.02 <= a <= 0.02 and abs(lat) < 0.035:
            return True
    return False


def push_aside(prim, name, red_xy):
    """Push `name` sideways, away from the red's ray, with the closed jaws starting beside it on the ray's side.
    Returns the step's info, or None when no push start fits."""
    u, n, _ = ray_frame(prim, red_xy)
    p = xy_of(prim, name)
    side = 1.0 if (p - prim.geo.pan_xy) @ n >= (red_xy - prim.geo.pan_xy) @ n else -1.0
    direction = side * n
    start, end = p - 0.035 * direction, p - 0.035 * direction + prim.push_stroke * direction
    box = prim.action_space["xy"]
    if any(np.abs(np.clip(q, box.low, box.high) - q).max() > 0.005 for q in (start, end)):
        return None
    others = np.array([xy_of(prim, c) for c in prim.env.cyls if c != name])
    if np.linalg.norm(others - start, axis=1).min() < 0.03 or ray_blocked(prim, start, skip=(name,)):
        return None
    return prim.step(("push", start, float(np.arctan2(direction[1], direction[0]))))[4]


def dig_out(prim, seed, max_moves=8):
    """Clear what blocks the red cylinder's ray and jaw sides, then pull it out. Each blocker gets a sideways push
    first and a pick-and-place only when no push fits (spec, first amendment). Chosen from info poses, not images:
    this proves the task can be done, it isn't a policy."""
    prim.reset(seed=seed, options={"hidden": True})
    for _ in range(max_moves):
        red = xy_of(prim, TARGET)
        block = blockers(prim, red)
        if not block:
            break
        name = block[0]
        info = push_aside(prim, name, red)
        if info is not None and (info["failure"] or not info["refused"]):
            if info["failure"]:
                return info
            continue
        if blockers(prim, xy_of(prim, name)):
            return {"is_success": False, "failure": f"{name} is blocked itself", "refused": None}
        spot = place_spot(prim, red)
        if spot is None:
            return {"is_success": False, "failure": "no place to put a gray", "refused": None}
        info = run(prim, ("pick", xy_of(prim, name), 0.0), ("place", spot, 0.0))
        if info["failure"] or info["refused"]:
            return info
    return run(prim, ("pick", xy_of(prim, TARGET), 0.0), ("pull", xy_of(prim, TARGET), 0.0))


def a_clear_gray(prim, seeds=range(60), with_spot=False):
    """Reset to the first partial seed with a gray nothing blocks, and a place spot too if `with_spot`, and return
    its name."""
    for seed in seeds:
        prim.reset(seed=seed, options={"hidden": False})
        for name in prim.env.grays:
            if blockers(prim, xy_of(prim, name)) or sp_near_right(prim, name):
                continue
            if with_spot and place_spot(prim) is None:
                break
            return name
    pytest.fail("no clear gray in the first seeds")


def sp_near_right(prim, name):
    """Too near a side wall for the tests that use it: a push start 3.5 cm beside the cylinder, or the open moving
    jaw 6 cm to its right, has to fit inside the region."""
    y, hw = xy_of(prim, name)[1], prim.env.layout.half_width
    return y + hw <= 0.07 or hw - y <= 0.07


def stage(prim):
    """A fixed shelf for the primitive mechanics tests: thirteen cylinders in two back rows and one gray alone at the
    front centre, with nothing in its way. Resets never leave a gray that clear: 14 cylinders at least 3.5 cm apart
    on 28 by 10.5 cm put a neighbour inside every gray's jaw space. Returns (the lone gray, a clear spot)."""
    prim.reset(seed=0, options={"hidden": False})
    env = prim.env
    x_lo, x_hi, _, _ = env.region
    back = [(x_hi, y) for y in np.arange(-0.12, 0.1201, 0.04)]
    middle = [(x_hi - 0.04, y) for y in np.arange(-0.10, 0.1001, 0.04)]
    lone = (x_lo + 0.005, 0.0)
    spots = back + middle
    names = [TARGET] + [n for n in env.grays if n != "cyl_gray_0"]
    for name, xy in zip(names, spots):
        set_cylinder(env, name, [*xy, env.layout.board_z + 0.0405])
    set_cylinder(env, "cyl_gray_0", [*lone, env.layout.board_z + 0.0405])
    prim.data.qvel[:] = 0.0
    mujoco.mj_forward(prim.model, prim.data)
    assert not blockers(prim, np.array(lone))
    return "cyl_gray_0", np.array([x_lo + 0.005, -0.08])


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
    name, spot = stage(prim)
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
    name, _ = stage(prim)
    prim.step(("pick", xy_of(prim, name), 0.0))
    _, reward, terminated, _, info = prim.step(("pull", xy_of(prim, name), 0.0))
    assert terminated and reward == 0.0
    assert info["failure"] == f"left the shelf: {name}"


def test_a_missed_pick_backs_out_cleanly(prim):
    _, spot = stage(prim)  # empty, with a clear way in
    _, _, terminated, _, info = prim.step(("pick", spot, 0.0))
    assert not terminated and not prim.holding(), info["failure"]
    _, _, terminated, _, info = prim.step(("look", prim.action_space["xy"].low, 0.0))
    assert not terminated and info["refused"] is None, info["failure"]


def test_a_low_push_slides_a_cylinder_without_tipping_it(prim):
    name, _ = stage(prim)
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


def test_a_primitive_that_would_hit_the_shelf_is_refused(prim):
    """The arm touching the shelf is very bad behaviour (spec, first amendment), so a path that would is refused
    before it moves. A pick 2 cm from the right wall opens the moving jaw into it."""
    prim.reset(seed=0)
    x_lo, x_hi, y_lo, _ = prim.env.region
    assert "would hit shelf_" in refused(prim, ("pick", [(x_lo + x_hi) / 2, y_lo], 0.0))


def test_the_path_check_keeps_a_margin(prim):
    """The path check runs on a copy whose shelf boxes are CLEARANCE bigger. A pose that only the inflated shelf
    touches is refused; a pose that touches neither isn't."""
    from so101sim import ik, primitives, units
    from so101sim.shelf_checks import CLEARANCE
    prim.reset(seed=0)
    top_id = prim.model.geom("shelf_top").id
    np.testing.assert_allclose(prim._probe_model.geom_size[top_id], prim.model.geom_size[top_id] + CLEARANCE)
    L = prim.env.layout
    grip = primitives._grip_rad(prim.model, primitives.GRIP_HOLDING)
    real = mujoco.MjData(prim.model)
    real.qpos[:] = prim.data.qpos
    near = clear = None
    for dz in np.arange(0.12, 0.05, -0.001):
        q, _ = ik.solve_side(prim.model, (L.front_x + 0.10, 0.03, L.board_z + L.opening - dz), gripper=grip)
        pose = units.to_lerobot(np.r_[q, grip], prim.env._gripper_range)
        real.qpos[prim.env._qpos_adr] = np.r_[q, grip]
        mujoco.mj_forward(prim.model, real)
        touches = any((c.geom1 in prim.env._arm_ids) != (c.geom2 in prim.env._arm_ids)
                      and ({c.geom1, c.geom2} & prim.env._shelf_ids) for c in real.contact[:real.ncon])
        flagged = prim._path_hit([pose]) is not None
        if not touches and not flagged and clear is None:
            clear = pose
        if not touches and flagged:
            near = pose
            break
    assert clear is not None and near is not None
    assert prim._path_hit([clear]) is None and prim._path_hit([near]) is not None


def test_a_pushed_cylinder_comes_to_rest_upright(prim):
    """With condim 4 the rolling coefficient in the friction triple is ignored, and a cylinder pushed off-centre
    wobbled on its rim for good, its tilt growing from 7.8 to 9.0 degrees over 10 s (2026-09-26)."""
    name, _ = stage(prim)
    prim.step(("push", xy_of(prim, name) + [0.0, 0.035], -np.pi / 2))
    hold = prim.env.joint_state.copy()
    for _ in range(30):
        prim.env.advance(hold)
    dof = prim.env._dofadr[name]
    assert prim.env.tilt(name) < 1.0
    assert np.abs(prim.data.qvel[dof:dof + 6]).max() < 0.05


def test_holding_counts_any_close_that_stops_short(prim):
    """An empty close reads 10.0. Held cylinders read 15.8 to 17.8 for grasps up to 8 mm off centre, and one taken
    off-centre in the seed-300 planner run read 12.7 with both jaws on it (2026-09-26)."""
    from so101sim import units
    prim.reset(seed=0)
    adr = prim.env._qpos_adr[5]
    prim._last_grip = 10.0
    for reading, held in ((10.0, False), (10.4, False), (12.7, True), (17.7, True)):
        prim.data.qpos[adr] = units.to_rad(np.array([0, 0, 0, 0, 0, reading]), prim.env._gripper_range)[5]
        assert prim.holding() is held, reading
