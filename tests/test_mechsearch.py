import numpy as np
import pytest

from so101sim import grasp, primitives
from so101sim.mechsearch import (PARTIAL_MAX_SHARE, TARGET, TASK, VISIBLE_MIN_PIXELS, MechSearchEnv,
                                 MechSearchPrimitiveEnv, face_pixels)

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


def test_face_pixels_drop_seams_and_keep_faces():
    mask = np.zeros((40, 40), bool)
    mask[5, :] = True  # a 1-pixel seam
    mask[10:14, :] = True  # a 4-pixel seam
    mask[0:3, 0:3] = True  # a sliver in the corner, which must not wrap around the edges
    assert face_pixels(mask) == 0
    mask[20:30, 20:30] = True  # a 10 x 10 face keeps its 6 x 6 middle
    assert face_pixels(mask) == 36


def test_a_buried_red_cube_sits_on_the_floor_boxed_in(env):
    env.reset(seed=0, options={"buried": True})
    red = env.cube_pose(TARGET)["pos"]
    floor = env.bin["floor_z"] + primitives.CUBE_HALF
    assert abs(red[2] - floor) < 0.003
    others = [env.cube_pose(n)["pos"] for n in env.grays]
    sides = [p for p in others if abs(p[2] - floor) < 0.005 and np.linalg.norm(p[:2] - red[:2]) < 0.045]
    above = [p for p in others if p[2] > red[2] + 0.02 and np.linalg.norm(p[:2] - red[:2]) < 0.02]
    assert len(sides) >= 4 and above


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


def test_partial_reset_shows_part_of_the_red_cube_but_not_most_of_it(env):
    _, info = env.reset(seed=0, options={"buried": False})
    assert info["buried"] is False
    assert max(info["target_pixels"].values()) >= VISIBLE_MIN_PIXELS
    assert all(share <= PARTIAL_MAX_SHARE for share in info["target_visible_share"].values()), info["target_visible_share"]


def test_hiding_the_gray_cubes_never_shows_less_of_the_red_one(env):
    env.reset(seed=1, options={"buried": False})
    shown, unoccluded = env.target_pixels(), env.target_pixels(hide_grays=True)
    assert all(unoccluded[cam] >= shown[cam] for cam in shown)
    assert max(unoccluded.values()) > max(shown.values())


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


def held_still(env, frames):
    """How far the most-moved cube goes while the arm holds its pose for `frames` frames, in metres."""
    before = positions(env)
    hold = env.joint_state.copy()
    for _ in range(frames):
        env.advance(hold)
    after = positions(env)
    return max(np.linalg.norm(after[n] - before[n]) for n in env.cubes)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_reset_hands_over_a_pile_at_rest(env, seed):
    # Until 2026-09-29 reset let the episode start at max |qvel| <= 0.2, and a shelf cylinder slid 22 mm in the
    # first second after it.
    env.reset(seed=seed)
    assert env.at_rest()
    moved = held_still(env, 30)
    assert moved < 0.001, f"a cube moved {moved * 1000:.1f} mm in the first second"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_resting_pile_does_not_creep(env, seed):
    # Without MuJoCo's noslip pass, stacked cubes crept 6 to 42 mm in 10 s with nothing touching them; seed 0 moved
    # 20 mm here. With it, measured over seeds 0 to 19 on 2026-09-29: median 0.85 mm, 16 of 20 under 2 mm, and two
    # piles over 5 mm where a single cube resettled (seeds 10 and 18). Seed 0's worst cube moves 2.7 mm.
    env.reset(seed=seed)
    moved = held_still(env, 300)
    assert moved < 0.003, f"a cube crept {moved * 1000:.1f} mm in 10 s"


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


def test_success_needs_a_real_lift_not_just_the_rim(env, monkeypatch):
    """With 5 cm walls, rim + 1 cm is where a third-layer cube already rests, so a grasp and a 2 mm twitch used to
    count. Success needs the cube's bottom 2 cm above where a top-layer cube rests."""
    env.reset(seed=0, options={"buried": False})
    monkeypatch.setattr(grasp, "is_grasped", lambda *args: True)
    resting_on_layer_3 = env.bin["floor_z"] + 5 * primitives.CUBE_HALF
    adr = env.model.jnt_qposadr[env.model.body(TARGET).jntadr[0]]
    for lift, retrieved in ((0.002, False), (0.035, True)):
        env.data.qpos[adr:adr + 3] = [*env.bin["center"], resting_on_layer_3 + lift]
        env.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)
        assert env._target_retrieved() is retrieved, lift


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


# ---- primitive env ------------------------------------------------------------------------------

# First of seeds 0 to 19 that dig_out clears. Re-found 2026-09-29 after stage.xml turned noslip on: 15 of the 20
# succeed, against 12 under the old creeping physics, and seed 0 now spills a gray.
SOLVABLE_SEED = 1


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


def drop_spot(prim, avoid):
    """Where to set a cleared cube down: the lowest part of the pile at least 6 cm from `avoid`, 2 cm in from the
    edge of the action box so a cube set on top doesn't tumble over a wall."""
    env, box = prim.env, prim.action_space["xyz"]
    floor = box.low[2] - primitives.CUBE_HALF
    best = None
    for x in np.linspace(box.low[0] + 0.02, box.high[0] - 0.02, 9):
        for y in np.linspace(box.low[1] + 0.02, box.high[1] - 0.02, 9):
            if np.hypot(x - avoid[0], y - avoid[1]) < 0.06:
                continue
            top = max((env.cube_pose(n)["pos"][2] + primitives.CUBE_HALF for n in env.cubes
                       if np.hypot(*(env.cube_pose(n)["pos"][:2] - [x, y])) < 0.035), default=floor)
            if best is None or top < best[0]:
                best = (top, x, y)
    top, x, y = best
    return [x, y, min(top + primitives.CUBE_HALF + 0.005, box.high[2])]


def dig_out(prim, seed):
    """Clear the grays over the red cube and the two flush against its jaw sides, then pick the red cube.

    A buried red cube is boxed in, so clearing the top isn't enough: the jaws close along the red cube's x axis,
    and the grays touching it on that axis have to go too. Each of those is gripped across its own width, so
    its jaws don't have to fit between it and the red cube.
    """
    prim.reset(seed=seed, options={"buried": True})
    env = prim.env

    def move(name, yaw):
        xyz, _ = grab(prim, name)
        _, _, terminated, _, info = prim.step(("pick", xyz, yaw))
        if terminated:
            return info
        _, _, terminated, _, info = prim.step(("place", drop_spot(prim, env.cube_pose(TARGET)["pos"]), yaw))
        return info if terminated else None

    for _ in range(4):
        red = env.cube_pose(TARGET)["pos"]
        above = [n for n in env.grays if covers(env.cube_pose(n)["pos"], red)]
        if not above:
            break
        top = max(above, key=lambda n: env.cube_pose(n)["pos"][2])
        if (info := move(top, env.cube_pose(top)["yaw"])) is not None:
            return info
    red = env.cube_pose(TARGET)
    along = np.array([np.cos(red["yaw"]), np.sin(red["yaw"])])
    across = np.array([-along[1], along[0]])
    for name in env.grays:
        d = env.cube_pose(name)["pos"] - red["pos"]
        if abs(d[2]) < primitives.CUBE_HALF and abs(d[:2] @ along) < 0.05 and abs(d[:2] @ across) < 0.025:
            if (info := move(name, red["yaw"] + np.pi / 2)) is not None:
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


def test_a_floor_cube_can_be_picked_at_the_pose_info_reports(prim):
    """A resting cube sinks a fraction of a millimetre into the floor, just under the box's lowest z."""
    _, info = prim.reset(seed=0, options={"buried": True})
    floor = prim.bin["floor_z"] + primitives.CUBE_HALF
    name = next(n for n, c in info["cubes"].items() if abs(c["pos"][2] - floor) < 0.002 and n != TARGET)
    p = info["cubes"][name]
    _, _, _, _, info = prim.step(("pick", p["pos"], p["yaw"]))
    assert info["refused"] is None or info["refused"].startswith("unreachable"), info["refused"]


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


def test_a_pick_can_ask_for_either_wrist_flip(prim):
    """The action dict takes an optional flip. The arm ends each pick hovering with the wrist where that flip's
    plan put it, half a turn apart."""
    ends = []
    for flip in (0, 1):
        prim.reset(seed=0, options={"buried": True})
        xyz = np.array([0.20, 0.0, prim.action_space["xyz"].high[2]])
        plan = prim._plan("pick", xyz, 0.3, flip=flip)
        _, _, _, _, info = prim.step({"kind": "pick", "xyz": xyz, "angle": 0.3, "flip": flip})
        assert info["refused"] is None, info["refused"]
        roll = prim.env.joint_state[4]
        assert abs((roll - plan[-1][4] + 180.0) % 360.0 - 180.0) < 5.0
        ends.append(roll)
    assert abs(abs((ends[1] - ends[0] + 180.0) % 360.0 - 180.0) - 180.0) < 8.0


def test_a_sweep_drags_the_closed_jaws_the_whole_stroke_at_the_chosen_height(prim):
    """A sweep is a push with its own stroke length and height, for clearing several cubes off the red at once."""
    prim.reset(seed=0, options={"buried": True})
    start = np.array([0.30, -0.06, prim.bin["rim_z"] + 0.008])
    track = []
    prim.on_frame = lambda p: track.append(primitives.tool_pose(p.model, p.env.joint_state)[0])
    try:
        _, _, _, _, info = prim.step({"kind": "sweep", "xyz": start, "angle": np.pi, "stroke": 0.12})
    finally:
        prim.on_frame = None
    assert info["refused"] is None, info["refused"]
    track = np.array(track)
    low = track[np.abs(track[:, 2] - start[2]) < 0.004]  # the frames at stroke height
    assert start[0] - low[:, 0].min() > 0.11


def test_a_sweep_that_would_run_into_a_wall_is_refused(prim):
    prim.reset(seed=0, options={"buried": True})
    start = np.array([0.30, 0.0, prim.bin["rim_z"] + 0.008])
    _, _, _, _, info = prim.step({"kind": "sweep", "xyz": start, "angle": 0.0, "stroke": 0.12})
    assert info["refused"] and info["frames"] == 0


def test_a_pick_can_end_higher_than_travel_height(prim):
    """The red has to clear the success line; a held tilted cube missed it by 0.7 mm at travel height (e400)."""
    prim.reset(seed=0, options={"buried": True})
    xyz = np.array([0.20, 0.0, prim.action_space["xyz"].high[2]])
    lift = prim.travel_z + 0.03
    _, _, _, _, info = prim.step({"kind": "pick", "xyz": xyz, "angle": 0.3, "lift_z": lift})
    assert info["refused"] is None, info["refused"]
    end = prim.env.joint_state.copy()
    end[5] = primitives.GRIP_PLAN  # measure the jaw midpoint where IK solved it
    assert abs(primitives.tool_pose(prim.model, end)[0][2] - lift) < 0.004


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
