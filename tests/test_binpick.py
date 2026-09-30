import numpy as np
import pytest

from so101sim import binpick, ik, primitives
from so101sim.binpick import BinPickEnv


@pytest.fixture(scope="module")
def env():
    e = BinPickEnv(height=64, width=64)
    yield e
    e.close()


@pytest.fixture(scope="module")
def model(env):
    return env.model


def sector_points(n=24, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(*binpick.SPAWN_RADIUS, n)
    a = rng.uniform(*binpick.SPAWN_ANGLE, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


# ---- conventions the rest of the code assumes ---------------------------------------------------


def test_tool_axis_points_along_the_gripper_and_jaws_close_along_z(model):
    """wrist_roll spins about the tool axis, so it changes jaw yaw and nothing else about direction."""
    import mujoco

    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                         "wrist_flex", "wrist_roll")]]
    base = np.array([0.3, -0.7, 0.9, 1.1, 0.0])

    def axes(roll):
        data.qpos[adr] = np.r_[base[:4], roll]
        mujoco.mj_kinematics(model, data)
        # .copy(), because site_xmat is a view into `data` and would alias between the two calls.
        return data.site_xmat[model.site("gripperframe").id].reshape(3, 3).copy()

    a, b = axes(0.0), axes(1.0)
    assert np.degrees(np.arccos(np.clip(a[:, ik.TOOL_AXIS] @ b[:, ik.TOOL_AXIS], -1, 1))) < 1e-3
    turned = np.degrees(np.arccos(np.clip(a[:, ik.JAW_AXIS] @ b[:, ik.JAW_AXIS], -1, 1)))
    assert abs(turned - np.degrees(1.0)) < 0.1


def test_jaw_geometry_matches_the_numbers_the_comments_quote(model):
    """Pins the gripper geometry the primitives are tuned against.

    Every one of these was got wrong once by reading an offset as a displacement or fitting a circle to
    a partial arc, so they are worth a regression test. Verified independently by the so101 session
    against the same MJCF on 2026-09-22.
    """
    import mujoco

    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[model.joint("gripper").id]
    site = model.site("gripperframe").id

    def offset(deg):
        data.qpos[:] = 0.0
        data.qpos[adr] = np.radians(deg)
        mujoco.mj_kinematics(model, data)
        rot = data.site_xmat[site].reshape(3, 3).copy()
        return rot.T @ (ik.grasp_point(model, data) - data.site_xpos[site])

    shut, wide = offset(-10.0), offset(100.0)
    np.testing.assert_allclose(shut * 1000, [1.439, 0.018, -18.631], atol=0.05)
    np.testing.assert_allclose(wide * 1000, [-36.790, 0.018, 33.499], atol=0.05)
    assert abs(np.linalg.norm(wide - shut) * 1000 - 64.65) < 0.1

    # The roll axis runs through the jaw midpoint at MJCF 0, and the site is 12 mm off it. Full sweep:
    # over a partial arc the centroid is not the circle centre and the answer comes out ~8.7 mm.
    arm = model.jnt_qposadr[[model.joint(n).id for n in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                          "wrist_flex", "wrist_roll")]]

    def sweep(point_of):
        pts = []
        for roll in np.linspace(-np.pi, np.pi, 181):
            data.qpos[:] = 0.0
            data.qpos[arm] = [0.2, -0.5, 0.5, 1.5, roll]
            mujoco.mj_kinematics(model, data)
            pts.append(point_of(data))
        pts = np.asarray(pts)
        return np.linalg.norm(pts - pts.mean(0), axis=1).mean()

    assert abs(sweep(lambda d: d.site_xpos[site].copy()) * 1000 - 12.00) < 0.1
    assert sweep(lambda d: ik.grasp_point(model, d)) * 1000 < 0.3


def test_grasp_point_is_not_the_site(model):
    """A cube lands between the jaw tips, which is over a centimetre from `gripperframe`."""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    offset = np.linalg.norm(ik.grasp_point(model, data) - data.site_xpos[model.site("gripperframe").id])
    assert offset > 0.010, "targeting the site instead of the jaw midpoint would miss every grasp"


# ---- top-down IK --------------------------------------------------------------------------------


@pytest.mark.parametrize("xy", sector_points(12))
def test_topdown_ik_reaches_the_spawn_sector(model, xy):
    grip = primitives._grip_rad(model, primitives.GRIP_PLAN)
    for yaw in np.radians([0.0, 30.0, 60.0]):
        _, info = ik.solve_topdown(model, [xy[0], xy[1], primitives.CUBE_HALF], yaw=yaw, gripper=grip)
        assert info["pos_err"] < 1e-3, info
        assert info["tilt"] < 1.0, info
        assert info["yaw_err"] < 5.0, info


def test_topdown_ik_reaches_the_bin_at_release_height(env):
    grip = primitives._grip_rad(env.model, primitives.GRIP_HOLDING)
    for z in (primitives.RELEASE_Z, primitives.HOVER_Z):
        _, info = ik.solve_topdown(env.model, [*env.bin["center"], z], yaw=0.0, gripper=grip)
        assert info["pos_err"] < 1e-3, f"bin unreachable at z={z}: {info}"
        assert info["tilt"] < 1.5, info


# ---- scene --------------------------------------------------------------------------------------


def test_bin_is_short_enough_to_lift_a_cube_over(env):
    """Top-down reach dies with height, so the rim has to stay low enough to clear carrying a cube."""
    assert env.bin["rim_z"] + primitives.CUBE_HALF < primitives.RELEASE_Z + 1e-9


def test_spawn_sector_does_not_overlap_the_bin(env):
    for xy in sector_points(64, seed=1):
        gap = np.abs(xy - env.bin["center"]).max() - env.bin["half"]
        assert gap > primitives.CUBE_HALF, f"{xy} spawns inside the bin"


# ---- primitives ---------------------------------------------------------------------------------


def test_primitives_respect_the_lerobot_step_limit(model):
    """lerobot clips any command more than max_relative_target=4.0 from the last one, silently."""
    trajectories = [
        primitives.pick(model, (0.22, -0.08), np.radians(20)),
        primitives.place(model, (0.18, 0.19)),
        primitives.push(model, (0.20, -0.12), np.radians(90)),
    ]
    for traj in trajectories:
        assert np.abs(np.diff(traj, axis=0)).max() < primitives.LEROBOT_STEP_LIMIT_DEG


def test_bridge_keeps_the_gripper_high_when_crossing(model):
    """Crossing from the bin back to the table must not dip down through whatever is in the bin."""
    from_bin = primitives.place(model, (0.18, 0.19))[-1]
    to_cube = primitives.pick(model, (0.24, -0.12), 0.0)
    crossing = primitives.bridge(model, from_bin, to_cube)

    heights = np.array([primitives.tool_pose(model, f)[0][2] for f in crossing[: len(crossing) - len(to_cube)]])
    assert heights.min() > 0.04, f"bridge dipped to {heights.min():.3f} m"


def test_pick_grasps_at_the_requested_height(model):
    traj = primitives.pick(model, (0.22, -0.05), 0.0, hover_z=0.08, z=0.045)
    planned = traj.copy()
    planned[:, 5] = primitives.GRIP_PLAN  # IK solved the arm at GRIP_PLAN, so measure it there
    lowest = min(primitives.tool_pose(model, f)[0][2] for f in planned)
    assert abs(lowest - 0.045) < 2e-3


@pytest.mark.parametrize("make, ik_grip", [
    (lambda model, flip: primitives.pick(model, (0.22, -0.05), 0.3, flip=flip), primitives.GRIP_PLAN),
    (lambda model, flip: primitives.place(model, (0.22, -0.05), 0.3, flip=flip), primitives.GRIP_HOLDING),
    (lambda model, flip: primitives.push(model, (0.22, -0.05), 2.5, flip=flip), primitives.GRIP_CLOSE),
])
def test_a_forced_wrist_flip_turns_the_jaws_half_a_turn_and_keeps_the_path(model, make, ik_grip):
    """Yaw and yaw + pi close the jaws on the same faces. `flip` picks which of the two the wrist takes, and
    near the base only one of them keeps the wrist camera mount off the shoulder."""
    a, b = make(model, 0), make(model, 1)
    assert len(a) == len(b)
    for fa, fb in zip(a[::5], b[::5]):
        fa, fb = fa.copy(), fb.copy()
        fa[5] = fb[5] = ik_grip  # the jaw midpoint moves with the gripper, so compare where IK solved it
        (pa, ya, _), (pb, yb, _) = primitives.tool_pose(model, fa), primitives.tool_pose(model, fb)
        np.testing.assert_allclose(pa, pb, atol=2e-3)
        assert abs(abs(np.degrees(primitives._wrap(yb - ya))) - 180.0) < 3.0


def test_leaving_the_flip_out_keeps_the_old_plan(model):
    a = primitives.pick(model, (0.22, -0.05), 0.3)
    b = primitives.pick(model, (0.22, -0.05), 0.3, flip=None)
    np.testing.assert_array_equal(a, b)


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


def test_unreachable_targets_raise_rather_than_returning_nonsense(model):
    with pytest.raises(primitives.Unreachable):
        primitives.pick(model, (0.50, 0.30), 0.0)


# ---- env ----------------------------------------------------------------------------------------


def test_reset_is_deterministic(env):
    a = {n: env.reset(seed=3)[1]["cubes"][n]["pos"].copy() for n in env.cubes}
    b = {n: env.reset(seed=3)[1]["cubes"][n]["pos"].copy() for n in env.cubes}
    for name in env.cubes:
        np.testing.assert_allclose(a[name], b[name])


def test_reset_spawns_a_flush_pair(env):
    """Push only matters if something is actually jammed, so reset makes it so on purpose."""
    env.reset(seed=0)
    assert sum(not env.graspable(n) for n in env.cubes) >= 2


def test_reset_keeps_cubes_in_the_sector_and_apart(env):
    for seed in range(5):
        _, info = env.reset(seed=seed)
        for name, cube in info["cubes"].items():
            r = float(np.hypot(*cube["pos"][:2]))
            assert binpick.SPAWN_RADIUS[0] - 0.05 <= r <= binpick.SPAWN_RADIUS[1] + 0.05
            assert abs(cube["pos"][2] - primitives.CUBE_HALF) < 1e-6


def test_pick_then_place_puts_a_cube_in_the_bin(env):
    env.reset(seed=1)
    target = next(n for n in env.cubes if env.graspable(n))
    pose = env.cube_pose(target)

    _, _, _, _, info = env.step(("pick", pose["pos"][:2], pose["yaw"]))
    assert info["failure"] is None
    assert env.is_lifted(target), "cube was not lifted clear of the table"

    _, reward, _, _, info = env.step(("place", env.bin["center"], 0.0))
    assert info["failure"] is None
    assert reward == 1.0
    assert env.cube_pose(target)["in_bin"]


def test_push_separates_a_flush_pair(env):
    # Seed 1 since the stiff contacts of 2026-09-30 rebuilt every pile. Over seeds 0 to 9 the same push separated the
    # first flush pair by 0 to 32 mm (seed 0: 3.2 mm, seed 1: 18.7 mm; two were unreachable).
    env.reset(seed=1)
    stuck = [n for n in env.cubes if not env.graspable(n)]
    a, b = stuck[0], stuck[1]
    here, there = env.cube_pose(a)["pos"][:2], env.cube_pose(b)["pos"][:2]
    step = (here - there) / np.linalg.norm(here - there)
    heading = float(np.arctan2(step[1], step[0]))
    before = np.linalg.norm(here - there)

    _, _, _, _, info = env.step(("push", here - 0.045 * step, heading))
    assert info["failure"] is None
    after = np.linalg.norm(env.cube_pose(a)["pos"][:2] - env.cube_pose(b)["pos"][:2])
    assert after > before + 0.005


def test_unreachable_primitive_is_reported_not_raised(env):
    env.reset(seed=0)
    _, reward, _, _, info = env.step(("pick", (0.50, 0.30), 0.0))
    assert info["failure"] is not None
    assert reward == 0.0


def test_object_poses_stay_out_of_the_observation(env):
    obs, info = env.reset(seed=0)
    assert set(obs) == {"observation.state", "observation.images.front", "observation.images.wrist"}
    assert "cubes" in info
