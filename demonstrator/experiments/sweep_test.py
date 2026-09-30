"""Sweep test on buried piles: long straight strokes of the closed jaws across the red's spot. Throwaway.

For each seed and each candidate sweep: reset, unpark the way tools.py does, run the stroke, park, settle, and
score with ground truth. Candidates: four headings, and two kinds of stroke:
  rim   start at the wall on the far side from where the stroke heads, jaws 8 mm above the rim, 12 cm inward
  mid   start 5 cm before the red, jaw tips at layer-2 centre height, 9 cm long
"""
import json
import multiprocessing as mp
import sys

import numpy as np

REPO = "/common/home/dm1487/robotics_research/rearrange"
OUT = "/common/home/dm1487/robotics_research/rearrange/outputs/sweep_pilot_2026-09-30"


def setup(seed, nudge=0.0):
    sys.path.insert(0, REPO + "/demonstrator")
    import so101sim  # noqa
    import mujoco
    from so101sim.mechsearch import MechSearchPrimitiveEnv
    prim = MechSearchPrimitiveEnv(height=64, width=64)
    prim.reset(seed=seed, options={"buried": True})
    if nudge:
        env = prim.env
        n = env.grays[0]
        a = env.model.jnt_qposadr[env.model.body(n).jntadr[0]]
        env.data.qpos[a] += nudge
        mujoco.mj_forward(env.model, env.data)
    return prim


def red_state(env):
    from so101sim.mechsearch import TARGET
    red = env.cube_pose(TARGET)["pos"]
    over = side = 0
    for n in env.grays:
        p = env.cube_pose(n)["pos"]
        if p[2] > -0.005:
            continue
        d = p - red
        h = np.linalg.norm(d[:2])
        if d[2] > 0.015 and h < 0.036:
            over += 1
        elif abs(d[2]) < 0.015 and h < 0.0375:
            side += 1
    return over, side


def red_pixels(prim, cams):
    from so101sim.mechsearch import TARGET
    m, d = prim.model, prim.env.data
    g = m.geom(TARGET).id
    return [int((cam.capture(d)[2] == g).sum()) for cam in cams]


def unpark(prim):
    from so101sim import primitives
    env = prim.env
    c = env.bin["center"]
    wp = primitives._Waypoint([c[0], c[1], prim.travel_z], 0.0, float(env.joint_state[5]), hold=5)
    traj = primitives._retime(*primitives._trace(prim.model, [wp], primitives.GRIP_PLAN))
    for row in primitives.lead_in(env.joint_state, traj):
        env.advance(row)


def park(prim):
    from so101sim import primitives, rest
    env = prim.env
    home = np.asarray(env.layout.home, dtype=np.float32)
    for row in primitives.lead_in(env.joint_state, home[None, :]):
        env.advance(row)
    hold = env.joint_state.copy()
    rest.come_to_rest(lambda: env.advance(hold), env.data, rest.free_dofs(prim.model, env.cubes))


def candidates(prim):
    from so101sim.mechsearch import TARGET
    env = prim.env
    red = env.cube_pose(TARGET)["pos"][:2]
    c, h = np.asarray(env.bin["center"]), env.bin["half"]
    rim = env.bin["rim_z"]
    out = []
    for deg in (0, 90, 180, 270):
        u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        # rim: start where the line through the red, heading u, meets the wall behind it, 5 mm inside
        t = min((h - 0.005 - s * (red[k] - c[k])) / (s * -u[k]) for k in range(2) for s in (1, -1)
                if s * -u[k] > 1e-9)
        start = red - t * u
        out.append({"kind": "rim", "deg": deg, "start": start.tolist(), "stroke": 0.12, "z": rim + 0.008})
        out.append({"kind": "mid", "deg": deg, "start": (red - 0.05 * u).tolist(), "stroke": 0.09,
                    "z": env.bin["floor_z"] + 0.045})
    return out


def run(args):
    seed, cand, nudge = args
    sys.path.insert(0, REPO + "/demonstrator")
    prim = setup(seed, nudge)
    from so101sim import primitives
    from reach import Clearance
    import mujoco
    env, m = prim.env, prim.model
    C = Clearance(m)
    before = {n: env.cube_pose(n)["pos"].copy() for n in env.cubes}
    state0 = red_state(env)
    unpark(prim)
    # both wrist rolls; keep the one that stays farther from the arm
    best = None
    probe = mujoco.MjData(m)
    adr = m.jnt_qposadr[[m.joint(n).id for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                  "wrist_roll", "gripper")]]
    from so101sim import units
    grip_range = tuple(m.jnt_range[m.joint("gripper").id])
    for flip in (0, 1):
        try:
            traj = primitives.push(m, cand["start"], np.radians(cand["deg"]), stroke=cand["stroke"],
                                   hover_z=prim.travel_z, push_z=cand["z"], flip=flip)
        except primitives.Unreachable:
            continue
        worst = 1.0
        for row in traj[::3]:
            probe.qpos[adr] = units.to_rad(np.asarray(row, float), grip_range)
            mujoco.mj_kinematics(m, probe)
            worst = min(worst, C.measure(probe)[0])
        if best is None or worst > best[0]:
            best = (worst, flip, traj)
    if best is None:
        prim.close()
        return {**cand, "seed": seed, "nudge": nudge, "reachable": False}
    worst, flip, traj = best
    traj = primitives.bridge(m, env.joint_state, traj, hover_z=prim.travel_z)
    failure = None
    for row in traj:
        _, term, _, info = env.advance(row)
        if term:
            failure = info.get("failure")
            break
    if failure is None:
        park(prim)
    after = {n: env.cube_pose(n)["pos"].copy() for n in env.cubes}
    moved = sorted(float(np.linalg.norm(after[n] - before[n])) for n in env.cubes)
    res = {**cand, "seed": seed, "nudge": nudge, "reachable": True, "flip": flip, "self_mm": round(worst * 1000, 1),
           "failure": failure, "over_side_before": state0,
           "over_side_after": red_state(env) if failure is None else None,
           "cubes_moved": sum(1 for x in moved if x > 0.003), "max_move_mm": round(moved[-1] * 1000, 1)}
    prim.close()
    return res


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in sys.argv[1:]] or [1006, 1012]
    jobs = []
    for seed in seeds:
        prim = setup(seed)
        for cand in candidates(prim):
            jobs.append((seed, cand, 0.0))
        prim.close()
    with mp.get_context("spawn").Pool(16) as pool:
        rows = pool.map(run, jobs)
    json.dump(rows, open(f"{OUT}/sweep_rows.json", "w"), indent=1)
    for r in rows:
        if not r["reachable"]:
            print(f"seed {r['seed']} {r['kind']:4s} {r['deg']:3d}: unreachable")
            continue
        print(f"seed {r['seed']} {r['kind']:4s} {r['deg']:3d}: over/side {r['over_side_before']} -> "
              f"{r['over_side_after']}, moved {r['cubes_moved']:2d} (max {r['max_move_mm']:5.1f} mm), "
              f"arm to itself {r['self_mm']:5.1f} mm, failure {r['failure']}")
