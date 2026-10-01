"""How hard does the arm hit things, and how hard do the objects get hit? (2026-10-01, dm1487 after the b5142 video:
"Why are you crashing into objects and then causing them to move? Thats damaging to the robot no", "Or the objects".)

Replays an episode from its record and, after every physics substep, reads every contact's normal force and the
actuator forces. Per op it reports:
  arm-object   peak normal force between any arm geom and a cube, and which cube
  arm-bin      peak force between the arm and the bin or floor
  object       peak force on any cube from anything (cube, bin, arm), with its resting weight for scale
  servo        share of substeps where an arm joint's actuator sits at 95% or more of its 2.94 N m limit, and
               the same for the gripper (holding a cube keeps it there by design)
  speed        the gripper's speed at the moment it first touches a cube it wasn't touching

    python demonstrator/experiments/contact_load.py b5142 b5000 b5002
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402  (imports so101sim before mujoco)
import mujoco  # noqa: E402

LIMIT = 2.94


def sets(m):
    """Arm geoms that collide, cube geom names, and the gripper body, from the model."""
    root = m.body_rootid[m.body("gripper").id]
    arm = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root
           and (m.geom_contype[g] or m.geom_conaffinity[g])}
    cubes = {g: m.geom(g).name for g in range(m.ngeom) if m.geom(g).name.startswith("cube_")}
    return arm, cubes, m.body("gripper").id


def measure(name):
    rec = {}
    state = {"op": -1, "on": False, "prev": set(), "sets": None}
    real_step = mujoco.mj_step
    f6, vel = np.zeros(6), np.zeros(6)

    def on_advance(ep, row):
        state["op"], state["on"] = ep.op_index, True

    def step(m, d, nstep=1):
        if state["sets"] is None:
            state["sets"] = sets(m)
        arm_geoms, cube_names, grip_body = state["sets"]
        arm_dofs = [m.jnt_dofadr[m.joint(j).id] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                          "wrist_roll")]
        for _ in range(nstep):
            real_step(m, d)
            if not state["on"]:
                continue  # reset and pile building
            r = rec.setdefault(state["op"], {"arm_obj": (0.0, None), "arm_bin": 0.0, "obj": (0.0, None), "servo": 0,
                                             "grip_sat": 0, "n": 0, "speed": 0.0, "over10": 0, "joint": 0.0})
            r["n"] += 1
            touching = set()
            for i in range(d.ncon):
                c = d.contact[i]
                mujoco.mj_contactForce(m, d, i, f6)
                fn = float(f6[0])
                a, b = c.geom1, c.geom2
                arm_a, arm_b = a in arm_geoms, b in arm_geoms
                cube_a, cube_b = cube_names.get(a), cube_names.get(b)
                if (arm_a and cube_b) or (arm_b and cube_a):
                    cube = cube_b if arm_a else cube_a
                    touching.add(cube)
                    if fn > r["arm_obj"][0]:
                        r["arm_obj"] = (fn, cube)
                elif arm_a or arm_b:
                    if not (arm_a and arm_b):
                        r["arm_bin"] = max(r["arm_bin"], fn)
                for cube in (cube_a, cube_b):
                    if cube and fn > r["obj"][0]:
                        r["obj"] = (fn, cube)
            peak_now = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                if (c.geom1 in arm_geoms) != (c.geom2 in arm_geoms) and \
                        (cube_names.get(c.geom1) or cube_names.get(c.geom2)):
                    mujoco.mj_contactForce(m, d, i, f6)
                    peak_now = max(peak_now, float(f6[0]))
            r["over10"] += int(peak_now > 10.0)
            # Torque the contacts put on the five arm joints (constraint forces on their dofs).
            r["joint"] = max(r["joint"], float(np.abs(d.qfrc_constraint[arm_dofs]).max()))
            if touching - state["prev"]:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, grip_body, vel, 0)
                r["speed"] = max(r["speed"], float(np.linalg.norm(vel[3:])))
            state["prev"] = touching
            af = np.abs(d.actuator_force)
            r["servo"] += int((af[:5] >= 0.95 * LIMIT).any())
            r["grip_sat"] += int(af[5] >= 0.95 * LIMIT)

    mujoco.mj_step = step
    try:
        ep = tools.Episode(name, size=64, on_advance=on_advance)
        weights = [float(ep.m.body_mass[ep.m.geom_bodyid[ep.m.geom(n).id]] * 9.81) for n in ep.env.cubes]
        ops = ep.rec["ops"]
        ep.close()
    finally:
        mujoco.mj_step = real_step
    print(f"\n{name}: cube weight {min(weights):.2f} to {max(weights):.2f} N")
    print(" op  kind    arm-object N (cube)  over 10 N ms  contact torque N m  object N (cube)  touch speed m/s")
    for k, op in enumerate(ops):
        r = rec.get(k)
        if r is None or op["op"] == "park":
            continue
        n = max(r["n"], 1)
        ao, aoc = r["arm_obj"]
        ob, obc = r["obj"]
        print(f"{k:3d}  {op['kind']:6s}  {ao:8.1f} ({tools.short(aoc) if aoc else '-':>4})  {r['over10'] / 0.6:11.0f}"
              f"  {r['joint']:18.2f}  {ob:8.1f} ({tools.short(obc) if obc else '-':>4})  {r['speed']:14.3f}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        measure(name)
