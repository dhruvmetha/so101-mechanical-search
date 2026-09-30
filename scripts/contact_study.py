"""Contact stability study (dm1487, 2026-09-30: "make sure the contact dynamics are stable overall ... we should not
be penetrating"). Measures how far objects sink into each other and whether piles hold still, under the scene's
contact settings and stiffer variants. Variants change the compiled model's per-geom solref and solimp, so nothing
in models/ changes until one wins.

Scenarios, per variant and seed:
  rest   reset the pile, then hold the arm still for 10 s: deepest penetration by contact class, cube drift, speeds
  grasp  one cube alone on the floor, a centred pick: lifted or not, gripper reading, penetrations
  press  the same cube, a pick aimed 9 mm off centre each way along the jaw axis, so a finger lands on its top
  stack  a cube on top of another, pressed the same way
  pile   the reset pile, a centred pick on the highest cube

    python scripts/contact_study.py --variants base stiff --seeds 0 1 2 3 --procs 8 --out outputs/contact_study
"""
import so101sim  # noqa: F401  before mujoco

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from so101sim.mechsearch import MechSearchPrimitiveEnv

CUBE_HALF = 0.015
TWO_STEPS = 2 / 300  # MuJoCo raises any contact time constant below two timesteps to two timesteps
STIFF_IMP = (0.95, 0.99, 0.001, 0.5, 2.0)
STIFFER_IMP = (0.99, 0.999, 0.001, 0.5, 2.0)
HALF_STEP = 1 / 600

VARIANTS = {
    "base": {},
    "scene_tc": {"scene_solref": (TWO_STEPS, 1.0)},
    "scene_imp": {"scene_solimp": STIFF_IMP},
    "scene_tc_imp": {"scene_solref": (TWO_STEPS, 1.0), "scene_solimp": STIFF_IMP},
    "all_tc_imp": {"scene_solref": (TWO_STEPS, 1.0), "scene_solimp": STIFF_IMP,
                   "grip_solref": (TWO_STEPS, 1.0), "grip_solimp": STIFF_IMP},
    "all_tc_imp999": {"scene_solref": (TWO_STEPS, 1.0), "scene_solimp": STIFFER_IMP,
                      "grip_solref": (TWO_STEPS, 1.0), "grip_solimp": STIFFER_IMP},
    "all_600": {"timestep": HALF_STEP, "scene_solref": (2 * HALF_STEP, 1.0), "scene_solimp": STIFF_IMP,
                "grip_solref": (2 * HALF_STEP, 1.0), "grip_solimp": STIFF_IMP},
    "all_600_999": {"timestep": HALF_STEP, "scene_solref": (2 * HALF_STEP, 1.0), "scene_solimp": STIFFER_IMP,
                    "grip_solref": (2 * HALF_STEP, 1.0), "grip_solimp": STIFFER_IMP},
}


def apply(env, variant):
    """Set solref and solimp per geom. Scene geoms are everything outside the arm; gripper geoms are the arm's
    priority-1 geoms, whose parameters win every contact they make."""
    m = env.model
    arm = env.env._arm_ids
    if "timestep" in variant:
        m.opt.timestep = variant["timestep"]
        env.env._n_substeps = int(round((1 / 30) / variant["timestep"]))
    for g in range(m.ngeom):
        if g in arm:
            if m.geom_priority[g] > 0:
                if "grip_solref" in variant:
                    m.geom_solref[g] = variant["grip_solref"]
                if "grip_solimp" in variant:
                    m.geom_solimp[g] = variant["grip_solimp"]
            continue
        if "scene_solref" in variant:
            m.geom_solref[g] = variant["scene_solref"]
        if "scene_solimp" in variant:
            m.geom_solimp[g] = variant["scene_solimp"]


class Probe:
    """Deepest penetration per contact class over a stretch of simulation."""

    def __init__(self, env):
        self.env = env
        self.cubes = env.env._cube_geoms
        self.arm = env.env._arm_ids
        self.reset()

    def reset(self):
        self.deep = {"cube-cube": 0.0, "cube-env": 0.0, "cube-arm": 0.0, "arm-env": 0.0}

    def __call__(self, *_):
        d = self.env.data
        n = d.ncon
        if not n:
            return
        dist = d.contact.dist[:n]
        g = d.contact.geom[:n]
        for k in np.flatnonzero(dist < 0):
            a, b = int(g[k, 0]), int(g[k, 1])
            ca, cb = a in self.cubes, b in self.cubes
            aa, ab = a in self.arm, b in self.arm
            if ca and cb:
                key = "cube-cube"
            elif (ca and ab) or (cb and aa):
                key = "cube-arm"
            elif ca or cb:
                key = "cube-env"
            elif aa or ab:
                key = "arm-env"
            else:
                continue
            self.deep[key] = max(self.deep[key], float(-dist[k]))

    def mm(self):
        return {k: round(v * 1000, 2) for k, v in self.deep.items()}


def cube_xyz(env):
    e = env.env
    return {n: e.data.qpos[e._qadr[n]:e._qadr[n] + 3].copy() for n in e.cubes}


def cube_speeds(env):
    e = env.env
    return np.array([np.linalg.norm(e.data.qvel[e._dofadr[n]:e._dofadr[n] + 3]) for n in e.cubes])


def hold(env, seconds, probe=None):
    """Step physics with the arm's targets unchanged."""
    e = env.env
    for _ in range(int(round(seconds * 30))):
        mujoco.mj_step(e.model, e.data, nstep=e._n_substeps)
        if probe is not None:
            probe()


def clear_bin(env, keep):
    """Park every cube except those in `keep` on the ground far from the arm, then place the kept ones.
    keep: {name: (x, y, z, yaw)}. Parked cubes are outside the bin, so the spill check is off from here on."""
    e = env.env
    e._left_bin = lambda: None
    for i, n in enumerate(e.cubes):
        if n in keep:
            continue
        e._place(n, (1.2 + 0.05 * (i % 6), -0.3 + 0.05 * (i // 6)), -0.75 + CUBE_HALF + 0.001, 0.0)
    for n, (x, y, z, yaw) in keep.items():
        e._place(n, (x, y), z, yaw)
    mujoco.mj_forward(e.model, e.data)


def pick(env, xyz, yaw, probe):
    probe.reset()
    before = cube_xyz(env)
    _, _, term, _, info = env.step({"kind": "pick", "xyz": list(map(float, xyz)), "angle": float(yaw)})
    after = cube_xyz(env)
    lifted = [n for n in before if after[n][2] - before[n][2] > 0.02]
    return {"lifted": lifted, "grip": round(float(env.env.joint_state[5]), 1), "refused": info["refused"],
            "failure": info["failure"], "deep_mm": probe.mm(), "max_speed_after": None}


def scenario_rest(env, probe):
    t0 = time.process_time()
    start = cube_xyz(env)
    probe.reset()
    hold(env, 10.0, probe)
    end = cube_xyz(env)
    drift = np.array([np.linalg.norm(end[n] - start[n]) for n in start]) * 1000
    return {"deep_mm": probe.mm(), "drift_mm_median": round(float(np.median(drift)), 2),
            "drift_mm_max": round(float(drift.max()), 2), "speed_mm_s_max_end": round(float(cube_speeds(env).max()) * 1000, 2),
            "cpu_s_per_sim_s": round((time.process_time() - t0) / 10.0, 3)}


def scenario_single(env, probe, offsets, stack):
    """One cube (two when stacked) at the bin's centre, picked at each offset along the jaw axis."""
    e = env.env
    c, f = e.bin["center"], e.bin["floor_z"]
    names = list(e.cubes)
    out = []
    for off in offsets:
        keep = {names[0]: (c[0], c[1], f + CUBE_HALF + 0.0005, 0.0)}
        top = f + CUBE_HALF
        if stack:
            keep[names[1]] = (c[0], c[1], f + 3 * CUBE_HALF + 0.001, 0.0)
            top = f + 3 * CUBE_HALF
        clear_bin(env, keep)
        hold(env, 1.0)
        probe.reset()
        hold(env, 0.5, probe)
        rest_deep = probe.mm()
        r = pick(env, (c[0] + off, c[1], top), 0.0, probe)
        speeds = []
        for _ in range(30):
            hold(env, 1 / 30)
            speeds.append(cube_speeds(env).max())
        r.update(offset_mm=round(off * 1000, 1), rest_deep_mm=rest_deep,
                 max_speed_after=round(float(max(speeds)) * 1000, 1))
        out.append(r)
    return out


def scenario_drop(env, probe):
    """A lone cube dropped flat from 5 cm onto the bin floor: highest rebound and deepest penetration."""
    e = env.env
    c, f = e.bin["center"], e.bin["floor_z"]
    name = list(e.cubes)[0]
    clear_bin(env, {name: (c[0] + 0.03, c[1] + 0.03, f + CUBE_HALF + 0.05, 0.3)})
    probe.reset()
    zs, landed = [], False
    for _ in range(60):
        hold(env, 1 / 30, probe)
        z = e.data.qpos[e._qadr[name] + 2] - (f + CUBE_HALF)
        landed = landed or z < 0.002
        if landed:
            zs.append(z)
    return {"rebound_mm": round(max(zs) * 1000, 2) if zs else None, "deep_mm": probe.mm()}


def scenario_sweep(env, probe, seed):
    """A fresh pile, one sweep from the rim across the middle; did anything leave the bin?"""
    env.reset(seed=seed)
    e = env.env
    c, h = e.bin["center"], e.bin["half"]
    probe.reset()
    start = (c[0] - h + 0.02, c[1], e.bin["rim_z"] + 0.008)
    _, _, term, _, info = env.step({"kind": "sweep", "xyz": list(start), "angle": 0.0, "stroke": 0.12})
    speeds = []
    for _ in range(30):
        hold(env, 1 / 30)
        speeds.append(cube_speeds(env).max())
    return {"refused": info["refused"], "failure": info["failure"], "deep_mm": probe.mm(),
            "max_speed_after": round(float(max(speeds)) * 1000, 1)}


def scenario_pile(env, probe):
    e = env.env
    xyz = cube_xyz(env)
    name = max(xyz, key=lambda n: xyz[n][2] if abs(xyz[n][0] - e.bin["center"][0]) < e.bin["half"] else -9)
    q = e.data.qpos[e._qadr[name] + 3:e._qadr[name] + 7]
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    r = pick(env, xyz[name], yaw, probe)
    r["cube"] = name
    return r


def run(job):
    vname, seed = job
    env = MechSearchPrimitiveEnv()
    apply(env, VARIANTS[vname])
    probe = Probe(env)
    env.on_frame = probe
    t0 = time.time()
    try:
        env.reset(seed=seed)
    except RuntimeError as exc:
        env.close()
        return {"variant": vname, "seed": seed, "reset_error": str(exc)}
    res = {"variant": vname, "seed": seed, "reset_s": round(time.time() - t0, 1),
           "reset_attempts": env.env._reset_info["reset_attempts"]}
    res["rest"] = scenario_rest(env, probe)
    res["pile"] = scenario_pile(env, probe)
    res["grasp"] = scenario_single(env, probe, [0.0], stack=False)[0]
    res["press"] = scenario_single(env, probe, [-0.009, 0.009], stack=False)
    res["stack"] = scenario_single(env, probe, [-0.009, 0.009], stack=True)
    res["drop"] = scenario_drop(env, probe)
    t0 = time.time()
    res["sweep"] = scenario_sweep(env, probe, seed)
    res["sweep_reset_s"] = round(time.time() - t0, 1)
    env.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="outputs/contact_study_2026-09-30")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s) for v in a.variants for s in a.seeds]
    with ProcessPoolExecutor(a.procs) as pool:
        results = list(pool.map(run, jobs))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    for v in a.variants:
        rows = [r for r in results if r["variant"] == v and "rest" in r]
        errs = [r for r in results if r["variant"] == v and "reset_error" in r]
        if not rows:
            print(v, "no resets", errs[:1])
            continue
        def worst(key, cls):
            vals = []
            for r in rows:
                x = r[key]
                for item in (x if isinstance(x, list) else [x]):
                    vals.append(item["deep_mm"][cls])
            return max(vals)
        print(f"{v:14s} resets ok {len(rows)}/{len(rows) + len(errs)}, attempts {[r['reset_attempts'] for r in rows]}, "
              f"reset s {np.median([r['reset_s'] for r in rows]):.0f}")
        print(f"   rest: deep cube-cube {max(r['rest']['deep_mm']['cube-cube'] for r in rows)} "
              f"cube-env {max(r['rest']['deep_mm']['cube-env'] for r in rows)} mm; drift median "
              f"{np.median([r['rest']['drift_mm_median'] for r in rows]):.2f} max {max(r['rest']['drift_mm_max'] for r in rows):.1f} mm; "
              f"end speed max {max(r['rest']['speed_mm_s_max_end'] for r in rows):.1f} mm/s; cpu/sim s "
              f"{np.median([r['rest']['cpu_s_per_sim_s'] for r in rows]):.2f}")
        print(f"   drop : rebound max {max(r['drop']['rebound_mm'] or 0 for r in rows)} mm, deepest cube-env "
              f"{max(r['drop']['deep_mm']['cube-env'] for r in rows)} mm")
        print(f"   sweep: spills {sum(bool(r['sweep']['failure']) for r in rows)}/{len(rows)}, refused "
              f"{sum(bool(r['sweep']['refused']) for r in rows)}, deepest cube-cube "
              f"{max(r['sweep']['deep_mm']['cube-cube'] for r in rows)} cube-arm {max(r['sweep']['deep_mm']['cube-arm'] for r in rows)} mm")
        for key in ("grasp", "press", "stack", "pile"):
            print(f"   {key:5s}: deepest cube-env {worst(key, 'cube-env')} cube-cube {worst(key, 'cube-cube')} "
                  f"cube-arm {worst(key, 'cube-arm')} mm", end="")
            items = [i for r in rows for i in (r[key] if isinstance(r[key], list) else [r[key]])]
            print(f"; lifted {sum(bool(i['lifted']) for i in items)}/{len(items)}, grips "
                  f"{sorted(set(i['grip'] for i in items))[:6]}, max speed after "
                  f"{max((i['max_speed_after'] or 0) for i in items)} mm/s")


if __name__ == "__main__":
    main()
