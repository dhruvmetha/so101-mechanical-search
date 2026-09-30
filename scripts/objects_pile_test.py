"""Drop scanned toys and household objects into a deep bin and measure the pile, against the 30-cube search pile.

Written 2026-09-30 for docs/reports/2026-09-30-realistic-objects.md. A feasibility test, not a task env.

It writes a test scene, models/objects_test_scene.xml, which includes the shared stage and a generated objects file:
the search task's table, riser and front camera, a bin with the same 22 cm footprint at x = 0.20 but 12 cm walls, a
camera straight above the bin, and 20 Google Scanned Objects from kevinzakka/mujoco_scanned_objects, each a free
body with its textured visual mesh and its convex hulls. Mesh paths are absolute, into /common/users/dm1487/assets,
because stage.xml has already pointed meshdir at the arm. With --hulls coacd it uses hulls from
scripts/objects_coacd.py instead and writes models/objects_test_coacd_scene.xml.

The GSO data has no masses. The visual mesh carries all of the mass at DENSITY and the hulls none, so the inertia
follows the scanned shape. The shipped model.xml files instead give every geom 1000 kg/m^3, and with the visual mesh
and 32 overlapping hulls all counted, that comes to about 4 times what 1000 kg/m^3 over the scanned volume gives.

Per seed: the arm waits swung aside (PARKED), because a full 12 cm bin reaches the jaws at the search start pose.
Objects drop in waves, as MechSearchEnv drops cubes, each with a random 3D orientation and its bounding sphere clear of
the walls and of the others in its wave. Then it counts frames until
so101sim.rest's rule holds (every object under 2 mm/s and 0.1 rad/s for 15 frames), allowing 10 s instead of rest's 3;
then it idles 10 s and records how far each object's centre moved and how long each physics step took. The cube
baseline runs MechSearchEnv.reset on the same seed, then the same idle measurement.

    python scripts/objects_pile_test.py --seeds 0 1 2 3 4 --render 0
    python scripts/objects_pile_test.py --seeds 0 1 2 3 4 --sleep 1e-3     # MuJoCo's sleeping islands, tolerance 1e-3
    python scripts/objects_pile_test.py --seeds 0 1 2 3 4 --noslip 0       # noslip off
    python scripts/objects_pile_test.py --seeds 0 1 2 3 4 --hulls coacd
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import so101sim  # noqa: F401  sets MUJOCO_GL before mujoco loads
import mujoco

from so101sim import SEARCH_SCENE_PATH, rest, units

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
GSO = Path("/common/users/dm1487/assets/mujoco_scanned_objects/models")
COACD = Path("/common/users/dm1487/assets/gso_coacd")
OUT = ROOT / "outputs" / "objects_2026-09-30"

# 15 toys and 5 household objects from the 80 GSO objects scripts/objects_catalog.py says fit the gripper.
OBJECTS = [
    "Great_Dinos_Triceratops_Toy", "Schleich_Lion_Action_Figure", "Schleich_Bald_Eagle",
    "Nintendo_Mario_Action_Figure", "PEEKABOO_ROLLER", "SCHOOL_BUS",
    "Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj", "Thomas_Friends_Wooden_Railway_Porter_5JzRhMm3a9o",
    "WHALE_WHISTLE_6PCS_SET", "LADYBUG_BEAD", "Toysmith_Windem_Up_Flippin_Animals_Dog",
    "Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi", "Android_Figure_Orange", "3D_Dollhouse_Happy_Brother",
    "Kong_Puppy_Teething_Rubber_Small_Pink", "CoQ10", "YumYum_D3_Liquid",
    "Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz", "Crayola_Crayons_24_count", "Cole_Hardware_Mini_Honey_Dipper",
]

DENSITY = 500  # kg/m^3 over the scanned volume, between hollow packaging and solid PVC figures
BIN_CENTER = np.array([0.20, 0.0])
BIN_HALF = 0.11  # inner half width, the search bin's
FLOOR_Z = -0.096  # top of the bin floor
WALL_HEIGHT = 0.12
HOME = np.array([0.0, -30.0, 30.0, 90.0, 0.0, 50.0])  # the search start pose, lerobot units
# HOME holds the jaws about 5 cm above a 12 cm rim, so a full pile reaches them. The arm waits swung aside, and moves to
# HOME without stepping only for the wrist camera picture.
PARKED = np.array([90.0, -30.0, 30.0, 90.0, 0.0, 50.0])

WAVE = 3
DROP_GAP = 0.01
WALL_CLEAR = 0.003
SETTLE_SPEED = 0.05  # m/s, between waves
SETTLE_SECONDS = 1.5
REST_LIMIT_SECONDS = 10.0
IDLE_SECONDS = 10.0
FRAME_HZ = 30


# ---- scene --------------------------------------------------------------------------------------------------------

def hull_files(name, hulls):
    if hulls == "vhacd":
        return sorted((GSO / name).glob("model_collision_*.obj"), key=lambda p: int(p.stem.split("_")[-1]))
    return sorted((COACD / name).glob("hull_*.obj"), key=lambda p: int(p.stem.split("_")[-1]))


def objects_xml(names, hulls):
    t = WALL_HEIGHT / 2
    asset, bodies = [], []
    for i, name in enumerate(names):
        folder = GSO / name
        asset.append(f'    <texture type="2d" name="o{i}_tex" file="{folder / "texture.png"}"/>')
        asset.append(f'    <material name="o{i}_mat" texture="o{i}_tex" specular="0.2" shininess="0.2"/>')
        asset.append(f'    <mesh name="o{i}_vis" file="{folder / "model.obj"}"/>')
        geoms = [f'      <geom name="o{i}_vis" class="gso_visual" mesh="o{i}_vis" material="o{i}_mat"/>']
        for k, h in enumerate(hull_files(name, hulls)):
            asset.append(f'    <mesh name="o{i}_c{k}" file="{h}"/>')
            geoms.append(f'      <geom name="o{i}_c{k}" class="gso_collision" mesh="o{i}_c{k}"/>')
        x = 1.2 + 0.2 * (i % 5)
        y = -0.4 + 0.2 * (i // 5)
        bodies.append(f'    <body name="obj{i}" pos="{x:.2f} {y:.2f} -0.74">\n      <freejoint name="obj{i}"/>\n'
                      + "\n".join(geoms) + "\n    </body>")
    return f"""<mujoco>
  <!-- Generated by scripts/objects_pile_test.py for a feasibility test, not a task. Hulls: {hulls}.
       Objects, in order obj0, obj1, ...: {", ".join(names)} -->
  <size memory="200M"/>
  <default>
    <default class="gso_visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" density="{DENSITY}"/>
    </default>
    <default class="gso_collision">
      <geom type="mesh" group="3" density="0" condim="4" friction="0.8 0.005 0.0001" rgba="0.5 0.6 0.8 1"/>
    </default>
  </default>
  <asset>
{chr(10).join(asset)}
  </asset>
  <worldbody>
    <geom name="table" type="box" size="0.35 0.45 0.02" pos="0.2 0 -0.12" material="table"/>
    <geom name="riser" type="box" size="0.06 0.06 0.05" pos="0.01 0 -0.05" rgba="0.45 0.35 0.25 1"/>
    <body name="front_cam_target" pos="0.2 0 -0.096"/>
    <camera name="front" mode="targetbody" target="front_cam_target" pos="0.7 0 0.45" fovy="50"/>
    <camera name="top" pos="0.2 0 0.55" xyaxes="0 -1 0 1 0 0" fovy="35"/>
    <body name="bin" pos="0.2 0 -0.1">
      <geom name="bin_floor" class="bin_wall" size="0.114 0.114 0.002" pos="0 0 0.002"/>
      <geom name="bin_wall_px" class="bin_wall" size="0.002 0.114 {t}" pos="0.112 0 {0.004 + t}"/>
      <geom name="bin_wall_nx" class="bin_wall" size="0.002 0.114 {t}" pos="-0.112 0 {0.004 + t}"/>
      <geom name="bin_wall_py" class="bin_wall" size="0.114 0.002 {t}" pos="0 0.112 {0.004 + t}"/>
      <geom name="bin_wall_ny" class="bin_wall" size="0.114 0.002 {t}" pos="0 -0.112 {0.004 + t}"/>
    </body>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""


def write_scene(names, hulls):
    tag = "" if hulls == "vhacd" else f"_{hulls}"
    objects = MODELS / f"objects_test{tag}_objects.xml"
    scene = MODELS / f"objects_test{tag}_scene.xml"
    objects.write_text(objects_xml(names, hulls))
    scene.write_text(f"""<mujoco model="so101_objects_test">
  <!-- Generated by scripts/objects_pile_test.py: the shared stage, then the test objects. -->
  <include file="stage.xml"/>
  <include file="{objects.name}"/>
</mujoco>
""")
    return scene


# ---- helpers ------------------------------------------------------------------------------------------------------

class Objects:
    """Per-object addresses and the collision vertices in each body's frame."""

    def __init__(self, model, n):
        self.names = [f"obj{i}" for i in range(n)]
        self.bodies = np.array([model.body(b).id for b in self.names])
        self.qadr = np.array([model.jnt_qposadr[model.body_jntadr[b]] for b in self.bodies])
        self.dofs = rest.free_dofs(model, self.names)
        self.verts = []
        for b in self.bodies:
            pts = []
            for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
                if model.geom_contype[g] == 0:
                    continue
                m = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, model.geom_quat[g])
                pts.append(v @ rot.reshape(3, 3).T + model.geom_pos[g])
            self.verts.append(np.concatenate(pts))
        self.center = [0.5 * (v.min(0) + v.max(0)) for v in self.verts]
        self.radius = [float(np.linalg.norm(v - c, axis=1).max()) for v, c in zip(self.verts, self.center)]

    def centres(self, data):
        return np.array([data.xpos[b] + data.xmat[b].reshape(3, 3) @ c for b, c in zip(self.bodies, self.center)])

    def top(self, data, idx):
        return max(float((self.verts[i] @ data.xmat[self.bodies[i]].reshape(3, 3).T
                          + data.xpos[self.bodies[i]])[:, 2].max()) for i in idx)

    def speeds(self, data):
        return np.array([np.linalg.norm(data.qvel[a:a + 3]) for a in self.dofs])


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def set_arm(model, data, pose=PARKED):
    ids = [model.joint(n).id for n in units.JOINT_NAMES]
    q = units.to_rad(pose, tuple(model.jnt_range[ids[5]]))
    data.qpos[model.jnt_qposadr[ids]] = q
    data.ctrl[:] = q


def load(scene, sleep, noslip):
    """`sleep` is False, or the sleep tolerance to turn MuJoCo's sleeping islands on with."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    if noslip is not None:
        model.opt.noslip_iterations = noslip
    if sleep:
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        model.opt.sleep_tolerance = sleep
    return model


def idle(model, data, dofs_centres, seconds=IDLE_SECONDS):
    """Step `seconds` with nothing touching the pile. Returns (per-object drift in m, ms per physics step, contacts)."""
    centres, bodies = dofs_centres
    before, rot_before = centres(data), [data.xmat[b].reshape(3, 3).copy() for b in bodies]
    n = int(round(seconds * FRAME_HZ))
    sub = int(round(1 / FRAME_HZ / model.opt.timestep))
    ncon = []
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data, nstep=sub)
        ncon.append(data.ncon)
    ms = (time.perf_counter() - t0) / (n * sub) * 1e3
    drift = np.linalg.norm(centres(data) - before, axis=1)
    turn = [np.degrees(np.arccos(np.clip((np.trace(r0.T @ data.xmat[b].reshape(3, 3)) - 1) / 2, -1, 1)))
            for r0, b in zip(rot_before, bodies)]
    return drift, np.array(turn), ms, float(np.mean(ncon))


# ---- one pile -----------------------------------------------------------------------------------------------------

def build_pile(scene, seed, sleep=0.0, noslip=None, n_objects=len(OBJECTS)):
    """Drop the objects into the bin and wait for rest's rule. Returns (model, data, Objects, stats)."""
    model = load(scene, sleep, noslip)
    data = mujoco.MjData(model)
    obj = Objects(model, n_objects)
    rng = np.random.default_rng(seed)
    sub = int(round(1 / FRAME_HZ / model.opt.timestep))
    set_arm(model, data)
    mujoco.mj_forward(model, data)
    root = model.body_rootid[model.body("gripper").id]
    arm = {b for b in range(model.nbody) if model.body_rootid[b] == root}

    touched = [False]
    objs = set(obj.bodies.tolist())
    physics = [0.0, 0]  # seconds inside mj_step, steps taken

    def frame():
        t = time.perf_counter()
        mujoco.mj_step(model, data, nstep=sub)
        physics[0] += time.perf_counter() - t
        physics[1] += sub
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (b1 in arm and b2 in objs) or (b2 in arm and b1 in objs):
                touched[0] = True

    queue = list(rng.permutation(n_objects))
    dropped, frames, t0 = [], 0, time.perf_counter()
    while queue:
        wave, spots = [], []
        for i in list(queue):
            if len(wave) == WAVE:
                break
            r = obj.radius[i]
            reach = max(BIN_HALF - r - WALL_CLEAR, 0.0)
            for _ in range(50):
                xy = BIN_CENTER + rng.uniform(-reach, reach, 2)
                if all(np.linalg.norm(xy - s) >= r + obj.radius[j] + 0.005 for j, s in spots):
                    break
            else:
                continue
            spots.append((i, xy))
            wave.append(i)
            queue.remove(i)
        if not wave:  # nothing fits beside anything: drop the next one alone
            wave, queue = [queue[0]], queue[1:]
            spots = [(wave[0], BIN_CENTER.copy())]
        base = obj.top(data, dropped) if dropped else FLOOR_Z
        z = base + DROP_GAP + max(obj.radius[i] for i in wave)
        for i, xy in spots:
            q = random_quat(rng)
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, q)
            pos = np.array([xy[0], xy[1], z]) - rot.reshape(3, 3) @ obj.center[i]
            a, d = obj.qadr[i], obj.dofs[i]
            data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = pos, q
            data.qvel[d:d + 6] = 0.0
            dropped.append(i)
        mujoco.mj_forward(model, data)
        for _ in range(int(SETTLE_SECONDS * FRAME_HZ)):
            frame()
            frames += 1
            if obj.speeds(data)[dropped].max() < SETTLE_SPEED:
                break
    drop_s = time.perf_counter() - t0
    drop_frames = frames

    # Frames until rest's rule holds, with a longer limit than rest.REST_SECONDS.
    run, rest_frames = 0, None
    physics[:] = [0.0, 0]
    for k in range(int(REST_LIMIT_SECONDS * FRAME_HZ)):
        frame()
        run = run + 1 if rest.still(data, obj.dofs) else 0
        if run >= rest.REST_FRAMES:
            rest_frames = k + 1
            break
    stats = dict(drop_frames=drop_frames, drop_wall_s=round(drop_s, 1), rest_frames=rest_frames,
                 settle_ms_per_step=round(physics[0] / physics[1] * 1e3, 3), touched_arm=touched[0])
    return model, data, obj, stats


def run_objects(args):
    scene, seed, sleep, noslip, n_objects, render = args
    model, data, obj, stats = build_pile(scene, seed, sleep, noslip, n_objects)
    rest_frames = stats["rest_frames"]
    c = obj.centres(data)
    spilled = int(np.sum((np.abs(c[:, :2] - BIN_CENTER).max(1) > BIN_HALF) | (c[:, 2] < FLOOR_Z - 0.01)))
    pile_top = obj.top(data, range(n_objects)) - FLOOR_Z
    drift, turn, ms, ncon = idle(model, data, (obj.centres, obj.bodies))
    asleep = int(np.sum(data.tree_asleep >= 0)) if sleep else 0
    overflow = int(data.warning[mujoco.mjtWarning.mjWARN_CNSTRFULL].number
                   + data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number)
    out = dict(kind="objects", seed=seed, sleep=sleep, noslip=int(model.opt.noslip_iterations),
               n=n_objects, hulls=int(sum(model.body_geomnum[obj.bodies]) - n_objects), **stats,
               rest_within_3s=bool(rest_frames is not None and rest_frames <= 90), spilled=spilled,
               pile_top_cm=round(pile_top * 100, 1),
               drift_median_mm=round(float(np.median(drift)) * 1e3, 3), drift_max_mm=round(float(drift.max()) * 1e3, 3),
               drift_worst=OBJECTS[int(np.argmax(drift))], drift_over_2mm=int(np.sum(drift > 0.002)),
               turn_max_deg=round(float(turn.max()), 3), idle_ms_per_step=round(ms, 3), idle_ncon=round(ncon, 1),
               trees_asleep=asleep, memory_warnings=overflow)
    if render:
        save_renders(model, data, obj, f"seed{seed}" + ("_sleep" if sleep else "") + (
            "" if "coacd" not in str(scene) else "_coacd"))
    return out


def run_cubes(args):
    seed, sleep, noslip = args
    from so101sim.mechsearch import MechSearchEnv
    env = MechSearchEnv(height=120, width=160)
    if noslip is not None:
        env.model.opt.noslip_iterations = noslip
    if sleep:  # has to be set before MjData exists, so rebuild data
        env.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        env.model.opt.sleep_tolerance = sleep
        env.data = mujoco.MjData(env.model)
    t0 = time.perf_counter()
    env.reset(seed=seed, options={"buried": True})
    reset_s = time.perf_counter() - t0
    names = list(env.cubes)
    bodies = [env.model.body(n).id for n in names]
    drift, turn, ms, ncon = idle(env.model, env.data,
                                 (lambda d: np.array([d.xpos[b] for b in bodies]), bodies))
    env.close()
    return dict(kind="cubes", seed=seed, sleep=sleep, noslip=int(env.model.opt.noslip_iterations), n=len(names),
                reset_wall_s=round(reset_s, 2),
                drift_median_mm=round(float(np.median(drift)) * 1e3, 3), drift_max_mm=round(float(drift.max()) * 1e3, 3),
                turn_max_deg=round(float(turn.max()), 3), idle_ms_per_step=round(ms, 3), idle_ncon=round(ncon, 1))


# ---- pictures -----------------------------------------------------------------------------------------------------

def save_renders(model, data, obj, tag):
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(model, 480, 640)
    for cam in ("top", "front"):
        r.update_scene(data, camera=cam)
        Image.fromarray(r.render()).save(OUT / f"pile_{tag}_{cam}.png")
    shown = mujoco.MjData(model)
    shown.qpos[:], shown.qvel[:] = data.qpos, 0.0
    set_arm(model, shown, HOME)
    mujoco.mj_forward(model, shown)
    r.update_scene(shown, camera="wrist_cam")
    Image.fromarray(r.render()).save(OUT / f"pile_{tag}_wrist_cam.png")
    # Front camera depth and per-object segmentation, to show the perception inputs come out of this scene too.
    r.enable_segmentation_rendering()
    r.update_scene(data, camera="front")
    seg = r.render()
    r.disable_segmentation_rendering()
    body = np.full(seg.shape[:2], -1)
    geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
    body[geom] = model.geom_bodyid[seg[..., 0][geom]]
    r.enable_depth_rendering()
    r.update_scene(data, camera="front")
    depth = r.render()
    r.disable_depth_rendering()
    # Stretch grey levels over the objects' depth range; everything else is shown flat.
    on = np.isin(body, obj.bodies)
    near, far = depth[on].min(), depth[on].max()
    grey = np.where(on, 40 + 215 * (1 - np.clip((depth - near) / (far - near), 0, 1)), 20)
    Image.fromarray(grey.astype(np.uint8)).save(OUT / f"pile_{tag}_front_depth.png")
    palette = np.random.default_rng(1).integers(40, 255, size=(model.nbody, 3))
    img = np.zeros(seg.shape[:2] + (3,), np.uint8)
    for b in obj.bodies:
        img[body == b] = palette[b]
    Image.fromarray(img).save(OUT / f"pile_{tag}_front_seg.png")
    r.close()
    contact_sheet(tag)


def contact_sheet(tag):
    """Tile one pile's five pictures into pile_<tag>_sheet.png: top, front, wrist over front depth and segmentation."""
    from PIL import Image
    names = ["top", "front", "wrist_cam", "front_depth", "front_seg"]
    sheet = Image.new("RGB", (3 * 640, 2 * 480), "white")
    for k, name in enumerate(names):
        sheet.paste(Image.open(OUT / f"pile_{tag}_{name}.png").convert("RGB"), ((k % 3) * 640, (k // 3) * 480))
    sheet.save(OUT / f"pile_{tag}_sheet.png")


# ---- main ---------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--hulls", choices=["vhacd", "coacd"], default="vhacd")
    ap.add_argument("--n-objects", type=int, default=len(OBJECTS))
    ap.add_argument("--sleep", type=float, nargs="?", const=1e-4, default=0.0,
                    help="turn on MuJoCo's sleeping islands, with this tolerance (MuJoCo 3.8's default 1e-4)")
    ap.add_argument("--noslip", type=int, default=None, help="override noslip_iterations (stage.xml has 10)")
    ap.add_argument("--render", type=int, nargs="*", default=[], help="seeds to save pictures for")
    ap.add_argument("--no-cubes", action="store_true", help="skip the cube baseline")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for the results file")
    args = ap.parse_args()

    scene = write_scene(OBJECTS[:args.n_objects], args.hulls)
    jobs = [(run_objects, (scene, s, args.sleep, args.noslip, args.n_objects, s in args.render)) for s in args.seeds]
    if not args.no_cubes:
        jobs += [(run_cubes, (s, args.sleep, args.noslip)) for s in args.seeds]
    with Pool(args.workers) as pool:
        async_results = [pool.apply_async(fn, (a,)) for fn, a in jobs]
        results = [r.get() for r in async_results]
    for r in results:
        print(json.dumps(r))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"pile_{args.hulls}_sleep{args.sleep:g}_noslip{args.noslip if args.noslip is not None else 10}{args.tag}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    for kind in ("objects", "cubes"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        keys = ["drift_median_mm", "drift_max_mm", "idle_ms_per_step", "idle_ncon"]
        if kind == "objects":
            keys += ["rest_frames", "settle_ms_per_step", "pile_top_cm", "spilled"]
        summary = {k: [r[k] for r in rows] for k in keys}
        print(kind, json.dumps(summary))


if __name__ == "__main__":
    main()
