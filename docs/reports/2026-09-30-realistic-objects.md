# Realistic objects for the bin task, and whether to leave MuJoCo

2026-09-30. Written by Claude for dm1487, who asked: "bring in realistic objects: toys? Look for available assets. If you
think we should switch to IsaacLab or something that is also okay."

Everything under "Loading test" and "Snapshot test" I measured on arrakis. Everything else comes from the linked page.
Where I could not find a primary source, the text says "unconfirmed".

## Short answer

- **Objects.** Use Google Scanned Objects through kevinzakka's MuJoCo port. The meshes are real scans in metres,
  licensed CC BY 4.0, and all 1030 models compile in our MuJoCo 3.8.1. Only 80 of them fit this gripper, and about 20 of
  those 80 are toys. For objects someone can also buy and put in front of the real arm, add YCB and its MuJoCo port.
- **Loading test.** 20 scanned toys and household objects came to rest in 0.5 to 2.1 s after the last drop, within
  rest.py's rule, in all 8 seeds. Median idle drift over 10 s was 0.3 mm, against 0.1 mm for the cube pile. A physics
  step costs 4.1 ms, 5.3 times the cube pile's 0.76 ms measured in the same session. Refitting the collision hulls
  with CoACD at 8 or fewer per object halves that to 2.1 ms, but 2 of 8 of those piles jittered and never met the rest
  rule.
- **MuJoCo has sleeping islands.** With them turned on, 7 of 8 piles froze with zero drift, and an idle step cost
  0.05 ms. When sleeping is on, a snapshot needs `mj_copyData`: restoring the saved state vector into a new `MjData` drifted
  in 2 of 4 seeds.
- **Simulator.** Stay on MuJoCo CPU. It is the only candidate where we have measured noslip, bit-exact replay and
  full mid-contact snapshots, and it runs the SO-101 we already use. Genesis documents all three but ships no SO arm, and
  its MJCF import drops geom groups. Switching would mean rewriting the demonstrator and the perception code, and no
  engine has shown better pile stability. The last section lists what would change this.

## Part 1: object sets

### Survey

"Fit" means an object roughly 2 to 10 cm across that a top-down grasp could close on. The sim gripper opens to
133 mm, the real one probably nearer 82 mm (CLAUDE.md).

| Set | License | Objects | Toys that fit | Collision shipped for MuJoCo | Scale | Format | Download |
|---|---|---|---|---|---|---|---|
| [Google Scanned Objects](https://arxiv.org/abs/2204.11918) (GSO) via [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) | Meshes CC BY 4.0, XML MIT ([README](https://github.com/kevinzakka/mujoco_scanned_objects)) | 1030 | 11 in toy categories, about 20 toy-like by name, out of 80 that fit (my catalog below) | 32 V-HACD hulls per object ([README](https://github.com/kevinzakka/mujoco_scanned_objects)) | Real scans in metres. I checked extents on all 1030 | MJCF + OBJ + PNG | 1.0 GB packed, 2.5 GB on disk with .git (measured) |
| [YCB](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) via [elpis-lab/YCB_Dataset](https://github.com/elpis-lab/YCB_Dataset) | CC BY 4.0 data, repo MIT | 78 in the port, of about 100 items | Duplo b to f, airplane parts c to e, 26 mm wood block, stacking cups, balls, Rubik's cube, foam brick ([object list](http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf)) | CoACD, at most 5 hulls ([README](https://github.com/elpis-lab/YCB_Dataset)) | Metres. Every inertia is set to `diag(1e-3)`, which is wrong for small objects ([README](https://github.com/elpis-lab/YCB_Dataset)) | MJCF + URDF | about 170 MB |
| YCB via [vikashplus/YCB_sim](https://github.com/vikashplus/YCB_sim) | Apache-2.0 | 10 | none | one mesh used as its own convex hull | metres | MJCF + .msh | about 100 MB |
| [Lightwheel-YCB](https://github.com/LightwheelAI/Lightwheel-YCB) | CC BY-NC 4.0 | 125 | YCB's | rebuilt by hand, per the README | unconfirmed | MJCF + USD | unconfirmed |
| [RoboCasa objects](https://github.com/robocasa/robocasa/blob/main/docs/assets/objects.md) ([paper](https://arxiv.org/abs/2406.02523)) | CC BY 4.0 assets, per the [README](https://github.com/robocasa/robocasa) | 2,509 in the paper, 3,134 in the current table | none: kitchen categories only | V-HACD, up to 32 hulls ([generator](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/model_zoo/mjcf_gen_utils.py)); which tool made each shipped object is unconfirmed | normalized, then hand-scaled per category | MJCF | about 10 GB |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) objects | MIT | 15 | none | one convex hull each | hand-made | MJCF | in the package |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) objects | MIT code, CC BY 4.0 data; redistribution terms for its TurboSquid meshes are unconfirmed | about 40 | 14 HOPE toy groceries | box primitives | HOPE meshes are in cm, scaled by 0.01 in the XML | MJCF | in the package |
| [ManiSkill 3 assets](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/assets/data.py) | README says CC BY-NC 4.0; the YCB zip's card says Apache-2.0 ([README](https://github.com/haosulab/ManiSkill)) | YCB, PartNet-Mobility subsets, scenes | YCB's | multi-convex collision files for SAPIEN | metres | OBJ/GLB, not MJCF | YCB zip 26 MB |
| [Objaverse](https://huggingface.co/datasets/allenai/objaverse) / LVIS subset | per object: 721K CC-BY, the rest SA, NC or CC0 | 800K; LVIS 46K in 1,156 categories | LVIS has figurine 126, teddy_bear 123, Lego 85, toy 80, doll 53 and more | none | no metric units | GLB | varies |
| [MolmoSpaces](https://arxiv.org/abs/2602.11337) (Objaverse converted to MJCF) | per object, mostly CC-BY; a `commercial_safe` filter drops NC ([repo](https://github.com/allenai/molmospaces)) | 129,725 | about 930 toy-like after a size filter | CoACD, median 9 hulls, at most 15 | GPT-4o estimates, heaped at round numbers | MJCF | about 1 MB per object, 142 GB total |
| [HOPE](https://github.com/swtyree/hope-dataset) | CC BY-SA 4.0 per [BOP](https://bop.felk.cvut.cz/datasets/); the repo states none | 28 toy groceries, buyable | all 28 | none | centimetres | OBJ | about 100 MB |
| [Toys4K](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) | mixed; a form gate and "please do not distribute" | 4,179 in 105 categories | all toys | none | metric scale unconfirmed | .blend / OBJ | 4.1 GB |
| [DTC](https://www.projectaria.com/datasets/dtc/) ([paper](https://arxiv.org/abs/2504.08541)) | CC BY-SA 4.0 plus a no-sale clause | about 2,000 | blocks 129, dinosaurs 103, figurines 77, fake ducks 48 | none | sub-mm scans | format unconfirmed | email gate |
| [OmniObject3D](https://github.com/omniobject3d/OmniObject3D) | CC BY 4.0 | 6,000 | 339 in its Toys split | none | the paper says real scale | OBJ | 1.2 TB |

Others that I ruled out: ABO (furniture), BEHAVIOR-1K (encrypted USD), GraspNet-1Billion, ACRONYM and DexGraspNet (all
NC), EGAD (abstract shapes), BigBIRD and KIT (unclear licenses). MuJoCo Menagerie has robots only.

A mesh geom in MuJoCo collides as its convex hull, so any non-convex object has to be split into convex pieces. The
docs point to CoACD for that ([computation docs](https://mujoco.readthedocs.io/en/stable/computation/index.html)).
[obj2mjcf](https://github.com/kevinzakka/obj2mjcf) (MIT) turns an OBJ into MJCF and runs
[CoACD](https://github.com/SarahWeiii/CoACD) (MIT). It switched from V-HACD to CoACD in v0.0.24, after the GSO port was
generated in July 2022.

### What the GSO port needs before use

I cataloged all 1030 models with `scripts/objects_catalog.py`: extents, hull count, whether each compiles, and the
scanned volume. Categories came from the Gazebo Fuel API
(`https://fuel.gazebosim.org/1.0/GoogleResearch/models`), which lists all 1030 as CC BY 4.0 with these counts: Shoe
254, Consumer Goods 248, none 216, Toys 147, Bottles and Cans and Cups 53, Action Figures 17, Board Games 17, Legos 10,
Stuffed Toys 3. They match the paper's table.

- **All 1030 compile in MuJoCo 3.8.1.** 1027 have 32 hulls and 3 have 1.
- **Few fit the gripper.** 80 objects are at least 1.5 cm thick, at most 8 cm wide and at most 12 cm long. Most are
  supplement bottles, cosmetics and 11 toys; by name about 20 are toy-like (Schleich figures, Mario, Thomas wooden trains,
  Android figures, a triceratops, a school bus, a wooden ABC block). Loosen the limits to 10 by 15 cm and 190 fit, 35
  of them in toy categories. Most GSO toys are boxed sets: the median toy is 7.5 x 16 x 23 cm.
- **The shipped masses are wrong.** Every geom gets MuJoCo's default 1000 kg/m^3, and the visual mesh counts on top
  of 32 overlapping hulls. That comes to about 4 times what 1000 kg/m^3 over the scanned volume gives. The
  [GSO paper](https://arxiv.org/abs/2204.11918) records real mass for only some objects. In my test scene the visual
  mesh carries all the mass at 500 kg/m^3 and the hulls carry none, so the inertia follows the scanned shape. 500 is a
  guess that sits between hollow packaging and solid PVC figures.
- **32 hulls per object is expensive.** See the loading test.
- **The XML has no freejoint, friction or condim.** The test scene adds them.

### Pick

GSO through the MuJoCo port, for toys and household objects. The meshes are real scans at real scale, the license
allows redistribution with attribution, and the port already loads. The catch is the size range: about 80 objects, 20
of them toys. That is enough for a 20 to 40 object bin with some variety, but not for hundreds of distinct toys.

YCB through elpis-lab's port comes second. Its value is that the objects can be bought for the real SO-101. Delete its
`<inertial>` lines so MuJoCo computes inertia from the hulls. HOPE's 28 toy groceries are also buyable, but they would
need collision geometry built.

If we ever need hundreds of toy shapes, MolmoSpaces has already converted Objaverse to MJCF with CoACD hulls, but every
size there is a language-model guess and would need rescaling by hand.

### Loading test

`scripts/objects_pile_test.py` writes `models/objects_test_scene.xml`. It uses the shared stage with noslip on, the
search task's table, riser and front camera, a camera above the bin, and a bin with the search bin's 22 cm footprint at
x = 0.20 but 12 cm walls. It drops 20 GSO objects in: 15 toys and 5 household objects, listed in the script. They fall
in waves the way `MechSearchEnv` drops cubes, each with a random 3D orientation. Collision geoms use condim 4 and friction
0.8. The arm waits at shoulder_pan 90 degrees, out of the drop column. The script then counts frames until rest.py's
rule holds (every object under 2 mm/s and 0.1 rad/s for 15 frames), idles 10 s, and records how far each object's
centre moved. The cube baseline is `MechSearchEnv.reset` on the same seed (buried start), then the same 10 s idle.
Timings come from a sequential run while the machine's load average was 28 to 41 on 32 cores, so they are noisy.

Results over seeds 0 to 7, from `outputs/objects_2026-09-30/*_seq.jsonl`:

| | Cubes, 30 | GSO, 32 V-HACD hulls each | GSO, CoACD at most 8 hulls | GSO, V-HACD, sleep on (tolerance 1e-3) |
|---|---|---|---|---|
| Collision geoms | 30 boxes | 640 hulls | 136 hulls | 640 hulls |
| Frames from last drop to rest | n/a | 16 to 64 | 18 to 208; 2 seeds never | 15 to 58 |
| Objects out of the bin | 0 | 0 | 1 in seed 2 | 0 |
| Idle drift over 10 s, median of seed medians | 0.12 mm | 0.30 mm | 0.15 mm | 0 |
| Idle drift, worst object | 0.4 to 3.9 mm | 0.4 to 18.7 mm | 0.4 to 34.5 mm | 0 in 7 seeds, 12.3 mm in 1 |
| Seeds with any object over 2 mm | 1 of 8 | 3 of 8 | 2 of 8 | 1 of 8 |
| ms per physics step, idle, median (range) | 0.76 (0.57 to 1.01) | 4.06 (3.71 to 10.05) | 2.11 (1.71 to 2.44) | 0.05 (0.05 to 0.93) |
| ms per step while settling, median | n/a | 4.7 | 2.1 | 2.1 |
| Contacts while idle | 118 to 151 | 101 to 159 | 70 to 87 | 0 in 7 seeds |
| Pile top above the bin floor | 3 layers | 13.4 to 16.4 cm | 14.7 to 18.3 cm | 13.7 to 16.8 cm |

What this shows:

- **Rest.** Scanned piles settle as fast as cubes. Every V-HACD pile met rest.py's rule inside its 3 s limit.
- **Creep.** It is worse than for cubes but still small. The worst movers were a school bus that slid 18.7 mm and
  turned 38 degrees (seed 5), a CoQ10 bottle at 7.4 mm, and the triceratops at 7.3 mm. The bottle probably needs
  condim 6 and rolling friction, as the shelf's cylinders did (CLAUDE.md). With noslip off, the median rose to 1.4 mm
  and every seed had an object move more than 2 mm (`pile_vhacd_sleep0_noslip0.jsonl`, run in parallel), so noslip still
  earns its cost here.
- **Cost.** A physics step costs 5.3 times the cube pile's. A 30 Hz frame is 10 steps, so the scanned pile runs at about
  0.8 times real time on one core, against about 4 times for cubes.
- **CoACD with fewer hulls.** Halving the hull work halves the cost, but those piles chatter. In the 2 s after the 10 s
  limit, a honey dipper in seed 0 kept spiking to 7 mm/s without going anywhere, and in seed 1 thirteen objects crossed
  2 mm/s now and then. Something between 8 and 32 hulls, or CoACD's default concavity without a hull cap, is the next
  thing to try. I did not test it.
- **Sleeping islands.** In 7 of 8 seeds every object fell asleep during idle, so drift was exactly zero and an idle
  step cost 0.05 ms. In seed 5 an eagle moved 12 mm before the pile slept. At MuJoCo 3.8's default tolerance of 1e-4,
  only 2 of 8 piles fell fully asleep within the 10 s (`pile_vhacd_sleep0.0001_noslip10.jsonl`). The default became 1e-3
  in MuJoCo 3.11 ([changelog](https://mujoco.readthedocs.io/en/stable/changelog.html)). A sleeping body stays put until
  something awake touches it or its qpos or qvel is edited, then "behaves exactly as if it was awake all along"
  ([docs](https://mujoco.readthedocs.io/en/stable/computation/index.html#sleeping)). Sleeping also changes how a pile
  builds, so the same seed gives a different pile with sleep on.
- **Replay.** It is deterministic across processes: the 8 V-HACD seeds run 8 at a time in parallel and then one at a
  time gave identical rest frames and drift to the last digit.
- **Pictures.** Front depth and per-object segmentation come out of the scene with no changes: every object gets its
  own body id in the segmentation render.
- **Arm clearance.** Twenty objects of this size stack 13 to 18 cm deep in a 22 cm bin, over the 12 cm rim. At the
  search start pose the jaws hang about 5 cm above that rim, and the wrist camera picture shows them inside the top of
  the pile. A sink-like bin needs the arm raised or the bin lowered, and CLAUDE.md already says top-down reach ends at
  z = 0.10 and walls over 6 cm catch the wrist camera mount.

Pictures, in `outputs/objects_2026-09-30/`:

- `pile_seed1_sheet.png` and `pile_seed0_sheet.png`: top view, front camera and wrist camera, over front depth and
  front segmentation.
- `pile_seed1_coacd_sheet.png` and `pile_seed1_sleep_sheet.png`: the same for the other two variants.
- The single pictures that go into each sheet.

### Snapshot test

`scripts/objects_snapshot_check.py` builds a pile, saves it three ways, shoves the top object for 0.5 s and lets
everything run 1.5 s more, then repeats that probe from each saved copy and compares every qpos with the first run.

| Restore method | Sleep off, seeds 0 to 3 | Sleep on at 1e-3, seeds 0 to 3 |
|---|---|---|
| `mj_setState(mjSTATE_INTEGRATION)` into the same `MjData` | exact | exact |
| the same vector into a fresh `MjData` | exact | largest qpos difference 2.6e-5 and 1.2e-4 after 2 s, in 2 of 4 seeds |
| `mj_copyData` | exact | exact |

This matches MuJoCo's docs. The integration state includes the warmstart, and replay is exact "within a single
version, on the same architecture" ([reproducibility](https://mujoco.readthedocs.io/en/stable/computation/index.html#reproducibility)).
Once islands are asleep, "the only way to fully save and restore the state of a simulation with sleeping elements is
to copy the entire mjData structure" ([sleep notes](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#sisleepnotes)).
So if the demonstrator's rollback turns sleeping on, it should keep whole `MjData` copies, not state vectors.
Upgrading MuJoCo will change how every saved episode replays.

## Part 2: stay with MuJoCo or switch

| | MuJoCo CPU (ours, 3.8.1) | MuJoCo Warp / MJX on GPU | Isaac Lab (PhysX 5) | ManiSkill 3 (SAPIEN, PhysX) | Genesis |
|---|---|---|---|---|---|
| Piles at rest | noslip; sleeping islands since 3.4.0. Measured above | no noslip ([README](https://github.com/google-deepmind/mujoco_warp/blob/main/README.md)); sleeping added June 2026, in mjx 3.10+ ([docs](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)), not in our 3.8.1 bundle (checked locally); float32. Our piles crept 22 to 47 mm over 30 s | no noslip equivalent found; PhysX staff say zero-penetration resting contact is not attainable ([forum](https://forums.developer.nvidia.com/t/physx-rigid-body-dynamics-resting-contact/283182)); no pile data found | no noslip; sleep_threshold 0.005 by default ([SceneConfig](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/utils/structs/types.html)); no pile data found | `noslip_iterations` exists, default 0; hibernation "hasn't been robustly tested" ([docs](https://genesis-world.readthedocs.io/en/latest/api_reference/engine/solvers/rigid_solver.html)); pile behaviour unconfirmed |
| Deterministic replay | yes, same version and machine (tested above) | "No" on GPU ([FAQ](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html), [issue 562](https://github.com/google-deepmind/mujoco_warp/issues/562)); Warp's deterministic mode on these kernels is unconfirmed | run to run 0 cm, parallel envs 4.21 cm apart ([GPUSimBench](https://arxiv.org/abs/2607.13059), inclined-plane test) | run to run 0, parallel 4.76 cm (same paper); replays "can fail due to non-determinism" ([docs](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)) | 0 and 0 in the same paper; bit-exact mode since v1.3.3 ([release](https://github.com/Genesis-Embodied-AI/genesis-world/releases/tag/v1.3.3)) |
| Mid-episode snapshot | full, `mj_getState` or `mj_copyData` (tested) | no documented API that covers warmstart and contacts (unconfirmed) | contact state "cannot currently be serialized" ([Omniverse physics limits](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/current_limitations.html)) | poses and velocities only ([API](https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/sapien_env/index.html)) | checkpoints hold contacts, constraint solution and warm-start caches, bit-for-bit in memory ([docs](https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/checkpoints.html)) |
| RGB, depth, segmentation | yes, EGL, in use | yes, batched ray tracer | yes (TiledCamera); needs RT cores ([requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)) | yes; depth in int16 mm ([docs](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html)); needs Vulkan | yes, rasterizer or batch renderer ([docs](https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html)) |
| SO-100/101 model | Menagerie SO-101, in use | same MJCF | SO-101 in [LeIsaac](https://lightwheelai.github.io/leisaac/docs/getting_started/installation) (no bin task) | SO-100 only ([docs](https://maniskill.readthedocs.io/en/latest/_sources/robots/so100/index.md.txt)) | none shipped |
| Cost for us | 4 ms per step for the scanned pile, 0.05 ms asleep | about 30k steps/s at 1024 worlds (our 2026-09-18 benchmark in `so101sim/mjx.py`) | unmeasured | unmeasured | unmeasured; its headline speed claims were disputed ([Stone Tao](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)) |
| Install on arrakis | 21 MB, installed | installed | current Isaac Sim needs driver 595.58.03 on Linux ([requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)); arrakis runs 570.195.03 (checked); 50 GB SSD minimum | about 150 MB plus torch | 84 MB plus dependencies |
| Porting our code | none | low: same model and data layout | high: USD scene, new env, new renderer for perception | high: whether our MJCF loads is unconfirmed | medium: MJCF import drops geom `group` ([issue 3307](https://github.com/Genesis-Embodied-AI/Genesis/issues/3307)), which the buried-red check relies on |

Other points:

- **Evidence on piles is thin.** Nobody has published an idle-pile creep comparison across engines that I found. The
  closest study is Erez, Tassa and Todorov 2015 ([pdf](https://roboti.us/lab/papers/ErezICRA15.pdf)). It measured
  consistency on 27 capsules, found MuJoCo most consistent and PhysX least, and was written by MuJoCo's authors.
- **The creep has an open issue.** [MuJoCo issue 3328](https://github.com/google-deepmind/mujoco/issues/3328), filed June
  2026 against 3.8.1, measured the same creep on a pinched capsule. It fell from 155 um/s at default settings to 1.4 um/s
  with noslip 10.
- **Newton will not fix it.** [Newton](https://github.com/newton-physics/newton) runs MuJoCo Warp underneath, so it
  inherits the missing noslip.
- **LeRobot has no SO-101 sim env.** [Its env configs](https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/configs.py)
  cover aloha, pusht, gym-hil, LIBERO, MetaWorld, RoboCasa and others. Our export writes LeRobotDataset directly, so no
  simulator choice changes that part.
- **Menagerie has two SO-101 fixes our pinned copy lacks.** PR #319 moves the wrist camera, and PR #324 raises the
  wrist_roll limit to match its ctrlrange ([commits](https://github.com/google-deepmind/mujoco_menagerie/commits/main/robotstudio_so101)).
  Re-vendoring changes the wrist camera picture, so it should be a deliberate step.

### Recommendation

Stay on MuJoCo on the CPU. Bring in the GSO objects in this order:

1. Keep noslip on and turn sleeping islands on at tolerance 1e-3. On this pile it removed idle drift in 7 of 8 seeds
   and made an idle step 80 times cheaper. The demonstrator's rollback then has to copy whole `MjData`. Check that a
   sleeping pile wakes and behaves when the jaws come down on it before relying on it; I did not test that.
2. Rebuild hulls with CoACD at a cap between 8 and 32. Check both cost and rest.py's rule.
3. Give round objects condim 6 and rolling friction. Put real masses on the objects that will be targets.

The problem is the physics of resting contact, and none of the other engines has shown it solved. MuJoCo is the only
option where we have measured noslip, bit-exact replay and full mid-contact snapshots. Genesis documents all three, but
nobody has shown its piles holding still. Current Isaac Sim releases would also need a driver upgrade on a shared box,
and ManiSkill has only the SO-100.

What would change this:

- **Move bulk rollouts, not the demonstrator, to the GPU** if MuJoCo Warp from mjx 3.10 or later, with sleeping on,
  holds a pile under 1 mm for 30 s and replays bit-exactly under Warp's deterministic mode. It keeps our model files,
  so this is an upgrade, not a switch. The test is cheap, but upgrading MuJoCo changes how saved episodes replay.
- **Consider Genesis** if it passes the same 30 s idle test with noslip on the GPU and imports geom groups intact.
- **Reconsider Isaac Lab** if we need photoreal rendering for sim-to-real, arrakis gets a driver of 595 or newer, and
  PhysX gains a snapshot that includes contacts.
- **Lean harder on sleeping and CoACD** if CPU cost becomes the limit, before changing engines. At 4 ms per step, one
  core simulates this pile at about 0.8 times real time, and arrakis has 32 cores.

## Unconfirmed

- Metric scale of Toys4K and the file format of DTC.
- Lightwheel-YCB's access terms and scale.
- Which decomposition produced RoboCasa's shipped objects.
- Whether MuJoCo Warp becomes bit-exact under Warp's deterministic mode, and whether it has a snapshot API covering
  contacts.
- Whether PhysX sleeping runs in the GPU pipeline.
- Genesis's behaviour on piles with noslip.
- Whether ManiSkill loads our MJCF.
- The GPUSimBench numbers come from an inclined-plane test, not piles. The venue attributed to it (IROS 2026) is
  unconfirmed.

## Files

- `scripts/objects_catalog.py` writes `outputs/objects_2026-09-30/gso_catalog.csv`: extents, hulls, volume and whether
  each of the 1030 GSO objects fits.
- `scripts/objects_pile_test.py` holds the pile test and writes `models/objects_test_scene.xml`,
  `models/objects_test_objects.xml`, the `_coacd` pair, the JSONL results and the pictures. The generated XML has absolute
  paths into `/common/users/dm1487/assets`, so it only loads on this machine.
- `scripts/objects_coacd.py` is the CoACD refit. CoACD isn't in the so101sim env; I installed it to a scratch directory
  with `pip install --no-deps --target`.
- `scripts/objects_snapshot_check.py` is the restore test.
- Downloads, 2.5 GB in total, are in `/common/users/dm1487/assets/`: `mujoco_scanned_objects` at commit 6ff8d27 (July
  2022), `gso_fuel_metadata.json` and `gso_coacd/`.
