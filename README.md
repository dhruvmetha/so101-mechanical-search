# rearrange

The SO-101 arm on a table in MuJoCo, running headless on arrakis. There's a table with three cubes and a
bin, a front camera and a wrist camera. Actions and observations use lerobot's joint units and key names.

The task is mechanical search: find the one red cube in a bin of gray ones and lift it out, rearranging
only inside the bin. `so101sim.mechsearch` has a 30 Hz env and a pick/place/push primitive env for it.
Whether this trains a VLA or an RL policy is still open. The older `BinPickEnv`, which moves loose cubes
into a small bin, still works.

## Run it

```bash
conda activate /common/users/dm1487/envs/so101sim
python -m pytest -q                               # about 2 minutes, most of it search resets
python scripts/smoke.py --out outputs/smoke.mp4   # arm hovers over each cube and the bin
```

```python
from so101sim.env import SO101Env, HOME

env = SO101Env()
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(HOME)  # 6 joint targets: degrees, gripper 0..100
obs["observation.state"], obs["observation.images.front"], obs["observation.images.wrist"]
env.close()
```

```python
from so101sim.mechsearch import MechSearchPrimitiveEnv

env = MechSearchPrimitiveEnv()
obs, info = env.reset(seed=0, options={"buried": True})
p = info["cubes"]["cube_gray_0"]
obs, reward, terminated, truncated, info = env.step(("pick", p["pos"], p["yaw"]))
env.close()
```

For batched GPU physics, `so101sim.mjx.put(model, worlds)` runs the same scene on MuJoCo Warp. That's
about 30,000 physics steps/s at 1024 worlds on one RTX 6000 Ada.

## Where things are

- `models/` holds the menagerie SO-101, pinned and untouched, the shared `stage.xml`, and one scene per task. See `models/README.md`.
- `so101sim/` has the unit conversion, IK, primitives, both task envs, the search layout generator and the MJX loader.
- `docs/superpowers/specs/2026-09-18-so101-sim-design.md` records what was decided and why.
- `docs/superpowers/specs/2026-09-25-mechanical-search-design.md` is the current task's spec.
- `CLAUDE.md` lists commands, conventions and the traps found so far.
