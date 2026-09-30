---
paths:
  - "so101sim/mjx.py"
  - "tests/test_mjx.py"
---

# Traps: MJX and Warp on the GPU

Claude Code loads this file when Claude reads a file matching the paths above. It is the only home of these traps,
which moved here from CLAUDE.md on 2026-09-30. Add a trap here when you find one in this area, dated, with the
evidence. The load follows the Read tool; if you read code with `cat` or `sed`, open this file yourself.

- Warp preallocates contact and constraint buffers. When one fills it prints `broadphase overflow`,
  `narrowphase overflow` or `nefc overflow`, drops contacts, and objects sink into the table. Raise the
  matching argument to `so101sim.mjx.put`. Don't filter those warnings out of logs.
- Warp's `mjx.make_data` takes the `MjModel`, not the `mjx` model.
- `mujoco_warp` does not implement `noslip` and raises if a model asks for it. `so101sim.mjx.put` drops it on a
  copy and warns. Piles creep on the GPU under every setting tried, 22 to 47 mm median over 30 s, more iterations
  included, likely because Warp computes in 32-bit floats. Warp's contact order also varies run to run, so the same
  start gives different numbers. Don't trust GPU results that need objects to stay where they were left, and
  compare against the CPU before trusting either for pushes.
