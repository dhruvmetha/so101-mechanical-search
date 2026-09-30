"""An episode kept alive in memory, with snapshots, so a planner can try a move and take it back for free.

tools.py rebuilds an episode by replaying every op from reset on each call, 5 to 30 s. Here the env stays up, and
`snapshot()` / `restore()` save and restore everything that decides what happens next: MuJoCo's integration state
(including the solver warm start), the env's step, success-streak and move counters, and the episode's own robot
time, arm state and op list. Measured 2026-09-30: saving the MuJoCo state takes 2 microseconds.

A line of ops committed this way replays exactly from reset, which is what export.py checks before it keeps a demo;
tests/test_demonstrator_live.py holds that to bit-exact qpos.
"""
import copy

import numpy as np

import tools  # sets up so101sim before mujoco
import mujoco

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class Live(tools.Episode):
    def snapshot(self):
        st = np.empty(mujoco.mj_stateSize(self.m, SPEC))
        mujoco.mj_getState(self.m, self.d, st, SPEC)
        return {"state": st, "steps": self.env._steps, "streak": self.env._streak, "n": self.prim._n,
                "robot_s": self.robot_s, "last": copy.deepcopy(self.last), "op_index": self.op_index,
                "n_ops": len(self.rec["ops"])}

    def restore(self, snap):
        # Set the state, recompute positions for rendering and perception, then set it again: the recompute may
        # touch the solver's warm start, and the next step must start from exactly the saved one.
        mujoco.mj_setState(self.m, self.d, snap["state"], SPEC)
        mujoco.mj_forward(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, snap["state"], SPEC)
        self.env._steps, self.env._streak, self.prim._n = snap["steps"], snap["streak"], snap["n"]
        self.robot_s, self.last, self.op_index = snap["robot_s"], copy.deepcopy(snap["last"]), snap["op_index"]
        del self.rec["ops"][snap["n_ops"]:]

    def run(self, ops):
        """Apply ops and record them; returns the last act's result, or None if there was no act."""
        res = None
        for op in ops:
            self.op_index = len(self.rec["ops"])
            r = self._apply(op)
            if op["op"] == "act":
                op = {**op, "result": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}}
                res = r
            self.rec["ops"].append(op)
            if res is not None and (res["terminated"] or res["success"]):
                break
        return res
