"""Snapshots in demonstrator/live.py must be exact: a line committed by trying moves on restored states has to replay
bit for bit from reset, or export.py throws the demo away."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demonstrator"))

from live import Live  # noqa: E402  (imports so101sim before mujoco)
import tools  # noqa: E402


def recorded(name):
    return json.loads((tools.EPISODES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["d1011", "d1006", "e402"])
def test_restoring_a_snapshot_replays_the_rest_of_the_episode_bit_for_bit(name):
    rec = recorded(name)
    k = len(rec["ops"]) // 2
    live = Live(name, rec={**rec, "ops": rec["ops"][:k]})
    try:
        snap = live.snapshot()
        rest = [dict(op) for op in rec["ops"][k:]]
        live.run([dict(op) for op in rest])
        first = live.d.qpos.copy(), [op.get("result") for op in live.rec["ops"][k:]]
        live.restore(snap)
        live.run([dict(op) for op in rest])
        second = live.d.qpos.copy(), [op.get("result") for op in live.rec["ops"][k:]]
    finally:
        live.close()
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[1] == [op.get("result") for op in rec["ops"][k:]]  # and both match what was played
