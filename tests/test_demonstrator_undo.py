"""Undo is only for spills (dm1487, 2026-09-30). Taking back a miss would pick the next move by how the hidden pile
answered, so tools.py refuses it and the miss stays in the demonstration."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demonstrator"))

import tools  # noqa: E402  (imports so101sim before mujoco)


class Stub:
    def __init__(self, failure):
        act = {"op": "act", "kind": "pick", "xyz": [0.2, 0.0, -0.08], "deg": 0.0,
               "result": {"success": False, "terminated": failure is not None, "failure": failure}}
        self.rec = {"ops": [{"op": "park"}, act]}
        self.saved = False

    def save(self):
        self.saved = True


def test_undo_takes_back_a_spill():
    ep = Stub("cube left the bin: cube_gray_3")
    tools.cmd_undo(ep, "spilled g3 over the far wall")
    assert ep.rec["ops"] == [] and ep.saved
    assert ep.rec["undone"][0]["why"] == "spilled g3 over the far wall"


def test_undo_refuses_a_miss():
    ep = Stub(None)
    with pytest.raises(SystemExit, match="only takes back a spill"):
        tools.cmd_undo(ep, "missed")
    assert len(ep.rec["ops"]) == 2 and not ep.saved
