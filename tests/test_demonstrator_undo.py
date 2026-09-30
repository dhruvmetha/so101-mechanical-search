"""Resets are privileged (dm1487, 2026-09-30). Undo takes back a spill and nothing else. A missed pick gets reset
with probability eps by a fixed draw, never by the player's choice, so which misses stay can't depend on how bad they
were. Everything else stays in the demonstration."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demonstrator"))

import tools  # noqa: E402  (imports so101sim before mujoco)


class Stub:
    def __init__(self, failure=None, grip=10.0, eps=None):
        act = {"op": "act", "kind": "pick", "xyz": [0.2, 0.0, -0.08], "deg": 0.0,
               "result": {"success": False, "terminated": failure is not None, "failure": failure, "grip": grip}}
        self.rec = {"seed": 1000, "ops": [{"op": "park"}, act]}
        if eps is not None:
            self.rec["miss_reset_eps"] = eps
        self.saved = False

    def save(self):
        self.saved = True


def test_undo_takes_back_a_spill():
    ep = Stub("cube left the bin: cube_gray_3")
    tools.cmd_undo(ep, "spilled g3 over the far wall")
    assert ep.rec["ops"] == [] and ep.saved
    assert ep.rec["undone"][0]["why"] == "spilled g3 over the far wall"
    assert ep.rec["undone"][0]["granted"] == "spill"


def test_undo_refuses_a_miss():
    ep = Stub()
    with pytest.raises(SystemExit, match="only takes back a spill"):
        tools.cmd_undo(ep, "missed")
    assert len(ep.rec["ops"]) == 2 and not ep.saved


def test_a_miss_is_an_empty_close_on_a_pick():
    assert tools.missed(Stub(grip=10.0).rec["ops"][1])
    assert not tools.missed(Stub(grip=18.0).rec["ops"][1])  # holding
    assert not tools.missed(Stub(grip=40.0).rec["ops"][1])  # open, as after a refused pick


@pytest.mark.parametrize("eps, reset", [(1.0, True), (0.0, False)])
def test_the_draw_decides_a_miss_reset(eps, reset):
    ep = Stub(eps=eps)
    got, u, _ = tools.reset_miss(ep)
    assert got == reset and 0.0 <= u < 1.0 and ep.saved
    if reset:
        assert ep.rec["ops"] == [] and ep.rec["undone"][0]["granted"] == "miss draw"
    else:
        assert len(ep.rec["ops"]) == 2 and not ep.rec.get("undone")


def test_the_draw_is_fixed_and_fair():
    assert tools.draw(1000, 3, 0) == tools.draw(1000, 3, 0)
    assert tools.draw(1000, 3, 0) != tools.draw(1000, 3, 1)  # a retry at the same move draws again
    share = sum(tools.draw(s, m, 0) < 0.1 for s in range(200) for m in range(1, 16)) / 3000
    assert 0.08 < share < 0.12
