"""Turn solved episodes into a LeRobot dataset: every 30 Hz frame the arm was commanded, with both camera images.

    python demonstrator/export.py --out demonstrator/dataset d1000 d1001 ...
    python demonstrator/export.py --out demonstrator/dataset --all-solved

Each episode is replayed from its op list, the same way tools.py rebuilds it, with a hook on every frame. A frame is
the observation before the command (joint state in lerobot units, front and wrist images) and the command itself.
Parks, settling and the unpark hover are frames too: they are motions the robot made. Undone moves are not, because
they are not in the op list.

The replay is checked against the result each act recorded. An episode whose replay disagrees is skipped, since
its frames would not be the demonstration that was played.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tools  # noqa: E402  sets up so101sim before mujoco
from so101sim import units  # noqa: E402

TASK = "find the red cube in the bin and lift it out"
FPS = 30


def record(name, size):
    """Frames of one episode: dict of arrays, plus whether the replay matched the record."""
    frames = {"state": [], "action": [], "front": [], "wrist": [], "op": []}

    def on_advance(ep, row):
        obs = ep.env._obs()
        frames["state"].append(obs["observation.state"])
        low, high = ep.env.action_space.low, ep.env.action_space.high
        frames["action"].append(np.clip(np.asarray(row, np.float32), low, high))
        frames["front"].append(obs["observation.images.front"])
        frames["wrist"].append(obs["observation.images.wrist"])
        frames["op"].append(ep.op_index)

    results = []
    original = tools.Episode._apply

    def spy(self, op):
        r = original(self, op)
        if op["op"] == "act":
            results.append((op["result"], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}))
        return r

    tools.Episode._apply = spy
    try:
        ep = tools.Episode(name, size=size, on_advance=on_advance)
    finally:
        tools.Episode._apply = original
    matched = all(a == b for a, b in results)
    solved = ep.outcome() == "solved"
    meta = {"name": name, "seed": ep.rec["seed"], "buried": ep.rec["buried"], "moves": ep.moves(),
            "undone": len(ep.rec.get("undone", [])), "robot_s": round(ep.robot_s, 1), "solved": solved,
            "replay_matched": matched}
    ep.close()
    return {k: np.asarray(v) for k, v in frames.items()}, meta


def features(size):
    image = {"dtype": "video", "shape": (size, size, 3), "names": ["height", "width", "channels"]}
    joints = {"dtype": "float32", "shape": (6,), "names": list(units.JOINT_NAMES)}
    return {
        "observation.state": joints,
        "action": dict(joints),
        "observation.images.front": image,
        "observation.images.wrist": dict(image),
        "op_index": {"dtype": "int64", "shape": (1,), "names": None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all-solved", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--repo-id", default="so101sim/mechsearch_bin_claude")
    args = ap.parse_args()

    names = list(args.names)
    if args.all_solved:
        for path in sorted(tools.EPISODES.glob("*.json")):
            rec = json.loads(path.read_text())
            acts = [op for op in rec["ops"] if op["op"] == "act"]
            if acts and acts[-1]["result"]["success"] and path.stem not in names:
                names.append(path.stem)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; pick a new --out")
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features(args.size), root=out,
                               robot_type="so101_follower", use_videos=True)
    index = []
    for name in names:
        frames, meta = record(name, args.size)
        if not meta["solved"] or not meta["replay_matched"]:
            print(f"skip {name}: {meta}")
            continue
        for k in range(len(frames["action"])):
            ds.add_frame({
                "observation.state": frames["state"][k],
                "action": frames["action"][k],
                "observation.images.front": frames["front"][k],
                "observation.images.wrist": frames["wrist"][k],
                "op_index": np.array([frames["op"][k]], dtype=np.int64),
                "task": TASK,
            })
        ds.save_episode()
        meta["frames"] = len(frames["action"])
        index.append(meta)
        print(f"saved {name}: {meta['frames']} frames, {meta['moves']} moves, {meta['robot_s']} s")
    ds.finalize()
    (out / "demos.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} episodes in {out}")


if __name__ == "__main__":
    main()
