"""Tally every demonstrator episode: outcome, moves, undos, missed picks, robot time.

    python demonstrator/report.py              counts from the episode files alone
    python demonstrator/report.py --time       also replays each episode for its robot time (parallel)
    python demonstrator/report.py --json OUT   write the per-episode rows as JSON too

A missed pick is a kept pick that ended with the gripper at 14 or below (empty). Undone attempts are counted apart,
since they are not in the demonstration.
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPISODES = HERE / "episodes"
MAX_MOVES = 15


def row(path):
    rec = json.loads(path.read_text())
    acts = [op for op in rec["ops"] if op["op"] == "act"]
    picks = [a for a in acts if a["kind"] == "pick"]
    missed = [a for a in picks if a["result"]["grip"] <= 14 and not a["result"]["success"]]
    end = "going"
    if acts and acts[-1]["result"]["success"]:
        end = "solved"
    elif acts and acts[-1]["result"]["terminated"]:
        end = "failed"
    elif len(acts) >= MAX_MOVES:
        end = "out of moves"
    undone = rec.get("undone", [])
    undone_acts = [op for u in undone for op in u["ops"] if op["op"] == "act"]
    return {"name": path.stem, "seed": rec["seed"], "start": "buried" if rec["buried"] else "partial", "end": end,
            "moves": len(acts), "picks": len(picks), "places": sum(a["kind"] == "place" for a in acts),
            "pushes": sum(a["kind"] == "push" for a in acts), "missed_picks": len(missed), "undos": len(undone),
            "undone_spills": sum(1 for a in undone_acts if a["result"]["terminated"] and not a["result"]["success"])}


def robot_time(name):
    sys.path.insert(0, str(HERE))
    import tools
    ep = tools.Episode(name)
    t = round(ep.robot_s, 1)
    ep.close()
    return t


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--prefix", default="d", help="only episodes whose name starts with this")
    args = ap.parse_args()
    rows = [row(p) for p in sorted(EPISODES.glob("*.json")) if p.stem.startswith(args.prefix)]
    if args.time:
        with mp.get_context("spawn").Pool(min(16, len(rows) or 1)) as pool:
            for r, t in zip(rows, pool.map(robot_time, [r["name"] for r in rows])):
                r["robot_s"] = t
    print(f"{'name':>7} {'seed':>5} {'start':>7} {'end':>12} {'moves':>5} {'picks':>5} {'missed':>6} {'undos':>5}"
          + (f" {'robot s':>7}" if args.time else ""))
    for r in rows:
        print(f"{r['name']:>7} {r['seed']:5d} {r['start']:>7} {r['end']:>12} {r['moves']:5d} {r['picks']:5d} "
              f"{r['missed_picks']:6d} {r['undos']:5d}" + (f" {r['robot_s']:7.1f}" if args.time else ""))
    done = [r for r in rows if r["end"] != "going"]
    solved = [r for r in done if r["end"] == "solved"]
    print(f"\n{len(rows)} episodes: {len(solved)} solved, {len(done) - len(solved)} not, "
          f"{len(rows) - len(done)} still going")
    for start in ("buried", "partial"):
        s = [r for r in solved if r["start"] == start]
        d = [r for r in done if r["start"] == start]
        if d:
            line = (f"  {start}: {len(s)}/{len(d)} solved, mean {mean([r['moves'] for r in s]):.1f} moves, "
                    f"{mean([r['undos'] for r in s]):.1f} undos")
            if args.time:
                line += f", {mean([r['robot_s'] for r in s]):.0f} s robot time"
            print(line)
    picks = sum(r["picks"] for r in done)
    if picks:
        print(f"  kept picks that missed: {sum(r['missed_picks'] for r in done)} of {picks}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
