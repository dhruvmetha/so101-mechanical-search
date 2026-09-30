"""Catalog the Google Scanned Objects MuJoCo models and pick the ones a top-down SO-101 grasp could hold.

Written 2026-09-30 for docs/reports/2026-09-30-realistic-objects.md. Reads kevinzakka/mujoco_scanned_objects
(one folder per object with model.xml, model.obj, texture.png and 32 V-HACD hulls) plus the category list from the
Gazebo Fuel API, which that repo doesn't carry. For every object it records the visual mesh's extents in its own
frame (metres, z up, origin at the base), the hull count, whether model.xml compiles in this MuJoCo, the volume
MuJoCo computes for the visual mesh alone, and the mass the shipped model.xml gets. That last one is inflated: every geom
gets MuJoCo's default density of 1000 kg/m^3, and the visual mesh counts once and the 32 overlapping hulls again.

An object "fits" when, sorted smallest to largest, its extents are at least MIN_THICK, at most MAX_WIDTH and at most
MAX_LENGTH. A top-down grasp closes across the middle extent of an object lying on its largest face, so MAX_WIDTH is
the jaw limit with margin: the sim opens to 133 mm, the real arm's calibration nearer 82 mm (CLAUDE.md).

    python scripts/objects_catalog.py                 # writes outputs/objects_2026-09-30/gso_catalog.csv
    python scripts/objects_catalog.py --workers 16
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import so101sim  # noqa: F401  sets MUJOCO_GL before mujoco loads
import mujoco

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path("/common/users/dm1487/assets")
GSO = ASSETS / "mujoco_scanned_objects" / "models"
FUEL_META = ASSETS / "gso_fuel_metadata.json"
OUT = ROOT / "outputs" / "objects_2026-09-30" / "gso_catalog.csv"

MIN_THICK = 0.015
MAX_WIDTH = 0.08
MAX_LENGTH = 0.12
VOLUME_XML = """<mujoco><compiler meshdir="{folder}"/><asset><mesh name="m" file="model.obj"/></asset>
<worldbody><body><freejoint/><geom type="mesh" mesh="m" density="1000"/></body></worldbody></mujoco>"""
TOY_CATEGORIES = {"Toys", "Action Figures", "Legos", "Stuffed Toys", "Board Games"}


def obj_vertices(path):
    rows = [line.split()[1:4] for line in open(path) if line.startswith("v ")]
    return np.asarray(rows, dtype=float)


def inspect(name):
    folder = GSO / name
    row = {"name": name}
    v = obj_vertices(folder / "model.obj")
    ext = v.max(0) - v.min(0)
    row.update(ext_x=ext[0], ext_y=ext[1], ext_z=ext[2], base_z=v[:, 2].min(), visual_verts=len(v))
    hulls = sorted(folder.glob("model_collision_*.obj"))
    row["hulls"] = len(hulls)
    row["hull_verts"] = sum(len(obj_vertices(h)) for h in hulls)
    try:
        m = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
        row["compiles"] = 1
        row["mass_shipped"] = float(m.body_mass[1])
        row["error"] = ""
        vis = mujoco.MjModel.from_xml_string(VOLUME_XML.format(folder=folder))
        row["volume_cm3"] = float(vis.body_mass[1]) / 1000.0 * 1e6
    except Exception as e:  # a bad mesh fails the whole model, so record it and move on
        row["compiles"] = 0
        row["mass_shipped"] = float("nan")
        row["volume_cm3"] = float("nan")
        row["error"] = str(e).splitlines()[0][:200]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    meta = [m for m in json.load(open(FUEL_META)) if isinstance(m, dict)]
    cats = {m["name"]: ";".join(m.get("categories") or []) for m in meta}
    names = sorted(p.name for p in GSO.iterdir() if p.is_dir())
    with Pool(args.workers) as pool:
        rows = pool.map(inspect, names, chunksize=8)
    for r in rows:
        r["category"] = cats.get(r["name"], "")
        a, b, c = sorted([r["ext_x"], r["ext_y"], r["ext_z"]])
        r["fits"] = int(a >= MIN_THICK and b <= MAX_WIDTH and c <= MAX_LENGTH)
        r["toy"] = int(bool(set(r["category"].split(";")) & TOY_CATEGORIES))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fits = [r for r in rows if r["fits"]]
    print(f"{len(rows)} objects, {sum(r['compiles'] for r in rows)} compile in MuJoCo {mujoco.__version__}")
    for r in rows:
        if not r["compiles"]:
            print("  fails:", r["name"], r["error"])
    print(f"{len(fits)} fit {MIN_THICK*100:.1f} cm <= thickness, width <= {MAX_WIDTH*100:.0f} cm, "
          f"length <= {MAX_LENGTH*100:.0f} cm")
    print(f"  of those, {sum(r['toy'] for r in fits)} are in a toy category "
          f"(of {sum(r['toy'] for r in rows)} toys overall)")
    by_cat = {}
    for r in fits:
        by_cat[r["category"] or "(none)"] = by_cat.get(r["category"] or "(none)", 0) + 1
    print("  fitting objects by category:", dict(sorted(by_cat.items(), key=lambda kv: -kv[1])))
    hulls = np.array([r["hulls"] for r in rows])
    print(f"hulls per object: min {hulls.min()}, median {np.median(hulls):.0f}, max {hulls.max()}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
