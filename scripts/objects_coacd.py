"""Re-decompose the pile test's objects with CoACD, to see what fewer hulls buy against the shipped 32 V-HACD hulls.

Written 2026-09-30 for docs/reports/2026-09-30-realistic-objects.md. CoACD (Wei et al., SIGGRAPH 2022) isn't in the
so101sim env; install it anywhere and put it on PYTHONPATH, for example
    pip install --no-deps --target $SCRATCH/pylib coacd
    PYTHONPATH=$SCRATCH/pylib python scripts/objects_coacd.py --max-hulls 8
It reads each object's visual mesh from kevinzakka/mujoco_scanned_objects and writes hull_0.obj, hull_1.obj, ... to
/common/users/dm1487/assets/gso_coacd/<object>/, where scripts/objects_pile_test.py --hulls coacd looks for them.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from objects_pile_test import COACD, GSO, OBJECTS  # noqa: E402


def decompose(job):
    name, threshold, max_hulls = job
    import coacd
    import trimesh
    coacd.set_log_level("error")
    mesh = trimesh.load(GSO / name / "model.obj", force="mesh", process=False)
    t0 = time.perf_counter()
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=max_hulls,
                            seed=0)
    seconds = time.perf_counter() - t0
    out = COACD / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("hull_*.obj"):
        old.unlink()
    for k, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f).export(out / f"hull_{k}.obj")
    return name, len(parts), seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity threshold (its default 0.05)")
    ap.add_argument("--max-hulls", type=int, default=8, help="-1 for no limit")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        for name, n, s in pool.imap_unordered(decompose, [(o, args.threshold, args.max_hulls) for o in OBJECTS]):
            print(f"{name}: {n} hulls in {s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
