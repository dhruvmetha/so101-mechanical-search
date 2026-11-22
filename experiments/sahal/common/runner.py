"""Running many independent jobs, and writing results: CSV rows and the config.json saved next to every output."""

from __future__ import annotations

import csv
import json
import multiprocessing
import platform
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import common  # noqa: F401
import mujoco
import numpy as np


def run_parallel(jobs: list, work: Callable, initializer: Callable, workers: int, label: str = "") -> list:
    """Run `work(job)` for every job, `workers` processes at a time, results in job order.

    Each worker calls `initializer` once (to build its own env). Every job must carry everything that decides its
    result (seed, settings), so the result does not depend on which worker ran it or what it ran before.
    """
    start = time.perf_counter()
    if workers <= 1:
        initializer()
        results = [work(job) for job in jobs]
    else:
        with multiprocessing.get_context("spawn").Pool(workers, initializer=initializer) as pool:
            results = pool.map(work, jobs, chunksize=2)
    print(f"{label}{len(jobs)} runs in {time.perf_counter() - start:.0f} s on {max(1, workers)} worker(s)", flush=True)
    return results


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_config(out_dir: Path, settings: dict) -> None:
    """config.json: the study's settings plus the command, versions and time, next to its outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "command": " ".join(sys.argv),
        "settings": settings,
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=_jsonable) + "\n")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
