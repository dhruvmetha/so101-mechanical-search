"""Shared code for the shelf studies in experiments/sahal/studies.

Importing this package puts the repo root on sys.path, so `so101sim` imports without PYTHONPATH, and sets
MUJOCO_GL=cgl on macOS. Both must happen before `so101sim` (and so mujoco) is imported.
"""

import os
import sys
from pathlib import Path

SAHAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SAHAL_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")  # so101sim defaults to egl, which is Linux only
