"""Skills built on the repo's shelf primitives without changing them. Importing sets up `common` first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/sahal, for `common`

import common  # noqa: E402,F401  sets sys.path and MUJOCO_GL before mujoco loads
