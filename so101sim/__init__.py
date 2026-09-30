import os
from pathlib import Path

# arrakis has no display. EGL renders offscreen on the GPU. Must be set before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

SCENE_PATH = Path(__file__).resolve().parent.parent / "models" / "scene.xml"
SEARCH_SCENE_PATH = Path(__file__).resolve().parent.parent / "models" / "search_scene.xml"
SHELF_SCENE_PATH = Path(__file__).resolve().parent.parent / "models" / "shelf_scene.xml"
