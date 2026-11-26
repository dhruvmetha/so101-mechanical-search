"""Demo videos of saved random-sweep trials (studies 10 and 11): replay a trial exactly, then render it.

    env = RecordingShelf()                                   # after cylinder_size.use(...)
    replay = replay_trial(settings, env, scene_index, row, after_failure_s=2.0)
    frames = compose(env, replay, label, layout="top+wrist")  # or "front+wrist" (study 10's videos)
    write_mp4(path, frames)

The replay stores only the state of every 30 Hz frame (qpos), whether the arm touches a cylinder and the phase;
images are rendered afterwards from those states on a separate MjData, so recording never touches the physics. The
replay uses common.random_sweeps' own sampling with the trial's seed, so it runs the trial's sweep exactly; the
caller compares its outcome with the saved row. With `after_failure_s` the physics keeps running for that long after
a failed sweep or retrieval (the arm holding still), for the video only: the outcome is decided before.

Top-down view: a free camera straight down on the board, framed on the cylinders and the sweep, with the shelf's top
board hidden, the arm see-through and softer light (rendering only). The sweep is drawn as an arrow from its start to
its planned end and the swept cylinder gets a yellow ring, during the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import common  # noqa: F401  sets sys.path and MUJOCO_GL before mujoco loads
import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import cylinder_size, random_sweeps, retrieval_check, scenes
from common.shelf_env import RedAndGraysShelf
from so101sim import grasp, shelf_layout
from so101sim.env import CAMERAS
from so101sim.shelf import TARGET

VIDEO_FPS = 30
SLOW_MOTION_REPEAT = 2  # each frame shown twice (0.5x) near an arm-cylinder touch
SLOW_MOTION_PAD_FRAMES = 6  # this many frames either side of a touch
END_HOLD_S = 1.0  # last frame held at the end
LABEL_BAR_PX = 48
LABEL_FONT_PX = 17
VIEW_HEIGHT_PX = 400  # top-down and wrist views are this tall
WRIST_WIDTH_PX = 533  # the wrist camera's 4:3 at VIEW_HEIGHT_PX
MACRO_BLOCK_PX = 16  # video width and height are padded to a multiple of this
TOP_MARGIN_M = 0.02  # framing: this much past the cylinders' edges and the sweep
TOP_PERSPECTIVE_SCALE = 1.25  # cylinder tops are nearer the camera than the board, so they spread outward
TOP_MIN_HALF_EXTENT_M = 0.09
ARM_ALPHA = 0.35  # arm geoms see-through in the top view
LIGHT_SCALE = 0.45  # scene lights dimmed in the top view (the board washes out otherwise)
HEADLIGHT_DIFFUSE = 0.3
ARROW_WIDTH_M = 0.004
ARROW_LIFT_M = 0.005  # arrow drawn this far above the cylinder tops, so no cylinder hides it
RING_EXTRA_M = 0.006  # the swept cylinder's ring reaches this far past its radius
RING_HEIGHT_M = 0.0005
ARROW_RGBA = (1.0, 0.85, 0.1, 1.0)
RING_RGBA = (1.0, 0.85, 0.1, 0.9)
FRONT_WRIST_HEIGHT_PX, FRONT_WRIST_WIDTH_PX = 480, 640  # study 10's front | wrist layout (the env's camera size)
QUALITY = 6  # imageio-ffmpeg quality (0 to 10)


@dataclass
class Replay:
    """What a replayed trial did and what the video needs to draw it."""

    outcome: str
    sample: dict
    refusals: int
    max_contact_force_n: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    cylinder: str
    retrieve_reason: str | None = None
    extra: dict = field(default_factory=dict)


class RecordingShelf(RedAndGraysShelf):
    """RedAndGraysShelf that stores, every 30 Hz frame, the state (qpos), whether the arm touches a cylinder at the
    frame's end, and the phase. Reads state only."""

    def __init__(self) -> None:
        super().__init__()
        self.clear_recording()

    def clear_recording(self) -> None:
        self.states: list[np.ndarray] = []
        self.touching: list[bool] = []
        self.frame_phase: list[str] = []

    def _on_frame(self, env) -> None:
        super()._on_frame(env)
        inner = self.env
        self.states.append(self.data.qpos.copy())
        self.touching.append(bool(grasp.touching(self.data, inner._cyl_geoms, inner._arm_ids)))
        self.frame_phase.append(self.phase or "")

    def hold(self, seconds: float, phase: str) -> None:
        """Keep the physics running with the arm's commands unchanged, recording, for the video only."""
        self.phase = phase
        for _ in range(round(seconds * VIDEO_FPS)):
            mujoco.mj_step(self.model, self.data, nstep=self.env._n_substeps)
            self._on_frame(self)


def replay_trial(settings: random_sweeps.Settings, env: RecordingShelf, scene_index: int, row: dict,
                 after_failure_s: float = 0.0) -> Replay:
    """Re-run a saved trial from the scene's settled start: the same sweep (same seed), then, unless the sweep
    already failed, the pick + pull the exact check simulates. After a failure, `after_failure_s` more of physics."""
    scene = scenes.load_scene(settings.scene_dir / f"{row['scene']}.json")
    scenes.place_scene(env, scene)
    env.log = type(env.log)()
    env.clear_recording()
    env.phase = "sweep"
    refusals: list[str] = []
    sample, plan, info = random_sweeps.first_executable_sweep(
        settings, env, random_sweeps.trial_rng(settings, scene_index, int(row["trial"])), sorted(scene.positions),
        refusals)
    outcome = random_sweeps.classify(env, info)
    replay = Replay(outcome=outcome or "", sample=sample, refusals=len(refusals),
                    max_contact_force_n=round(max(env.log.max_contact_force_n, env.log.max_gray_contact_force_n), 1),
                    start_xy=tuple(plan.start_xy_m), end_xy=tuple(plan.end_xy_m), cylinder=sample["cylinder"])
    if outcome is not None:
        if after_failure_s:
            env.hold(after_failure_s, f"after the failure ({outcome})")
        return replay
    env.phase = "pick + pull"
    result = retrieval_check.pick_and_pull(env, env.env.cylinder_pose(TARGET)["pos"][:2].copy())
    replay.outcome = "success" if result["success"] else "still blocked"
    replay.retrieve_reason = result["reason"]
    if not result["success"] and after_failure_s:
        env.hold(after_failure_s, "after the failed retrieve")
    return replay


def matches(replay: Replay, row: dict) -> bool:
    """The replay ran the saved trial's sweep and ended the same way."""
    return (replay.outcome == row["outcome"] and replay.refusals == row["refusals"]
            and replay.sample["direction_deg"] == row["direction_deg"]
            and replay.max_contact_force_n == row["max_contact_force_n"])


def slow_frames(env: RecordingShelf) -> np.ndarray:
    """True for frames within SLOW_MOTION_PAD_FRAMES of an arm-cylinder touch."""
    touching = np.array(env.touching, dtype=bool)
    return np.convolve(touching, np.ones(2 * SLOW_MOTION_PAD_FRAMES + 1), mode="same") > 0


def compose(env: RecordingShelf, replay: Replay, label: str, layout: str) -> list[np.ndarray]:
    """Render every recorded state as video frames: label bar, then the views ("top+wrist" or "front+wrist"); frames
    near a touch repeated (0.5x); the last held END_HOLD_S."""
    view = _Views(env, replay, layout)
    font = ImageFont.load_default(size=LABEL_FONT_PX)
    frames = []
    try:
        for state, slow, phase in zip(env.states, slow_frames(env), env.frame_phase):
            image = view.render(state, sweeping=phase == "sweep")
            canvas = Image.new("RGB", _padded(image.shape[1], image.shape[0] + LABEL_BAR_PX), (20, 20, 20))
            canvas.paste(Image.fromarray(image), (0, LABEL_BAR_PX))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 3), label, fill=(240, 240, 240), font=font)
            draw.text((8, 5 + LABEL_FONT_PX), f"{phase}{'   0.5x (contact)' if slow else '   1x'}",
                      fill=(255, 200, 80) if slow else (170, 170, 170), font=font)
            frame = np.asarray(canvas)
            frames += [frame] * (SLOW_MOTION_REPEAT if slow else 1)
    finally:
        view.close()
    return frames + [frames[-1]] * round(END_HOLD_S * VIDEO_FPS)


def write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=VIDEO_FPS, codec="libx264", quality=QUALITY)


def _padded(width: int, height: int) -> tuple[int, int]:
    return (-(-width // MACRO_BLOCK_PX) * MACRO_BLOCK_PX, -(-height // MACRO_BLOCK_PX) * MACRO_BLOCK_PX)


class _Views:
    """Renders a stored state: the top-down view with the sweep drawn, or the front camera, next to the wrist."""

    def __init__(self, env: RecordingShelf, replay: Replay, layout: str) -> None:
        self.env, self.replay, self.layout = env, replay, layout
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        if layout == "front+wrist":
            self.renderer = mujoco.Renderer(self.model, FRONT_WRIST_HEIGHT_PX, FRONT_WRIST_WIDTH_PX)
        else:
            self.renderer = mujoco.Renderer(self.model, VIEW_HEIGHT_PX, VIEW_HEIGHT_PX)
            self.wrist = mujoco.Renderer(self.model, VIEW_HEIGHT_PX, WRIST_WIDTH_PX)
            self.camera = self._frame_top_camera()
            root = self.model.body_rootid[self.model.geom_bodyid[next(iter(grasp.jaw_geom_ids(self.model)))]]
            self.arm_geoms = [g for g in range(self.model.ngeom)
                              if self.model.body_rootid[self.model.geom_bodyid[g]] == root]

    def close(self) -> None:
        self.renderer.close()
        if self.layout != "front+wrist":
            self.wrist.close()

    def render(self, state: np.ndarray, sweeping: bool) -> np.ndarray:
        self.data.qpos[:] = state
        mujoco.mj_forward(self.model, self.data)
        if self.layout == "front+wrist":
            images = [self._camera(self.renderer, CAMERAS["front"]), self._camera(self.renderer, CAMERAS["wrist"])]
        else:
            images = [self._top(sweeping), self._camera(self.wrist, CAMERAS["wrist"])]
        return np.hstack(images)

    def _camera(self, renderer: mujoco.Renderer, name: str) -> np.ndarray:
        renderer.update_scene(self.data, camera=name)
        return renderer.render().copy()

    def _frame_top_camera(self) -> mujoco.MjvCamera:
        """Straight down, image up = away from the robot, framed on everywhere the cylinders stood before the pick
        (the pull carries the red out of the shelf) and on the sweep."""
        board_z = self.env.env.layout.board_z
        points = [np.asarray(self.replay.start_xy), np.asarray(self.replay.end_xy)]
        for state, phase in zip(self.env.states, self.env.frame_phase):
            if phase == "pick + pull":
                continue
            self.data.qpos[:] = state
            mujoco.mj_kinematics(self.model, self.data)
            points += [self.data.xpos[self.model.body(n).id][:2].copy() for n in self.env.env.cyls]
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        reach = float(np.max(high - low)) / 2 + shelf_layout.CYL_RADIUS + TOP_MARGIN_M
        half = max(reach * TOP_PERSPECTIVE_SCALE, TOP_MIN_HALF_EXTENT_M)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [*((low + high) / 2), board_z]
        camera.distance = half / np.tan(np.radians(self.model.vis.global_.fovy / 2))  # the free camera's fovy
        camera.elevation, camera.azimuth = -90.0, 0.0
        return camera

    def _top(self, sweeping: bool) -> np.ndarray:
        model = self.model
        saved = (model.geom_rgba.copy(), model.light_diffuse.copy(), model.light_castshadow.copy(),
                 model.vis.headlight.diffuse.copy())
        try:
            model.geom_rgba[model.geom("shelf_top").id, 3] = 0.0
            for g in self.arm_geoms:
                model.geom_rgba[g, 3] = min(model.geom_rgba[g, 3], ARM_ALPHA)
            model.light_diffuse[:] *= LIGHT_SCALE
            model.light_castshadow[:] = 0
            model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
            self.renderer.update_scene(self.data, camera=self.camera)
            if sweeping:
                self._draw_sweep()
            return self.renderer.render().copy()
        finally:
            model.geom_rgba[:], model.light_diffuse[:], model.light_castshadow[:] = saved[:3]
            model.vis.headlight.diffuse[:] = saved[3]

    def _draw_sweep(self) -> None:
        """The sweep's arrow (start to planned end) and a ring under the swept cylinder, as extra scene geoms."""
        scene, board_z = self.renderer.scene, self.env.env.layout.board_z
        body = self.model.body(self.replay.cylinder).id
        centre = self.data.xpos[body]
        ring = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(ring, mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([shelf_layout.CYL_RADIUS + RING_EXTRA_M, 0, RING_HEIGHT_M]),
                            np.array([centre[0], centre[1], board_z + RING_HEIGHT_M]), np.eye(3).ravel(),
                            np.array(RING_RGBA, dtype=np.float32))
        scene.ngeom += 1
        arrow = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.array(ARROW_RGBA, dtype=np.float32))
        z = board_z + 2 * shelf_layout.CYL_HALF_HEIGHT + ARROW_LIFT_M
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH_M,
                             np.array([*self.replay.start_xy, z]), np.array([*self.replay.end_xy, z]))
        scene.ngeom += 1


@dataclass(frozen=True)
class Pick:
    """A video to make: the first trial in trials.csv of `scene` with this outcome, cylinder (None: any) and stroke."""

    name: str
    scene: str
    outcome: str
    cylinder: str | None = None
    length_cm: tuple[float, float] = (0.0, 100.0)


def pick_trial(rows: list[dict], pick: Pick) -> dict:
    match = next((r for r in rows if r["scene"] == pick.scene and r["outcome"] == pick.outcome
                  and (pick.cylinder is None or r["cylinder"] == pick.cylinder)
                  and pick.length_cm[0] <= r["length_cm"] < pick.length_cm[1]), None)
    if match is None:
        raise SystemExit(f"no trial in {pick.scene} for {pick.name}")
    return match


def make_videos(settings: random_sweeps.Settings, rows: list[dict], picks: list[Pick], video_dir: Path, layout: str,
                after_failure_s: float, max_seconds: float) -> list[dict]:
    """Replay each pick's trial, render it and write <name>.mp4 and videos.csv into `video_dir`."""
    scene_index = {s.name: i for i, s in enumerate(random_sweeps.scene_list(settings, False))}
    cylinder_size.use(settings.size)
    env = RecordingShelf()
    report = []
    try:
        for pick in picks:
            row = pick_trial(rows, pick)
            replay = replay_trial(settings, env, scene_index[row["scene"]], row, after_failure_s)
            label = (f"{row['scene']}  trial {row['trial']}  swept {row['cylinder'].replace('cyl_', '')}"
                     f"  dir {row['direction_deg']:+.0f} deg  {row['length_cm']:.1f} cm  ->  {row['outcome'].upper()}")
            frames = compose(env, replay, label, layout)
            path = video_dir / f"{pick.name}.mp4"
            write_mp4(path, frames)
            seconds = len(frames) / VIDEO_FPS
            report.append({"video": path.name, "scene": row["scene"], "trial": row["trial"],
                           "cylinder": row["cylinder"], "direction_deg": row["direction_deg"],
                           "length_cm": row["length_cm"], "saved_outcome": row["outcome"],
                           "replayed_outcome": replay.outcome, "replay_matches": matches(replay, row),
                           "retrieve_reason": replay.retrieve_reason, "seconds": round(seconds, 1),
                           "over_limit": seconds > max_seconds, "size_mb": round(path.stat().st_size / 1e6, 2),
                           "cylinder_size": cylinder_size.current().name})
            print(report[-1], flush=True)
    finally:
        env.close()
    return report
