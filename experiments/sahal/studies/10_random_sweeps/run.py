"""Study 10: how often does one random but sensible sweep make the red retrievable? (Dhruv's idea)

    experiments/sahal/.venv/bin/python experiments/sahal/studies/10_random_sweeps/run.py [--quick] [--workers 10]
        [--parts scenes trials analyse]

scenes: random scenes of the red and 3 grays (fixed seeds, no overlaps, no hand-tuning), kept only if the red stands
in study 01's empty-shelf graspable zone and can_retrieve(mode="exact") says no at the start. The first N_SCENES
candidates that pass are saved to experiments/sahal/scenes/random4/ (JSON and a picture each).

trials: for each scene, TRIALS_PER_SCENE trials from the same settled start. A trial samples a sweep: any of the 4
cylinders, a direction anywhere not toward the robot, a stroke of 2 to 12 cm, starting just behind that cylinder
(skills.sweep's start, 5 mm short of it) moved up to 1 cm sideways and up to 1 cm further back. A sweep refused
before moving (IK, path check, off the board) is logged and a new one sampled; nothing that ran is resampled. After
the sweep: toppled, jammed against a wall, gray or red off the board, glitch, or can_retrieve(mode="exact") decides
success or still blocked. Friction 0.55. analyse: redo the summary and plots from an existing trials.csv. --quick: the first scene, QUICK_TRIALS trials, into outputs/quick/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/sahal, for `common` and `skills`

import common  # noqa: E402,F401

from common import random_sweeps, runner, scenes, sweep_videos  # noqa: E402
from skills import sweep as sweep_skill  # noqa: E402
from so101sim.shelf import TARGET  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "outputs"
SETTINGS = random_sweeps.Settings(study_seed=10, scene_dir=scenes.SCENES_DIR / "random4", out_dir=OUT_DIR,
                                  scene_prefix="random4", friction=0.55, n_scenes=10, n_grays=3,
                                  trials_per_scene=500, quick_trials=20)  # the rest: random_sweeps.Settings defaults
# videos part: one scene, and per video the first trial (lowest number) in trials.csv that matches
VIDEO_SCENE = "random4_0019"
VIDEO_KEY_GRAY = "cyl_gray_0"  # this scene's gray whose sweep most often frees the red (by_cylinder.csv)
LONG_STROKE_CM = 8.0
VIDEO_PICKS = [
    sweep_videos.Pick("success_key_gray", VIDEO_SCENE, "success", VIDEO_KEY_GRAY, (0.0, LONG_STROKE_CM)),
    sweep_videos.Pick("success_red", VIDEO_SCENE, "success", TARGET, (0.0, LONG_STROKE_CM)),
    sweep_videos.Pick("success_long", VIDEO_SCENE, "success", None, (LONG_STROKE_CM, 100.0)),
    sweep_videos.Pick("fail_still_blocked", VIDEO_SCENE, "still blocked", VIDEO_KEY_GRAY),
    sweep_videos.Pick("fail_toppled", VIDEO_SCENE, "toppled"),
    sweep_videos.Pick("fail_jammed", VIDEO_SCENE, "jammed"),
]
VIDEO_LAYOUT = "front+wrist"
MAX_VIDEO_S = 30.0


# ---- videos ----------------------------------------------------------------------------------------------------

def part_videos(out: Path) -> None:
    """Replay the VIDEO_PICKS trials with the front and wrist cameras; one mp4 each in outputs/videos/."""
    rows = [random_sweeps.typed(r) for r in runner.read_csv(out / "trials.csv")]
    video_dir = out / "videos"
    report = sweep_videos.make_videos(SETTINGS, rows, VIDEO_PICKS, video_dir, VIDEO_LAYOUT, after_failure_s=0.0,
                                      max_seconds=MAX_VIDEO_S)
    runner.write_csv(video_dir / "videos.csv", report)
    runner.save_config(video_dir, {"picks": [p.__dict__ for p in VIDEO_PICKS], "layout": VIDEO_LAYOUT,
                                   "fps": sweep_videos.VIDEO_FPS, "slow_motion_repeat": sweep_videos.SLOW_MOTION_REPEAT,
                                   "slow_motion_pad_frames": sweep_videos.SLOW_MOTION_PAD_FRAMES,
                                   "end_hold_s": sweep_videos.END_HOLD_S})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="first scene, quick_trials trials, into outputs/quick/")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--parts", nargs="+", default=["trials"], choices=["scenes", "trials", "analyse", "videos"])
    args = parser.parse_args()
    out = OUT_DIR / "quick" if args.quick else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if "scenes" in args.parts:
        random_sweeps.part_scenes(SETTINGS, args.workers)
    if "trials" in args.parts:
        random_sweeps.part_trials(SETTINGS, out, args.quick, args.workers)
    if "analyse" in args.parts:
        random_sweeps.part_analyse(SETTINGS, out, args.quick)
    if "videos" in args.parts:
        part_videos(out)
    if set(args.parts) == {"videos"}:
        return  # the videos keep their own config.json; the study's stays as the trials left it
    runner.save_config(out, {"quick": args.quick, "parts": args.parts, **SETTINGS.config(),
                             "sweep_heights_m": sweep_skill.SWEEP_HEIGHTS_M,
                             "contact_speed_m_s": sweep_skill.CONTACT_SPEED_M_S})


if __name__ == "__main__":
    main()
