"""Does a softer actuator gain reproduce the real arm's measured gravity droop?

The so101 session measured the physical SO-101 sagging below its commanded pose by an amount proportional
to horizontal reach: -9.06 mm per 100 mm, R^2 0.59 over 303 samples, never positive. It then solved for
the joint stiffness that would explain that, getting about 21 N.m/rad against menagerie's kp=998.22, and
asked for this check.

The interesting question is not whether kp=21 fits. It is whether a STATIC run fits. If commanding each
point and letting it settle reproduces the slope, the droop is pure joint compliance and raising P on the
real servo should fix it. If only the moving run reproduces it, the droop is velocity-dependent, it comes
from damping or friction rather than stiffness, and raising P will help less than expected.

Nothing here touches the bin-picking sim. Gains are overridden on an in-memory copy of the model.

    python scripts/sag_experiment.py
"""

import argparse

import mujoco
import numpy as np

import so101sim
from so101sim import ik, units

# so101's spec, to make the numbers comparable with its hardware run.
CIRCLE_CENTRE = np.array([0.300, 0.000, 0.220])
CIRCLE_RADIUS = 0.070
CIRCLE_POINTS = 72
STREAM_HZ = 30
STREAM_DEG_PER_S = 12.0

# The URDF frame so101 measured at, given as an offset from the MJCF site. Note this disagrees with
# docs/.../2026-09-18-so101-sim-design.md, which records the same 19.9 mm along the site's x instead of
# its z. The offset is rigid either way, so it barely moves a commanded-minus-achieved difference, but
# somebody should pin down which axis is right before it matters.
TOOL_OFFSET = np.array([0.000000, 0.000002, -0.019899])

TARGET_SLOPE = -0.09063  # mm of sag per mm of reach, measured on hardware
ARM = units.JOINT_NAMES[:5]


def set_kp(model, kp):
    """Override the position actuators' stiffness, leaving damping and force limits alone."""
    for i in range(model.nu):
        model.actuator_gainprm[i][0] = kp
        model.actuator_biasprm[i][1] = -kp


def set_frictionloss(model, value):
    """Dry friction on the arm joints, in N.m. Creates a deadband: a joint stops moving once the
    actuator and gravity disagree by less than this, so it droops only where gravity torque exceeds it."""
    for name in ARM:
        model.dof_frictionloss[model.jnt_dofadr[model.joint(name).id]] = value


def tool_point(model, data):
    rot = data.site_xmat[model.site("gripperframe").id].reshape(3, 3)
    return data.site_xpos[model.site("gripperframe").id] + rot @ TOOL_OFFSET


def plan_circle(model):
    """IK for each point on the circle. Returns (joint targets, planned tool points)."""
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in ARM]]
    targets, planned, errors = [], [], []
    seed = None
    for k in range(CIRCLE_POINTS):
        angle = 2 * np.pi * k / CIRCLE_POINTS
        goal = CIRCLE_CENTRE + CIRCLE_RADIUS * np.array([np.cos(angle), np.sin(angle), 0.0])
        q, err = ik.solve_ik(model, goal, q_init=seed)
        seed = q
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        targets.append(q)
        planned.append(tool_point(model, data).copy())
        errors.append(err)
    return np.array(targets), np.array(planned), np.array(errors)


def fit(sag_mm, reach_mm):
    """Slope of sag against horizontal reach, which is the reach-independent comparison."""
    slope, intercept = np.polyfit(reach_mm, sag_mm, 1)
    pred = slope * reach_mm + intercept
    ss_res = float(np.sum((sag_mm - pred) ** 2))
    ss_tot = float(np.sum((sag_mm - sag_mm.mean()) ** 2))
    return {
        "slope": float(slope),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mean": float(sag_mm.mean()),
        "sd_raw": float(sag_mm.std()),
        "sd_resid": float(np.std(sag_mm - pred)),
        "min": float(sag_mm.min()),
        "max": float(sag_mm.max()),
    }


def run_static(model, targets, planned, settle_s=3.0):
    """Command each point, let it settle, then read. Isolates static compliance."""
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in ARM]]
    steps = int(settle_s / model.opt.timestep)
    sag, reach = [], []
    for q, want in zip(targets, planned):
        data.qpos[adr] = q
        data.qvel[:] = 0.0
        data.ctrl[:5] = q
        data.ctrl[5] = 0.0
        for _ in range(steps):
            mujoco.mj_step(model, data)
            if np.abs(data.qvel).max() < 1e-4:
                break
        sag.append((tool_point(model, data)[2] - want[2]) * 1000.0)
        reach.append(np.hypot(*want[:2]) * 1000.0)
    return np.array(sag), np.array(reach)


def run_streamed(model, targets, planned):
    """Drive the circle at 12 deg/s, sampling every control step, like the hardware run."""
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[[model.joint(n).id for n in ARM]]
    kin = mujoco.MjData(model)
    substeps = int(round((1 / STREAM_HZ) / model.opt.timestep))
    per_frame = STREAM_DEG_PER_S / STREAM_HZ

    data.qpos[adr] = targets[0]
    data.ctrl[:5] = targets[0]
    mujoco.mj_forward(model, data)

    sag, reach = [], []
    loop = list(targets) + [targets[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        frames = max(1, int(np.ceil(np.degrees(np.abs(b - a)).max() / per_frame)))
        for i in range(1, frames + 1):
            cmd = a + (i / frames) * (b - a)
            data.ctrl[:5] = cmd
            data.ctrl[5] = 0.0
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            kin.qpos[adr] = cmd
            mujoco.mj_kinematics(model, kin)
            want = tool_point(model, kin)
            sag.append((tool_point(model, data)[2] - want[2]) * 1000.0)
            reach.append(np.hypot(*want[:2]) * 1000.0)
    return np.array(sag), np.array(reach)


def show(label, stats):
    print(f"  {label:26s} slope {stats['slope'] * 100:+7.2f} mm/100mm   mean {stats['mean']:+7.2f}   "
          f"R2 {stats['r2']:5.2f}   resid sd {stats['sd_resid']:5.2f}   "
          f"range {stats['min']:+6.2f}..{stats['max']:+6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp", type=float, nargs="+", default=[998.22, 200.0, 100.0, 50.0, 21.0, 10.0])
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(so101sim.SCENE_PATH))
    targets, planned, errors = plan_circle(model)
    reach = np.hypot(planned[:, 0], planned[:, 1]) * 1000
    print(f"circle: {CIRCLE_POINTS} points, radius {CIRCLE_RADIUS * 1000:.0f} mm at z={CIRCLE_CENTRE[2]:.3f}")
    print(f"IK residual: max {errors.max() * 1000:.3f} mm, mean {errors.mean() * 1000:.3f} mm")
    print(f"horizontal reach spans {reach.min():.0f} to {reach.max():.0f} mm")
    if errors.max() > 2e-3:
        print("  WARNING: IK cannot reach part of this circle, so the sag numbers below are contaminated")
    print(f"\nhardware reference: slope {TARGET_SLOPE * 100:+.2f} mm/100mm, mean -8.84, "
          f"R2 0.59, resid sd 3.25\n")

    for kp in args.kp:
        set_kp(model, kp)
        s_sag, s_reach = run_static(model, targets, planned)
        d_sag, d_reach = run_streamed(model, targets, planned)
        print(f"kp = {kp:7.2f}")
        show("static (settled)", fit(s_sag, s_reach))
        show(f"streamed ({STREAM_DEG_PER_S:.0f} deg/s)", fit(d_sag, d_reach))

    set_kp(model, 21.0)
    s_sag, s_reach = run_static(model, targets, planned)
    print("\nkp=21 static, binned by reach the way the hardware run was binned:")
    for lo in range(280, 360, 20):
        sel = (s_reach >= lo) & (s_reach < lo + 20)
        if sel.sum():
            print(f"  {lo}-{lo + 20} mm  n={sel.sum():3d}  sag {s_sag[sel].mean():+7.2f} mm  "
                  f"sd {s_sag[sel].std():5.2f}")


if __name__ == "__main__":
    main()
