"""When the loose objects in a scene count as at rest. Both task envs call this, so their numbers can't drift apart.

Reset used to hand over a scene once max |qvel| over every DOF fell to 0.2, arm joints included. That let an episode
start with a shelf cylinder sliding 22 mm in its first second. This checks only the objects' free joints, in plain
units, and asks for the stillness to last.
"""
import numpy as np

import mujoco

REST_SPEED = 0.002  # m/s, the fastest any object's centre may move
REST_SPIN = 0.1  # rad/s, the fastest any object may turn
REST_FRAMES = 15  # consecutive 30 Hz frames both have to hold for, 0.5 s
REST_SECONDS = 3.0  # give up after this long
FRAME_HZ = 30


def free_dofs(model, bodies):
    """The first DOF address of each body's free joint. Raises if a body isn't free."""
    out = []
    for name in bodies:
        body = model.body(name)
        joint = body.jntadr[0]
        if body.jntnum[0] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{name} has no free joint")
        out.append(int(model.jnt_dofadr[joint]))
    return np.array(out, dtype=int)


def still(data, dofs):
    """True when every object moves slower than REST_SPEED and turns slower than REST_SPIN right now."""
    qvel = data.qvel
    linear = np.stack([qvel[a:a + 3] for a in dofs])
    angular = np.stack([qvel[a + 3:a + 6] for a in dofs])
    return bool(np.linalg.norm(linear, axis=1).max() <= REST_SPEED
                and np.linalg.norm(angular, axis=1).max() <= REST_SPIN)


def come_to_rest(step, data, dofs):
    """Call `step`, one 30 Hz frame, until the objects stay still for REST_FRAMES frames in a row.

    Returns False if REST_SECONDS pass first. Returns at once if they are already still for long enough, so a scene
    that is truly at rest costs REST_FRAMES frames.
    """
    run = 0
    for _ in range(int(round(REST_SECONDS * FRAME_HZ))):
        step()
        run = run + 1 if still(data, dofs) else 0
        if run >= REST_FRAMES:
            return True
    return False
