"""Contact checks against the arm, shared by the task envs."""

from so101sim import ik

JAW_PREFIXES = ("fixed_jaw", "moving_jaw")


def jaw_geom_ids(model):
    """Every geom on either jaw: the tip spheres plus the boxes named fixed_jaw_* and moving_jaw_*."""
    ids = {model.geom(name).id for name in ik.JAW_TIP_GEOMS}
    ids |= {i for i in range(model.ngeom) if model.geom(i).name.startswith(JAW_PREFIXES)}
    return frozenset(ids)


def arm_geom_ids(model):
    """Every geom on a body below the arm's `base`, which is every link that moves."""
    base = model.body("base").id
    moving = set()
    for b in range(model.nbody):
        a = b
        while a not in (0, base):
            a = model.body_parentid[a]
        if a == base and b != base:
            moving.add(b)
    return frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] in moving)


def is_grasped(model, data, geom_id, jaw_ids):
    """True when geoms from both jaws are touching `geom_id`."""
    touched = set()
    for c in data.contact[: data.ncon]:
        if geom_id in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == geom_id else c.geom1
            if other in jaw_ids:
                touched.add(model.geom(other).name.split("_jaw")[0])
    return {"fixed", "moving"} <= touched


def touching(data, group_a, group_b):
    """True if any current contact joins a geom in `group_a` to a geom in `group_b`."""
    for c in data.contact[: data.ncon]:
        if (c.geom1 in group_a and c.geom2 in group_b) or (c.geom2 in group_a and c.geom1 in group_b):
            return True
    return False
