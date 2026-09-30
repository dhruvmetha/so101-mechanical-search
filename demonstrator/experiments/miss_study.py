"""Why top-down picks closed on nothing. Replays every missed pick, kept or undone, and the successful picks of the same
episodes as a control group, frame by frame with the simulator's true state.

    python demonstrator/experiments/miss_study.py [--procs 8] [--out outputs/miss_study_2026-09-30] [--pictures]

Ground truth throughout: this is diagnosis after the fact, never play. A miss is a pick whose recorded gripper reads
<= 14 after the act. An undone pick is replayed on the line as it stood when it was played: the kept line's first
at_move-1 acts, plus any ops later undos took back from that stretch, then the undone ops. Every replayed act must
reproduce its recorded result exactly, or its rows are flagged `replay_ok: false`.

Per pick it records: the intended cube (parsed from the act's `why`) and the true nearest cube; the aim against the
intended cube's true pose (xy along and across the jaw axis, z, yaw against the nearest face, tilt); how far the arm
got down (the jaw midpoint at the planning gripper value, GRIP_PLAN, from the actual arm joints) against the aim;
every arm contact during descent, close and lift; where the target sat relative to the two jaw tips when the close
began; how it moved and turned during the close; the gripper reading at the end of the close and at the end; and
what tools.check said before the move.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

CUBE = 0.030
HALF = 0.015
LIFTED_DZ = 0.02
NAMES = r"(g\d+|red)"


# ---- which ops to replay ------------------------------------------------------------------------------------------

def acts_of(ops):
    return [k for k, op in enumerate(ops) if op["op"] == "act"]


def through_act(ops, n):
    """The ops up to and including the n-th act (n >= 0; n == 0 gives the ops before the first act... none)."""
    if n == 0:
        return []
    return ops[:acts_of(ops)[n - 1] + 1]


def undone_lines(rec):
    """For each undone entry, the full op line as it stood just before that undo (see the module docstring).

    Walking the entries latest first: the line before undo j is the first at_move-1 acts of the line before the next
    undo (or the final kept line), followed by the ops undo j took back. This rebuilds e402's undo of move 9, which
    happened while move 8, later undone too, was still in the line."""
    entries = rec.get("undone", [])
    line = rec["ops"]
    out = [None] * len(entries)
    for j in range(len(entries) - 1, -1, -1):
        u = entries[j]
        line = through_act(line, u["at_move"] - 1) + u["ops"]
        out[j] = line
    return out


def is_miss(op):
    return op["op"] == "act" and op["kind"] == "pick" and op["result"]["grip"] <= 14 and not op["result"]["terminated"]


def jobs_for(names, episodes):
    jobs = []
    for name in names:
        rec = json.loads((episodes / f"{name}.json").read_text())
        jobs.append({"name": name, "line": "kept", "undo_index": None, "ops": rec["ops"], "focus": None})
        for j, line in enumerate(undone_lines(rec)):
            u = rec["undone"][j]
            act = [op for op in u["ops"] if op["op"] == "act"][0]
            if not is_miss(act):
                continue
            jobs.append({"name": name, "line": f"undone{j}", "undo_index": j, "ops": line,
                         "focus": u["at_move"]})  # the act number (1-based) in `line` to study
    return jobs


# ---- the intended cube ------------------------------------------------------------------------------------------

VERB = re.compile(r"\b(lift|take|clear|grasp|pick|retry|regrasp|try|nudge|re-grasp)\b((?:\s+[\w'-]+,?){0,4}?)\s+"
                  + NAMES + r"\b(?!'s)", re.I)
RED_CONTEXT = re.compile(r"\bred\s+(line|pixels?|seams?|px|seen)\b", re.I)


def parse_target(why):
    """(the cube the `why` names as the one to grasp, how it was found). Returns short names: 'g4' or 'red'."""
    text = RED_CONTEXT.sub("", why)
    m = VERB.search(text)
    if m:
        return m.group(3).lower(), "verb"
    m = re.search(r"\b" + NAMES + r"\b(?!'s)", text, re.I)
    if m:
        return m.group(1).lower(), "first"
    m = re.search(r"\b" + NAMES + r"\b", text, re.I)
    return (m.group(1).lower(), "first-possessive") if m else (None, "none")


def full_name(short):
    if short is None:
        return None
    return "cube_red" if short == "red" else f"cube_gray_{short[1:]}"


def short_name(name):
    return "red" if name == "cube_red" else "g" + name.rsplit("_", 1)[1]


# ---- geometry helpers ---------------------------------------------------------------------------------------------

def quat_mat(q):
    import mujoco
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def tilt_deg(R):
    """Angle between the cube's most vertical axis and world z."""
    return float(np.degrees(np.arccos(np.clip(np.abs(R[2, :]).max(), 0, 1))))


def face_misalign_deg(R, u):
    """Angle between jaw axis u (unit, 3D) and the nearest cube face normal."""
    return float(np.degrees(np.arccos(np.clip(np.abs(R.T @ u).max(), 0, 1))))


def corners(pos, R):
    k = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * HALF
    return pos + k @ R.T


def rot_angle_deg(R0, R1):
    c = (np.trace(R0.T @ R1) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def other_of_(m, gid, names, intended):
    nm = m.geom(gid).name
    if nm in names:
        return "TARGET" if nm == intended else short_name(nm)
    return nm or f"geom{gid}"


def fold90(a):
    """Absolute angle difference modulo 90, in [0, 45]."""
    a = abs(a) % 90.0
    return min(a, 90.0 - a)


# ---- the replay ---------------------------------------------------------------------------------------------------

def study(job):
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    import tools  # sets up so101sim before mujoco
    import mujoco
    from so101sim import grasp, ik, primitives, units

    t0 = time.time()
    rec = json.loads((tools.EPISODES / f"{job['name']}.json").read_text())
    rec = {**rec, "ops": [dict(op) for op in job["ops"]]}
    rec.pop("undone", None)

    ctx = {"on": False}
    rows = []
    frames_npz = {}

    def on_advance(ep, row):
        if not ctx["on"]:
            return
        ctx["frames"].append(snap(ep, row))

    probe = {}

    def snap(ep, row):
        m, d = ep.m, ep.d
        if "data" not in probe:
            probe["data"] = mujoco.MjData(m)
            probe["grip22"] = primitives._grip_rad(m, primitives.GRIP_PLAN)
            probe["arm_ids"] = grasp.arm_geom_ids(m)
            probe["fixed_body"] = m.body("gripper").id
            probe["moving_body"] = m.body("moving_jaw_so101_v1").id
            probe["tips"] = [m.geom(g).id for g in ik.JAW_TIP_GEOMS]
            probe["site"] = m.site("gripperframe").id
            probe["qadr"] = [m.jnt_qposadr[m.body(n).jntadr[0]] for n in ep.env.cubes]
            probe["dadr"] = [m.jnt_dofadr[m.body(n).jntadr[0]] for n in ep.env.cubes]
            probe["cube_index"] = {m.geom(n).id: i for i, n in enumerate(ep.env.cubes)}
        pd = probe["data"]
        pd.qpos[:] = d.qpos
        pd.qpos[ep.adr[5]] = probe["grip22"]
        mujoco.mj_kinematics(m, pd)
        fk22 = ik.grasp_point(m, pd).copy()
        cubes = np.array([d.qpos[a:a + 7] for a in probe["qadr"]])
        speed = np.array([np.linalg.norm(d.qvel[a:a + 3]) for a in probe["dadr"]])
        arm = probe["arm_ids"]
        con = []
        # deepest penetration of each cube into anything that is not the arm (floor, walls, other cubes), metres
        pen = np.zeros(len(probe["qadr"]))
        for c in d.contact[:d.ncon]:
            for g_cube, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                i = probe["cube_index"].get(g_cube)
                if i is not None and g_other not in arm:
                    pen[i] = max(pen[i], -c.dist)
            if c.geom1 in arm and c.geom2 not in arm:
                con.append((int(c.geom1), int(c.geom2), float(c.dist), c.pos.copy()))
            elif c.geom2 in arm and c.geom1 not in arm:
                con.append((int(c.geom2), int(c.geom1), float(c.dist), c.pos.copy()))
        return {"row": None if row is None else np.asarray(row, float).copy(),
                "js": np.asarray(ep.env.joint_state, float).copy(),
                "fixed_tip": d.geom_xpos[probe["tips"][0]].copy(), "moving_tip": d.geom_xpos[probe["tips"][1]].copy(),
                "site": d.site_xmat[probe["site"]].reshape(3, 3).copy(), "fk22": fk22, "cubes": cubes, "speed": speed, "pen": pen,
                "con": con}

    original = tools.Episode._apply
    act_no = {"n": 0}

    def spy(self, op):
        if op["op"] != "act":
            return original(self, op)
        act_no["n"] += 1
        n = act_no["n"]
        if op["kind"] != "pick" or (job["focus"] is not None and n != job["focus"]):
            res = original(self, op)
            rows.append({"act": n, "kind": op["kind"], "replay_ok": same(op, res), "studied": False})
            return res
        pre = pre_act(self, op)
        ctx.update(on=True, frames=[])
        res = original(self, op)
        ctx["frames"].append(snap(self, None))
        ctx["on"] = False
        r = summarize(self, op, n, pre, ctx["frames"], res)
        r["replay_ok"] = same(op, res)
        rows.append(r)
        if r["miss"] or n == job["focus"]:
            frames_npz[n] = ctx["frames"]
        return res

    def same(op, res):
        got = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}
        return got == op["result"]

    def pre_act(ep, op):
        """What the demonstrator's check said, and the plan, before the move. Neither touches the physics state;
        the replay check (`replay_ok`) would catch it if they did."""
        m = ep.m
        before = (ep.d.qpos.copy(), ep.d.qvel.copy(), ep.d.qacc_warmstart.copy())
        chk = None
        try:
            ep.see()
            c = ep.check("pick", op["xyz"], op["deg"], op.get("flip"))
            if c.get("runs"):
                chk = {"hits_mm": {tools.short(k) if k not in ("floor",) else k: round(v * 1000, 1)
                                   for k, v in c["hits"].items()},
                       "unseen_mm": round(c["unseen_mm"], 1), "wall_mm": round(c["wall_mm"], 1),
                       "wall_gap_mm": None if not np.isfinite(c["wall_gap_mm"]) else round(float(c["wall_gap_mm"]), 1), "self_mm": None if not np.isfinite(c["self_mm"]) else round(float(c["self_mm"]), 1),
                       "yaw_err": round(c["yaw_err"], 1), "fixed_deg": round(c["fixed_deg"], 1)}
                seen = ep.view["cubes"]
                chk["seen"] = {tools.short(k): {"pos": np.round(v["pos"], 4).tolist(), "yaw": round(v["yaw"], 1),
                                                "cov": round(v["cov"], 2), "how": v["how"],
                                                "tilt": None if v["tilt"] is None else round(v["tilt"], 1)}
                               for k, v in seen.items()}
            else:
                chk = {"runs": False, "why": c.get("why")}
        except Exception as exc:  # the check is optional
            chk = {"error": repr(exc)}
        after = (ep.d.qpos, ep.d.qvel, ep.d.qacc_warmstart)
        untouched = all(np.array_equal(a, b) for a, b in zip(before, after))
        traj = ep.prim._plan("pick", np.asarray(op["xyz"], float), float(np.radians(op["deg"])), op.get("flip"),
                             op.get("lift_z"))
        return {"check": chk, "check_left_state": untouched, "traj": np.asarray(traj, float)}

    def summarize(ep, op, n, pre, fr, res):
        m = ep.m
        names = list(ep.env.cubes)
        traj = pre["traj"]
        R = np.array([f["row"] for f in fr[:-1]])
        N = len(traj)
        g = traj[:, 5]
        first_close = int(np.argmax(g < primitives.GRIP_OPEN - 1e-6))
        close_done = int(np.argmax(g <= primitives.GRIP_CLOSE + 1e-6))
        lift_start = close_done + 9
        # the primitive's rows sit at the end of the commanded rows, before the settle frames; an act that succeeded
        # mid-lift stops early, so match as much of the primitive as ran (at least through the close)
        off = None
        for o in range(len(R) - lift_start):
            if np.allclose(R[o:o + lift_start + 1], traj[:lift_start + 1], atol=1e-3):
                off = o
                break
        S0 = fr[0]
        if off is None:
            return {"act": n, "kind": "pick", "error": "primitive rows not found", "miss": op["result"]["grip"] <= 14}
        s_desc = fr[off + first_close]  # after the last open row: the close is about to start
        s_close = fr[min(off + lift_start, len(fr) - 1)]  # after the 8-frame hold: the lift is about to start
        s_lift = fr[min(off + N, len(fr) - 1)]
        s_end = fr[-1]

        # the intended cube and the nearest cube
        why = op.get("why", "")
        tgt_short, how = parse_target(why)
        aim = np.asarray(op["xyz"], float)
        pos0 = S0["cubes"][:, :3]
        d3 = np.linalg.norm(pos0 - aim, axis=1)
        dxy = np.linalg.norm(pos0[:, :2] - aim[:2], axis=1)
        near3 = names[int(np.argmin(d3))]
        nearxy = names[int(np.argmin(dxy))]
        # The parse names the wrong cube when the `why` mentions others first ("nudge: red is boxed; grasp across
        # 148 where g9 has room"). A parsed cube whose centre is more than 20 mm from the aim in xy is not the one
        # the jaws were put on: use the nearest cube and say so.
        intended, how_used = near3, "nearest"
        if full_name(tgt_short) in names:
            if dxy[names.index(full_name(tgt_short))] < 0.020:
                intended, how_used = full_name(tgt_short), how
            else:
                how_used = f"parsed {tgt_short} is {dxy[names.index(full_name(tgt_short))] * 1000:.0f} mm off; nearest"
        ti = names.index(intended)

        def pose(s, i):
            return s["cubes"][i, :3].copy(), quat_mat(s["cubes"][i, 3:])

        p0, R0 = pose(S0, ti)
        yaw = np.radians(op["deg"])
        u_cmd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        v_cmd = np.array([-u_cmd[1], u_cmd[0], 0.0])
        e = aim - p0
        # jaw frame at the end of descent: u from moving tip to fixed tip, horizontal
        ft, mt = s_desc["fixed_tip"], s_desc["moving_tip"]
        u = (ft - mt) * np.array([1, 1, 0])
        u /= np.linalg.norm(u)
        w = np.array([-u[1], u[0], 0.0])
        site = s_desc["site"]
        jaw_axis = site[:, ik.JAW_AXIS].copy()
        jaw_yaw = float(np.degrees(np.arctan2(jaw_axis[1], jaw_axis[0])))

        # the target where the close began
        pD, RD = pose(s_desc, ti)
        rel = pD - s_desc["fk22"]
        cz = corners(pD, RD)[:, 2]
        tips_z = 0.5 * (ft[2] + mt[2])
        seg = ft - mt
        tpar = float(((pD - mt) @ seg) / (seg @ seg))
        # neighbours along the jaw axis (true, at the start): gap from the target's face to theirs, per end
        ends = {"fixed_end": [], "moving_end": []}
        for i, nm in enumerate(names):
            if i == ti:
                continue
            dd = pos0[i] - p0
            along, across = dd @ u, dd @ w
            if abs(across) < CUBE and -CUBE < dd[2] < CUBE + 0.01 and 0 < abs(along) < CUBE + 0.035:
                ends["fixed_end" if along > 0 else "moving_end"].append(
                    (round((abs(along) - CUBE) * 1000, 1), short_name(nm), round(float(dd[2]) * 1000, 1)))
        for k in ends:
            ends[k].sort()

        # when the target first moved, and whether the arm touched anything before that
        tpos = np.array([f["cubes"][ti, :3] for f in fr])
        moved = np.nonzero(np.linalg.norm(tpos - p0, axis=1) > 0.001)[0]
        first_move = None if len(moved) == 0 else int(moved[0])
        contact_before_move = sorted({other_of_(m, g2, names, intended) for f in fr[:first_move + 1]
                                      for _, g2, _, _ in f["con"]}) if first_move is not None else None
        # contacts: first frame of each (side, other) pair, by phase
        body_side = {probe["fixed_body"]: "fixed", probe["moving_body"]: "moving"}

        def side_of(gid):
            b = m.geom_bodyid[gid]
            return body_side.get(b, m.body(b).name)

        def other_of(gid):
            nm = m.geom(gid).name
            if nm in names:
                return "TARGET" if nm == intended else short_name(nm)
            return nm or f"geom{gid}"

        def where_on(f, gid, cp):
            """'top' if a contact point sits on the cube's upward face (within 2 mm of it and inside its outline),
            'side' otherwise; with the point in the cube's frame, upward axis last, in mm."""
            nm = m.geom(gid).name
            if nm not in names:
                return None
            i = names.index(nm)
            pc, Rc = f["cubes"][i, :3], quat_mat(f["cubes"][i, 3:])
            loc = (cp - pc) @ Rc
            up = int(np.argmax(np.abs(Rc[2, :])))
            h = loc[up] * np.sign(Rc[2, up])
            lat = np.delete(loc, up)
            kind = "top" if h > HALF - 0.002 and np.abs(lat).max() < HALF + 0.001 else "side"
            return f"{kind} ({lat[0] * 1000:.0f},{lat[1] * 1000:.0f},{h * 1000:.0f})"

        # the descent: how low the arm got, and when it stopped following the plan
        desc = fr[off:off + first_close + 1]
        fk_z = np.array([f["fk22"][2] for f in desc])
        tip_low = np.array([min(f["fixed_tip"][2], f["moving_tip"][2]) for f in desc])
        # planned fk22 for each row: the commanded arm at GRIP_PLAN
        pd = probe["data"]

        def plan_fk(row):
            q = units.to_rad(np.asarray(row, float), ep.grip_range)
            pd.qpos[ep.adr] = q
            pd.qpos[ep.adr[5]] = probe["grip22"]
            mujoco.mj_kinematics(m, pd)
            return ik.grasp_point(m, pd).copy()

        plan_arrive = plan_fk(traj[first_close - 1])
        dev = np.array([f["fk22"][2] - plan_fk(traj[k])[2] for k, f in enumerate(desc[1:])])  # after row k
        stall = np.nonzero(dev > 0.002)[0]
        stall_frame = None if len(stall) == 0 else int(stall[0])
        blockers = []
        if stall_frame is not None:
            for k in range(max(0, stall_frame - 2), min(len(desc), stall_frame + 3)):
                for ga, go, _, _ in desc[k]["con"]:
                    key = f"{side_of(ga)}>{other_of(go)}"
                    if key not in blockers:
                        blockers.append(key)
        lag_desc = max(float(np.abs(f["js"][:5] - traj[k][:5]).max()) for k, f in enumerate(desc[1:]))
        track = s_desc["fk22"] - plan_arrive
        joint_lag = np.abs(s_desc["js"][:5] - traj[first_close - 1][:5])

        phases = [("approach", 0, off), ("descent", off, off + first_close), ("close", off + first_close,
                  off + lift_start), ("lift", off + lift_start, off + N), ("settle", off + N, len(fr))]
        firsts = {}
        for ph, a, b in phases:
            for k in range(a, min(b, len(fr))):
                for ga, go, dist, cp in fr[k]["con"]:
                    key = f"{side_of(ga)}>{other_of(go)}"
                    if key not in firsts:
                        firsts[key] = {"phase": ph, "frame": k - off, "geom": m.geom(ga).name or f"geom{ga}",
                                       "where": where_on(fr[k], go, cp)}
        # who touched the target during the close, and was it held between both jaws at the end of it
        tgt_gid = m.geom(intended).id
        touch = {"fixed": [], "moving": []}
        for k in range(off, off + lift_start):
            for ga, go, dist, _ in fr[k]["con"]:
                if go == tgt_gid and side_of(ga) in touch:
                    touch[side_of(ga)].append(k - off)

        # the close: target motion, gripper, and what was between both jaws at its end
        pC, RC = pose(s_close, ti)
        pE, RE = pose(s_end, ti)
        pL, RL = pose(s_lift, ti)
        jaw_bodies = {probe["fixed_body"]: "fixed", probe["moving_body"]: "moving"}
        held_at_close = []
        for i, nm in enumerate(names):
            gid = m.geom(nm).id
            sides = {jaw_bodies[m.geom_bodyid[ga]] for ga, go, _, _ in s_close["con"] if go == gid
                     and m.geom_bodyid[ga] in jaw_bodies}
            if sides == {"fixed", "moving"}:
                held_at_close.append(short_name(nm))
        grip_trace = np.array([f["js"][5] for f in fr])
        # where each jaw touched the target at the end of the close, in the target's own frame (mm): a face grasp
        # touches near mid-height, an edge pinch near a corner row
        grip_pts = {}
        for ga, go, _, cp in s_close["con"]:
            if go == tgt_gid and m.geom_bodyid[ga] in jaw_bodies:
                grip_pts.setdefault(jaw_bodies[m.geom_bodyid[ga]], []).append((cp - pC) @ RC)
        grip_local = {k: np.round(np.mean(v, axis=0) * 1000, 1).tolist() for k, v in grip_pts.items()}
        grip_world_dz = {k: round(float(np.mean([p[2] for p in v]) * 1000), 1) for k, v in
                         {k: [RC @ q for q in v] for k, v in grip_pts.items()}.items()}
        lifted = [short_name(nm) for i, nm in enumerate(names) if s_end["cubes"][i, 2] - pos0[i, 2] > LIFTED_DZ]
        # frame the gripper first read <= 14 after the close began, if it held something first
        g_close = float(s_close["js"][5])
        g_lift = float(s_lift["js"][5])
        drop_frame = None
        if g_close > 14:
            after = np.nonzero(grip_trace[off + lift_start:] <= 14)[0]
            drop_frame = None if len(after) == 0 else int(after[0] + lift_start)

        out = {
            "act": n, "kind": "pick", "xyz": op["xyz"], "deg": op["deg"], "flip": op.get("flip"),
            "lift_z": op.get("lift_z"), "why": why, "grip": op["result"]["grip"],
            "miss": op["result"]["grip"] <= 14, "nudge": is_nudge(why),
            "intended": short_name(intended), "intended_how": how_used,
            "parsed": tgt_short, "nearest3d": short_name(near3), "nearest_xy": short_name(nearxy),
            "nearest3d_mm": round(float(d3.min()) * 1000, 1),
            "target_layer": int(round((p0[2] - ep.bin["floor_z"] - HALF) / CUBE)) + 1,
            "target_start": np.round(p0, 4).tolist(),
            # aim vs truth, at the act's start
            "aim_xy_mm": round(float(np.linalg.norm(e[:2])) * 1000, 1),
            "aim_along_mm": round(float(e @ u_cmd) * 1000, 1),
            "aim_across_mm": round(float(e @ v_cmd) * 1000, 1),
            "aim_z_mm": round(float(e[2]) * 1000, 1),
            # + : the aim sits toward the moving finger, so the fixed finger comes down closer to (or onto) the cube
            "aim_off_fixed_mm": round(float((p0 - aim) @ u) * 1000, 1),
            # horizontal distance from the target's face to each open jaw tip at the end of the descent (negative:
            # the tip is over the cube's own top)
            "fixed_tip_out_mm": round(float((ft - p0) @ u) * 1000 - HALF * 1000, 1),
            "moving_tip_out_mm": round(float((p0 - mt) @ u) * 1000 - HALF * 1000, 1),
            "yaw_cmd_vs_face": round(face_misalign_deg(R0, u_cmd), 1),
            "tilt_start": round(tilt_deg(R0), 1),
            "target_top_mm_above_aim": round(float((corners(p0, R0)[:, 2].max() - aim[2]) * 1000), 1),
            # the arm
            "jaw_yaw_err": round(abs((jaw_yaw - op["deg"] + 90.0) % 180.0 - 90.0), 1),
            "jaw_vs_face_at_close": round(face_misalign_deg(RD, jaw_axis), 1),
            "track_mm": np.round(track * 1000, 1).tolist(),
            "stopped_high_mm": round(float(track[2]) * 1000, 1),
            "joint_lag_deg": round(float(joint_lag.max()), 1),
            "lowest_fk22_vs_aim_mm": round(float((fk_z.min() - aim[2]) * 1000), 1),
            "lowest_tip_z": round(float(tip_low.min()), 4),
            # target at the start of the close, in the jaw frame at the planning gripper value
            "tgt_rel_fixed_mm": round(float(rel @ u) * 1000, 1),  # + toward the fixed jaw
            "tgt_rel_across_mm": round(float(rel @ w) * 1000, 1),
            "tgt_rel_z_mm": round(float(rel[2]) * 1000, 1),  # target centre minus jaw midpoint (at GRIP_PLAN)
            "tips_below_top_mm": round(float((cz.max() - tips_z) * 1000), 1),  # > 0: tips reach below the cube top
            "tips_above_bottom_mm": round(float((tips_z - cz.min()) * 1000), 1),
            "tgt_between_t": round(tpar, 2),  # 0 at the moving tip, 1 at the fixed tip
            "tilt_at_close_start": round(tilt_deg(RD), 1),
            "tgt_speed_start_mm_s": round(float(S0["speed"][ti]) * 1000, 2),
            "first_move_frame": None if first_move is None else first_move - off,
            "arm_contacts_before_first_move": contact_before_move,
            "moved_in_approach_mm": round(float(np.linalg.norm(fr[off]["cubes"][ti, :3] - p0)) * 1000, 1),
            "moved_in_descent_mm": round(float(np.linalg.norm(pD - fr[off]["cubes"][ti, :3])) * 1000, 1),
            "moved_in_close_mm": round(float(np.linalg.norm(pC - pD)) * 1000, 1),
            "moved_in_close_xyz_mm": np.round((pC - pD) * 1000, 1).tolist(),
            "turned_in_close_deg": round(rot_angle_deg(RD, RC), 1),
            "tilt_at_close_end": round(tilt_deg(RC), 1),
            "rose_in_lift_mm": round(float(pL[2] - pC[2]) * 1000, 1),
            "moved_total_mm": round(float(np.linalg.norm(pE - p0)) * 1000, 1),
            "target_end": np.round(pE, 4).tolist(), "tilt_end": round(tilt_deg(RE), 1),
            "grip_close_end": round(g_close, 1), "grip_lift_end": round(g_lift, 1), "drop_frame": drop_frame,
            "held_at_close": held_at_close, "lifted": lifted,
            "touch_fixed_frames": [touch["fixed"][0], touch["fixed"][-1]] if touch["fixed"] else None,
            "touch_moving_frames": [touch["moving"][0], touch["moving"][-1]] if touch["moving"] else None,
            "first_contacts": firsts, "jaw_axis_ends": ends,
            "prim_offset": off,
            "phase_frames": {"descent_end": first_close, "close_done": close_done, "lift_start": lift_start,
                             "lift_end": N},
            "tgt_pen_max_mm": round(float(max(f["pen"][ti] for f in fr[off:off + lift_start])) * 1000, 1),
            "tgt_below_floor_mm": round(float(ep.bin["floor_z"] - cz.min()) * 1000, 1),
            "stall_frame": stall_frame, "blockers": blockers, "joint_lag_descent_deg": round(lag_desc, 1),
            "grip_contact_local_mm": grip_local, "grip_contact_dz_mm": grip_world_dz,
            "check": pre["check"], "check_left_state": pre["check_left_state"],
        }
        return out

    tools.Episode._apply = spy
    ep = None
    try:
        ep = tools.Episode(job["name"], on_advance=on_advance, rec=rec)
    finally:
        tools.Episode._apply = original
    for r in rows:
        r.update(episode=job["name"], line=job["line"], seed=rec["seed"], buried=rec["buried"])
    # keep raw frames of the misses and the focus act, for pictures and later looks
    raw = {}
    for n, fr in frames_npz.items():
        raw[n] = {"fk22": np.array([f["fk22"] for f in fr]), "js": np.array([f["js"] for f in fr]),
                  "fixed_tip": np.array([f["fixed_tip"] for f in fr]),
                  "moving_tip": np.array([f["moving_tip"] for f in fr]),
                  "cubes": np.array([f["cubes"] for f in fr]), "speed": np.array([f["speed"] for f in fr])}
    if ep is not None:
        ep.close()
    return {"job": {k: v for k, v in job.items() if k != "ops"}, "rows": rows, "raw": raw,
            "seconds": round(time.time() - t0, 1)}


# ---- classification ------------------------------------------------------------------------------------------------

def landings(r):
    """(a finger came down on the target's upward face, [neighbours whose upward face a finger came down on]),
    from the first contact of each finger with each cube before the close."""
    own, nb = False, []
    for key, v in r["first_contacts"].items():
        if v["phase"] not in ("approach", "descent") or not (v["where"] or "").startswith("top"):
            continue
        other = key.split(">")[1]
        if other == "TARGET":
            own = True
        elif re.fullmatch(NAMES, other):
            nb.append(other)
    return own, nb


def classify(r):
    """One cause per miss, from the replay's numbers, in the order the events happen. Thresholds come from the
    control group: no hit had an arm lag over 4.5 deg or an aim offset toward the moving finger over 3.4 mm."""
    if r["joint_lag_descent_deg"] > 10:
        return "arm lagged its plan (wrist roll jammed at home, before the unpark step)"
    if r["moved_in_approach_mm"] > 10:
        return "target fell before the jaws arrived, no arm contact"
    own, nb = landings(r)
    if r["tilt_start"] >= 20:
        return ("fixed finger came down on a tilted cube's upper face" if own
                else "tilted cube squeezed out of the jaws")
    if own and r["yaw_cmd_vs_face"] > 30:
        return "fixed finger came down on the cube's top corner (jaws across the diagonal)"
    if own and r["aim_off_fixed_mm"] > 4 and (perception_xy_mm(r) or 0) > 5:
        return "fixed finger came down on the target's own top (perceived pose > 5 mm off)"
    if own and r["aim_off_fixed_mm"] > 4:
        return "fixed finger came down on the target's own top (aim shifted > 4 mm toward the moving finger)"
    if nb:
        return ("nudge: a finger came down on a flush neighbour's top" if r["nudge"]
                else "a finger came down on a flush neighbour's top")
    return "other"


def perception_xy_mm(r):
    """How far the pose the cameras gave for the intended cube was from its true centre, in xy (mm)."""
    seen = (r.get("check") or {}).get("seen") or {}
    got = seen.get("RED" if r["intended"] == "red" else r["intended"])
    if got is None:
        return None
    return float(np.linalg.norm(np.array(got["pos"][:2]) - np.array(r["target_start"][:2])) * 1000)


def is_nudge(why):
    return bool(re.search(r"\bnudge\b", why, re.I)) and not re.search(r"after the nudge", why, re.I)


def unplanned_high(r):
    """How far above the cube centre the jaws were when the close began, beyond any height the aim asked for (mm):
    the arm stopping short plus the cube sinking under a finger."""
    return -r["tgt_rel_z_mm"] - max(r["aim_z_mm"], 0.0)


# The misses the brief of 2026-09-30 listed: 12 kept ones from outputs/postmortem_2026-09-30.json (finished d* episodes)
# and 10 undone ones. The rest of the 32 are in episodes the postmortem skipped (e400 to e402, unfinished d* episodes)
# or were undone but left off that list (d1000 move 9, d1022 move 12).
BRIEF = {("d1000", "kept", 9), ("d1000", "kept", 14), ("d1003", "kept", 5), ("d1004", "kept", 5),
         ("d1012", "kept", 7), ("d1013", "kept", 9), ("d1013", "kept", 12), ("d1015", "kept", 3),
         ("d1018", "kept", 9), ("d1027", "kept", 1), ("d1031", "kept", 1), ("d1033", "kept", 9),
         ("d1001", "undone0", 3), ("d1005", "undone0", 1), ("d1009", "undone0", 7), ("d1012", "undone0", 7),
         ("d1016", "undone0", 5), ("d1031", "undone0", 6), ("e402", "undone0", 5), ("e402", "undone1", 7),
         ("e402", "undone2", 9), ("e402", "undone3", 8)}


def summarize(rows):
    """Class counts, and misses against the control group (kept picks that lifted the cube they were aimed at)."""
    ok = [r for r in rows if "error" not in r]
    for r in ok:
        r["nudge"] = is_nudge(r["why"])
    miss = [r for r in ok if r["miss"]]
    hit = [r for r in ok if not r["miss"] and 14 < r["grip"] < 30 and r["intended"] in r["lifted"]]
    out = {"n_miss": len(miss), "n_hit": len(hit), "n_brief": sum((r["episode"], r["line"], r["act"]) in BRIEF
                                                                 for r in miss)}
    for key, group in (("all", miss), ("brief", [r for r in miss if (r["episode"], r["line"], r["act"]) in BRIEF])):
        counts = {}
        for r in group:
            counts.setdefault(r["cause"], []).append(f"{r['episode']}:{r['line']}:{r['act']}")
        out[f"classes_{key}"] = {k: {"n": len(v), "picks": v} for k, v in sorted(counts.items(), key=lambda kv: -len(kv[1]))}

    def stats(v):
        v = np.array([x for x in v if x is not None], float)
        return {"n": len(v), "median": round(float(np.median(v)), 1), "p10": round(float(np.percentile(v, 10)), 1),
                "p90": round(float(np.percentile(v, 90)), 1)}

    flat_m = [r for r in miss if r["tilt_start"] < 20]
    flat_h = [r for r in hit if r["tilt_start"] < 20]
    for r in ok:
        r["unplanned_high_mm"] = round(unplanned_high(r), 1)
        pe = perception_xy_mm(r)
        r["perception_xy_mm"] = None if pe is None else round(pe, 1)
    cmp = {}
    for k in ("unplanned_high_mm", "perception_xy_mm", "aim_xy_mm", "aim_z_mm", "aim_off_fixed_mm", "fixed_tip_out_mm", "moving_tip_out_mm", "yaw_cmd_vs_face",
              "tilt_start", "stopped_high_mm", "tgt_pen_max_mm", "moved_in_descent_mm", "tgt_rel_z_mm",
              "tips_below_top_mm", "grip_close_end", "turned_in_close_deg"):
        cmp[k] = {"miss": stats([r[k] for r in miss]), "hit": stats([r[k] for r in hit]),
                  "miss_flat": stats([r[k] for r in flat_m]), "hit_flat": stats([r[k] for r in flat_h])}
    for side in ("fixed", "moving"):
        cmp[f"grip_contact_dz_mm_{side}"] = {"miss": stats([r["grip_contact_dz_mm"].get(side) for r in miss]),
                                             "hit": stats([r["grip_contact_dz_mm"].get(side) for r in hit])}
    out["miss_vs_hit"] = cmp

    def rate(pred, group_m, group_h):
        m, h = sum(pred(r) for r in group_m), sum(pred(r) for r in group_h)
        return {"misses": m, "hits": h}

    high = lambda r: -r["tgt_rel_z_mm"] >= 5  # noqa: E731  jaws 5 mm or more above the cube centre as the close starts
    out["splits"] = {
        "flat, jaws >= 5 mm above centre at close start": rate(high, flat_m, flat_h),
        "flat, jaws < 5 mm above centre at close start": rate(lambda r: not high(r), flat_m, flat_h),
        "flat, jaws >= 5 mm higher than the aim asked": rate(lambda r: unplanned_high(r) >= 5, flat_m, flat_h),
        "flat, jaws < 5 mm higher than the aim asked": rate(lambda r: unplanned_high(r) < 5, flat_m, flat_h),
        "all, jaws >= 5 mm higher than the aim asked": rate(lambda r: unplanned_high(r) >= 5, miss, hit),
        "aim offset toward the moving finger > 4 mm": rate(lambda r: r["aim_off_fixed_mm"] > 4, miss, hit),
        "aim offset toward the moving finger 2 to 4 mm": rate(lambda r: 2 < r["aim_off_fixed_mm"] <= 4, miss, hit),
        "flat, target pressed > 4 mm into what it rests on": rate(lambda r: r["tgt_pen_max_mm"] > 4, flat_m, flat_h),
        "flat, arm stopped > 2 mm high": rate(lambda r: r["stopped_high_mm"] > 2, flat_m, flat_h),
        "flat, a finger came down on a cube's upward face": rate(lambda r: any(landings(r)[:1]) or bool(landings(r)[1]),
                                                                 flat_m, flat_h),
        "tilted >= 20 deg": rate(lambda r: r["tilt_start"] >= 20, miss, hit),
        "tilted >= 20 deg, fixed finger on its upper face": rate(
            lambda r: r["tilt_start"] >= 20 and landings(r)[0], miss, hit),
        "tilted >= 20 deg, no finger on its upper face": rate(
            lambda r: r["tilt_start"] >= 20 and not landings(r)[0], miss, hit),
        "nudge picks": rate(lambda r: r["nudge"], miss, hit),
    }
    return out


# ---- pictures ------------------------------------------------------------------------------------------------------

def pictures(task):
    """Three views of one act from a free camera looking across the jaw axis, bin walls see-through: before the act,
    at the end of the descent (jaws open at depth), and at the end of the close."""
    job, acts, out_dir = task
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    import tools
    import mujoco
    from PIL import Image, ImageDraw
    rec = json.loads((tools.EPISODES / f"{job['name']}.json").read_text())
    rec = {**rec, "ops": [dict(op) for op in job["ops"]]}
    rec.pop("undone", None)
    shots = {}
    state = {"n": 0, "k": None}
    original = tools.Episode._apply
    rend = {}

    def render(ep, a):
        m = ep.m
        # rendering only: see-through walls and bystander cubes, the target blue (or red), cubes a finger landed on
        # orange
        for w in ("bin_wall_px", "bin_wall_nx", "bin_wall_py", "bin_wall_ny"):
            m.geom_rgba[m.geom(w).id, 3] = 0.15
        mount = m.body("camera_mount").id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == mount:
                m.geom_rgba[g, 3] = 0.15
                m.geom_matid[g] = -1
        for nm in ep.env.cubes:
            g = m.geom(nm).id
            if nm == a["target"]:
                m.geom_rgba[g] = (0.85, 0.15, 0.15, 1) if nm == "cube_red" else (0.2, 0.4, 0.95, 1)
            elif short_name(nm) in a["landed"]:
                m.geom_rgba[g] = (1.0, 0.55, 0.1, 1)
            else:
                m.geom_rgba[g, 3] = 0.25
        if "r" not in rend:
            rend["r"] = mujoco.Renderer(m, 360, 480)
        r = rend["r"]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = a["lookat"]
        cam.distance = 0.24
        # look across the jaw axis from the side away from the arm's base, so the wrist doesn't hide the jaws
        look = np.asarray(a["lookat"][:2])
        az = max(((a["deg"] + 90.0) % 360.0, (a["deg"] - 90.0) % 360.0), key=lambda z: np.linalg.norm(
            look - 0.24 * np.array([np.cos(np.radians(z)), np.sin(np.radians(z))])))
        cam.azimuth = az
        cam.elevation = -40.0
        r.update_scene(ep.d, camera=cam)
        return r.render().copy()

    def spy(self, op):
        if op["op"] == "act":
            state["n"] += 1
            if state["n"] in acts:
                state["k"] = 0
                shots[(state["n"], 0)] = render(self, acts[state["n"]])
        res = original(self, op)
        state["k"] = None
        return res

    def on_advance(ep, row):
        if state["k"] is None:
            return
        a = acts[state["n"]]
        for j, f in ((1, a["descent_frame"]), (2, a["close_frame"])):
            if state["k"] == f:
                shots[(state["n"], j)] = render(ep, a)
        state["k"] += 1

    tools.Episode._apply = spy
    try:
        ep = tools.Episode(job["name"], on_advance=on_advance, rec=rec)
    finally:
        tools.Episode._apply = original
    ep.close()
    if "r" in rend:
        rend["r"].close()
    paths = []
    for n, a in acts.items():
        if not all((n, j) in shots for j in range(3)):
            continue
        img = Image.new("RGB", (1440, 400), "white")
        for j, title in enumerate(("before the act", "end of descent, jaws open", "end of the close")):
            img.paste(Image.fromarray(shots[(n, j)]), (480 * j, 40))
            ImageDraw.Draw(img).text((480 * j + 6, 24), title, fill=(0, 0, 0))
        ImageDraw.Draw(img).text((6, 4), f"{job['name']} {job['line']} move {n}: {a['label']}", fill=(0, 0, 0))
        ImageDraw.Draw(img).text((1100, 4), "target blue (or red); orange: a finger landed on it", fill=(0, 0, 0))
        path = out_dir / f"{job['name']}_{job['line']}_move{n}.png"
        img.save(path)
        paths.append(str(path))
    return paths


def picture_tasks(rows, picks, episodes, out_dir):
    groups = {}
    for ep, line, act in picks:
        r = next(r for r in rows if r["episode"] == ep and r["line"] == line and r["act"] == act)
        off, pf = r["prim_offset"], r["phase_frames"]
        groups.setdefault((ep, line), {})[act] = {
            "lookat": r["target_start"], "deg": r["deg"], "descent_frame": off + pf["descent_end"],
            "target": full_name(r["intended"]), "landed": landings(r)[1],
            "close_frame": off + pf["lift_start"], "label": f"{r['intended']} ({r.get('cause', 'hit')}) | {r['why'][:90]}"}
    tasks = []
    for (ep, line), acts in groups.items():
        rec = json.loads((episodes / f"{ep}.json").read_text())
        ops = rec["ops"] if line == "kept" else undone_lines(rec)[int(line[len("undone"):])]
        tasks.append(({"name": ep, "line": line, "ops": ops}, acts, out_dir))
    return tasks


# ---- main ----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "miss_study_2026-09-30"))
    ap.add_argument("--summarize", action="store_true", help="only reclassify and summarize an existing rows.json")
    ap.add_argument("--pictures", nargs="*", metavar="EP:LINE:ACT",
                    help="only render these acts from an existing rows.json, e.g. d1031:undone0:6")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = HERE.parent / "episodes"

    if args.summarize:
        rows = json.loads((out / "rows.json").read_text())
        for r in rows:
            if "error" not in r:
                r["nudge"] = is_nudge(r["why"])
            if r.get("miss") and "error" not in r:
                r["cause"] = classify(r)
                r["in_brief"] = (r["episode"], r["line"], r["act"]) in BRIEF
        summary = summarize(rows)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if k != "miss_vs_hit"}, indent=1))
        return
    if args.pictures is not None:
        rows = json.loads((out / "rows.json").read_text())
        picks = [(a.split(":")[0], a.split(":")[1], int(a.split(":")[2])) for a in args.pictures]
        tasks = picture_tasks(rows, picks, episodes, out)
        with mp.get_context("spawn").Pool(min(args.procs, len(tasks))) as pool:
            for paths in pool.map(pictures, tasks, chunksize=1):
                print("\n".join(paths))
        return

    names = args.names
    if not names:
        # every episode with a missed pick, kept or undone
        for p in sorted(episodes.glob("*.json")):
            rec = json.loads(p.read_text())
            ops = rec["ops"] + [op for u in rec.get("undone", []) for op in u["ops"]]
            if any(is_miss(op) for op in ops):
                names.append(p.stem)
    jobs = jobs_for(names, episodes)
    jobs.sort(key=lambda j: -len(j["ops"]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.procs, len(jobs))) as pool:
        results = pool.map(study, jobs, chunksize=1)
    rows, bad = [], []
    for res in results:
        for r in res["rows"]:
            if not r["replay_ok"]:
                bad.append((r["episode"], r["line"], r["act"]))
            if r.get("studied") is False:
                continue
            rows.append(r)
        for n, raw in res["raw"].items():
            np.savez_compressed(out / f"raw_{res['job']['name']}_{res['job']['line']}_move{n}.npz", **raw)
    for r in rows:
        if r.get("miss") and "error" not in r:
            r["cause"] = classify(r)
            r["in_brief"] = (r["episode"], r["line"], r["act"]) in BRIEF
    rows.sort(key=lambda r: (r["episode"], r["line"], r["act"]))
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(jobs)} replays, {time.time() - t0:.0f} s wall; slowest {max(res['seconds'] for res in results)} s")
    print(f"acts whose replay did not reproduce the record: {bad or 'none'}")


if __name__ == "__main__":
    main()
