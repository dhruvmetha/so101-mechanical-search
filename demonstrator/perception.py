"""Perception from MuJoCo's own renders: colour, depth and segmentation from a camera, turned into cube poses.

Depth stands in for an RGB-D camera. Segmentation stands in for an instance segmenter (SAM or similar) and is the
optimistic part: it is exact and never merges two touching cubes. Only surfaces the camera sees produce points, so
a hidden cube gets no pose.

Measured 2026-09-29 on six piles from the front camera with the arm parked: cubes with at least 60% of the top face
seen come out within 1.5 mm median, 2.5 mm 90th percentile, and 0.5 deg of yaw. Less-seen faces drift toward the
visible part, up to 14 mm; the coverage figure says how far to trust a pose.
"""
import numpy as np

import mujoco

CUBE = 0.030
HALF = CUBE / 2


class Camera:
    def __init__(self, model, name, height=480, width=640):
        self.model, self.name, self.h, self.w = model, name, height, width
        self.r = mujoco.Renderer(model, height, width)
        self.cid = model.camera(name).id
        sensor = model.cam_sensorsize[self.cid]
        if sensor[0] > 0:
            # A camera given a sensor size and focal length (the wrist camera: 5.76 x 3.24 mm, 3.6 mm) renders that
            # frustum into whatever image size is asked for, so its pixels aren't square unless the aspect matches.
            # Assuming square pixels from fovy put x off by a third (2026-09-30).
            focal = model.cam_intrinsic[self.cid][:2]
            self.fx, self.fy = width * focal[0] / sensor[0], height * focal[1] / sensor[1]
        else:
            self.fx = self.fy = (height / 2) / np.tan(np.radians(model.cam_fovy[self.cid]) / 2)
        self.f = self.fy

    def capture(self, data):
        """rgb (H,W,3), depth (H,W) in metres along the view axis, geom id per pixel (-1 for none), world points."""
        r = self.r
        r.update_scene(data, camera=self.name)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.name)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=self.name)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        pts = self.backproject(data, depth)
        return rgb, depth, geom, pts

    def backproject(self, data, depth):
        pos = data.cam_xpos[self.cid].copy()
        rot = data.cam_xmat[self.cid].reshape(3, 3).copy()
        v, u = np.mgrid[0:self.h, 0:self.w]
        x = (u + 0.5 - self.w / 2) / self.fx * depth
        y = -(v + 0.5 - self.h / 2) / self.fy * depth
        cam = np.stack([x, y, -depth], axis=-1)
        return pos + cam @ rot.T

    def close(self):
        self.r.close()


def top_face_pose(points):
    """Pose of a cube from the points on its visible top face: (centre xyz, yaw in [0, pi/2), coverage 0..1).

    Takes the highest points (within 3 mm of the top), finds the yaw that gives the smallest bounding square, and
    places the centre half a cube below the square's middle. Coverage is how much of a full 30 mm face was seen.
    """
    z = points[:, 2]
    top = points[z > z.max() - 0.003]
    xy = top[:, :2]
    best = None
    for yaw in np.radians(np.arange(0, 90, 1.0)):
        c, s = np.cos(yaw), np.sin(yaw)
        local = xy @ np.array([[c, -s], [s, c]])
        lo, hi = local.min(0), local.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, yaw, lo, hi)
    _, yaw, lo, hi = best
    c, s = np.cos(yaw), np.sin(yaw)
    mid_local = (lo + hi) / 2
    extent = hi - lo
    # local = xy rotated by -yaw, so rotate the middle back by +yaw. A partly hidden face biases this toward the
    # visible part; coverage says how far to trust it.
    centre_xy = np.array([c * mid_local[0] - s * mid_local[1], s * mid_local[0] + c * mid_local[1]])
    coverage = float(np.clip(extent / CUBE, 0, 1).prod())
    return np.array([centre_xy[0], centre_xy[1], top[:, 2].mean() - HALF]), float(yaw), coverage, len(top)


def box_pose(points, cam_pos, init_yaw=0.0):
    """Fit a 30 mm box to a cube's visible points: (centre, rotation matrix, rms residual in metres).

    Uses every face the camera sees, so a tilted cube or one seen mostly from the side still gets a centre. The top
    face alone drifts toward whichever edge is highest (e401: a leaning g3 was missed because of it). Starts half a
    cube behind the visible surface along the view direction, which is where the hidden centre usually is.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    centroid = points.mean(0)
    away = centroid - cam_pos
    away /= np.linalg.norm(away)
    c0 = centroid + away * HALF * 0.6
    r0 = Rotation.from_euler("z", init_yaw).as_rotvec()
    sample = points if len(points) <= 600 else points[np.random.default_rng(0).choice(len(points), 600, replace=False)]

    def residual(x):
        rot = Rotation.from_rotvec(x[3:]).as_matrix()
        q = (sample - x[:3]) @ rot  # into the box frame
        a = np.abs(q)
        on_surface = a.max(1) - HALF
        outside = np.clip(a - HALF, 0, None).sum(1)
        return np.concatenate([on_surface, outside])

    fit = least_squares(residual, np.concatenate([c0, r0]), loss="soft_l1", f_scale=0.002)
    rot = Rotation.from_rotvec(fit.x[3:]).as_matrix()
    rms = float(np.sqrt(np.mean(residual(fit.x)[:len(sample)] ** 2)))
    return fit.x[:3], rot, rms


def yaw_and_tilt(rot):
    """(jaw yaw in deg, tilt in deg) for a fitted box.

    The box axis closest to vertical is its top. Of the other two, the one closest to horizontal is the normal of the
    pair of faces that stays most nearly vertical when the cube leans. Closing the jaws across those faces holds a
    tilted cube best, so the yaw returned is that axis's heading.
    """
    up = int(np.argmax(np.abs(rot[2])))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(rot[2, up])))))
    sides = [k for k in range(3) if k != up]
    grip = min(sides, key=lambda k: abs(rot[2, k]))
    axis = rot[:, grip]
    yaw = float(np.degrees(np.arctan2(axis[1], axis[0]))) % 180.0
    return yaw, tilt


def clean_mask(mask, pts, min_blob=10, near=0.03):
    """Segmentation is anti-aliased, so edges carry stray ids: single pixels of one cube's id on another's border.
    Trim one pixel off the mask's edge (depth is mixed there too), keep the largest piece, and keep other pieces only
    if their points sit within `near` of it (a cube split in two by an occluder)."""
    from scipy import ndimage
    mask = ndimage.binary_erosion(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    centre = pts[lab == main].mean(0)
    keep = np.zeros_like(mask)
    for k in range(1, n + 1):
        if sizes[k - 1] < min_blob and k != main:
            continue
        if k == main or np.linalg.norm(pts[lab == k].mean(0) - centre) < near:
            keep |= lab == k
    return keep


def perceive(model, data, cams, cube_geoms, min_points=40):
    """Cube poses from every camera in `cams`. Returns {geom id: dict(pos, yaw, coverage, n, cams)}."""
    pooled = {}
    for cam in cams:
        rgb, depth, geom, pts = cam.capture(data)
        for g in cube_geoms:
            mask = geom == g
            if mask.sum() < min_points:
                continue
            mask = clean_mask(mask, pts)
            if mask.sum() < min_points:
                continue
            pooled.setdefault(g, []).append(pts[mask])
    out = {}
    for g, chunks in pooled.items():
        p = np.concatenate(chunks)
        pos, yaw, cov, n = top_face_pose(p)
        out[g] = {"pos": pos, "yaw": yaw, "coverage": cov, "n_top": n, "n": len(p)}
    return out
