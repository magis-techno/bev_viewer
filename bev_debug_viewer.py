#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# -----------------------------
# Geometry utilities
# -----------------------------


def ensure_array(x: Any, shape: Optional[Tuple[int, ...]] = None, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


S_EGO = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
S_CAM = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

CAMERA_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def yaw_to_rotmat(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_xyzw_to_rotmat(q_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_hmat(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def invert_hmat(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {pts.shape}")
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    out = (T @ homog.T).T
    return out[:, :3]


def parse_pose_to_world(data: Dict[str, Any]) -> np.ndarray:
    translation = ensure_array(data.get("translation", [0, 0, 0]), shape=(3,))
    if "rotation_matrix" in data:
        R = ensure_array(data["rotation_matrix"], shape=(3, 3))
    elif "quaternion_xyzw" in data:
        R = quat_xyzw_to_rotmat(data["quaternion_xyzw"])
    elif "yaw_rad" in data:
        R = yaw_to_rotmat(float(data["yaw_rad"]))
    else:
        R = np.eye(3, dtype=np.float64)
    return make_hmat(R, translation)


def parse_extrinsic_to_ego(data: Dict[str, Any]) -> np.ndarray:
    return parse_pose_to_world(data)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def load_lidar_points(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        pts = np.load(path)
    elif ext == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        for k in (3, 4, 5):
            if raw.size % k == 0:
                pts = raw.reshape(-1, k)
                break
        else:
            raise ValueError("Unsupported .bin point layout")
    else:
        raise ValueError(f"Unsupported lidar file extension: {ext}")
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"LiDAR points must be Nx3+, got {pts.shape}")
    return pts[:, :3]


def make_box_corners(size_xyz: Sequence[float]) -> np.ndarray:
    l, w, h = [float(v) for v in size_xyz]
    x = l / 2.0
    y = w / 2.0
    z = h / 2.0
    return np.array(
        [
            [x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z],
            [x, y, -z], [x, -y, -z], [-x, -y, -z], [-x, y, -z],
        ],
        dtype=np.float64,
    )


def transform_box(corners_local: np.ndarray, center: np.ndarray, yaw_rad: float) -> np.ndarray:
    R = yaw_to_rotmat(yaw_rad)
    return (R @ corners_local.T).T + center[None, :]


def bottom_face_polygon_xy(center: np.ndarray, size_xyz: Sequence[float], yaw_rad: float) -> np.ndarray:
    l, w, _ = [float(v) for v in size_xyz]
    local = np.array(
        [[l / 2.0, w / 2.0, 0.0], [l / 2.0, -w / 2.0, 0.0], [-l / 2.0, -w / 2.0, 0.0], [-l / 2.0, w / 2.0, 0.0]],
        dtype=np.float64,
    )
    R = yaw_to_rotmat(yaw_rad)
    pts = (R @ local.T).T + center[None, :]
    return pts[:, :2]


def project_points_to_image(pts_cam: np.ndarray, K: np.ndarray, image_size: Tuple[int, int], min_depth: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    z = pts_cam[:, 2]
    valid = z > min_depth
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64), valid
    pts = pts_cam[valid]
    uvw = (K @ pts.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    w, h = image_size
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] = in_bounds
    return uv[in_bounds], valid


def dense_polyline(poly: np.ndarray, step: float = 0.5) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) <= 1:
        return poly
    chunks = []
    for a, b in zip(poly[:-1], poly[1:]):
        d = np.linalg.norm(b - a)
        n = max(2, int(math.ceil(d / step)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        chunks.append(a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None])
    chunks.append(poly[-1:])
    return np.concatenate(chunks, axis=0)


# -----------------------------
# Records
# -----------------------------


@dataclass
class CameraRecord:
    name: str
    image_path: str
    intrinsic: np.ndarray
    T_cam_to_ego: np.ndarray

    def mirrored(self, base_dir: Path) -> "CameraRecord":
        img = Image.open(self.image_path).convert("RGB")
        mirrored_path = str((base_dir / f"{Path(self.image_path).stem}_mirrored.png").resolve())
        img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(mirrored_path)
        R = self.T_cam_to_ego[:3, :3]
        t = self.T_cam_to_ego[:3, 3]
        T_m = make_hmat(S_EGO @ R @ S_CAM, S_EGO @ t)
        return CameraRecord(name=self.name, image_path=mirrored_path, intrinsic=self.intrinsic.copy(), T_cam_to_ego=T_m)


@dataclass
class LidarRecord:
    points_xyz: np.ndarray
    T_lidar_to_ego: np.ndarray

    def points_ego(self) -> np.ndarray:
        return apply_transform(self.T_lidar_to_ego, self.points_xyz)

    def mirrored(self) -> "LidarRecord":
        pts_ego = self.points_ego()
        pts_ego[:, 1] *= -1.0
        return LidarRecord(points_xyz=pts_ego, T_lidar_to_ego=np.eye(4, dtype=np.float64))


@dataclass
class BoxRecord:
    label: str
    center_ego: np.ndarray
    size_lwh: np.ndarray
    yaw_rad: float

    def mirrored(self) -> "BoxRecord":
        center = self.center_ego.copy()
        center[1] *= -1.0
        return BoxRecord(self.label, center, self.size_lwh.copy(), -self.yaw_rad)


@dataclass
class MapRecord:
    raster_path: Optional[str]
    polylines_world: List[np.ndarray] = field(default_factory=list)
    polylines_ego: List[np.ndarray] = field(default_factory=list)

    def mirrored(self, ego_T_world: np.ndarray) -> "MapRecord":
        mirrored_world = []
        for poly in self.polylines_world:
            poly_ego = apply_transform(invert_hmat(ego_T_world), poly)
            poly_ego[:, 1] *= -1.0
            mirrored_world.append(apply_transform(ego_T_world, poly_ego))
        mirrored_ego = []
        for poly in self.polylines_ego:
            p = poly.copy()
            p[:, 1] *= -1.0
            mirrored_ego.append(p)
        return MapRecord(raster_path=self.raster_path, polylines_world=mirrored_world, polylines_ego=mirrored_ego)


@dataclass
class HistoryRecord:
    poses_world: List[np.ndarray] = field(default_factory=list)
    trajectory_ego: Optional[np.ndarray] = None

    def mirrored(self, current_ego_T_world: np.ndarray) -> "HistoryRecord":
        poses_world_new = []
        current_world_T_ego = current_ego_T_world
        current_ego_inv = invert_hmat(current_world_T_ego)
        for pose_w in self.poses_world:
            rel = current_ego_inv @ pose_w
            rel[:3, :3] = S_EGO @ rel[:3, :3] @ S_EGO
            rel[:3, 3] = S_EGO @ rel[:3, 3]
            poses_world_new.append(current_world_T_ego @ rel)
        traj = None
        if self.trajectory_ego is not None:
            traj = self.trajectory_ego.copy()
            traj[:, 1] *= -1.0
        return HistoryRecord(poses_world=poses_world_new, trajectory_ego=traj)


@dataclass
class SceneRecord:
    ego_T_world: np.ndarray
    cameras: List[CameraRecord]
    lidar: Optional[LidarRecord]
    boxes: List[BoxRecord]
    map_record: MapRecord
    history: HistoryRecord = field(default_factory=HistoryRecord)
    tag: str = "scene"
    base_dir: Optional[str] = None

    def world_to_ego(self, pts_world: np.ndarray) -> np.ndarray:
        return apply_transform(invert_hmat(self.ego_T_world), pts_world)

    def ego_to_cam(self, camera: CameraRecord, pts_ego: np.ndarray) -> np.ndarray:
        return apply_transform(invert_hmat(camera.T_cam_to_ego), pts_ego)

    def lidar_to_ego(self) -> Optional[np.ndarray]:
        return None if self.lidar is None else self.lidar.points_ego()

    def mirrored(self) -> "SceneRecord":
        base_dir = Path(self.base_dir or ".").resolve()
        cams = [cam.mirrored(base_dir) for cam in self.cameras]
        lidar = None if self.lidar is None else self.lidar.mirrored()
        boxes = [b.mirrored() for b in self.boxes]
        map_record = self.map_record.mirrored(self.ego_T_world)
        history = self.history.mirrored(self.ego_T_world)
        return SceneRecord(
            ego_T_world=self.ego_T_world.copy(),
            cameras=cams,
            lidar=lidar,
            boxes=boxes,
            map_record=map_record,
            history=history,
            tag=self.tag + "_mirrored",
            base_dir=self.base_dir,
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        def pose_dict(T: np.ndarray) -> Dict[str, Any]:
            return {"translation": T[:3, 3].tolist(), "rotation_matrix": T[:3, :3].tolist()}

        out = {
            "tag": self.tag,
            "ego_pose": pose_dict(self.ego_T_world),
            "cameras": [
                {
                    "name": c.name,
                    "image_path": str(Path(c.image_path).name),
                    "intrinsic": c.intrinsic.tolist(),
                    "extrinsic_to_ego": pose_dict(c.T_cam_to_ego),
                }
                for c in self.cameras
            ],
            "bboxes": [
                {"label": b.label, "center_ego": b.center_ego.tolist(), "size_lwh": b.size_lwh.tolist(), "yaw_rad": b.yaw_rad}
                for b in self.boxes
            ],
            "map": {
                "polylines_world": [p.tolist() for p in self.map_record.polylines_world],
                "polylines_ego": [p.tolist() for p in self.map_record.polylines_ego],
            },
            "history": {
                "poses_world": [pose_dict(p) for p in self.history.poses_world],
                "trajectory_ego": None if self.history.trajectory_ego is None else self.history.trajectory_ego.tolist(),
            },
        }
        if self.lidar is not None:
            out["lidar"] = {
                "points_path": "synthetic_lidar.npy",
                "extrinsic_to_ego": pose_dict(self.lidar.T_lidar_to_ego),
            }
        return out

    def visualize(self, out_path: Optional[str], bev_range: float = 45.0, frustum_far: float = 35.0):
        visualize_scene(self, out_path, bev_range=bev_range, frustum_far=frustum_far, show_lidar=True, show_boxes=True, show_map=True)


# -----------------------------
# Parsing
# -----------------------------


def parse_scene(manifest_path: str) -> SceneRecord:
    data = load_json(manifest_path)
    base_dir = Path(manifest_path).resolve().parent
    ego_T_world = parse_pose_to_world(data.get("ego_pose", {}))

    cameras: List[CameraRecord] = []
    for cam in data.get("cameras", []):
        image_path = resolve_path(base_dir, cam["image_path"])
        intrinsic = ensure_array(cam["intrinsic"], shape=(3, 3))
        T_cam_to_ego = parse_extrinsic_to_ego(cam["extrinsic_to_ego"])
        cameras.append(CameraRecord(cam["name"], image_path, intrinsic, T_cam_to_ego))

    lidar = None
    lidar_block = data.get("lidar")
    if lidar_block:
        points_xyz = load_lidar_points(resolve_path(base_dir, lidar_block["points_path"]))
        T_lidar_to_ego = parse_extrinsic_to_ego(lidar_block.get("extrinsic_to_ego", {}))
        lidar = LidarRecord(points_xyz=points_xyz, T_lidar_to_ego=T_lidar_to_ego)

    boxes = [
        BoxRecord(
            label=box.get("label", "box"),
            center_ego=ensure_array(box["center_ego"], shape=(3,)),
            size_lwh=ensure_array(box["size_lwh"], shape=(3,)),
            yaw_rad=float(box.get("yaw_rad", 0.0)),
        )
        for box in data.get("bboxes", [])
    ]

    map_block = data.get("map", {})
    polylines_world: List[np.ndarray] = []
    for key in ("polylines_world",):
        for poly in map_block.get(key, []):
            arr = np.asarray(poly, dtype=np.float64)
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float64)], axis=1)
            polylines_world.append(arr[:, :3])
    polylines_ego: List[np.ndarray] = []
    for key in ("polylines_ego",):
        for poly in map_block.get(key, []):
            arr = np.asarray(poly, dtype=np.float64)
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float64)], axis=1)
            polylines_ego.append(arr[:, :3])
    map_record = MapRecord(raster_path=resolve_path(base_dir, map_block.get("raster_path")), polylines_world=polylines_world, polylines_ego=polylines_ego)

    hist_block = data.get("history", {})
    poses_world = [parse_pose_to_world(p) for p in hist_block.get("poses_world", [])]
    trajectory_ego = None
    if hist_block.get("trajectory_ego") is not None:
        trajectory_ego = np.asarray(hist_block["trajectory_ego"], dtype=np.float64)
        if trajectory_ego.shape[1] == 2:
            trajectory_ego = np.concatenate([trajectory_ego, np.zeros((trajectory_ego.shape[0], 1), dtype=np.float64)], axis=1)
    history = HistoryRecord(poses_world=poses_world, trajectory_ego=trajectory_ego)

    return SceneRecord(ego_T_world=ego_T_world, cameras=cameras, lidar=lidar, boxes=boxes, map_record=map_record, history=history, tag=data.get("tag", "scene"), base_dir=str(base_dir))


# -----------------------------
# Visualization
# -----------------------------


def polylines_ego(scene: SceneRecord) -> List[np.ndarray]:
    out = [p.copy() for p in scene.map_record.polylines_ego]
    for poly in scene.map_record.polylines_world:
        out.append(scene.world_to_ego(poly))
    return out


def compute_camera_ground_footprint(camera: CameraRecord, image_size: Tuple[int, int], far: float, n_side: int = 20) -> np.ndarray:
    w, h = image_size
    border = []
    xs = np.linspace(0, w - 1, n_side)
    ys = np.linspace(0, h - 1, n_side)
    for x in xs:
        border.append([x, h - 1, 1.0])
    for y in ys[::-1]:
        border.append([w - 1, y, 1.0])
    for x in xs[::-1]:
        border.append([x, 0, 1.0])
    for y in ys:
        border.append([0, y, 1.0])
    uv1 = np.asarray(border, dtype=np.float64)
    rays_cam = (np.linalg.inv(camera.intrinsic) @ uv1.T).T
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
    R = camera.T_cam_to_ego[:3, :3]
    t = camera.T_cam_to_ego[:3, 3]
    rays_ego = (R @ rays_cam.T).T
    origin = np.repeat(t[None, :], len(rays_ego), axis=0)
    footprints = []
    for o, d in zip(origin, rays_ego):
        if abs(d[2]) < 1e-6:
            continue
        tau = -o[2] / d[2]
        if tau <= 0:
            continue
        p = o + tau * d
        if np.linalg.norm(p[:2] - t[:2]) <= far:
            footprints.append(p[:2])
    if len(footprints) < 3:
        heading = R[:, 2]
        c = t[:2]
        left = np.array([-heading[1], heading[0]])
        a = c + heading[:2] * far + left * far * 0.4
        b = c + heading[:2] * far - left * far * 0.4
        return np.vstack([c, a, b])
    return np.asarray(footprints, dtype=np.float64)


def draw_rig(ax: plt.Axes, T_to_ego: np.ndarray, label: str, color: str, scale: float = 1.4, text: bool = False):
    o = T_to_ego[:3, 3]
    R = T_to_ego[:3, :3]
    fwd = R[:, 2][:2]
    left = np.array([-fwd[1], fwd[0]])
    nose = o[:2] + fwd * scale
    rear = o[:2] - fwd * scale * 0.8
    p1 = rear + left * scale * 0.45
    p2 = rear - left * scale * 0.45
    ax.add_patch(patches.Polygon(np.vstack([nose, p1, p2]), closed=True, fill=False, edgecolor=color, linewidth=1.6, alpha=0.95))
    ax.scatter([o[0]], [o[1]], s=18, c=color, alpha=0.95)
    if text:
        ax.text(o[0] + 0.25, o[1] + 0.25, label, fontsize=8, color=color, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def draw_camera_overlay(ax: plt.Axes, scene: SceneRecord, camera: CameraRecord, show_lidar: bool, show_boxes: bool):
    img = np.array(Image.open(camera.image_path).convert("RGB"))
    ax.imshow(img)
    ax.set_title(camera.name, fontsize=10)
    ax.axis("off")
    h, w = img.shape[:2]

    if show_lidar and scene.lidar is not None:
        pts_ego = scene.lidar_to_ego()
        pts_cam = scene.ego_to_cam(camera, pts_ego)
        uv, mask = project_points_to_image(pts_cam, camera.intrinsic, (w, h), min_depth=0.2)
        if len(uv) > 0:
            depths = pts_cam[mask, 2]
            ax.scatter(uv[:, 0], uv[:, 1], s=2, c=depths, cmap="plasma", alpha=0.55, linewidths=0)

    polys = polylines_ego(scene)
    for poly in polys:
        dense = dense_polyline(poly, step=0.5)
        pts_cam = scene.ego_to_cam(camera, dense)
        uv, mask = project_points_to_image(pts_cam, camera.intrinsic, (w, h), min_depth=0.2)
        if np.count_nonzero(mask) >= 2:
            uv_full = np.full((len(dense), 2), np.nan)
            uv_full[np.where(mask)[0]] = uv
            valid = np.isfinite(uv_full[:, 0])
            start = None
            for i, ok in enumerate(valid):
                if ok and start is None:
                    start = i
                if (not ok or i == len(valid) - 1) and start is not None:
                    end = i if not ok else i + 1
                    if end - start >= 2:
                        ax.plot(uv_full[start:end, 0], uv_full[start:end, 1], color="#F6E05E", linewidth=2.2, alpha=0.95)
                    start = None

    if show_boxes:
        for box in scene.boxes:
            corners_ego = transform_box(make_box_corners(box.size_lwh), box.center_ego, box.yaw_rad)
            corners_cam = scene.ego_to_cam(camera, corners_ego)
            uv, mask = project_points_to_image(corners_cam, camera.intrinsic, (w, h), min_depth=0.2)
            if np.count_nonzero(mask) < 8:
                continue
            uv_full = (camera.intrinsic @ corners_cam.T).T
            uv_full = uv_full[:, :2] / uv_full[:, 2:3]
            for i, j in CAMERA_BOX_EDGES:
                ax.plot([uv_full[i, 0], uv_full[j, 0]], [uv_full[i, 1], uv_full[j, 1]], color="#FB923C", linewidth=1.8)
            ax.text(uv_full[0, 0] + 3, uv_full[0, 1] - 5, box.label, fontsize=8, color="white", bbox=dict(boxstyle="round,pad=0.15", fc="#9A3412", ec="none", alpha=0.75))


def draw_bev(ax: plt.Axes, scene: SceneRecord, bev_range: float, frustum_far: float, show_map: bool, show_lidar: bool, show_boxes: bool):
    ax.set_title(f"BEV (ego frame) - {scene.tag}", fontsize=16)
    ax.set_aspect("equal")
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    ax.grid(True, alpha=0.18)
    ax.set_facecolor("#F8FAFC")

    ego_poly = np.array([[2.4, 1.0], [2.4, -1.0], [-2.0, -1.0], [-2.0, 1.0]])
    ax.add_patch(patches.Polygon(ego_poly, fill=True, facecolor="#CBD5E1", edgecolor="black", linewidth=1.6, label="ego"))
    ax.arrow(0.0, 0.0, 3.0, 0.0, width=0.06, head_width=0.7, head_length=0.9, color="black", length_includes_head=True)

    if show_map:
        for poly in polylines_ego(scene):
            ax.plot(poly[:, 0], poly[:, 1], color="#2563EB", linewidth=2.2, alpha=0.85, label="map")

    if scene.history.trajectory_ego is not None and len(scene.history.trajectory_ego) > 1:
        tr = scene.history.trajectory_ego
        ax.plot(tr[:, 0], tr[:, 1], color="#16A34A", linewidth=2.8, alpha=0.95, label="history path")
        ax.scatter(tr[:, 0], tr[:, 1], s=np.linspace(18, 48, len(tr)), c=np.linspace(0.35, 1.0, len(tr)), cmap="Greens", alpha=0.8, zorder=4)

    if scene.history.poses_world:
        for idx, pose_w in enumerate(scene.history.poses_world):
            rel = invert_hmat(scene.ego_T_world) @ pose_w
            draw_rig(ax, rel, f"hist_{idx}", color="#059669", scale=1.1, text=False)

    if show_lidar and scene.lidar is not None:
        pts = scene.lidar_to_ego()
        mask = (np.abs(pts[:, 0]) <= bev_range) & (np.abs(pts[:, 1]) <= bev_range)
        pts = pts[mask]
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=2, c="#64748B", alpha=0.45, label="lidar")

    cam_label_done = False
    for cam in scene.cameras:
        img = Image.open(cam.image_path)
        footprint = compute_camera_ground_footprint(cam, img.size, far=frustum_far)
        center = cam.T_cam_to_ego[:2, 3]
        ax.plot(np.r_[footprint[:, 0], footprint[0, 0]], np.r_[footprint[:, 1], footprint[0, 1]], color="#A855F7", linewidth=1.4, alpha=0.55, label=None)
        draw_rig(ax, cam.T_cam_to_ego, cam.name, color="#7C3AED", scale=0.9, text=True)
        if not cam_label_done:
            ax.plot([], [], color="#A855F7", linewidth=1.4, alpha=0.7, label="camera footprint")
            cam_label_done = True

    if scene.lidar is not None:
        draw_rig(ax, scene.lidar.T_lidar_to_ego, "lidar", color="#DC2626", scale=1.0, text=True)

    if show_boxes:
        for box in scene.boxes:
            poly = bottom_face_polygon_xy(box.center_ego, box.size_lwh, box.yaw_rad)
            ax.add_patch(patches.Polygon(poly, fill=False, edgecolor="#EA580C", linewidth=2.2))
            center = poly.mean(axis=0)
            heading = np.array([math.cos(box.yaw_rad), math.sin(box.yaw_rad)]) * 1.8
            ax.arrow(center[0], center[1], heading[0], heading[1], width=0.05, head_width=0.5, head_length=0.6, color="#EA580C", length_includes_head=True)
            ax.text(center[0] + 0.2, center[1] + 0.2, box.label, fontsize=9, color="#7C2D12", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            new_h.append(h)
            new_l.append(l)
            seen.add(l)
    ax.legend(new_h, new_l, loc="upper right", fontsize=9, framealpha=0.95)


def visualize_scene(scene: SceneRecord, out_path: Optional[str], bev_range: float, frustum_far: float, show_lidar: bool, show_boxes: bool, show_map: bool):
    n_cams = max(1, len(scene.cameras))
    ncols = min(4, max(2, n_cams))
    nrows_cam = int(math.ceil(n_cams / ncols))
    fig = plt.figure(figsize=(5.2 * ncols, 3.6 * nrows_cam + 8.0))
    gs = fig.add_gridspec(nrows_cam + 1, ncols, height_ratios=[1.0] * nrows_cam + [1.9])

    for idx, cam in enumerate(scene.cameras):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        draw_camera_overlay(ax, scene, cam, show_lidar=show_lidar, show_boxes=show_boxes)
    for idx in range(len(scene.cameras), nrows_cam * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.axis("off")

    bev_ax = fig.add_subplot(gs[nrows_cam, :])
    draw_bev(bev_ax, scene, bev_range=bev_range, frustum_far=frustum_far, show_map=show_map, show_lidar=show_lidar, show_boxes=show_boxes)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
        print(f"Saved visualization to: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic multi-modal BEV debug viewer")
    p.add_argument("--manifest", required=True, help="Path to scene manifest JSON")
    p.add_argument("--out", default=None, help="Optional output image path")
    p.add_argument("--bev-range", type=float, default=45.0)
    p.add_argument("--frustum-far", type=float, default=35.0)
    p.add_argument("--mirror", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    scene = parse_scene(args.manifest)
    if args.mirror:
        scene = scene.mirrored()
    visualize_scene(scene, args.out, bev_range=args.bev_range, frustum_far=args.frustum_far, show_lidar=True, show_boxes=True, show_map=True)


if __name__ == "__main__":
    main()
