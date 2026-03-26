#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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


def project_points_to_image(pts_cam: np.ndarray, K: np.ndarray, image_size: Tuple[int, int], min_depth: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
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


def make_box_corners(size_xyz: Sequence[float]) -> np.ndarray:
    l, w, h = [float(v) for v in size_xyz]
    x = l / 2.0
    y = w / 2.0
    z = h / 2.0
    return np.array(
        [[x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z], [x, y, -z], [x, -y, -z], [-x, -y, -z], [-x, y, -z]],
        dtype=np.float64,
    )


def transform_box(corners_local: np.ndarray, center: np.ndarray, yaw_rad: float) -> np.ndarray:
    R = yaw_to_rotmat(yaw_rad)
    return (R @ corners_local.T).T + center[None, :]


def bottom_face_polygon_xy(center: np.ndarray, size_xyz: Sequence[float], yaw_rad: float) -> np.ndarray:
    l, w, _ = [float(v) for v in size_xyz]
    local = np.array([[l / 2.0, w / 2.0, 0.0], [l / 2.0, -w / 2.0, 0.0], [-l / 2.0, -w / 2.0, 0.0], [-l / 2.0, w / 2.0, 0.0]], dtype=np.float64)
    R = yaw_to_rotmat(yaw_rad)
    pts = (R @ local.T).T + center[None, :]
    return pts[:, :2]


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
        if raw.size % 5 == 0:
            pts = raw.reshape(-1, 5)
        elif raw.size % 4 == 0:
            pts = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            pts = raw.reshape(-1, 3)
        else:
            raise ValueError("Unsupported .bin point layout; expected Nx3/Nx4/Nx5 float32")
    else:
        raise ValueError(f"Unsupported lidar file extension: {ext}")
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"LiDAR points must be Nx3+, got {pts.shape}")
    return pts[:, :3]


# -----------------------------
# Records
# -----------------------------


@dataclass
class CameraRecord:
    name: str
    image_path: str
    intrinsic: np.ndarray
    T_cam_to_ego: np.ndarray


@dataclass
class LidarRecord:
    points_xyz: np.ndarray
    T_lidar_to_ego: np.ndarray


@dataclass
class BoxRecord:
    label: str
    center_ego: np.ndarray
    size_lwh: np.ndarray
    yaw_rad: float


@dataclass
class MapRecord:
    raster_path: Optional[str]
    polylines_world: List[np.ndarray]
    polylines_ego: List[np.ndarray]


@dataclass
class SceneRecord:
    ego_T_world: np.ndarray
    cameras: List[CameraRecord]
    lidar: Optional[LidarRecord]
    boxes: List[BoxRecord]
    map_record: MapRecord

    def mirrored(self, image_dir: Optional[str] = None) -> "SceneRecord":
        S_ego = np.diag([1.0, -1.0, 1.0])
        S_cam = np.diag([-1.0, 1.0, 1.0])

        new_cams: List[CameraRecord] = []
        if image_dir is not None:
            out_dir = Path(image_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None

        for cam in self.cameras:
            R = cam.T_cam_to_ego[:3, :3]
            t = cam.T_cam_to_ego[:3, 3]
            Rm = S_ego @ R @ S_cam
            tm = S_ego @ t
            new_path = cam.image_path
            if out_dir is not None:
                img = Image.open(cam.image_path).convert("RGB")
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                new_path = str((out_dir / Path(cam.image_path).name).resolve())
                img.save(new_path)
            new_cams.append(CameraRecord(name=cam.name, image_path=new_path, intrinsic=cam.intrinsic.copy(), T_cam_to_ego=make_hmat(Rm, tm)))

        new_lidar = None
        if self.lidar is not None:
            pts_ego = lidar_to_ego(self)
            assert pts_ego is not None
            pts_ego_m = pts_ego.copy()
            pts_ego_m[:, 1] *= -1.0
            new_lidar = LidarRecord(points_xyz=pts_ego_m, T_lidar_to_ego=np.eye(4, dtype=np.float64))

        new_boxes = [
            BoxRecord(label=b.label, center_ego=np.array([b.center_ego[0], -b.center_ego[1], b.center_ego[2]], dtype=np.float64), size_lwh=b.size_lwh.copy(), yaw_rad=-b.yaw_rad)
            for b in self.boxes
        ]

        new_map = MapRecord(
            raster_path=self.map_record.raster_path,
            polylines_world=self.map_record.polylines_world,
            polylines_ego=[np.column_stack([p[:, 0], -p[:, 1], p[:, 2]]) for p in self.all_map_polylines_ego()],
        )
        return SceneRecord(ego_T_world=self.ego_T_world.copy(), cameras=new_cams, lidar=new_lidar, boxes=new_boxes, map_record=new_map)

    def all_map_polylines_ego(self) -> List[np.ndarray]:
        polys = list(self.map_record.polylines_ego)
        for poly_world in self.map_record.polylines_world:
            polys.append(world_to_ego(self, poly_world))
        return polys


# -----------------------------
# Parsing and transforms
# -----------------------------


def parse_scene(manifest_path: str) -> SceneRecord:
    data = load_json(manifest_path)
    base_dir = Path(manifest_path).resolve().parent

    ego_T_world = parse_pose_to_world(data.get("ego_pose", {}))

    cameras: List[CameraRecord] = []
    for cam in data.get("cameras", []):
        image_path = resolve_path(base_dir, cam["image_path"])
        if image_path is None:
            raise ValueError(f"Camera {cam.get('name', 'unknown')} missing image_path")
        cameras.append(CameraRecord(name=cam["name"], image_path=image_path, intrinsic=ensure_array(cam["intrinsic"], shape=(3, 3)), T_cam_to_ego=parse_extrinsic_to_ego(cam["extrinsic_to_ego"])))

    lidar_block = data.get("lidar")
    lidar: Optional[LidarRecord] = None
    if lidar_block:
        lidar_path = resolve_path(base_dir, lidar_block["points_path"])
        if lidar_path is None:
            raise ValueError("lidar.points_path missing")
        lidar = LidarRecord(points_xyz=load_lidar_points(lidar_path), T_lidar_to_ego=parse_extrinsic_to_ego(lidar_block.get("extrinsic_to_ego", {})))

    boxes = [
        BoxRecord(label=box.get("label", "box"), center_ego=ensure_array(box["center_ego"], shape=(3,)), size_lwh=ensure_array(box["size_lwh"], shape=(3,)), yaw_rad=float(box.get("yaw_rad", 0.0)))
        for box in data.get("bboxes", [])
    ]

    map_block = data.get("map", {})
    raster_path = resolve_path(base_dir, map_block.get("raster_path"))
    polylines_world: List[np.ndarray] = []
    polylines_ego: List[np.ndarray] = []

    def _norm_poly(poly: Any) -> np.ndarray:
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Polyline must be Nx2/Nx3, got {arr.shape}")
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float64)], axis=1)
        return arr[:, :3]

    for poly in map_block.get("polylines_world", []):
        polylines_world.append(_norm_poly(poly))
    for poly in map_block.get("polylines_ego", []):
        polylines_ego.append(_norm_poly(poly))

    return SceneRecord(
        ego_T_world=ego_T_world,
        cameras=cameras,
        lidar=lidar,
        boxes=boxes,
        map_record=MapRecord(raster_path=raster_path, polylines_world=polylines_world, polylines_ego=polylines_ego),
    )


def world_to_ego(scene: SceneRecord, pts_world: np.ndarray) -> np.ndarray:
    return apply_transform(invert_hmat(scene.ego_T_world), pts_world)


def ego_to_cam(camera: CameraRecord, pts_ego: np.ndarray) -> np.ndarray:
    return apply_transform(invert_hmat(camera.T_cam_to_ego), pts_ego)


def lidar_to_ego(scene: SceneRecord) -> Optional[np.ndarray]:
    if scene.lidar is None:
        return None
    return apply_transform(scene.lidar.T_lidar_to_ego, scene.lidar.points_xyz)


# -----------------------------
# Visualization
# -----------------------------


CAMERA_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def compute_camera_ground_footprint(camera: CameraRecord, image_size: Tuple[int, int], max_range: float = 40.0, samples_per_edge: int = 32) -> np.ndarray:
    w, h = image_size
    K_inv = np.linalg.inv(camera.intrinsic)
    cam_origin_ego = camera.T_cam_to_ego[:3, 3]
    R = camera.T_cam_to_ego[:3, :3]

    border = []
    xs = np.linspace(0.0, w - 1.0, samples_per_edge)
    ys = np.linspace(0.0, h - 1.0, samples_per_edge)
    for x in xs:
        border.append([x, 0.0, 1.0])
    for y in ys[1:]:
        border.append([w - 1.0, y, 1.0])
    for x in xs[-2::-1]:
        border.append([x, h - 1.0, 1.0])
    for y in ys[-2:0:-1]:
        border.append([0.0, y, 1.0])

    pts_xy = []
    for uv1 in np.asarray(border, dtype=np.float64):
        ray_cam = K_inv @ uv1
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_ego = R @ ray_cam
        dz = ray_ego[2]
        if abs(dz) < 1e-9:
            continue
        t = -cam_origin_ego[2] / dz
        if t <= 0:
            continue
        hit = cam_origin_ego + t * ray_ego
        if np.linalg.norm(hit[:2] - cam_origin_ego[:2]) <= max_range:
            pts_xy.append(hit[:2])
    if len(pts_xy) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts_xy, dtype=np.float64)


def draw_camera_overlay(ax: plt.Axes, scene: SceneRecord, camera: CameraRecord, show_lidar: bool, show_boxes: bool):
    img = np.array(Image.open(camera.image_path).convert("RGB"))
    ax.imshow(img)
    ax.set_title(camera.name, fontsize=10)
    ax.axis("off")
    h, w = img.shape[:2]

    if show_lidar and scene.lidar is not None:
        pts_ego = lidar_to_ego(scene)
        if pts_ego is not None and pts_ego.size > 0:
            pts_cam = ego_to_cam(camera, pts_ego)
            uv, mask = project_points_to_image(pts_cam, camera.intrinsic, image_size=(w, h), min_depth=0.3)
            if uv.shape[0] > 0:
                depths = pts_cam[mask, 2]
                ax.scatter(uv[:, 0], uv[:, 1], s=1, c=depths, cmap="viridis", alpha=0.8)

    if show_boxes:
        for box in scene.boxes:
            corners_ego = transform_box(make_box_corners(box.size_lwh), box.center_ego, box.yaw_rad)
            corners_cam = ego_to_cam(camera, corners_ego)
            uv, mask = project_points_to_image(corners_cam, camera.intrinsic, image_size=(w, h), min_depth=0.3)
            if np.count_nonzero(mask) < 8:
                continue
            uv_full = (camera.intrinsic @ corners_cam.T).T
            uv_full = uv_full[:, :2] / uv_full[:, 2:3]
            for i, j in CAMERA_BOX_EDGES:
                ax.plot([uv_full[i, 0], uv_full[j, 0]], [uv_full[i, 1], uv_full[j, 1]], linewidth=1.0, color="red")
            ax.text(uv_full[0, 0], uv_full[0, 1], box.label, fontsize=7, color="white", bbox=dict(facecolor="black", alpha=0.4, pad=1))


def draw_bev(ax: plt.Axes, scene: SceneRecord, bev_range: float, frustum_far: float, show_map: bool, show_lidar: bool, show_boxes: bool):
    ax.set_title("BEV (ego frame)")
    ax.set_aspect("equal")
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    ax.grid(True, alpha=0.2)

    ego_poly = np.array([[2.0, 1.0], [2.0, -1.0], [-2.0, -1.0], [-2.0, 1.0]])
    ax.add_patch(patches.Polygon(ego_poly, fill=False, linewidth=1.5, label="ego"))

    if show_map:
        for poly_ego in scene.all_map_polylines_ego():
            ax.plot(poly_ego[:, 0], poly_ego[:, 1], linestyle="--", linewidth=1.2, alpha=0.9, color="orange")
        if scene.map_record.raster_path:
            try:
                raster = np.array(Image.open(scene.map_record.raster_path).convert("RGB"))
                ax.imshow(raster, extent=[-bev_range, bev_range, -bev_range, bev_range], origin="lower", alpha=0.25)
            except Exception as exc:
                ax.text(0.02, 0.02, f"Map raster load failed: {exc}", transform=ax.transAxes, fontsize=8)

    if show_lidar and scene.lidar is not None:
        pts_ego = lidar_to_ego(scene)
        if pts_ego is not None and pts_ego.size > 0:
            mask = (np.abs(pts_ego[:, 0]) <= bev_range) & (np.abs(pts_ego[:, 1]) <= bev_range)
            pts = pts_ego[mask]
            if pts.shape[0] > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.6, label="lidar")

    for cam in scene.cameras:
        try:
            img = Image.open(cam.image_path)
            footprint = compute_camera_ground_footprint(cam, image_size=img.size, max_range=frustum_far)
            cam_center = cam.T_cam_to_ego[:2, 3]
            if footprint.shape[0] >= 3:
                poly = np.vstack([cam_center[None, :], footprint, cam_center[None, :]])
                ax.plot(poly[:, 0], poly[:, 1], linewidth=1.0)
            ax.text(cam_center[0], cam_center[1], cam.name, fontsize=7)
        except Exception as exc:
            ax.text(0.02, 0.95, f"Frustum error for {cam.name}: {exc}", transform=ax.transAxes, fontsize=8)

    if show_boxes:
        for box in scene.boxes:
            poly = bottom_face_polygon_xy(box.center_ego, box.size_lwh, box.yaw_rad)
            ax.add_patch(patches.Polygon(poly, fill=False, linewidth=1.2, color="red"))
            ax.text(poly[0, 0], poly[0, 1], box.label, fontsize=7, color="red")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)


def visualize_scene(scene: SceneRecord, out_path: Optional[str], bev_range: float, frustum_far: float, show_lidar: bool, show_boxes: bool, show_map: bool):
    n_cams = max(1, len(scene.cameras))
    ncols = min(3, n_cams)
    nrows_cam = int(math.ceil(n_cams / ncols))
    fig = plt.figure(figsize=(6 * ncols, 4 * (nrows_cam + 1)))
    gs = fig.add_gridspec(nrows_cam + 1, ncols)

    for idx, cam in enumerate(scene.cameras):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        draw_camera_overlay(ax, scene, cam, show_lidar=show_lidar, show_boxes=show_boxes)

    for idx in range(len(scene.cameras), nrows_cam * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    bev_ax = fig.add_subplot(gs[nrows_cam, :])
    draw_bev(bev_ax, scene, bev_range=bev_range, frustum_far=frustum_far, show_map=show_map, show_lidar=show_lidar, show_boxes=show_boxes)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved visualization to: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic multi-modal BEV debug viewer")
    p.add_argument("--manifest", required=True, help="Path to scene manifest JSON")
    p.add_argument("--out", default=None, help="Optional output image path")
    p.add_argument("--bev-range", type=float, default=60.0, help="Half width of BEV plot in meters")
    p.add_argument("--frustum-far", type=float, default=40.0, help="Max range for camera ground footprint")
    p.add_argument("--mirror", action="store_true", help="Mirror the whole scene before visualization")
    p.add_argument("--mirror-image-dir", default=None, help="Optional dir to write mirrored camera images into")
    p.add_argument("--no-lidar", action="store_true", help="Hide lidar points")
    p.add_argument("--no-boxes", action="store_true", help="Hide 3D boxes")
    p.add_argument("--no-map", action="store_true", help="Hide map overlays")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    scene = parse_scene(args.manifest)
    if args.mirror:
        scene = scene.mirrored(image_dir=args.mirror_image_dir)
    visualize_scene(scene, out_path=args.out, bev_range=args.bev_range, frustum_far=args.frustum_far, show_lidar=not args.no_lidar, show_boxes=not args.no_boxes, show_map=not args.no_map)


if __name__ == "__main__":
    main()
