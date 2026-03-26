#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import bev_debug_viewer as bev

ROOT = Path(__file__).resolve().parent
W, H = 1280, 720
FX = FY = 720.0
CX, CY = W / 2.0, H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n


def make_cam_rot_from_target(cam_pos, target, world_up=np.array([0.0, 0.0, 1.0])):
    z_fwd = unit(target - cam_pos)
    y_down = -world_up
    y_down = unit(y_down - np.dot(y_down, z_fwd) * z_fwd)
    x_right = unit(np.cross(y_down, z_fwd))
    return np.stack([x_right, y_down, z_fwd], axis=1)


def camera_record(name, pos, yaw_deg, pitch_down_deg=8.0):
    yaw = math.radians(yaw_deg)
    forward_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    target = np.asarray(pos, dtype=np.float64) + forward_xy * 20.0 + np.array([0.0, 0.0, -20.0 * math.tan(math.radians(pitch_down_deg))])
    R = make_cam_rot_from_target(np.asarray(pos, dtype=np.float64), target)
    return {
        "name": name,
        "image_path": f"./{name}.png",
        "intrinsic": K.tolist(),
        "extrinsic_to_ego": {"translation": list(map(float, pos)), "rotation_matrix": R.tolist()},
    }


def generate_box_points(center, size_lwh, yaw, n_per_face=240, seed=0):
    rng = np.random.default_rng(seed)
    l, w, h = size_lwh
    local = []
    per_face = max(8, n_per_face // 6)
    for axis, val in [(0, l / 2), (0, -l / 2), (1, w / 2), (1, -w / 2), (2, h / 2), (2, -h / 2)]:
        pts = rng.uniform(-1.0, 1.0, size=(per_face, 3))
        pts[:, 0] *= l / 2
        pts[:, 1] *= w / 2
        pts[:, 2] *= h / 2
        pts[:, axis] = val
        local.append(pts)
    local = np.concatenate(local, axis=0)
    R = bev.yaw_to_rotmat(yaw)
    return (R @ local.T).T + np.asarray(center)[None, :]


def generate_lane_points(y_values, x_min=-28.0, x_max=45.0, z=0.0, samples=260):
    xs = np.linspace(x_min, x_max, samples)
    pts = []
    for y in y_values:
        jitter = 0.03 * np.sin(xs * 0.3 + y)
        lane = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z) + jitter], axis=1)
        pts.append(lane)
    return np.concatenate(pts, axis=0)


def dense_polyline(poly, step=0.4):
    poly = np.asarray(poly, dtype=np.float64)
    segs = []
    for a, b in zip(poly[:-1], poly[1:]):
        d = np.linalg.norm(b - a)
        n = max(2, int(math.ceil(d / step)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        segs.append(a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None])
    segs.append(poly[-1:])
    return np.concatenate(segs, axis=0)


def chunk_visible_polyline(cam, pts_ego):
    pts_cam = bev.ego_to_cam(cam, pts_ego)
    _, valid = bev.project_points_to_image(pts_cam, cam.intrinsic, (W, H), min_depth=0.2)
    if not np.any(valid):
        return []
    valid_idx = np.where(valid)[0]
    chunks = []
    start = 0
    while start < len(valid_idx):
        end = start + 1
        while end < len(valid_idx) and valid_idx[end] == valid_idx[end - 1] + 1:
            end += 1
        idxs = valid_idx[start:end]
        pts_cam_seg = pts_cam[idxs]
        uvw = (cam.intrinsic @ pts_cam_seg.T).T
        uv_seg = uvw[:, :2] / uvw[:, 2:3]
        if len(uv_seg) >= 2:
            chunks.append([tuple(map(float, p)) for p in uv_seg])
        start = end
    return chunks


def render_camera_image(scene, cam, boxes, map_polys, out_path):
    img = Image.new("RGB", (W, H), (210, 230, 245))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, H * 0.45, W, H], fill=(60, 65, 72, 255))
    draw.line([(0, H * 0.45), (W, H * 0.45)], fill=(180, 190, 205, 255), width=2)

    lane_colors = [(255, 220, 80, 220), (255, 255, 255, 220)]
    for i, poly in enumerate(map_polys):
        pts_dense = dense_polyline(poly, step=0.25)
        for chunk in chunk_visible_polyline(cam, pts_dense):
            if len(chunk) >= 2:
                draw.line(chunk, fill=lane_colors[i % len(lane_colors)], width=5)

    for box in boxes:
        corners = bev.transform_box(bev.make_box_corners(box["size_lwh"]), np.asarray(box["center_ego"]), box["yaw_rad"])
        pts_cam = bev.ego_to_cam(cam, corners)
        _, valid = bev.project_points_to_image(pts_cam, cam.intrinsic, (W, H), min_depth=0.2)
        if valid.sum() < 6:
            continue
        uv_all = np.full((8, 2), np.nan)
        valid_idx = np.where(valid)[0]
        uv = (cam.intrinsic @ pts_cam[valid].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv_all[valid_idx] = uv
        faces = [[0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5], [0, 3, 7, 4], [3, 2, 6, 7]]
        for face in faces:
            if np.all(np.isfinite(uv_all[face])):
                draw.polygon([tuple(uv_all[i]) for i in face], fill=(220, 60, 60, 45))
        for i0, i1 in bev.CAMERA_BOX_EDGES:
            if np.all(np.isfinite(uv_all[[i0, i1]])):
                draw.line([tuple(uv_all[i0]), tuple(uv_all[i1])], fill=(170, 20, 20, 255), width=3)

    img.save(out_path)


def main():
    camera_defs = [
        camera_record("cam_front_left", [1.9, 0.35, 1.45], 28),
        camera_record("cam_front_right", [1.9, -0.35, 1.45], -28),
        camera_record("cam_left_front", [0.7, 0.95, 1.45], 92),
        camera_record("cam_left_rear", [-0.9, 0.95, 1.45], 145),
        camera_record("cam_right_front", [0.7, -0.95, 1.45], -92),
        camera_record("cam_right_rear", [-0.9, -0.95, 1.45], -145),
        camera_record("cam_rear", [-1.9, 0.00, 1.45], 180),
    ]

    boxes = [
        {"label": "car_gt", "center_ego": [18.0, 0.4, 0.85], "size_lwh": [4.6, 1.9, 1.7], "yaw_rad": 0.10},
        {"label": "car_left", "center_ego": [14.0, 4.2, 0.85], "size_lwh": [4.6, 1.9, 1.7], "yaw_rad": -0.08},
        {"label": "car_right", "center_ego": [16.0, -4.0, 0.85], "size_lwh": [4.6, 1.9, 1.7], "yaw_rad": 0.05},
    ]

    map_polys = [
        [[-28.0, -1.8, 0.0], [-10.0, -1.8, 0.0], [8.0, -1.8, 0.0], [26.0, -1.8, 0.0], [45.0, -1.8, 0.0]],
        [[-28.0, 1.8, 0.0], [-10.0, 1.8, 0.0], [8.0, 1.8, 0.0], [26.0, 1.8, 0.0], [45.0, 1.8, 0.0]],
    ]

    lidar_points_ego = np.concatenate(
        [
            generate_box_points(boxes[0]["center_ego"], boxes[0]["size_lwh"], boxes[0]["yaw_rad"], n_per_face=300, seed=7),
            generate_box_points(boxes[1]["center_ego"], boxes[1]["size_lwh"], boxes[1]["yaw_rad"], n_per_face=220, seed=8),
            generate_box_points(boxes[2]["center_ego"], boxes[2]["size_lwh"], boxes[2]["yaw_rad"], n_per_face=220, seed=9),
            generate_lane_points([-1.8, 1.8], x_min=-25.0, x_max=42.0, z=0.0, samples=260),
        ],
        axis=0,
    )

    lidar_t = np.array([0.0, 0.0, 1.8], dtype=np.float64)
    lidar_points_sensor = lidar_points_ego - lidar_t[None, :]
    np.save(ROOT / "synthetic_lidar.npy", lidar_points_sensor.astype(np.float32))

    manifest = {
        "tag": "synthetic_rig_scene",
        "ego_pose": {"translation": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        "cameras": camera_defs,
        "lidar": {"points_path": "./synthetic_lidar.npy", "extrinsic_to_ego": {"translation": lidar_t.tolist(), "rotation_matrix": np.eye(3).tolist()}},
        "bboxes": boxes,
        "map": {"polylines_ego": map_polys},
    }

    with open(ROOT / "synthetic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    scene = bev.parse_scene(str(ROOT / "synthetic_manifest.json"))
    for cam in scene.cameras:
        render_camera_image(scene, cam, boxes, [np.asarray(p, dtype=np.float64) for p in map_polys], ROOT / f"{cam.name}.png")

    scene = bev.parse_scene(str(ROOT / "synthetic_manifest.json"))
    bev.visualize_scene(scene, str(ROOT / "synthetic_debug.png"), bev_range=35.0, frustum_far=28.0, show_lidar=True, show_boxes=True, show_map=True)

    mirrored_dir = ROOT / "mirrored_images"
    mirrored_scene = scene.mirrored(image_dir=str(mirrored_dir))
    bev.visualize_scene(mirrored_scene, str(ROOT / "synthetic_debug_mirror.png"), bev_range=35.0, frustum_far=28.0, show_lidar=True, show_boxes=True, show_map=True)

    mirror_manifest = {
        "tag": "synthetic_rig_scene_mirror",
        "ego_pose": manifest["ego_pose"],
        "cameras": [
            {
                "name": c.name,
                "image_path": str(Path("mirrored_images") / Path(c.image_path).name),
                "intrinsic": c.intrinsic.tolist(),
                "extrinsic_to_ego": {"translation": c.T_cam_to_ego[:3, 3].tolist(), "rotation_matrix": c.T_cam_to_ego[:3, :3].tolist()},
            }
            for c in mirrored_scene.cameras
        ],
        "lidar": {"points_path": "./synthetic_lidar_mirror.npy", "extrinsic_to_ego": {"translation": [0.0, 0.0, 0.0], "rotation_matrix": np.eye(3).tolist()}},
        "bboxes": [
            {"label": b.label, "center_ego": b.center_ego.tolist(), "size_lwh": b.size_lwh.tolist(), "yaw_rad": b.yaw_rad}
            for b in mirrored_scene.boxes
        ],
        "map": {"polylines_ego": [p.tolist() for p in mirrored_scene.map_record.polylines_ego]},
    }
    if mirrored_scene.lidar is not None:
        np.save(ROOT / "synthetic_lidar_mirror.npy", mirrored_scene.lidar.points_xyz.astype(np.float32))
    with open(ROOT / "synthetic_manifest_mirror.json", "w", encoding="utf-8") as f:
        json.dump(mirror_manifest, f, indent=2)

    print(f"Wrote synthetic scene bundle to {ROOT}")


if __name__ == "__main__":
    main()
