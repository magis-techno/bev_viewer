#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import importlib.util
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
VIEWER_PATH = ROOT / 'bev_debug_viewer.py'
spec = importlib.util.spec_from_file_location('bev_debug_viewer', VIEWER_PATH)
bev = importlib.util.module_from_spec(spec)
sys.modules['bev_debug_viewer'] = bev
spec.loader.exec_module(bev)

W, H = 1280, 720
FX = FY = 760.0
CX, CY = W / 2.0, H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def make_cam_rot_from_target(cam_pos, target, world_up=np.array([0.0, 0.0, 1.0])):
    z_fwd = unit(target - cam_pos)
    y_down = -world_up
    y_down = unit(y_down - np.dot(y_down, z_fwd) * z_fwd)
    x_right = unit(np.cross(y_down, z_fwd))
    return np.stack([x_right, y_down, z_fwd], axis=1)


def camera_record(name, pos, yaw_deg, pitch_down_deg=8.0):
    yaw = math.radians(yaw_deg)
    forward_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    target = np.asarray(pos, dtype=np.float64) + forward_xy * 24.0 + np.array([0.0, 0.0, -24.0 * math.tan(math.radians(pitch_down_deg))])
    R = make_cam_rot_from_target(np.asarray(pos, dtype=np.float64), target)
    return {
        'name': name,
        'image_path': f'./{name}.png',
        'intrinsic': K.tolist(),
        'extrinsic_to_ego': {'translation': list(map(float, pos)), 'rotation_matrix': R.tolist()},
    }


def generate_box_points(center, size_lwh, yaw, n_per_face=420, seed=0):
    rng = np.random.default_rng(seed)
    l, w, h = size_lwh
    local = []
    face_count = max(24, n_per_face // 6)
    for axis, val in [(0, l / 2), (0, -l / 2), (1, w / 2), (1, -w / 2), (2, h / 2), (2, -h / 2)]:
        pts = rng.uniform(-1.0, 1.0, size=(face_count, 3))
        pts[:, 0] *= l / 2
        pts[:, 1] *= w / 2
        pts[:, 2] *= h / 2
        pts[:, axis] = val
        local.append(pts)
    local = np.concatenate(local, axis=0)
    R = bev.yaw_to_rotmat(yaw)
    return (R @ local.T).T + np.asarray(center)[None, :]


def lane_centerline(xs):
    ys = 0.6 * np.sin((xs - 6.0) / 14.0) + 0.012 * (xs - 10.0)
    return ys


def lane_poly(offset, x_min=-35.0, x_max=42.0, samples=180):
    xs = np.linspace(x_min, x_max, samples)
    ys = lane_centerline(xs) + offset
    zs = np.zeros_like(xs)
    return np.stack([xs, ys, zs], axis=1)


def lane_lidar_samples(poly, jitter=0.05, seed=0):
    rng = np.random.default_rng(seed)
    pts = bev.dense_polyline(poly, step=0.35)
    pts = pts.copy()
    pts[:, 0] += rng.normal(scale=jitter, size=len(pts))
    pts[:, 1] += rng.normal(scale=jitter, size=len(pts))
    pts[:, 2] += rng.normal(scale=jitter * 0.2, size=len(pts))
    return pts


def history_poses_world():
    poses = []
    xs = np.array([-16.0, -12.0, -8.5, -5.0, -2.0])
    ys = lane_centerline(xs)
    yaws = np.arctan2(np.gradient(ys), np.gradient(xs))
    for x, y, yaw in zip(xs, ys, yaws):
        T = bev.make_hmat(bev.yaw_to_rotmat(float(yaw)), np.array([x, y, 0.0], dtype=np.float64))
        poses.append(T)
    return poses


def history_trajectory_ego():
    xs = np.linspace(-16.0, 0.0, 13)
    ys = lane_centerline(xs)
    zs = np.zeros_like(xs)
    return np.stack([xs, ys, zs], axis=1)


def chunk_visible_polyline(scene, cam, pts_ego):
    pts_cam = scene.ego_to_cam(cam, pts_ego)
    uv, valid = bev.project_points_to_image(pts_cam, cam.intrinsic, (W, H), min_depth=0.2)
    if not np.any(valid):
        return []
    uv_full = np.full((len(pts_ego), 2), np.nan)
    uv_full[np.where(valid)[0]] = uv
    chunks = []
    start = None
    for i, ok in enumerate(np.isfinite(uv_full[:, 0])):
        if ok and start is None:
            start = i
        if (not ok or i == len(uv_full) - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start >= 2:
                chunks.append([tuple(map(float, p)) for p in uv_full[start:end]])
            start = None
    return chunks


def render_camera_image(scene, cam, out_path):
    img = Image.new('RGB', (W, H), (196, 220, 243))
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.rectangle([0, int(H * 0.46), W, H], fill=(56, 62, 70, 255))
    draw.line([(0, int(H * 0.46)), (W, int(H * 0.46))], fill=(220, 228, 235, 180), width=2)

    lane_colors = [(255, 232, 120, 255), (255, 255, 255, 250), (255, 232, 120, 255)]
    for idx, poly in enumerate(bev.polylines_ego(scene)):
        dense = bev.dense_polyline(poly, step=0.25)
        for chunk in chunk_visible_polyline(scene, cam, dense):
            if len(chunk) >= 2:
                draw.line(chunk, fill=lane_colors[idx % len(lane_colors)], width=6)

    for box in scene.boxes:
        corners = bev.transform_box(bev.make_box_corners(box.size_lwh), box.center_ego, box.yaw_rad)
        pts_cam = scene.ego_to_cam(cam, corners)
        uv, mask = bev.project_points_to_image(pts_cam, cam.intrinsic, (W, H), min_depth=0.2)
        if mask.sum() < 8:
            continue
        uv_full = (cam.intrinsic @ pts_cam.T).T
        uv_full = uv_full[:, :2] / uv_full[:, 2:3]
        faces = [[0,1,2,3], [0,1,5,4], [1,2,6,5], [0,3,7,4], [3,2,6,7]]
        for face in faces:
            draw.polygon([tuple(uv_full[i]) for i in face], fill=(255, 120, 50, 40))
        for i0, i1 in bev.CAMERA_BOX_EDGES:
            draw.line([tuple(uv_full[i0]), tuple(uv_full[i1])], fill=(175, 48, 16, 255), width=3)

    banner = [18, 18, 240, 56]
    draw.rounded_rectangle(banner, radius=10, fill=(15, 23, 42, 160))
    draw.text((30, 29), cam.name, fill=(255, 255, 255, 255))
    img.save(out_path)


camera_defs = [
    camera_record('cam_front_left',  [1.9,  0.35, 1.45],  28),
    camera_record('cam_front_right', [1.9, -0.35, 1.45], -28),
    camera_record('cam_left_front',  [0.7,  0.95, 1.45],  96),
    camera_record('cam_left_rear',   [-0.9, 0.95, 1.45],  150),
    camera_record('cam_right_front', [0.7, -0.95, 1.45], -96),
    camera_record('cam_right_rear',  [-0.9,-0.95, 1.45], -150),
    camera_record('cam_rear',        [-1.9, 0.00, 1.45], 180),
]

boxes = [
    {'label': 'lead_car', 'center_ego': [18.0, 0.8, 0.85], 'size_lwh': [4.6, 1.9, 1.7], 'yaw_rad': 0.18},
    {'label': 'car_left', 'center_ego': [11.5, 4.4, 0.85], 'size_lwh': [4.4, 1.9, 1.7], 'yaw_rad': 0.26},
    {'label': 'car_right', 'center_ego': [14.5, -4.2, 0.85], 'size_lwh': [4.5, 1.9, 1.7], 'yaw_rad': -0.10},
]

map_polys = [lane_poly(-3.5), lane_poly(0.0), lane_poly(3.5)]

lidar_points_ego = np.concatenate([
    *(generate_box_points(b['center_ego'], b['size_lwh'], b['yaw_rad'], n_per_face=420, seed=11 + i) for i, b in enumerate(boxes)),
    *(lane_lidar_samples(poly, seed=40 + i) for i, poly in enumerate(map_polys)),
], axis=0)

lidar_t = np.array([0.0, 0.0, 1.8], dtype=np.float64)
lidar_points_sensor = lidar_points_ego - lidar_t[None, :]
np.save(ROOT / 'synthetic_lidar.npy', lidar_points_sensor.astype(np.float32))

history_world = history_poses_world()
history_traj = history_trajectory_ego()

manifest = {
    'tag': 'synthetic_rig_scene',
    'ego_pose': {'translation': [0.0, 0.0, 0.0], 'yaw_rad': 0.0},
    'cameras': camera_defs,
    'lidar': {
        'points_path': './synthetic_lidar.npy',
        'extrinsic_to_ego': {'translation': lidar_t.tolist(), 'rotation_matrix': np.eye(3).tolist()},
    },
    'bboxes': boxes,
    'map': {'polylines_ego': [p.tolist() for p in map_polys]},
    'history': {
        'poses_world': [{'translation': T[:3, 3].tolist(), 'rotation_matrix': T[:3, :3].tolist()} for T in history_world],
        'trajectory_ego': history_traj.tolist(),
    },
}

with open(ROOT / 'synthetic_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

scene = bev.parse_scene(str(ROOT / 'synthetic_manifest.json'))
for cam in scene.cameras:
    render_camera_image(scene, cam, ROOT / f'{cam.name}.png')

scene = bev.parse_scene(str(ROOT / 'synthetic_manifest.json'))
scene.visualize(str(ROOT / 'synthetic_debug.png'), bev_range=40.0, frustum_far=28.0)
scene_m = scene.mirrored()
with open(ROOT / 'synthetic_manifest_mirror.json', 'w', encoding='utf-8') as f:
    json.dump(scene_m.to_manifest_dict(), f, indent=2)
scene_m.visualize(str(ROOT / 'synthetic_debug_mirror.png'), bev_range=40.0, frustum_far=28.0)

print('Wrote synthetic scene to', ROOT)
