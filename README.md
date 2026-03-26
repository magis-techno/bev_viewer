# BEV Debug Viewer Project

一个可直接下载运行的轻量级项目，用来构造和可视化多模态自动驾驶样本。

支持内容：
- 7 路相机图像
- LiDAR 点云
- ego pose
- 相机内外参
- 3D bbox GT
- 地图 GT（`polylines_ego` 和 `polylines_world` 都支持）
- 原始场景和镜像场景可视化

## 安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 一键运行

Linux / macOS:

```bash
bash run_demo.sh
```

Windows PowerShell:

```powershell
python generate_synthetic_scene.py
```

运行后会生成：
- `synthetic_manifest.json`
- `synthetic_manifest_mirror.json`
- `synthetic_lidar.npy`
- `synthetic_lidar_mirror.npy`
- `synthetic_debug.png`
- `synthetic_debug_mirror.png`
- `mirrored_images/*.png`

## 单独可视化

原始场景：

```bash
python bev_debug_viewer.py --manifest synthetic_manifest.json --out synthetic_debug.png
```

镜像场景：

```bash
python bev_debug_viewer.py --manifest synthetic_manifest.json --out synthetic_debug_mirror.png --mirror --mirror-image-dir mirrored_images
```

或者直接读取已经导出的镜像 manifest：

```bash
python bev_debug_viewer.py --manifest synthetic_manifest_mirror.json --out synthetic_debug_mirror.png
```

## 坐标约定

- ego frame: `x` 向前，`y` 向左，`z` 向上
- camera frame: `x` 向右，`y` 向下，`z` 向前
- bbox: `center_ego / size_lwh / yaw_rad` 全部在 ego frame 下

## 这版已修正的点

- 支持 `polylines_ego` 和 `polylines_world`
- frustum 画的是相机视场与地面 `z=0` 的交线，而不是远平面投影
- 镜像图像由脚本生成，不再依赖图像内部的反向文字
