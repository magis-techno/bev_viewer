# BEV Debug Viewer

A lightweight, runnable project for validating multi-camera + LiDAR + GT geometry before training.

## What it shows
- 7 camera views
- LiDAR projected into each camera
- 3D boxes projected into each camera
- BEV view with:
  - ego vehicle
  - map polylines
  - LiDAR points
  - camera ground-footprint polygons
  - 3D box footprints and headings
  - history trajectory
  - history pose rigs (6DoF poses rendered as oriented triangles)
  - LiDAR rig marker

## Files
- `bev_debug_viewer.py`: parser, mirror logic, visualization
- `generate_synthetic_scene.py`: creates a curved synthetic scene and exports original / mirrored results
- `synthetic_manifest.json`: generated source scene
- `synthetic_manifest_mirror.json`: generated mirrored scene
- `synthetic_debug.png`: original visualization
- `synthetic_debug_mirror.png`: mirrored visualization
- `MIRRORING_STRATEGY.md`: concise explanation of online virtual mirroring

## Run
```bash
pip install -r requirements.txt
bash run_demo.sh
```

Or directly:
```bash
python generate_synthetic_scene.py
```

## Notes
This project is intentionally independent from training code. The goal is to debug geometry and mirroring at the sample level first, then integrate the same interface into a dataloader.
