#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python generate_synthetic_scene.py
printf '\nGenerated files:\n'
ls -1 synthetic_debug.png synthetic_debug_mirror.png synthetic_manifest.json synthetic_manifest_mirror.json
