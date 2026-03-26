#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 generate_synthetic_scene.py
printf '\nDone. Outputs:\n'
ls -1 synthetic_debug.png synthetic_debug_mirror.png synthetic_manifest.json synthetic_manifest_mirror.json
