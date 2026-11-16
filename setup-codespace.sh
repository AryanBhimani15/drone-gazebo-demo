#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile

with ZipFile('drone-gazebo-browser.zip') as source:
    for entry in source.namelist():
        relative = Path(entry).relative_to('drone-gazebo-browser')
        if relative.parts[0] == '.devcontainer':
            continue
        relative.parent.mkdir(parents=True, exist_ok=True)
        relative.write_bytes(source.read(entry))
PY

apt-get update
apt-get install -y --no-install-recommends \
  ros-humble-ros-gz ros-humble-message-filters python3-pip \
  xvfb xauth x11vnc openbox novnc websockify libgl1-mesa-dri git ca-certificates
python3 -m pip install --no-cache-dir -r requirements.txt
printf '\nGazebo dependencies installed. Start with: bash ros/browser.sh 300 oracle\n'
