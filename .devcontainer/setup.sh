#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ros-humble-ros-gz ros-humble-message-filters python3-pip \
  xvfb xauth x11vnc openbox novnc websockify libgl1-mesa-dri ca-certificates
python3 -m pip install --no-cache-dir -r requirements.txt
