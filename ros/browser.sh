#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir/.."

if [[ "${GAZEBO_BROWSER_DISPLAY:-0}" != 1 ]]; then
  exec xvfb-run -a --server-args="-screen 0 1280x800x24 -nolisten tcp" \
    env GAZEBO_BROWSER_DISPLAY=1 bash "$script_dir/browser.sh" "$@"
fi

results="${GAZEBO_RESULTS:-gazebo_runs}"
mkdir -p "$results"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT

openbox --sm-disable > "$results/browser-desktop.log" 2>&1 &
pids+=("$!")
x11vnc -display "$DISPLAY" -auth "$XAUTHORITY" -localhost -forever -shared -nopw \
  -rfbport 5901 > "$results/browser-vnc.log" 2>&1 &
pids+=("$!")
websockify --web /usr/share/novnc 127.0.0.1:6080 127.0.0.1:5901 \
  > "$results/browser-websocket.log" 2>&1 &
pids+=("$!")

if ! python3 - <<'PY'
import time
from urllib.request import urlopen

for _ in range(30):
    try:
        with urlopen('http://127.0.0.1:6080/vnc.html', timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(.2)
else:
    raise SystemExit('The browser desktop did not start. Check the browser logs in gazebo_runs/.')
PY
then
  exit 1
fi

printf '\nOpen port 6080 from the Codespaces Ports panel. Keep its visibility Private.\n'
printf 'Open /vnc.html?autoconnect=true&resize=scale on that forwarded address.\n'
read -r -p 'Once the browser desktop connects, press Enter here to start Gazebo. '
GAZEBO_GUI=1 GAZEBO_TIMEOUT="${GAZEBO_TIMEOUT:-240}" bash ros/run.sh "$@"
