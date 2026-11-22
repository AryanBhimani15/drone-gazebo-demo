#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
seed="${1:-1000}"
mode="${2:-midas}"
shift "$(( $# >= 2 ? 2 : $# ))"
results="${GAZEBO_RESULTS:-gazebo_runs}"
mkdir -p "$results"
work="$(mktemp -d)"
export IGN_PARTITION="drone-${HOSTNAME}-$$"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  wait || true
  rm -f -- "$work/course.sdf"
  rmdir -- "$work"
}
trap cleanup EXIT
case "$mode" in midas|oracle|baseline) ;; *) exit 2;; esac
extra=()
if [[ "${1:-}" == "--safety" ]]; then extra+=(--safety); fi
name="${mode}_${seed}${extra:+_safety}"
python3 ros/world.py "$seed" "$work/course.sdf" "${extra[@]}"
if [[ "${GAZEBO_GUI:-0}" == "1" ]]; then
  : "${DISPLAY:?The Gazebo GUI requires an X display}"
  run_flags=(-r)
  if [[ "${GAZEBO_START_PAUSED:-0}" == "1" ]]; then
    run_flags=()
  fi
  setsid ign gazebo -s "${run_flags[@]}" -v 3 "$work/course.sdf" > "$results/$name-gazebo.log" 2>&1 &
  pids+=("$!")
  setsid ign gazebo -g -v 3 > "$results/$name-gui.log" 2>&1 &
  pids+=("$!")
else
  setsid xvfb-run -a ign gazebo -s -r -v 3 "$work/course.sdf" > "$results/$name-gazebo.log" 2>&1 &
  pids+=("$!")
fi
setsid ros2 run ros_gz_bridge parameter_bridge \
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock' \
    '/drone/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image' \
    '/drone/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image' \
    '/drone/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo' \
    '/drone/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry' \
    '/drone/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist' > "$results/$name-bridge.log" 2>&1 &
pids+=("$!")
if [[ "$mode" == "midas" ]]; then
  python3 -c 'from depth import download_weights; download_weights("models/model-small.onnx")'
  setsid python3 ros/nodes.py perception > "$results/$name-perception.log" 2>&1 &
  pids+=("$!")
fi
capture_pid=''
if [[ -n "${CAPTURE_COUNT:-}" ]]; then
  setsid timeout "${GAZEBO_TIMEOUT:-130}" python3 ros/nodes.py capture --seed "$seed" \
      --count "$CAPTURE_COUNT" --out "$results/${name}_rgbd" &
  capture_pid="$!"
  pids+=("$capture_pid")
fi
run_status=0
timeout "${GAZEBO_TIMEOUT:-130}" python3 ros/nodes.py control --seed "$seed" --mode "$mode" \
    --out "$results/$name.json" "$@" || run_status=$?
if [[ -n "$capture_pid" ]]; then wait "$capture_pid"; fi
if [[ "${GAZEBO_GUI:-0}" == "1" && -t 0 ]]; then
  if [[ "$run_status" == 0 ]]; then
    cat "$results/$name.json"
  else
    printf 'Control exited with code %s. See %s/%s-*.log.\n' "$run_status" "$results" "$name" >&2
  fi
  read -r -p 'Press Enter to close Gazebo. ' || true
fi
exit "$run_status"
