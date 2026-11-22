#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for mode in midas oracle baseline; do
  for seed in $(seq 1000 1049); do bash ros/run.sh "$seed" "$mode"; done
done
for seed in $(seq 2000 2019); do bash ros/run.sh "$seed" midas --safety; done
