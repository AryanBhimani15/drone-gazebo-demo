# Autonomous Drone Obstacle Avoidance Using Monocular Depth Estimation

This repository contains a runnable simulation prototype for B.Tech AI & ML Project 6. A perspective camera renders seeded indoor corridors and outdoor groves. The official MiDaS v2.1 Small model predicts inverse depth from each RGB frame. A reactive controller navigates toward the goal using the estimated depth or runs the same course with oracle depth or no depth.

## Quick start

Open **../drone-simulation.html** in Chrome, Edge or Firefox for recorded runs. The file works offline. Choose the same seed with MiDaS, ground truth or no depth to compare the trajectories. The stop-only examples include a successful stop and a collision. Use the slider to inspect RGB, the depth input and geometric clearance.

Use Python 3.11 or 3.12 for the local simulator. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python replay.py --serve 8765
```

Open **http://127.0.0.1:8765**. Choose a seed and controller, then select **Run simulation**. The first MiDaS launch fetches the 66.8 MB public release and verifies SHA-256. Subsequent runs reuse it. The live viewer uses the same Python control loop that produced the measured evaluation. Each run completes before playback starts. Use Ctrl+C in the terminal to close the local server.

On Linux or macOS, use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Reproduce the portable results

```sh
python -m unittest discover -s tests -v
python experiment.py --split smoke --modes oracle --skip-safety --out smoke
python experiment.py --calibration results/calibration.json --settings results/settings.json --threads 4 --out reproduced
python benchmark.py --calibration results/calibration.json --threads 4 --out benchmark-local
python export_rgbd.py data/raycast-rgbd-v1
```

The seeded control run produces 150 mission outcomes and 40 dedicated stop-only outcomes. Fitted calibration and policy parameters are shipped as separate JSON files. To refit the calibration, omit `--calibration`. To inspect tuning, install `requirements-analysis.txt` and run:

```sh
python tune.py --calibration results/calibration.json --out tuning --trials 6 --courses 4 --threads 4
python monitor.py --evidently --out monitoring
python track.py --output tracking
```

Re-running the profiler or flight evaluation can produce different duration metrics. The dynamics use a fixed 0.1 s simulation step, while timing uses wall-clock measurements. Scene and outcome reproducibility depend on maintaining compatible library versions.

The archive includes the generated 5,000 pairs in `data/raycast-rgbd-v1/`. Each `.npz` shard contains arrays named `rgb` and `depth_mm`. Read the camera model and checksum record in the same directory. The MLflow database is saved as an evidence snapshot in `results/mlflow/`. To recreate locally resolvable artifact links, run `python track.py --output tracking` after extracting the archive.

### Measured local evaluation

| Arm | Collision-free completion | Collisions |
|---|---:|---:|
| MiDaS-small | 46/50 (92.0%, 95% Wilson interval 81.2%–96.8%) | 4/50 |
| Ground-truth depth | 50/50 | 0/50 |
| Straight line | 0/50 | 50/50 |

MiDaS collided in **14/20 stop-only tests**. Oracle depth stopped in all 20. Estimated metric depth can be wrong near obstacles even when steering succeeds overall. This prototype is limited to simulation and has no flight certification.

ONNX Runtime CPU had a local camera-to-command p95 of about **47 ms** in the 50-course run, with one measurement above 100 ms. In the separate 100-frame serving profile, loop p95 was **45.1 ms** for ONNX Runtime CPU, **84.2 ms** for PyTorch FP32 CPU and **9.5 ms** for TensorRT FP16 on the local RTX 2060. Review the report and `results/` for exact metrics and provenance.

## Linux ROS 2 Humble and Gazebo Fortress

The project includes an SDF world generator, RGB perception and navigation nodes, a ROS–Gazebo bridge, a paired camera exporter, and a Docker Compose entry point. The recorded GitHub Codespaces demo used Gazebo Fortress and ROS 2 Humble on Linux. It uses a fixed-altitude quadrotor body driven by velocity commands. Collision clearance is computed independently from odometry and scene geometry. The fixed-altitude model omits motor and wind dynamics.

```sh
docker compose build
docker compose run --rm simulation bash ros/run.sh 300 oracle
docker compose run --rm simulation bash ros/run.sh 300 midas
docker compose run --rm simulation bash ros/run.sh 2000 midas --safety
```

Results and process logs are written to `gazebo_runs/`. Set `CAPTURE_COUNT=100` for paired capture, for example `docker compose run --rm -e CAPTURE_COUNT=100 simulation bash ros/run.sh 1000 oracle`. Camera pairs are joined by timestamp. The calibration was fitted on the portable renderer. Re-evaluate its metric error in Gazebo before interpreting safety results. The batch runner `ros/batch.sh` runs the test and safety seeds. The full Gazebo batch and the hosted five-course CI gate remain unverified.

On a two-core Codespace, use `GAZEBO_START_PAUSED=1 GAZEBO_REAL_TIME_FACTOR=.15 bash ros/browser.sh 301 midas`. Open the private forwarded port 6080 in the browser, return to the terminal and press Enter. Use the orange Play control in Gazebo when ready to start the flight. The submission includes the recorded run and its matching JSON and CSV.

## External benchmark inputs

- **NYU Depth V2.** Obtain the [official labelled MATLAB release](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and the [archived test split MAT](https://github.com/ankurhanda/nyuv2-meta-data/blob/master/splits.mat). Run `python benchmark.py --skip-sim --nyu data/nyu_depth_v2_labeled.mat data/splits.mat --out benchmark-nyu`. The loader verifies that the split contains 654 distinct IDs. Add `--inverse-affine-diagnostic` to reproduce the separate analysis that fits inverse-depth scale and shift using the test labels.
- **KITTI.** Download the [raw sync and calibration archives](https://www.cvlibs.net/datasets/kitti/raw_data.php) for `2011_09_26_drive_0002` and the date `2011_09_26`. Use the [Monodepth2 Eigen test list](https://github.com/nianticlabs/monodepth2/blob/master/splits/eigen/test_files.txt). Run `python benchmark.py --skip-sim --kitti data/drive.zip data/calib.zip data/eigen_test_files.txt --out benchmark-kitti`. Evaluation uses the 25 available frame IDs from the one drive.
- Depth metrics use a valid-pixel mask and per-image median alignment with ground truth for offline comparison. In-loop inference uses the fixed calibration from independent synthetic development seeds. The `results/` folder identifies the evaluated inputs and frame IDs. The package omits third-party dataset assets. Use the providers' terms when obtaining local copies.

## Serving profiles

```sh
python profile_backends.py --backend onnx --threads 4 --out profile
```

For PyTorch CPU, install `torch==2.8.0` from the [PyTorch CPU index](https://download.pytorch.org/whl/cpu) and `geffnet==1.0.2`. Obtain the [MiDaS v2.1 source](https://github.com/isl-org/MiDaS/tree/v2_1) and its [small checkpoint](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt).

```sh
python profile_backends.py --backend torch --torch-source models/MiDaS-2_1 --torch-weights models/midas_v21_small_256.pt --threads 4 --out profile
```

The measured TensorRT profile used Windows, an NVIDIA GeForce RTX 2060, TensorRT 10.9.0.34 and CUDA 12.9. Install the matching NVIDIA runtime in a GPU environment and make its shared libraries visible to that process. Use `tensorrt-cu12==10.9.0.34` and `cuda-bindings==12.9.5`, then run:

```sh
python profile_backends.py --backend tensorrt --trt-native --out profile
```

The native engine parses the complete ONNX graph and enables FP16. The timed sample excludes the engine build. Another available path is `--backend tensorrt` with a compatible [ONNX Runtime TensorRT provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html). That path rejects CPU fallback. The report records the GPU used and identifies a Colab T4 run as pending.

## What to inspect

| Evidence | File |
|---|---|
| Mission and safety results | `results/courses.csv`, `results/summary.json` |
| Frame-level commands, visual descriptors and timing | `results/frames.csv.gz` |
| Calibration and tuning | `results/calibration.json`, `results/settings.json`, `results/optuna_trials.csv` |
| Image-level and aggregate depth evaluation | `results/*_images.csv`, `results/*_summary.json` |
| Synthetic data manifest and camera intrinsics | `results/rgbd_manifest.csv`, `results/rgbd_data_card.json` |
| Fog injection and fallback | `results/drift_descriptors.csv`, `results/drift_summary.json` |
| Diagnostic drift report | `results/evidently_drift.html` |
| Near misses and collisions | `results/failure_review.csv` |
| Source attribution, intended use and hazards | `MODEL_AND_DATA_CARD.md` |

Report cover metadata and contribution records are institution-specific. The editable report source and cover metadata sit under `report/` at the repository root. To rebuild the PDF on Windows after editing `report/report-details.json`, install ReportLab and use:

```sh
python -m pip install reportlab
python report/report.py --results results --replays report/replays.json --out drone-project-report.pdf
```

Add the department's actual certificate and declaration, and use a repository selected by the group if a hosted URL is required.
