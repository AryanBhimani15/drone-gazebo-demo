# Model and data card

## Model

MiDaS v2.1 Small is an upstream pretrained monocular depth model. The [published ONNX weights and source](https://github.com/isl-org/MiDaS/releases/tag/v2_1) are provided under the upstream repository terms. The checkpoint used here has SHA-256 `2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58`.

The input is a single RGB image resized to 256 × 256, scaled to [0, 1] and packed channel-first. The exit tensor encodes relative inverse depth. The controller uses a fixed robust linear inverse-depth fit from development frames and clamps the resulting range to 0.25–20 m. Benchmark alignment uses labelled depth offline. Training or fine-tuning was outside the implementation.

Intended use is simulation-only research and teaching for civil inspection. The fixed-altitude environment covers stationary boxes, corridor surfaces and simplified grove geometry. The scope excludes operational aircraft. A low frontal estimate can stop motion, and a missing or overestimated obstacle can prevent that stop. The safety-only arm's 14 collisions in 20 attempts make independent measurement and safe braking prerequisites for any future operational investigation.

## Synthetic export

- Version `raycast-rgbd-v1`
- 5,000 paired frames from seeds 400–419, with one compressed shard per seed
- RGB uint8 and axial depth uint16 millimetres, with a 20,000 mm far-clip sentinel
- 128 × 80 camera, horizontal FOV 88°, altitude 1.25 m, principal point (63.5, 39.5)
- Labels produced by the same geometry rays that render RGB
- The manifest records seed, pose, shard index and content hash. `export_rgbd.py` reproduces the local export
- The frozen mission seeds 1000–1049, calibration seeds 100–111 and tuning seeds 200–203 are separate from the export

Simplified lighting and materials underrepresent glare, thin structures, fast motion, complex weather and real camera noise. The synthetic data contain no people. They support optical and controller development within this simulator.

## External sources

- NYU Depth V2 [release and citation terms](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), indoor aligned Kinect RGB-depth captures.
- KITTI [raw dataset and calibration](https://www.cvlibs.net/datasets/kitti/raw_data.php), with the [Eigen list distributed by Monodepth2](https://github.com/nianticlabs/monodepth2/blob/master/splits/eigen/test_files.txt). The measured subset has 25 frames from a single synchronised drive. This is a local comparison.

Project artifacts retain benchmark file identifiers, aggregate results and per-image metrics. Third-party RGB and depth data should be downloaded under the providers' applicable conditions. No demographic attributes are evaluated. Geometry, camera and scene properties are the supported analysis strata.
