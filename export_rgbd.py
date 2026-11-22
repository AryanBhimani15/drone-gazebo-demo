"""Generate and version 5,000 aligned synthetic RGB-depth frames."""

import argparse
import hashlib
from pathlib import Path
import random

import numpy as np

from depth import digest
from experiment import dump, write_rows
from sim import Camera, FAR, clearance, course


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    camera = Camera()
    manifest, shards, descriptors = [], [], []
    for seed in range(400, 420):
        layout = course(seed)
        rng = random.Random(seed + 785)
        rgb_shard, depth_shard = [], []
        while len(rgb_shard) < 250:
            x, y, yaw = rng.uniform(0, 15.5), rng.uniform(-2.5, 2.5), rng.uniform(-.52, .52)
            if clearance(layout, x, y) < .18:
                continue
            rgb, depth = camera.capture(layout, x, y, yaw)
            depth_mm = np.rint(depth * 1000).astype(np.uint16)
            index = len(rgb_shard)
            rgb_shard.append(rgb)
            depth_shard.append(depth_mm)
            frame_hash = hashlib.sha256(rgb.tobytes() + depth_mm.astype('<u2').tobytes()).hexdigest()
            filename = f'seed_{seed}.npz'
            manifest.append(dict(frame_id=f'{seed}:{index}', shard=filename, index=index,
                                 seed=seed, scene=layout.scene, x=x, y=y, yaw_rad=yaw, sha256=frame_hash))
            depth_m = depth_mm.astype(np.float32) / 1000
            valid = depth_m < FAR
            descriptors.append(dict(frame_id=f'{seed}:{index}', scene=layout.scene,
                                    brightness=float(rgb.mean() / 255),
                                    contrast=float(rgb.mean(axis=2).std() / 255),
                                    valid_fraction=float(valid.mean()),
                                    near_fraction=float((depth_m < 2).mean()),
                                    median_depth_m=float(np.median(depth_m[valid])) if valid.any() else None))
        shard = args.out / filename
        np.savez_compressed(shard, rgb=np.asarray(rgb_shard), depth_mm=np.asarray(depth_shard))
        shards.append(dict(file=filename, sha256=digest(shard), bytes=shard.stat().st_size))
        print(seed, shard.stat().st_size, flush=True)
    write_rows(args.out / 'rgbd_manifest.csv', manifest)
    write_rows(args.out / 'rgbd_descriptors.csv', descriptors)
    dump(args.out / 'rgbd_data_card.json', dict(
        version='raycast-rgbd-v1', pairs=len(manifest), scenes=['Corridor', 'Grove'],
        seeds=list(range(400, 420)), intrinsics=camera.intrinsics, far_clip_m=FAR,
        rgb='uint8 RGB', depth='uint16 axial millimetres, 20000 denotes the far clip',
        total_shard_bytes=sum(x['bytes'] for x in shards), shards=shards,
        unique_content_hashes=len({r['sha256'] for r in manifest})))


if __name__ == '__main__':
    main()
