"""Evaluate official NYU/KITTI inputs and deterministic held-out synthetic RGB-D."""

import argparse
from collections import defaultdict
import io
from pathlib import Path
import tarfile
from zipfile import ZipFile

import numpy as np
from PIL import Image

from depth import Calibration, depth_metrics, digest
from experiment import dump, load_model, write_rows
from sim import Camera, FAR, TEST, course


def nyu_indices(split):
    from scipy.io import loadmat

    test_ids = loadmat(split)['testNdxs'].ravel() - 1
    if len(test_ids) != 654 or len(np.unique(test_ids)) != 654:
        raise ValueError('Expected the official list of 654 NYU Depth V2 test frames.')
    return test_ids


def nyu_mask(truth):
    mask = np.zeros(truth.shape, dtype=bool)
    mask[45:471, 41:601] = True
    truth[~mask] = 0
    return truth


def nyu_pairs(mat, split, limit):
    import h5py

    test_ids = nyu_indices(split)
    with h5py.File(mat, 'r') as source:
        for index in test_ids[:limit]:
            rgb = np.transpose(source['images'][int(index)], (2, 1, 0))
            truth = source['depths'][int(index)].T.copy()
            yield str(int(index) + 1), rgb, nyu_mask(truth)


def nyu_mirror_pairs(split, archives):
    import h5py

    ids = set(map(int, nyu_indices(split) + 1))
    seen = set()
    for archive in archives:
        with tarfile.open(archive, 'r|') as source:
            for member in source:
                name = Path(member.name)
                if not member.isfile() or name.suffix != '.h5' or not name.stem.isdigit():
                    continue
                identifier = int(name.stem)
                if identifier not in ids or identifier in seen:
                    raise ValueError(f'Unexpected NYU mirror sample ID {identifier}')
                seen.add(identifier)
                with h5py.File(io.BytesIO(source.extractfile(member).read()), 'r') as frame:
                    yield str(identifier), frame['rgb'][:].transpose(1, 2, 0), nyu_mask(frame['depth'][:])


def calibration_values(data):
    values = {}
    for line in data.decode().splitlines():
        if ':' in line:
            key, numbers = line.split(':', 1)
            try:
                values[key] = np.array([float(x) for x in numbers.split()])
            except ValueError:
                continue
    return values


def kitti_pairs(raw_zip, calib_zip, split, limit):
    with ZipFile(raw_zip) as raw, ZipFile(calib_zip) as calib:
        entries = set(raw.namelist())
        used = 0
        for line in Path(split).read_text().splitlines():
            folder, index, side = line.split()
            date = folder.split('/')[0]
            camera = '02' if side == 'l' else '03'
            identifier = f'{int(index):010d}'
            image_path = f'{folder}/image_{camera}/data/{identifier}.png'
            lidar_path = f'{folder}/velodyne_points/data/{identifier}.bin'
            if image_path not in entries or lidar_path not in entries:
                continue
            rgb = np.asarray(Image.open(io.BytesIO(raw.read(image_path))).convert('RGB'))
            velo = calibration_values(calib.read(f'{date}/calib_velo_to_cam.txt'))
            cam = calibration_values(calib.read(f'{date}/calib_cam_to_cam.txt'))
            transform = np.eye(4)
            transform[:3, :3] = velo['R'].reshape(3, 3)
            transform[:3, 3] = velo['T']
            rectification = np.eye(4)
            rectification[:3, :3] = cam['R_rect_00'].reshape(3, 3)
            projection = cam[f'P_rect_{camera}'].reshape(3, 4)
            points = np.frombuffer(raw.read(lidar_path), dtype='<f4').reshape(-1, 4).copy()
            points = points[points[:, 0] > 0]
            points[:, 3] = 1
            projected = points @ (projection @ rectification @ transform).T
            projected = projected[projected[:, 2] > .1]
            xy = np.rint(projected[:, :2] / projected[:, 2:]).astype(int) - 1
            h, w = rgb.shape[:2]
            keep = ((xy[:, 0] >= int(.03594771 * w)) & (xy[:, 0] < int(.96405229 * w)) &
                    (xy[:, 1] >= int(.40810811 * h)) & (xy[:, 1] < int(.99189189 * h)))
            truth = np.full((h, w), np.inf, dtype=np.float32)
            np.minimum.at(truth, (xy[keep, 1], xy[keep, 0]), projected[keep, 2])
            yield f'{folder}/{identifier}/{side}', rgb, truth
            used += 1
            if limit is not None and used >= limit:
                break


def simulated_pairs():
    camera = Camera()
    for seed in TEST:
        layout = course(seed)
        for idx, (x, y, yaw) in enumerate(((0., 0., 0.), (2.8, -.7, .20), (6.5, .5, -.12),
                                         (10.5, -.7, .14), (14.0, .3, -.22))):
            rgb, truth = camera.capture(layout, x, y, yaw)
            yield f'{seed}/{idx}/{layout.scene}', rgb, truth


def benchmark(model, pairs, dataset, transform, cap):
    rows = []
    for index, rgb, truth in pairs:
        disparity = model(rgb)
        for label, calibration in transform:
            rows.append(dict(dataset=dataset, frame=index, alignment=label,
                             **depth_metrics(disparity, truth, cap, calibration,
                                             inverse_affine=label == 'per_image_gt_inverse_affine')))
    if not rows:
        raise ValueError(f'No evaluable {dataset} image pairs')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--nyu', type=Path, nargs=2, metavar=('LABELLED_MAT', 'OFFICIAL_SPLIT_MAT'))
    parser.add_argument('--nyu-mirror', type=Path, nargs='+', metavar='FILE',
                        help='Official split MAT followed by one or both Hugging Face validation TAR shards')
    parser.add_argument('--kitti', type=Path, nargs=3, metavar=('RAW_ZIP', 'CALIB_ZIP', 'EIGEN_LIST'))
    parser.add_argument('--nyu-limit', type=int)
    parser.add_argument('--kitti-limit', type=int)
    parser.add_argument('--skip-sim', action='store_true')
    parser.add_argument('--inverse-affine-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.nyu_mirror and (len(args.nyu_mirror) < 2 or args.nyu):
        parser.error('The mirror requires a split MAT and one or more TARs. Choose one NYU input format.')
    model = load_model(args)
    rows = []
    median = [('per_image_gt_median', None)]
    if args.inverse_affine_diagnostic:
        median.append(('per_image_gt_inverse_affine', None))
    if not args.skip_sim:
        transform = median + ([('fixed_calibration', Calibration.load(args.calibration))]
                              if args.calibration else [])
        rows.extend(benchmark(model, simulated_pairs(), 'raycast', transform, FAR))
    if args.nyu:
        rows.extend(benchmark(model, nyu_pairs(*args.nyu, args.nyu_limit), 'nyu', median, 10.))
    if args.nyu_mirror:
        rows.extend(benchmark(model, nyu_mirror_pairs(args.nyu_mirror[0], args.nyu_mirror[1:]),
                              'nyu', median, 10.))
    if args.kitti:
        rows.extend(benchmark(model, kitti_pairs(*args.kitti, args.kitti_limit), 'kitti', median, 80.))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['alignment'])].append(row)
    summary = []
    for (dataset, alignment), values in grouped.items():
        summary.append(dict(dataset=dataset, alignment=alignment, images=len(values),
                            valid_pixels=sum(row['valid_pixels'] for row in values),
                            **{key: float(np.mean([row[key] for row in values])) for key in
                               ('delta1', 'delta2', 'absrel', 'rmse', 'mae', 'mse')}))
        if dataset == 'nyu' and args.nyu_mirror:
            summary[-1]['mirror'] = 'https://huggingface.co/datasets/heuliuh/nyu_depth_v2'
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / 'depth_images.csv', rows)
    dump(args.out / 'depth_summary.json', summary)
    dump(args.out / 'dataset_sources.json', {
        'nyu': {x.name: digest(x) for x in (args.nyu or args.nyu_mirror or [])},
        'kitti': {x.name: digest(x) for x in args.kitti} if args.kitti else {},
    })
    print(summary)


if __name__ == '__main__':
    main()
