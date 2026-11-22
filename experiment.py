"""Run deterministic courses, log telemetry, and compute outcome summaries."""

import argparse
import csv
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from control import Pilot, Settings
from depth import Calibration, OnnxDepth, download_weights, fit_calibration
from sim import (Camera, DT, FAR, SAFETY, TEST, VALIDATION,
                 clearance, course, heading_error, swept_clearance)


MODES = ('midas', 'oracle', 'baseline')


def dump(path, value):
    with Path(path).open('w', encoding='utf-8') as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write('\n')


def write_rows(path, rows):
    if rows:
        opener = gzip.open if Path(path).suffix == '.gz' else open
        with opener(path, 'wt', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def descriptors(rgb, depth):
    brightness = float(np.mean(rgb) / 255)
    contrast = float(np.std(rgb.mean(axis=2)) / 255)
    if depth is None:
        return dict(brightness=brightness, contrast=contrast, entropy=None, near_fraction=None)
    bins = np.minimum((depth / FAR * 16).astype(int), 15)
    counts = np.bincount(bins.ravel(), minlength=16)
    probabilities = counts[counts > 0] / bins.size
    return dict(brightness=brightness, contrast=contrast,
                entropy=float(-np.sum(probabilities * np.log2(probabilities))),
                near_fraction=float(np.mean(depth < 2)))


def simulate(seed, mode, model=None, calibration=None, settings=Settings(),
             safety=False, fog=0, degraded=False, snapshot=None, seconds=None):
    if mode not in MODES or (mode == 'midas' and (model is None or calibration is None)):
        raise ValueError('Choose a recognised mode and supply model and calibration for MiDaS.')
    layout = course(seed, safety=safety)
    camera = Camera()
    pilot = Pilot(settings)
    x = y = yaw = 0.0
    minimum = clearance(layout, x, y)
    rows = []
    status = 'timeout'
    max_steps = int((seconds or (8 if safety else 19)) / DT)
    for i in range(max_steps):
        start = perf_counter()
        rgb, truth = camera.capture(layout, x, y, yaw, fog=fog)
        captured = perf_counter()
        depth = truth if mode == 'oracle' else None
        if mode == 'midas':
            depth = calibration.apply(model(rgb))
        inferred = perf_counter()
        control = pilot.act(depth, heading_error(x, y, yaw, layout.goal),
                            math.radians(camera.intrinsics['hfov_deg']),
                            mode='safety' if safety else 'baseline' if mode == 'baseline' else 'avoid',
                            degraded=degraded)
        decided = perf_counter()
        new_yaw = yaw + control.yaw_rate * DT
        mean_yaw = (yaw + new_yaw) / 2
        new_x = x + math.cos(mean_yaw) * control.speed * DT
        new_y = y + math.sin(mean_yaw) * control.speed * DT
        committed = perf_counter()
        gap = swept_clearance(layout, x, y, new_x, new_y)
        minimum = min(minimum, gap)
        row = dict(seed=seed, scene=layout.scene, mode=mode, safety=safety, fog=fog,
                   degraded=degraded, step=i, time_s=round(i * DT, 3),
                   x=x, y=y, yaw=yaw, next_x=new_x, next_y=new_y, speed=control.speed,
                   yaw_rate=control.yaw_rate, frontal_m=control.frontal, stop=control.stopped,
                   clearance_m=gap, **descriptors(rgb, depth),
                   capture_ms=(captured - start) * 1000, depth_ms=(inferred - captured) * 1000,
                   policy_ms=(decided - inferred) * 1000, command_ms=(committed - decided) * 1000,
                   loop_ms=(committed - start) * 1000)
        rows.append(row)
        x, y, yaw = new_x, new_y, new_yaw
        if gap <= 0:
            status = 'collision'
        elif math.hypot(x - layout.goal[0], y - layout.goal[1]) <= .55:
            status = 'goal'
        elif safety and control.stopped:
            # Test the stop for the remainder of the run, including unintended restart.
            status = 'stopped'
        elif safety:
            status = 'timeout'
        terminal = status in ('collision', 'goal') or i == max_steps - 1
        if snapshot and (i % 3 == 0 or terminal):
            snapshot(row, rgb, depth, terminal)
        if terminal:
            break
    stops = sum(row['stop'] for row in rows)
    outcome = dict(seed=seed, scene=layout.scene, mode=mode, safety=safety, fog=fog,
                   degraded=degraded, status=status, time_s=round(len(rows) * DT, 3),
                   clearance_m=minimum, stop_frames=stops, steps=len(rows),
                   final_x=x, final_y=y, goal_distance_m=math.hypot(x - layout.goal[0], y - layout.goal[1]))
    return outcome, rows


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0, center - radius), min(1, center + radius)]


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(('p05', 'p50', 'p95', 'p99'), map(float, np.quantile(values, [.05, .50, .95, .99]))))


def summarize(outcomes, rows):
    summary = {}
    for mode in sorted({r['mode'] for r in outcomes}):
        normal = [r for r in outcomes if r['mode'] == mode and not r['safety']]
        safety = [r for r in outcomes if r['mode'] == mode and r['safety']]
        frames = [r for r in rows if r['mode'] == mode and not r['safety']]
        if normal:
            goal = [r for r in normal if r['status'] == 'goal']
            summary[mode] = dict(runs=len(normal), goals=len(goal),
                                 collisions=sum(r['status'] == 'collision' for r in normal),
                                 timeouts=sum(r['status'] == 'timeout' for r in normal),
                                 wilson95=wilson(len(goal), len(normal)),
                                 time_to_goal_s=quantiles([r['time_s'] for r in goal]),
                                 clearance_m=quantiles([r['clearance_m'] for r in normal]),
                                 latency={key: quantiles([r[key] for r in frames]) for key in
                                          ('capture_ms', 'depth_ms', 'policy_ms', 'command_ms', 'loop_ms')},
                                 over_100_ms=sum(r['loop_ms'] > 100 for r in frames),
                                 frame_count=len(frames))
        if safety:
            summary.setdefault(mode, {})['safety'] = dict(
                runs=len(safety), collided=sum(r['status'] == 'collision' for r in safety),
                stopped=sum(r['status'] == 'stopped' for r in safety),
                timeout=sum(r['status'] == 'timeout' for r in safety),
                clearance_m=quantiles([r['clearance_m'] for r in safety]))
    return summary


def load_model(args):
    if not args.model.is_file():
        download_weights(args.model)
    return OnnxDepth(args.model, args.threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--split', choices=('test', 'validation', 'smoke'), default='test')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--skip-safety', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = calibration = None
    settings = Settings(**json.loads(args.settings.read_text())) if args.settings else Settings()
    if 'midas' in args.modes:
        model = load_model(args)
        calibration = Calibration.load(args.calibration) if args.calibration else fit_calibration(model)
        calibration.save(args.out / 'calibration.json')
    seeds = {'test': TEST, 'validation': VALIDATION, 'smoke': VALIDATION[:5]}[args.split]
    outcomes, rows = [], []
    for mode in args.modes:
        for seed, safety in [(seed, False) for seed in seeds] + (
            [(seed, True) for seed in SAFETY] if mode != 'baseline' and not args.skip_safety else []
        ):
            outcome, frames = simulate(seed, mode, model, calibration, settings, safety=safety)
            outcomes.append(outcome)
            rows.extend(frames)
            print(mode, seed, outcome['status'], flush=True)
    write_rows(args.out / 'courses.csv', outcomes)
    write_rows(args.out / 'frames.csv.gz', rows)
    dump(args.out / 'summary.json', summarize(outcomes, rows))
    dump(args.out / 'config.json', dict(version='1.0', split=args.split, seeds=list(seeds),
                                      settings=asdict(settings), camera=Camera().intrinsics,
                                      platform=platform.platform(), processor=platform.processor(),
                                      python=platform.python_version(), numpy=np.__version__,
                                      onnx_threads=args.threads))


if __name__ == '__main__':
    main()
