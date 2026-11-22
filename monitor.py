"""Replay matched clean and foggy frames, measure drift and exercise degraded control."""

import argparse
from pathlib import Path
import math

import numpy as np

from control import Settings
from depth import Calibration
from experiment import descriptors, dump, load_model, simulate, write_rows
from sim import Camera, VALIDATION, course


FEATURES = ('brightness', 'contrast', 'entropy', 'near_fraction')


def psi(reference, current):
    bounds = np.unique(np.quantile(reference, np.linspace(0, 1, 9)))
    bins = np.r_[-np.inf, bounds[1:-1], np.inf]
    rf, _ = np.histogram(reference, bins=bins)
    cf, _ = np.histogram(current, bins=bins)
    r = (rf + .5) / (rf.sum() + .5 * len(rf))
    c = (cf + .5) / (cf.sum() + .5 * len(cf))
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--calibration', type=Path, default=Path('results/calibration.json'))
    parser.add_argument('--settings', type=Path, default=Path('results/settings.json'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--evidently', action='store_true')
    args = parser.parse_args()
    import json
    settings = Settings(**json.loads(args.settings.read_text()))
    model, cal, camera = load_model(args), Calibration.load(args.calibration), Camera()
    rows = []
    for seed in VALIDATION:
        layout = course(seed)
        for i in range(12):
            for fog in (0., .17):
                rgb, _ = camera.capture(layout, .8 + i, .7 * math.sin(i), .18 * math.cos(i), fog)
                rows.append(dict(seed=seed, pose=i, condition='fog' if fog else 'clean',
                                 **descriptors(rgb, cal.apply(model(rgb)))))
    ref = [row for row in rows if row['condition'] == 'clean']
    drift = [row for row in rows if row['condition'] == 'fog']
    scores = {key: psi([r[key] for r in ref], [r[key] for r in drift]) for key in FEATURES}
    alert = [key for key, value in scores.items() if value > .25 and key != 'contrast']
    degraded = len(alert) >= 2
    examples = []
    for fog, safe in ((0., False), (.17, False), (.17, degraded)):
        result, _ = simulate(301, 'midas', model, cal, settings, fog=fog, degraded=safe)
        examples.append(result)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / 'drift_descriptors.csv', rows)
    dump(args.out / 'drift_summary.json', dict(
        psi=scores, rule_features=['brightness', 'entropy', 'near_fraction'],
        alerts=alert, threshold=.25, degraded_triggered=degraded,
        degraded_max_speed_mps=settings.speed * .4, degraded_stop_m=settings.stop + .45,
        per_condition=len(ref), scenarios=examples))
    if args.evidently:
        import pandas as pd
        from evidently import Report
        from evidently.presets import DataDriftPreset
        dashboard = Report([DataDriftPreset(method='psi', threshold=.25)])
        snapshot = dashboard.run(pd.DataFrame(drift)[list(FEATURES)],
                                 pd.DataFrame(ref)[list(FEATURES)])
        snapshot.save_html(str(args.out / 'evidently_drift.html'))


if __name__ == '__main__':
    main()
