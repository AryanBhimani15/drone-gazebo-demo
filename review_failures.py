"""Extract measured near-miss and collision evidence from the frozen run logs."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from control import Pilot, Settings
from experiment import write_rows
from sim import Camera, HFOV, course


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    root = args.results
    with (root / 'courses.csv').open(encoding='utf-8', newline='') as file:
        selected = [r for r in csv.DictReader(file) if r['mode'] == 'midas' and
                    (r['status'] == 'collision' or
                     (r['safety'] == 'False' and float(r['clearance_m']) < .15))]
    frames = defaultdict(list)
    with gzip.open(root / 'frames.csv.gz', 'rt', encoding='utf-8', newline='') as file:
        for frame in csv.DictReader(file):
            if frame['mode'] == 'midas':
                frames[frame['seed'], frame['safety']].append(frame)
    settings = Settings(**json.loads((root / 'settings.json').read_text(encoding='utf-8')))
    camera = Camera()
    evidence = []
    for item in selected:
        seed = int(item['seed'])
        safety = item['safety'] == 'True'
        path = frames[item['seed'], item['safety']]
        event = min(path, key=lambda r: float(r['clearance_m']))
        pose = [float(event[key]) for key in ('x', 'y', 'yaw')]
        _, truth = camera.capture(course(seed, safety=safety), *pose)
        oracle = Pilot(settings).act(truth, 0, HFOV, mode='safety')
        observed = event['stop'] == 'True'
        indication = ('oracle_stop_only' if oracle.stopped and not observed else
                      'both_stop' if oracle.stopped and observed else
                      'estimated_stop_only' if observed else 'neither_front_stop')
        evidence.append(dict(seed=seed, scene=item['scene'], arm='stop_only' if safety else 'navigation',
                             status=item['status'], evidence_time_s=event['time_s'],
                             geometric_clearance_m=event['clearance_m'],
                             estimated_frontal_m=event['frontal_m'], oracle_frontal_m=oracle.frontal,
                             speed_mps=event['speed'], yaw_rate_rad_s=event['yaw_rate'],
                             estimate_stopped=observed, oracle_would_stop=oracle.stopped,
                             event_category=indication))
    write_rows(root / 'failure_review.csv', evidence)


if __name__ == '__main__':
    main()
