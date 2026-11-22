"""Import measured course results and Optuna trials into a local MLflow SQLite store."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from depth import WEIGHTS_SHA256, digest
from experiment import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('tracking'))
    args = parser.parse_args()
    import mlflow

    args.output.mkdir(parents=True, exist_ok=True)
    database = args.output.resolve() / 'mlflow.db'
    mlflow.set_tracking_uri(f'sqlite:///{database.as_posix()}')
    experiment = mlflow.get_experiment_by_name('Monocular obstacle avoidance')
    exp_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        'Monocular obstacle avoidance', artifact_location=(args.output.resolve() / 'artifacts').as_uri())
    run_ids = {}
    root = Path(__file__).parent
    source_revision = hashlib.sha256(''.join(digest(root / name) for name in
        ('control.py', 'sim.py', 'depth.py', 'experiment.py')).encode()).hexdigest()
    summary = json.loads((args.results / 'summary.json').read_text())
    configs = json.loads((args.results / 'config.json').read_text())
    with mlflow.start_run(experiment_id=exp_id, run_name='Frozen raycast ablation') as run:
        run_ids['ablation'] = run.info.run_id
        mlflow.log_params({**configs['settings'], 'threads': configs['onnx_threads'],
                           'tested_seeds': '1000-1049', 'safety_seeds': '2000-2019',
                           'model_sha256': WEIGHTS_SHA256, 'source_sha256': source_revision})
        for arm, values in summary.items():
            mlflow.log_metrics({f'{arm}_{key}': values[key] for key in ('goals', 'runs', 'collisions', 'timeouts')})
            mlflow.log_metric(f'{arm}_camera_command_p95_ms', values['latency']['loop_ms']['p95'])
            if 'safety' in values:
                mlflow.log_metric(f'{arm}_safety_collisions', values['safety']['collided'])
        for name in ('calibration.json', 'config.json', 'courses.csv', 'summary.json', 'depth_summary.json',
                     'rgbd_data_card.json', 'nyu_sources.json', 'kitti_sources.json',
                     'nyu_summary.json', 'kitti_summary.json', 'profile_onnx.json',
                     'profile_torch.json', 'profile_tensorrt.json', 'failure_review.csv', 'drift_summary.json'):
            path = args.results / name
            if path.exists():
                mlflow.log_artifact(str(path))
    with mlflow.start_run(experiment_id=exp_id, run_name='Optuna trial diagnostics') as run:
        run_ids['optuna'] = run.info.run_id
        path = args.results / 'optuna_trials.csv'
        with path.open(encoding='utf-8', newline='') as source:
            for row in csv.DictReader(source):
                mlflow.log_metric('trial_objective', float(row['objective']), step=int(row['trial']))
        mlflow.log_artifact(str(path))
        for name in ('study.json', 'settings.json', 'tuned_validation.csv'):
            mlflow.log_artifact(str(args.results / name))
    dump(args.results / 'mlflow_run_ids.json', dict(experiment_id=exp_id, runs=run_ids,
                                                   source_sha256=source_revision))
    print(run_ids)


if __name__ == '__main__':
    main()
