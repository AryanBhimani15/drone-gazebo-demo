"""Tune policy parameters on development seeds with an Optuna TPE study."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import optuna

from control import Settings
from depth import Calibration
from experiment import dump, load_model, simulate, write_rows
from sim import TUNING, VALIDATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--courses', type=int, default=4, choices=range(2, 9))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    calibration = Calibration.load(args.calibration)
    base = Settings()
    tuning = TUNING[:args.courses]
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=17, n_startup_trials=5))
    for stop, gain, window, lookahead, goal_weight, half_width in (
        (.90, 3.2, 2, 3.0, .48, .49),
        (1.15, 3.7, 2, 4.0, .43, .65),
        (1.25, 4.1, 1, 4.7, .36, .68),
        (.83, 3.7, 3, 4.1, .45, .62),
    ):
        study.enqueue_trial(dict(stop=stop, gain=gain, window=window, lookahead=lookahead,
                                 goal_weight=goal_weight, half_width=half_width))

    def objective(trial):
        settings = replace(base, stop=trial.suggest_float('stop', .78, 1.42),
                           gain=trial.suggest_float('gain', 2.8, 4.6),
                           window=trial.suggest_int('window', 1, 4),
                           lookahead=trial.suggest_float('lookahead', 3.0, 5.0),
                           goal_weight=trial.suggest_float('goal_weight', .31, .54),
                           half_width=trial.suggest_float('half_width', .47, .74))
        results = [simulate(seed, 'midas', model, calibration, settings)[0] for seed in tuning]
        goal = sum(r['status'] == 'goal' for r in results)
        collision = sum(r['status'] == 'collision' for r in results)
        clearance = sum(min(.8, r['clearance_m']) for r in results) / len(results)
        trial.set_user_attr('goals', goal)
        trial.set_user_attr('collisions', collision)
        trial.set_user_attr('mean_clearance_m', clearance)
        return 3 * goal - 4 * collision + clearance

    start = perf_counter()
    study.optimize(objective, n_trials=args.trials)
    winner = replace(base, **study.best_params)
    trials = [dict(trial=t.number, objective=t.value, **t.params, **t.user_attrs)
              for t in study.trials if t.value is not None]
    dump(args.out / 'settings.json', asdict(winner))
    write_rows(args.out / 'optuna_trials.csv', trials)
    checks = [simulate(seed, 'midas', model, calibration, winner)[0] for seed in VALIDATION]
    write_rows(args.out / 'tuned_validation.csv', checks)
    dump(args.out / 'study.json', dict(tuning_seeds=list(tuning), validation_seeds=list(VALIDATION),
                                      sampler='TPESampler', sampler_seed=17, best_trial=study.best_trial.number,
                                      objective=study.best_value, elapsed_s=perf_counter() - start,
                                      validation_goals=sum(r['status'] == 'goal' for r in checks)))


if __name__ == '__main__':
    main()
