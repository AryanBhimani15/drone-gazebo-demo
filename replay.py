"""Record actual simulation frames and optionally serve new local simulation runs."""

import argparse
import base64
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from control import Settings
from depth import Calibration
from experiment import MODES, load_model, simulate
from sim import course


def jpeg(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='JPEG', quality=77, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def palette(depth):
    # Use one fixed logarithmic colour scale for every run and depth source.
    stops = np.array([[251, 211, 101], [250, 126, 83], [186, 76, 134],
                      [93, 55, 130], [26, 43, 79]], dtype=np.float32)
    t = np.clip(np.log(np.maximum(depth, .25) / .25) / np.log(20 / .25), 0, 1) * 4
    low = np.minimum(t.astype(int), 3)
    fraction = t[..., None] - low[..., None]
    return np.rint(stops[low] * (1 - fraction) + stops[low + 1] * fraction).astype(np.uint8)


def record(seed, mode, model, calibration, settings, safety=False):
    frames = []

    def snapshot(row, rgb, depth, terminal):
        fields = ('time_s', 'x', 'y', 'yaw', 'speed', 'yaw_rate', 'frontal_m', 'clearance_m', 'stop')
        frame = {key: round(row[key], 4) if isinstance(row[key], float) else row[key] for key in fields}
        frame.update(rgb=jpeg(rgb), depth=jpeg(palette(depth)) if depth is not None else None)
        frames.append(frame)

    outcome, rows = simulate(seed, mode, model, calibration, settings, safety=safety, snapshot=snapshot)
    layout = course(seed, safety=safety)
    track = [[round(r['x'], 4), round(r['y'], 4)] for r in rows]
    track.append([outcome['final_x'], outcome['final_y']])
    return dict(outcome=outcome, course=layout.dictionary(), frames=frames, track=track,
                stop_threshold_m=settings.stop)


def choices(results):
    with (results / 'courses.csv').open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    corridor = next(r for r in rows if r['mode'] == 'midas' and r['scene'] == 'Corridor' and r['status'] == 'goal')
    grove = next(r for r in rows if r['mode'] == 'midas' and r['scene'] == 'Grove' and r['status'] == 'goal')
    fault = next((r for r in rows if r['mode'] == 'midas' and r['safety'] == 'False' and r['status'] != 'goal'), None)
    stop = next((r for r in rows if r['mode'] == 'midas' and r['safety'] == 'True' and r['status'] == 'stopped'), None)
    stop_failure = next((r for r in rows if r['mode'] == 'midas' and
                         r['safety'] == 'True' and r['status'] == 'collision'), None)
    selected = [(int(corridor['seed']), mode, False) for mode in MODES] + [(int(grove['seed']), 'midas', False)]
    if stop:
        selected.append((int(stop['seed']), 'midas', True))
    if stop_failure:
        selected.append((int(stop_failure['seed']), 'midas', True))
    if fault:
        selected.append((int(fault['seed']), 'midas', False))
    return selected


def encode_replay(runs, live=False):
    template = Path(__file__).with_name('viewer.html').read_text(encoding='utf-8')
    return template.replace('__RUN_DATA__', json.dumps(runs, separators=(',', ':'), allow_nan=False)
                            ).replace('__LIVE_MODE__', 'true' if live else 'false')


def serve(port, model, calibration, settings, presets):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, mime, status=200):
            body = data.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            uri = urlsplit(self.path)
            if uri.path == '/':
                self.reply(encode_replay(presets, live=True), 'text/html; charset=utf-8')
            elif uri.path == '/simulate':
                query = parse_qs(uri.query)
                try:
                    seed = int(query['seed'][0])
                    mode = query['mode'][0]
                    safety = query.get('safety', ['0'])[0] == '1'
                    if not 0 <= seed <= 1_000_000 or mode not in MODES or (safety and mode == 'baseline'):
                        raise ValueError('Invalid course seed or run mode.')
                    output = record(seed, mode, model, calibration, settings, safety)
                    self.reply(json.dumps(output, allow_nan=False), 'application/json')
                except (KeyError, ValueError) as exc:
                    self.reply(json.dumps({'error': str(exc)}), 'application/json', 400)
            else:
                self.reply('Page unavailable.', 'text/plain', 404)

    with HTTPServer(('127.0.0.1', port), Handler) as server:
        print(f'Simulation viewer running on http://127.0.0.1:{port}', flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--html', type=Path)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--serve', type=int, metavar='PORT')
    parser.add_argument('--run', nargs=2, action='append', metavar=('SEED', 'MODE'))
    args = parser.parse_args()
    model = load_model(args)
    calibration = Calibration.load(args.results / 'calibration.json')
    settings = Settings(**json.loads((args.results / 'settings.json').read_text()))
    specs = [(int(seed), mode, False) for seed, mode in args.run] if args.run else (
        [] if args.serve else choices(args.results))
    presets = [record(seed, mode, model, calibration, settings, safety) for seed, mode, safety in specs]
    if args.html:
        args.html.write_text(encode_replay(presets), encoding='utf-8')
    if args.json:
        args.json.write_text(json.dumps(presets, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    if args.serve:
        serve(args.serve, model, calibration, settings, presets)


if __name__ == '__main__':
    main()
