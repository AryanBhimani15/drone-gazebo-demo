"""MiDaS v2.1 Small ONNX backend and depth benchmarking transforms."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import urllib.request

import numpy as np

from sim import CALIBRATION, FAR, Camera, course


WEIGHTS_URL = 'https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx'
WEIGHTS_SHA256 = '2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58'


def digest(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def download_weights(path):
    path = Path(path)
    if path.is_file() and digest(path) == WEIGHTS_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.download', delete=False) as scratch:
        temporary = Path(scratch.name)
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, temporary)
        if digest(temporary) != WEIGHTS_SHA256:
            raise ValueError('The downloaded MiDaS weight SHA-256 does not match.')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class OnnxDepth:
    def __init__(self, path, threads=2, tensorrt=False):
        import cv2
        import onnxruntime as ort

        cv2.setNumThreads(1)
        self.cv = cv2
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
        if tensorrt:
            providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
            if 'TensorrtExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('This ONNX Runtime build has no TensorRT provider.')
            # Including a CPU fallback would invalidate a pure TensorRT timing arm.
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if tensorrt and self.session.get_providers()[0] != 'TensorrtExecutionProvider':
            raise RuntimeError('TensorRT initialization fell back to another provider.')
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, rgb):
        crop = self.cv.resize(rgb, (256, 256), interpolation=self.cv.INTER_CUBIC)
        # The published ONNX graph accepts RGB in [0, 1].
        batch = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = self.session.run(None, {self.input_name: batch})[0].reshape(256, 256)
        return self.cv.resize(output, (rgb.shape[1], rgb.shape[0]), interpolation=self.cv.INTER_CUBIC)


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float

    def apply(self, disparity):
        inv = self.slope * disparity + self.intercept
        return (1 / np.clip(inv, 1 / FAR, 4.0)).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def pairs_for_calibration(camera):
    for seed in CALIBRATION:
        layout = course(seed)
        for x in (.0, 2., 3., 3.7, 5.8, 7., 9.6, 10.8, 14.):
            for y, yaw in ((-.65, -.22), (0., 0.), (.65, .22)):
                yield camera.capture(layout, x, y, yaw)


def fit_calibration(model):
    features, labels = [], []
    for rgb, depth in pairs_for_calibration(Camera()):
        p = model(rgb)[30:50:2, ::3].ravel()
        d = depth[30:50:2, ::3].ravel()
        valid = np.isfinite(p) & (d >= .35) & (d < FAR)
        features.extend(p[valid])
        labels.extend(1 / d[valid])
    design = np.column_stack((features, np.ones(len(features))))
    # Reduce influence of the rare extreme residuals without consulting test seeds.
    kept = np.ones(len(labels), dtype=bool)
    labels = np.asarray(labels)
    for _ in range(4):
        slope, offset = np.linalg.lstsq(design[kept], labels[kept], rcond=None)[0]
        residuals = np.abs(design @ (slope, offset) - labels)
        kept = residuals <= np.quantile(residuals, .94)
    if slope <= 0:
        raise ValueError('Calibration produced a non-positive inverse-depth slope.')
    return Calibration(float(slope), float(offset))


def depth_metrics(disparity, truth, max_depth, calibration=None, inverse_affine=False):
    good = (truth > .1) & (truth < max_depth) & np.isfinite(truth) & np.isfinite(disparity)
    if not good.any():
        raise ValueError('Image has no valid benchmark pixels.')
    truth = truth[good].astype(np.float64)
    d = disparity[good].astype(np.float64)
    if inverse_affine:
        # This diagnostic fits both inverse-depth scale and offset using test
        # labels. It is never used to drive the simulator.
        d_centered = d - d.mean()
        inv = 1 / truth
        variance = np.dot(d_centered, d_centered)
        slope = np.dot(d_centered, inv - inv.mean()) / variance if variance else 0.
        prediction = 1 / np.maximum(slope * d_centered + inv.mean(), 1 / max_depth)
    elif calibration is None:
        prediction = 1 / np.maximum(d, .001)
        prediction *= np.median(truth) / np.median(prediction)
    else:
        prediction = calibration.apply(d)
    prediction = np.clip(prediction, .1, max_depth)
    ratio = np.maximum(truth / prediction, prediction / truth)
    diff = prediction - truth
    return dict(delta1=float(np.mean(ratio < 1.25)), delta2=float(np.mean(ratio < 1.25**2)),
                absrel=float(np.mean(np.abs(diff) / truth)), rmse=float(np.sqrt(np.mean(diff**2))),
                mae=float(np.mean(np.abs(diff))), mse=float(np.mean(diff**2)),
                valid_pixels=int(truth.size))
