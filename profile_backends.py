"""Run a repeatable CPU or TensorRT profile on identical synthetic scene poses."""

import argparse
import importlib
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

from control import Pilot
from depth import Calibration, OnnxDepth, depth_metrics, download_weights
from experiment import dump, quantiles, write_rows
from sim import Camera, FAR, HFOV, course


class TorchDepth:
    def __init__(self, source, checkpoint, threads):
        import cv2
        import geffnet
        import torch

        cv2.setNumThreads(1)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.cv, self.torch = cv2, torch
        sys.path.insert(0, str(source.resolve()))
        small = importlib.import_module('midas.midas_net_custom')
        # geffnet 1.0.2 provides the same Lite3 implementation as the upstream
        # torch.hub dependency, with an explicit version and no hub-cache writes.
        with patch.object(torch.hub, 'load', side_effect=lambda repo, name, **kwargs:
                          geffnet.create_model(name, **{**kwargs, 'pretrained': False})):
            self.model = small.MidasNet_small(None, features=64, backbone='efficientnet_lite3',
                                               exportable=True, non_negative=True, blocks={'expand': True})
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model.eval()
        self.mean = np.array([.485, .456, .406], dtype=np.float32)
        self.std = np.array([.229, .224, .225], dtype=np.float32)

    def __call__(self, rgb):
        image = self.cv.resize(rgb, (256, 256), interpolation=self.cv.INTER_CUBIC)
        image = (image.astype(np.float32) / 255 - self.mean) / self.std
        batch = self.torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)[None]))
        with self.torch.inference_mode():
            inv = self.model(batch).cpu().numpy().reshape(256, 256)
        return self.cv.resize(inv, (rgb.shape[1], rgb.shape[0]), interpolation=self.cv.INTER_CUBIC)


class NativeTensorRtDepth:
    def __init__(self, path):
        import cv2
        import tensorrt as trt
        from cuda.bindings import runtime as cuda

        self.cv, self.cuda = cv2, cuda
        cv2.setNumThreads(1)
        self.logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(self.logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(path.resolve())):
            errors = '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f'TensorRT parser rejected the complete graph. {errors}')
        build = builder.create_builder_config()
        build.set_flag(trt.BuilderFlag.FP16)
        build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024**2)
        build.builder_optimization_level = 2
        start = perf_counter()
        serialized = builder.build_serialized_network(network, build)
        self.build_seconds = perf_counter() - start
        if serialized is None:
            raise RuntimeError('TensorRT could not build an engine for this ONNX graph.')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self.stream = self.check(cuda.cudaStreamCreate())[0]
        self.input_name, self.output_name = (self.engine.get_tensor_name(i) for i in range(2))
        self.in_buffer = self.check(cuda.cudaMalloc(3 * 256 * 256 * 4))[0]
        self.out = np.empty(tuple(self.engine.get_tensor_shape(self.output_name)), dtype=np.float32)
        self.out_buffer = self.check(cuda.cudaMalloc(self.out.nbytes))[0]
        self.context.set_tensor_address(self.input_name, int(self.in_buffer))
        self.context.set_tensor_address(self.output_name, int(self.out_buffer))
        self.runtime_version = trt.__version__

    @staticmethod
    def check(result):
        code, *values = result
        if int(code) != 0:
            raise RuntimeError(f'CUDA runtime error {code}')
        return values

    def __call__(self, rgb):
        cuda = self.cuda
        image = self.cv.resize(rgb, (256, 256), interpolation=self.cv.INTER_CUBIC)
        batch = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        self.check(cuda.cudaMemcpy(self.in_buffer, batch.ctypes.data, batch.nbytes,
                                   cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError('TensorRT execution failed.')
        self.check(cuda.cudaStreamSynchronize(self.stream))
        self.check(cuda.cudaMemcpy(self.out.ctypes.data, self.out_buffer, self.out.nbytes,
                                   cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return self.cv.resize(self.out.reshape(256, 256), (rgb.shape[1], rgb.shape[0]),
                              interpolation=self.cv.INTER_CUBIC)

    def close(self):
        self.check(self.cuda.cudaFree(self.in_buffer))
        self.check(self.cuda.cudaFree(self.out_buffer))
        self.check(self.cuda.cudaStreamDestroy(self.stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('onnx', 'torch', 'tensorrt'), required=True)
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--torch-source', type=Path)
    parser.add_argument('--torch-weights', type=Path)
    parser.add_argument('--calibration', type=Path, default=Path('results/calibration.json'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--trt-native', action='store_true',
                        help='Use a native full-graph TensorRT engine instead of the ORT provider')
    args = parser.parse_args()
    if args.backend == 'torch':
        if not args.torch_source or not args.torch_weights:
            parser.error('PyTorch requires both the upstream v2_1 source and the v2.1 small checkpoint.')
        model = TorchDepth(args.torch_source, args.torch_weights, args.threads)
    else:
        if not args.model.is_file():
            download_weights(args.model)
        model = (NativeTensorRtDepth(args.model) if args.backend == 'tensorrt' and args.trt_native else
                 OnnxDepth(args.model, args.threads, tensorrt=args.backend == 'tensorrt'))
    cal = Calibration.load(args.calibration)
    camera, pilot = Camera(), Pilot()
    courses = [course(seed) for seed in range(300, 306)]
    summary, rows = {}, []
    for i in range(args.frames + 10):
        layout = courses[i % len(courses)]
        x, y = .5 + (i % 14), -.55 * math.sin(i)
        start = perf_counter()
        rgb, truth = camera.capture(layout, x, y, .18 * math.sin(i*.53))
        captured = perf_counter()
        disparity = model(rgb)
        depth = cal.apply(disparity)
        inferred = perf_counter()
        command = pilot.act(depth, 0, HFOV)
        decided = perf_counter()
        _ = (float(command.speed), float(command.yaw_rate))
        committed = perf_counter()
        if i >= 10:
            rows.append(dict(frame=i-10, capture_ms=(captured-start)*1000,
                             depth_ms=(inferred-captured)*1000, policy_ms=(decided-inferred)*1000,
                             command_ms=(committed-decided)*1000, loop_ms=(committed-start)*1000,
                             **depth_metrics(disparity, truth, FAR)))
    for key in ('capture_ms', 'depth_ms', 'policy_ms', 'command_ms', 'loop_ms'):
        summary[key] = quantiles([row[key] for row in rows])
    labels = {'onnx':'ONNX Runtime CPU', 'torch':'PyTorch FP32 CPU', 'tensorrt':'TensorRT FP16'}
    summary.update(backend=args.backend, label=labels[args.backend],
                   frames=len(rows), threads=args.threads, warmup=10,
                   platform=platform.platform(), processor=platform.processor(),
                   mean_inference_fps=1000 / np.mean([r['depth_ms'] for r in rows]),
                   mean_loop_fps=1000 / np.mean([r['loop_ms'] for r in rows]),
                   over_100_ms=sum(r['loop_ms'] > 100 for r in rows),
                   median_aligned_delta1=float(np.mean([r['delta1'] for r in rows])),
                   median_aligned_absrel=float(np.mean([r['absrel'] for r in rows])))
    if args.backend == 'tensorrt':
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, check=False)
        summary['gpu'] = gpu.stdout.strip().splitlines()[0] if gpu.stdout.strip() else 'NVIDIA GPU'
        summary['implementation'] = 'TensorRT native full graph' if args.trt_native else 'TensorRT EP in ONNX Runtime'
        if args.trt_native:
            summary.update(tensorrt_version=model.runtime_version, build_seconds=model.build_seconds)
            model.close()
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / f'profile_{args.backend}.csv', rows)
    dump(args.out / f'profile_{args.backend}.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
