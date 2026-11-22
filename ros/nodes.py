"""MiDaS perception, reactive control and optional RGB-D capture for ROS 2 Humble."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image

from control import Pilot, Settings
from depth import Calibration, OnnxDepth
from experiment import descriptors, dump
from sim import Camera, FAR, HFOV, clearance, course, heading_error, swept_clearance


SENSOR = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability)


def rgb_array(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Unsupported encoding {msg.encoding}')
    rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    rgb = rgb.reshape(msg.height, msg.width, 3)
    return rgb if msg.encoding == 'rgb8' else rgb[..., ::-1].copy()


def depth_array(msg):
    if msg.encoding != '32FC1' or msg.step % 4:
        raise ValueError(f'Unexpected depth encoding or stride {msg.encoding}/{msg.step}')
    dtype = '>f4' if msg.is_bigendian else '<f4'
    return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 4)[:, :msg.width]


class Perception(Node):
    def __init__(self, args):
        super().__init__('midas_perception')
        self.model = OnnxDepth(args.model, args.threads)
        self.calibration = Calibration.load(args.calibration)
        self.output = self.create_publisher(Image, '/drone/estimated_depth', SENSOR)
        self.create_subscription(Image, '/drone/camera/image', self.image, SENSOR)

    def image(self, msg):
        try:
            depth = self.calibration.apply(self.model(rgb_array(msg))).astype('<f4', copy=False)
        except (ValueError, RuntimeError) as exc:
            self.get_logger().error(str(exc))
            return
        result = Image(header=msg.header, height=msg.height, width=msg.width,
                       encoding='32FC1', is_bigendian=0, step=msg.width * 4,
                       data=depth.tobytes())
        self.output.publish(result)


class Control(Node):
    def __init__(self, args):
        super().__init__('reactive_avoidance', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.args = args
        self.done = False
        self.started = None
        self.last_odom = self.last_depth = -math.inf
        # Allow the same interval between depth frames when the cloud demo runs in slow motion.
        self.depth_watchdog = .30 / float(os.environ.get('GAZEBO_REAL_TIME_FACTOR', '1'))
        self.pose = None
        self.rgb_by_stamp = {}
        self.pilot = Pilot(Settings(**json.loads(args.settings.read_text())))
        self.layout = course(args.seed, safety=args.safety)
        self.minimum = clearance(self.layout, 0, 0)
        self.stops = self.frames = 0
        self.latency = []
        self.motion_commands = 0
        self.output = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.create_subscription(Odometry, '/drone/odom', self.odom, SENSOR)
        self.create_subscription(Image, '/drone/camera/image', self.rgb, SENSOR)
        if args.mode != 'baseline':
            topic = '/drone/estimated_depth' if args.mode == 'midas' else '/drone/camera/depth_image'
            self.create_subscription(Image, topic, self.depth, SENSOR)
        self.create_timer(.05, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.log = args.out.with_name(args.out.stem + '_frames.csv').open('w', newline='', encoding='utf-8')
        self.writer = None

    def watchdog(self):
        now = monotonic()
        if self.done or now - self.last_odom > .30 or now - self.last_depth > self.depth_watchdog:
            self.output.publish(Twist())

    def finish(self, status):
        if self.done:
            return
        self.done = True
        self.output.publish(Twist())
        latency = np.asarray(self.latency)
        dump(self.args.out, dict(seed=self.args.seed, scene=self.layout.scene, mode=self.args.mode,
                                 safety=self.args.safety, status=status, frames=self.frames,
                                 stop_frames=self.stops, min_clearance_m=self.minimum,
                                 motion_commands=self.motion_commands,
                                 latency_p95_sim_ms=float(np.quantile(latency, .95)) if len(latency) else None,
                                 stamp_basis='Gazebo simulated clock', elapsed_sim_s=self.elapsed()))
        self.log.close()

    def elapsed(self):
        return 0.0 if self.started is None else (self.get_clock().now().nanoseconds - self.started) / 1e9

    def odom(self, msg):
        if self.done:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.z**2 + q.y**2))
        old = self.pose
        self.pose = (p.x, p.y, yaw)
        self.last_odom = monotonic()
        if old:
            self.minimum = min(self.minimum, swept_clearance(self.layout, *old[:2], p.x, p.y))
        if self.minimum <= 0:
            self.finish('collision')
        elif math.hypot(p.x - 16, p.y) <= .55:
            self.finish('goal')
        elif self.elapsed() >= (8 if self.args.safety else 19):
            stopped = self.args.safety and self.stops and abs(msg.twist.twist.linear.x) < .1
            self.finish('stopped' if stopped else 'timeout')

    def rgb(self, msg):
        stamp = Time.from_msg(msg.header.stamp).nanoseconds
        try:
            self.rgb_by_stamp[stamp] = rgb_array(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        while len(self.rgb_by_stamp) > 8:
            self.rgb_by_stamp.pop(next(iter(self.rgb_by_stamp)))
        if self.args.mode == 'baseline':
            self.update(None, stamp)

    def depth(self, msg):
        try:
            depth = depth_array(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        if self.args.mode == 'oracle':
            depth = np.where(np.isposinf(depth), FAR, depth)
        self.update(depth, Time.from_msg(msg.header.stamp).nanoseconds)

    def update(self, depth, stamp):
        if self.done or self.pose is None or monotonic() - self.last_odom > .30:
            return
        now = self.get_clock().now().nanoseconds
        if not now or abs(now - stamp) > 250_000_000:
            return
        if self.started is None:
            self.started = now
        self.last_depth = monotonic()
        x, y, yaw = self.pose
        control = self.pilot.act(depth, heading_error(x, y, yaw, self.layout.goal), HFOV,
                                 mode='safety' if self.args.safety else 'baseline' if depth is None else 'avoid',
                                 degraded=self.args.degraded)
        message = Twist()
        message.linear.x = control.speed * math.cos(yaw)
        message.linear.y = control.speed * math.sin(yaw)
        message.angular.z = control.yaw_rate
        self.output.publish(message)
        latency = max(0, self.get_clock().now().nanoseconds - stamp) / 1e6
        self.latency.append(latency)
        self.stops += int(control.stopped)
        self.frames += 1
        self.motion_commands += int(control.speed > 0)
        rgb = self.rgb_by_stamp.get(stamp)
        valid_depth = depth if depth is not None and np.isfinite(depth).all() and (depth > 0).all() else None
        stats = descriptors(rgb, valid_depth) if rgb is not None else dict(
            brightness=None, contrast=None, entropy=None, near_fraction=None)
        row = dict(sim_time_s=stamp / 1e9, x=x, y=y, yaw=yaw, speed=control.speed,
                   yaw_rate=control.yaw_rate, stop=control.stopped, frontal_m=control.frontal,
                   geometric_min_clearance_m=self.minimum, camera_to_command_sim_ms=latency, **stats)
        if self.writer is None:
            self.writer = csv.DictWriter(self.log, fieldnames=row.keys())
            self.writer.writeheader()
        self.writer.writerow(row)


class Capture(Node):
    def __init__(self, args):
        from message_filters import Subscriber, TimeSynchronizer
        super().__init__('rgbd_capture')
        self.output = args.out
        self.seed = args.seed
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames, self.rgb, self.depth = [], [], []
        self.count = args.count
        inputs = [Subscriber(self, Image, topic, qos_profile=SENSOR) for topic in
                  ('/drone/camera/image', '/drone/camera/depth_image')]
        self.sync = TimeSynchronizer(inputs, 12)
        self.sync.registerCallback(self.pair)

    def pair(self, rgb, depth):
        if len(self.frames) >= self.count:
            return
        self.frames.append(Time.from_msg(rgb.header.stamp).nanoseconds)
        self.rgb.append(rgb_array(rgb).copy())
        self.depth.append(depth_array(depth).copy())
        if len(self.frames) == self.count:
            np.savez_compressed(self.output / 'gazebo_rgbd.npz', rgb=np.asarray(self.rgb),
                                depth_m=np.asarray(self.depth), stamp_ns=self.frames)
            dump(self.output / 'manifest.json', dict(seed=self.seed, intrinsics=Camera().intrinsics,
                                                     pairs=self.count, stamps_ns=self.frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('node', choices=('perception', 'control', 'capture'))
    parser.add_argument('--model', type=Path, default=Path('models/model-small.onnx'))
    parser.add_argument('--calibration', type=Path, default=Path('results/calibration.json'))
    parser.add_argument('--settings', type=Path, default=Path('results/settings.json'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mode', default='midas', choices=('midas', 'oracle', 'baseline'))
    parser.add_argument('--safety', action='store_true')
    parser.add_argument('--degraded', action='store_true')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--out', type=Path, default=Path('gazebo_result.json'))
    args = parser.parse_args()
    if args.safety and args.mode == 'baseline':
        parser.error('Stop-only runs require a source of depth.')
    rclpy.init(args=[])
    node = {'perception': Perception, 'control': Control, 'capture': Capture}[args.node](args)
    try:
        while rclpy.ok() and not getattr(node, 'done', False):
            rclpy.spin_once(node, timeout_sec=.20)
            if args.node == 'capture' and len(node.frames) >= args.count:
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
