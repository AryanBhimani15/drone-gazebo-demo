"""Fixed-altitude simulator with a perspective RGB-D camera and independent collision checks."""

from dataclasses import asdict, dataclass
from functools import cached_property
import math
import random

import numpy as np


DT = 0.1
ALTITUDE = 1.25
RADIUS = 0.28
WIDTH, HEIGHT = 128, 80
HFOV = math.radians(88)
FAR = 20.0
CALIBRATION = range(100, 112)
TUNING = range(200, 208)
VALIDATION = range(300, 306)
TEST = range(1000, 1050)
SAFETY = range(2000, 2020)


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    d: float
    h: float
    color: tuple[int, int, int]
    z: float = 0.0


@dataclass(frozen=True)
class Course:
    seed: int
    scene: str
    goal: tuple[float, float]
    boxes: tuple[Box, ...]

    @cached_property
    def bounds(self):
        return np.asarray([(b.x - b.w / 2, b.y - b.d / 2, b.z,
                            b.x + b.w / 2, b.y + b.d / 2, b.z + b.h)
                           for b in self.boxes], dtype=np.float32)

    @cached_property
    def footprint(self):
        bounds = self.bounds
        return bounds[(bounds[:, 2] <= ALTITUDE) & (bounds[:, 5] >= ALTITUDE)]

    def dictionary(self):
        return asdict(self)


def course(seed, safety=False):
    rng = random.Random(seed)
    scene = 'Corridor' if seed % 2 == 0 else 'Grove'
    boxes = []
    if scene == 'Corridor':
        for side in (-1, 1):
            boxes.append(Box(8, side * 3.25, 24, .22, 3.0, (170, 181, 183)))
        boxes.append(Box(18.8, 0, .25, 6.5, 3.0, (154, 166, 172)))
        # Ceiling only affects the camera.
        boxes.append(Box(8, 0, 24, 6.5, .15, (146, 155, 164), 3.0))
    else:
        for side in (-1, 1):
            for x in (2, 6, 10, 14, 18, 22):
                boxes.append(Box(x + rng.uniform(-.7, .7), side * rng.uniform(3.4, 5.0),
                                 .67, .67, rng.uniform(2.7, 3.7), (104, 74, 46)))
    if safety:
        boxes.append(Box(rng.uniform(4.1, 6.1), rng.uniform(-.22, .22),
                         .70, rng.uniform(1.05, 1.55), 2.25, (186, 108, 55)))
    else:
        first_side = rng.choice((-1, 1))
        for index, x in enumerate((4.7, 8.8, 12.5)):
            side = first_side * (-1 if index == 1 else 1)
            center = rng.uniform(0.04, .28) if index == 0 else rng.uniform(.48, 1.24)
            depth = rng.uniform(.92, 1.65)
            color = rng.choice(((160, 104, 58), (66, 125, 132), (157, 143, 108)))
            boxes.append(Box(x + rng.uniform(-.5, .5), side * center,
                             rng.uniform(.52, .95), depth,
                             rng.uniform(1.68, 2.55), color))
    return Course(seed, scene, (16.0, 0.0), tuple(boxes))


class Camera:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width, self.height = width, height
        self.focal = width / (2 * math.tan(HFOV / 2))
        cx, cy = (width - 1) / 2, (height - 1) / 2
        u = (cx - np.arange(width, dtype=np.float32)) / self.focal
        v = (cy - np.arange(height, dtype=np.float32)) / self.focal
        self.u, self.v = np.meshgrid(u, v)
        self.ground = np.full_like(self.v, FAR)
        np.divide(-ALTITUDE, self.v, out=self.ground, where=self.v < -.00001)
        np.clip(self.ground, .02, FAR, out=self.ground)
        self.intrinsics = dict(fx=self.focal, fy=self.focal, cx=cx, cy=cy,
                               width=width, height=height, hfov_deg=math.degrees(HFOV))

    def capture(self, layout, x, y, yaw, fog=0.0):
        """Return RGB uint8 and axial depth in metres from the same pinhole rays."""
        cosine, sine = math.cos(yaw), math.sin(yaw)
        dx, dy = cosine - self.u * sine, sine + self.u * cosine
        directions = [np.where(np.abs(a) < 1e-6, 1e-6, a) for a in (dx, dy, self.v)]
        inv = [1 / a for a in directions]
        depth = self.ground.copy()
        index = np.full(depth.shape, -1, dtype=np.int16)
        normal = np.zeros(depth.shape, dtype=np.int8)
        origin = (x, y, ALTITUDE)
        for i, bounds in enumerate(layout.bounds):
            near = [np.minimum((bounds[k] - origin[k]) * inv[k],
                               (bounds[k + 3] - origin[k]) * inv[k]) for k in range(3)]
            far = [np.maximum((bounds[k] - origin[k]) * inv[k],
                              (bounds[k + 3] - origin[k]) * inv[k]) for k in range(3)]
            enter = np.maximum(np.maximum(near[0], near[1]), near[2])
            leave = np.minimum(np.minimum(far[0], far[1]), far[2])
            hit = (leave >= enter) & (enter > .015) & (enter < depth)
            depth[hit] = enter[hit]
            index[hit] = i
            normal[hit] = np.where(near[0][hit] == enter[hit], 0,
                                   np.where(near[1][hit] == enter[hit], 1, 2))
        wx, wy, wz = x + depth * dx, y + depth * dy, ALTITUDE + depth * self.v
        indoor = layout.scene == 'Corridor'
        base = np.array((127, 125, 116) if indoor else (111, 128, 83))
        sky = np.array((159, 183, 200) if indoor else (145, 185, 217))
        colors = np.array([b.color for b in layout.boxes], dtype=np.float32)
        tiled = (np.floor(wx * 1.15) + np.floor(wy * 1.15)) % 2
        ground_texture = (0.89 + .14 * tiled if indoor else
                          .93 + .08 * np.sin(14 * wx) * np.cos(12 * wy) +
                          .10 * np.sin(3.7 * wx + 2.4 * wy))
        floor = base * ground_texture[..., None]
        rgb = np.where((depth < FAR)[..., None], floor, sky)
        surface = np.where(normal == 0, wy, wx)
        texture = .90 + .11 * np.cos(9 * surface) * np.cos(10 * wz)
        shade = np.choose(normal, [.78, .94, 1.02])
        hit_rgb = colors[np.maximum(index, 0)] * (texture * shade)[..., None]
        rgb = np.where((index >= 0)[..., None], hit_rgb, rgb)
        haze = 1 - np.exp(-depth * (.008 + fog))
        rgb = rgb * (1 - haze[..., None]) + np.array((194, 202, 208)) * haze[..., None]
        return np.clip(rgb, 0, 255).astype(np.uint8), depth


def heading_error(x, y, yaw, goal):
    angle = math.atan2(goal[1] - y, goal[0] - x) - yaw
    return math.atan2(math.sin(angle), math.cos(angle))


def clearance(layout, x, y):
    b = layout.footprint
    dx = np.maximum(np.maximum(b[:, 0] - x, x - b[:, 3]), 0)
    dy = np.maximum(np.maximum(b[:, 1] - y, y - b[:, 4]), 0)
    return float(np.hypot(dx, dy).min() - RADIUS)


def swept_clearance(layout, x, y, x_new, y_new):
    # Distance from a point to a box is 1-Lipschitz. The half-spacing bound
    # conservatively covers the motion between adjacent samples.
    spacing = math.hypot(x_new - x, y_new - y) / 4
    return min(clearance(layout, x + i * (x_new - x) / 4, y + i * (y_new - y) / 4)
               for i in range(5)) - spacing / 2
