"""Shared reactive controller. Input depth is axial distance in metres."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class Settings:
    speed: float = 2.0
    stop: float = .90
    gain: float = 3.2
    window: int = 2
    lookahead: float = 3.0
    goal_weight: float = .48
    half_width: float = .49


@dataclass(frozen=True)
class Command:
    speed: float
    yaw_rate: float
    frontal: float | None
    stopped: bool


@lru_cache(maxsize=16)
def ray_geometry(width, hfov):
    focal = width / (2 * math.tan(hfov / 2))
    return ((width - 1) / 2 - np.arange(width)) / focal


class Pilot:
    def __init__(self, settings=Settings()):
        self.settings = settings
        self.history = deque(maxlen=settings.window)
        self.previous = 0.0

    def act(self, depth, goal_error, hfov, mode='avoid', degraded=False):
        s = self.settings
        if mode == 'baseline':
            return Command(s.speed, float(np.clip(s.gain * goal_error, -1.45, 1.45)), None, False)
        if depth is None or depth.ndim != 2 or min(depth.shape) < 8:
            self.history.clear()
            return Command(0.0, 0.0, None, True)
        height, width = depth.shape
        band = depth[height * 46 // 100: max(height * 46 // 100 + 1, height * 54 // 100)]
        valid = np.isfinite(band) & (band > 0)
        if valid.mean() < .98:
            self.history.clear()
            return Command(0.0, 0.0, None, True)
        scan = np.min(np.where(valid, band, np.inf), axis=0)
        slopes = ray_geometry(width, hfov)
        lateral = slopes * scan
        front = np.abs(lateral) <= s.half_width
        frontal = float(scan[front].min()) if front.any() else None
        threshold = s.stop + (.45 if degraded else 0)
        stopped = frontal is not None and frontal <= threshold
        max_speed = s.speed * (.40 if degraded else 1.0)
        if mode == 'safety':
            return Command(0.0 if stopped else max_speed, 0.0, frontal, stopped)

        self.history.append(scan)
        smooth = np.mean(self.history, axis=0)
        # Retain sudden near returns immediately, while smoothing their recovery.
        smooth = np.minimum(smooth, scan)
        side = smooth * slopes
        candidates = np.linspace(-.66, .66, 17)[:, None]
        cosines, sines = np.cos(candidates), np.sin(candidates)
        forward = cosines * smooth + sines * side
        across = cosines * side - sines * smooth
        occupied = (forward > 0) & (np.abs(across) < s.half_width)
        free = np.min(np.where(occupied, forward, np.inf), axis=1)
        preferred = float(np.clip(goal_error, -.66, .66))
        angles = candidates[:, 0]
        score = (np.minimum(free, s.lookahead) / s.lookahead
                 - s.goal_weight * np.abs(angles - preferred)
                 - .085 * np.abs(angles - self.previous)
                 - .6 * (free <= threshold))
        selected = int(np.argmax(score))
        angle = float(angles[selected])
        if stopped and abs(angle) < .10:
            angle = .50 if self.previous >= 0 else -.50
        self.previous = angle
        yaw_rate = float(np.clip(s.gain * angle, -1.45, 1.45))
        factor = 1.0 if frontal is None else np.clip((frontal - threshold) / .85, 0, 1)
        speed = 0.0 if stopped else max_speed * float(factor) * max(.26, 1 - abs(angle) * 1.04)
        return Command(speed, yaw_rate, frontal, stopped)
