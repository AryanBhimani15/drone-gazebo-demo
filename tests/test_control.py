import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from control import Pilot
from depth import depth_metrics
from experiment import simulate, wilson
from ros.world import generate
from sim import Camera, SAFETY, TEST, course


class ProjectTests(unittest.TestCase):
    def test_repeatable_seed_and_shapes(self):
        c = course(TEST[0])
        self.assertEqual(c, course(TEST[0]))
        self.assertNotEqual(c.boxes, course(TEST[0] + 2).boxes)
        rgb, depth = Camera().capture(c, 0, 0, 0)
        self.assertEqual(rgb.shape, (80, 128, 3))
        self.assertEqual(depth.shape, (80, 128))
        self.assertTrue(np.all(np.isfinite(depth) & (depth > 0) & (depth <= 20)))

    def test_controller_halts_for_obstacle_or_invalid_input(self):
        pilot = Pilot()
        angle = math.radians(88)
        for depth in (None, np.full((80, 128), np.nan), np.full((80, 128), .5)):
            act = pilot.act(depth, 0, angle)
            self.assertTrue(act.stopped)
            self.assertEqual(act.speed, 0)
        self.assertGreater(pilot.act(np.full((80, 128), 4.0), 0, angle).speed, 0)

    def test_stop_only_simulation(self):
        run, _ = simulate(SAFETY[0], 'oracle', safety=True)
        self.assertEqual(run['status'], 'stopped')
        self.assertGreater(run['clearance_m'], 0)

    def test_distance_metric_recovers_known_scale(self):
        truth = np.array([[2.0, 4.0, 7.0], [2.0, 4.0, 7.0]])
        result = depth_metrics(10 / truth, truth, 20)
        self.assertEqual(result['delta1'], 1)
        self.assertAlmostEqual(result['absrel'], 0)
        self.assertLess(wilson(45, 50)[0], .9)

    def test_gazebo_world_matches_course(self):
        layout = course(TEST[0])
        world = ET.fromstring(generate(layout)).find('world')
        self.assertEqual(len(world.findall('model')), len(layout.boxes) + 2)
        self.assertEqual(world.find("model[@name='drone']/link/sensor/topic").text, '/drone/camera')


if __name__ == '__main__':
    unittest.main()
