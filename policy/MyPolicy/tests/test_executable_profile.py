"""executable clip profile = 45d9b9c official simulator URDF minus 1.5 deg.
Run: python -m unittest policy/MyPolicy/tests/test_executable_profile.py  (from the dir that holds policy/)."""
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
AP = importlib.import_module("policy.MyPolicy.action_post")

# official simulator URDF at 45d9b9c, raw rad
TABLE = {
    "J1": (-2.591820061220085, 2.591820061220085),
    "J2": (0.0, 3.113820061220085),
    "J3": (-2.940820061220085, 0.0),
    "J4": (-1.7188200612200852, 1.7188200612200852),
    "J5": (-1.2738200612200852, 1.2738200612200852),
    "J6": (-2.961820061220085, 2.961820061220085),
}
M = math.radians(1.5)


class TestExecutable(unittest.TestCase):
    def test_values_match_066_table(self):
        for (lo, hi), (j, (tlo, thi)) in zip(AP.PROFILES["executable"], TABLE.items()):
            self.assertAlmostEqual(lo, tlo, places=12, msg=j)
            self.assertAlmostEqual(hi, thi, places=12, msg=j)

    def test_degrees(self):
        deg = {j: (round(math.degrees(lo), 5), round(math.degrees(hi), 5)) for j, (lo, hi) in TABLE.items()}
        self.assertEqual(deg["J1"], (-148.50035, 148.50035))
        self.assertEqual(deg["J5"], (-72.98451, 72.98451))
        self.assertEqual(deg["J6"], (-169.69979, 169.69979))

    def test_margin_is_1p5_deg_off_the_45d9b9c_urdf(self):
        urdf = [(-2.618, 2.618), (0.0, 3.14), (-2.967, 0.0),
                (-1.745, 1.745), (-1.3, 1.3), (-2.988, 2.988)]
        for i, ((lo, hi), (ulo, uhi)) in enumerate(zip(AP.PROFILES["executable"], urdf), start=1):
            if i == 2:      # J2 lower = mechanical zero
                self.assertEqual(lo, 0.0)
                self.assertAlmostEqual(hi, uhi - M, places=12)
            elif i == 3:    # J3 upper = mechanical zero
                self.assertEqual(hi, 0.0)
                self.assertAlmostEqual(lo, ulo + M, places=12)
            else:
                self.assertAlmostEqual(lo, ulo + M, places=12)
                self.assertAlmostEqual(hi, uhi - M, places=12)

    def test_does_not_inherit_old_deploy_margin_j6(self):
        # old DEPLOY_CLIP had J6 +-160 deg; the finals default must be +-169.70
        ex, dm = AP.PROFILES["executable"], AP.PROFILES["deploy_margin"]
        self.assertGreater(ex[5][1], dm[5][1])
        self.assertAlmostEqual(math.degrees(ex[5][1]), 169.699789, places=5)

    def test_clips_both_arms(self):
        p = AP.ActionPost(dict(clip_profile="executable", log_dir=""))
        a = np.zeros(14)
        a[0] = math.radians(149.0)       # J1 outside
        a[1] = -0.01                     # J2 below mechanical zero
        a[2] = 0.01                      # J3 above mechanical zero
        a[3] = math.radians(99.0)        # J4 outside
        a[4] = math.radians(74.0)        # J5 outside (72.98)
        a[5] = math.radians(175.0)       # J6 outside (169.70)
        a[10] = -math.radians(99.0)      # right J4
        a[11] = -math.radians(74.0)      # right J5
        a[12] = -math.radians(175.0)     # right J6
        out = np.asarray(p.process(a))
        self.assertAlmostEqual(math.degrees(out[0]), 148.500351, delta=1e-5)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 0.0)
        self.assertAlmostEqual(math.degrees(out[3]), 98.481135, delta=1e-5)
        self.assertAlmostEqual(math.degrees(out[4]), 72.984513, delta=1e-5)
        self.assertAlmostEqual(math.degrees(out[5]), 169.699789, delta=1e-5)
        self.assertAlmostEqual(math.degrees(out[10]), -98.481135, delta=1e-5)
        self.assertAlmostEqual(math.degrees(out[11]), -72.984513, delta=1e-5)
        self.assertAlmostEqual(math.degrees(out[12]), -169.699789, delta=1e-5)

    def test_default_is_executable(self):
        self.assertEqual(AP.DEFAULTS["clip_profile"], "executable")
        self.assertEqual(AP.parse_cfg({"clip_profile": "executable"})["clip_profile"], "executable")


if __name__ == "__main__":
    unittest.main()
