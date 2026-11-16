"""CPU tests for the wrist-blank switch (WRIST_BLANK=1 / configs wrist_blank: true); off by default.

Run from a directory that holds policy/:
    python policy/MyPolicy/tests/test_wrist_blank.py -v
"""
import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE))

DP = importlib.import_module("policy.MyPolicy.deploy_policy")
WB = importlib.import_module("policy.MyPolicy.wrist_blank")
SK = importlib.import_module("test_skeleton")


class FakeEnvImg(SK.FakeEnv):
    def obs(self):
        o = super().obs()
        for c in o["sensor"]:
            o["sensor"][c]["color"] = np.full((1, 48, 64, 3), 200, np.uint8)
        return o


def _args(**entry):
    cfg = SK._yml()
    cfg["task_name"] = "click_bell"
    cfg["task_policies"]["click_bell"] = {"backend": "hold", "hold_chunk": 4, **entry}
    return cfg


def _run(ua, steps=12):
    m = DP.get_model(ua)
    seen = []
    orig = m.backend.eval

    def spy(env, inner, obs):
        seen.append(obs)
        out = orig(env, inner, obs)
        seen.append(out[0])
        return out
    m.backend = type("B", (), {"eval": staticmethod(spy), "reset_model": staticmethod(m.backend.reset_model)})
    env = FakeEnvImg(max_steps=steps)
    DP.reset_model(m)
    obs = env.obs()
    while env.t < steps:
        obs, info, trunc, _ = DP.eval(env, m, obs)
    return m, np.stack(env.received), seen


class TestWristBlank(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("WRIST_BLANK", None)

    def tearDown(self):
        os.environ.pop("WRIST_BLANK", None)
        if self._old is not None:
            os.environ["WRIST_BLANK"] = self._old

    def test_default_off_unchanged(self):
        m, acts, seen = _run(_args())
        self.assertIsNone(m.wrist_blank)
        for o in seen:
            self.assertTrue((o["sensor"]["cam_left_wrist"]["color"] == 200).all())
        m0, acts0, _ = _run(_args())
        self.assertTrue(np.array_equal(acts, acts0))

    def test_env_var_blanks_wrists_only(self):
        os.environ["WRIST_BLANK"] = "1"
        m, acts, seen = _run(_args())
        self.assertEqual(m.wrist_blank, "black")
        self.assertGreater(len(seen), 2)
        for o in seen:                                   # first obs and every step's obs
            for c in WB.WRIST_CAMS:
                self.assertTrue((o["sensor"][c]["color"] == 0).all())
                self.assertEqual(o["sensor"][c]["color"].shape, (1, 48, 64, 3))
                self.assertEqual(o["sensor"][c]["color"].dtype, np.uint8)
            self.assertTrue((o["sensor"]["cam_high"]["color"] == 200).all())

    def test_config_key_and_not_passed_to_backend(self):
        ua = _args(wrist_blank=True)
        m = DP.get_model(ua)
        self.assertEqual(m.wrist_blank, "black")
        self.assertNotIn("wrist_blank", ua)
        self.assertEqual(DP.get_model(_args(wrist_blank="gray")).wrist_blank, "gray")
        os.environ["WRIST_BLANK"] = "0"                  # the variable wins over the file
        self.assertIsNone(DP.get_model(_args(wrist_blank=True)).wrist_blank)

    def test_env_var_gray(self):
        os.environ["WRIST_BLANK"] = "gray"
        m, acts, seen = _run(_args())
        self.assertEqual(m.wrist_blank, "gray")
        for o in seen:
            for c in WB.WRIST_CAMS:
                self.assertTrue((o["sensor"][c]["color"] == 128).all())
                self.assertEqual(o["sensor"][c]["color"].dtype, np.uint8)
            self.assertTrue((o["sensor"]["cam_high"]["color"] == 200).all())
        os.environ["WRIST_BLANK"] = "purple"
        with self.assertRaises(ValueError):
            DP.get_model(_args())

    def test_blank_obs_does_not_touch_input(self):
        o = FakeEnvImg().obs()
        b = WB.blank_obs(o)
        self.assertTrue((o["sensor"]["cam_right_wrist"]["color"] == 200).all())
        self.assertTrue((b["sensor"]["cam_right_wrist"]["color"] == 0).all())


if __name__ == "__main__":
    unittest.main()
