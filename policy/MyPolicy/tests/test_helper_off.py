"""CPU tests: the click_bell helper switched off behaves exactly like the package without it.

Run from a directory that holds policy/:
    python policy/MyPolicy/tests/test_helper_off.py -v
"""

import copy
import hashlib
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY_DIR = HERE.parent
sys.path.insert(0, str(POLICY_DIR.parent.parent))
sys.path.insert(0, str(HERE))

DP = importlib.import_module("policy.MyPolicy.deploy_policy")
SK = importlib.import_module("test_skeleton")
HELPER_MOD = "policy.MyPolicy.helpers.click_bell.bell_press"
TC = importlib.import_module("policy.MyPolicy.task_config")


def _run(ua, steps=60, chunk=8):
    m = DP.get_model(ua)
    env = SK.FakeEnv(max_steps=steps)
    DP.reset_model(m)
    obs = env.obs()
    while env.t < steps:
        obs, info, trunc, _ = DP.eval(env, m, obs)
        if bool(np.asarray(trunc).any()):
            break
    DP.close_model(m)
    return m, np.stack(env.received)


class TestHelperOff(unittest.TestCase):
    def args(self, helper):
        cfg = SK._yml()
        cfg["task_name"] = "click_bell"
        entry = {"backend": "hold", "hold_chunk": 8, "hold_offset": [0.01 * i for i in range(14)]}
        if helper is not None:
            entry["helper"] = helper
        cfg["task_policies"]["click_bell"] = entry
        return cfg

    def test_shipped_yml_helper_is_off_and_file_camera(self):
        h = TC.task_entry(SK._yml()["task_policies"], "click_bell")["helper"]
        self.assertIs(h["enabled"], False)            # delivered off (总-580); one switch to turn it on
        self.assertEqual(h["camera"], "file")
        calib = POLICY_DIR / "helpers" / "click_bell" / "calib" / f"platform_{h['platform']}.json"
        self.assertTrue(hashlib.sha256(calib.read_bytes()).hexdigest().startswith(str(h["calib_sha256"])))

    def test_every_shipped_helper_is_off_and_names_resolve(self):
        import ast
        cfg = SK._yml()
        for t in cfg["task_policies"]:
            h = TC.task_entry(cfg["task_policies"], t).get("helper")
            if h is None:
                continue
            self.assertIs(h["enabled"], False, t)
            if "module" not in h:
                continue
            folder = HERE.parent / "helpers" / t
            for mod, name in ((h["module"], h["class"]),
                              tuple(h["env_class"].rsplit(".", 1)) if "." in h["env_class"] else (h["module"], h["env_class"])):
                src = (folder / f"{mod}.py").read_text()
                names = {n.name for n in ast.parse(src).body if isinstance(n, ast.ClassDef)}
                self.assertIn(name, names, f"{t}: {mod}.{name}")

    def test_off_equals_no_helper_bitwise(self):
        shipped = TC.task_entry(SK._yml()["task_policies"], "click_bell")["helper"]
        off = dict(copy.deepcopy(shipped), enabled=False)
        m0, a0 = _run(self.args(None))
        m1, a1 = _run(self.args(off))
        self.assertIsNone(m1.helper)
        self.assertEqual(a0.shape, a1.shape)
        self.assertTrue(np.array_equal(a0, a1))       # bit-identical action sequence
        self.assertNotIn(HELPER_MOD, sys.modules)     # the helper code is not even imported

    def test_every_task_has_its_config_file(self):
        table = SK._yml()["task_policies"]
        self.assertEqual(sorted(table), sorted(SK.TASKS))
        for t in SK.TASKS:
            self.assertEqual(table[t], {"config": f"configs/{t}.yml"})
            self.assertTrue((POLICY_DIR / "configs" / f"{t}.yml").is_file())
            self.assertIn("backend", TC.task_entry(table, t))

    def test_helper_key_not_passed_to_backend(self):
        ua = self.args({"enabled": False})
        DP.get_model(ua)
        self.assertNotIn("helper", ua)                # routing key, not a backend argument


if __name__ == "__main__":
    unittest.main()
