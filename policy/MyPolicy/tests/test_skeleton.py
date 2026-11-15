"""CPU unit tests for the finals policy skeleton (no simulator, no GPU, torch optional).

Run from final/delivery (or from a RoboSynChallenge checkout with policy/MyPolicy in it):
    python policy/MyPolicy/tests/test_skeleton.py -v
Optional cross-checks run when the files are there (else skipped):
    REAL_LIMITS_DIR  final/assets/urdf_real_limits (default: found relative to this file)
    OFFICIAL_REPO    an official RoboSynChallenge checkout (URDF limits, scripts/eval_policy.py)
    PRELIM_REPO      the preliminary delivery clone (git show b50f70a, byte identity of ofi/)
    OFI_CKPT_DIR     a local OfI checkpoint dir (manifest sha256 check; slow on large files)
"""

import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY_DIR = HERE.parent                      # .../policy/MyPolicy
ROOT = POLICY_DIR.parent.parent               # dir that holds policy/
sys.path.insert(0, str(ROOT))

AP = importlib.import_module("policy.MyPolicy.action_post")
DP = importlib.import_module("policy.MyPolicy.deploy_policy")

TASKS = ["click_bell", "drawer_open_place", "handle_basket", "item_assembly", "items_handover",
         "manipulate_pipette", "mixer_operating", "sample_loading", "table_rearrangement", "water_pouring"]


def _find_real_limits():
    env = os.environ.get("REAL_LIMITS_DIR")
    cands = [Path(env)] if env else []
    for up in Path(__file__).resolve().parents:
        cands.append(up / "assets" / "urdf_real_limits")
    for c in cands:
        if (c / "real_limits.py").is_file():
            return c
    return None


def _official_repo():
    for c in (os.environ.get("OFFICIAL_REPO"),):
        if c and (Path(c) / "scripts" / "eval_policy.py").is_file():
            return Path(c)
    return None


def _yml():
    import yaml

    return yaml.safe_load(open(POLICY_DIR / "deploy_policy.yml"))


class FakeSpace:
    shape = (14,)
    low = np.full(14, -10.0)
    high = np.full(14, 10.0)


class FakeUnwrapped:
    def __init__(self):
        self.single_action_space = FakeSpace()
        self.device = "cpu"
        self._robosyn_episode_seed = 1234


class FakeEnv:
    """Minimal stand-in for the official eval env: records every action it receives."""

    def __init__(self, qpos=None, max_steps=1000, success_at=None):
        self.unwrapped = FakeUnwrapped()
        self.received = []
        self.max_steps = max_steps
        self.success_at = success_at
        self.qpos = np.zeros(14) if qpos is None else np.asarray(qpos, dtype=np.float64)
        self.t = 0

    def obs(self):
        return {"robot": {"qpos": self.qpos.reshape(1, -1).copy()},
                "sensor": {c: {"color": np.zeros((1, 48, 64, 3), np.uint8)}
                           for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        self.received.append(a.copy())
        self.t += 1
        trunc = np.array([self.t >= self.max_steps])
        return self.obs(), 0.0, np.array([False]), trunc, {"elapsed_steps": np.array([self.t])}

    def get_wrapper_attr(self, name):
        if name == "is_task_success":
            return lambda: self.success_at is not None and self.t >= self.success_at
        raise AttributeError(name)


def deg(x):
    return math.radians(x)


class TestProfiles(unittest.TestCase):
    def test_default_is_executable(self):
        self.assertEqual(AP.DEFAULTS["clip_profile"], "executable")
        self.assertEqual(_yml()["action_post"]["clip_profile"], "executable")

    def test_profile_values(self):
        dm, rh = AP.PROFILES["deploy_margin"], AP.PROFILES["real_hard"]
        for (lo, hi), d in zip(dm[3:], (103.0, 74.5, 160.0)):
            self.assertAlmostEqual(hi, deg(d), places=9)
            self.assertAlmostEqual(lo, -deg(d), places=9)
        for (lo, hi), d in zip(rh[3:], (105.0, 75.0, 171.2)):
            self.assertAlmostEqual(hi, deg(d), delta=deg(0.01))
            self.assertAlmostEqual(lo, -hi, places=12)
        # deploy_margin is inside real_hard on every joint
        for (a, b), (c, d) in zip(dm, rh):
            self.assertGreaterEqual(a, c)
            self.assertLessEqual(b, d)

    def test_matches_real_limits_module(self):
        rl_dir = _find_real_limits()
        if rl_dir is None:
            self.skipTest("final/assets/urdf_real_limits not found")
        sys.path.insert(0, str(rl_dir))
        try:
            RL = importlib.import_module("real_limits")
        finally:
            sys.path.pop(0)
        for (a, b), (c, d) in zip(AP.PROFILES["deploy_margin"], RL.DEPLOY_CLIP):
            self.assertAlmostEqual(a, c, places=9)
            self.assertAlmostEqual(b, d, places=9)
        for (a, b), (c, d) in zip(AP.PROFILES["real_hard"], RL.hard_limits()):
            self.assertAlmostEqual(a, c, places=6)
            self.assertAlmostEqual(b, d, places=6)

    def test_official_sim_matches_official_urdf(self):
        repo = _official_repo()
        if repo is None:
            self.skipTest("official repo not found")
        root = ET.parse(str(repo / "assets/CobotMagicArm/CobotMagicWithGripperV100.urdf")).getroot()
        lim = {}
        for j in root.iter("joint"):
            L = j.find("limit")
            if L is not None and L.get("lower") is not None:
                lim[j.get("name")] = (float(L.get("lower")), float(L.get("upper")))
        for i, (lo, hi) in enumerate(AP.PROFILES["official_sim"], start=1):
            ulo, uhi = lim[f"joint{i}"]
            if i == 5 and (ulo, uhi) == (-1.22, 1.22):
                # local checkout is a485f91; the finals pin 45d9b9c changes J5 to +-1.3 rad
                self.assertAlmostEqual(lo, -1.3, places=6)
                self.assertAlmostEqual(hi, 1.3, places=6)
                continue
            self.assertAlmostEqual(lo, ulo, places=6)
            self.assertAlmostEqual(hi, uhi, places=6)


class TestActionPost(unittest.TestCase):
    def mk(self, **cfg):
        return AP.ActionPost(cfg, task="t")

    def test_in_range_unchanged(self):
        p = self.mk()
        a = np.array([0.1, 1.0, -1.0, 0.5, -0.5, 1.0, 0.7, -0.1, 0.9, -0.8, -0.4, 0.3, -2.0, 0.2], np.float32)
        out = p.process(a)
        np.testing.assert_array_equal(out, a)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(p.n_arm_clip_steps, 0)

    def test_clip_j4_j5_j6_both_arms(self):
        p = self.mk(clip_profile="deploy_margin")
        a = np.zeros(14)
        a[3], a[4], a[5] = deg(110), -deg(80), deg(170)
        a[10], a[11], a[12] = -deg(104), deg(75), -deg(161)
        out = p.process(a.reshape(1, 14))
        self.assertEqual(out.shape, (1, 14))
        o = out.reshape(-1)
        self.assertAlmostEqual(o[3], deg(103)); self.assertAlmostEqual(o[4], -deg(74.5)); self.assertAlmostEqual(o[5], deg(160))
        self.assertAlmostEqual(o[10], -deg(103)); self.assertAlmostEqual(o[11], deg(74.5)); self.assertAlmostEqual(o[12], -deg(160))
        self.assertEqual(p.n_arm_clip_steps, 1)
        self.assertEqual(sorted(k for k, v in p.per_joint.items() if v["n"]), ["L_J4", "L_J5", "L_J6", "R_J4", "R_J5", "R_J6"])
        self.assertAlmostEqual(p.per_joint["L_J6"]["max_excess"], deg(10), places=6)

    def test_real_hard_profile(self):
        p = self.mk(clip_profile="real_hard")
        a = np.zeros(14)
        a[4] = deg(74.9)    # inside real_hard, outside deploy_margin
        a[5] = deg(170)
        o = p.process(a)
        self.assertAlmostEqual(o[4], deg(74.9)); self.assertAlmostEqual(o[5], deg(170))
        self.assertEqual(p.n_arm_clip_steps, 0)

    def test_j2_j3_official(self):
        p = self.mk()
        a = np.zeros(14)
        a[1], a[2] = -0.05, 0.05   # real rest frames can sit just outside
        o = p.process(a)
        self.assertEqual(o[1], 0.0); self.assertEqual(o[2], 0.0)

    def test_gripper(self):
        p = self.mk()
        a = np.zeros(14); a[6], a[13] = 1.33, -0.2
        o = p.process(a)
        self.assertEqual(o[6], 1.0); self.assertEqual(o[13], 0.0)
        self.assertEqual(p.n_grip_clip_steps, 1); self.assertEqual(p.n_arm_clip_steps, 0)
        p2 = self.mk(gripper_src_range=[0.0, 0.1])  # e.g. a model that outputs metres of opening
        a = np.zeros(14); a[6], a[13] = 0.05, 0.1
        o = p2.process(a)
        self.assertAlmostEqual(o[6], 0.5); self.assertAlmostEqual(o[13], 1.0)
        p3 = self.mk(gripper_clip=False)
        a = np.zeros(14); a[6] = 1.2
        self.assertAlmostEqual(p3.process(a)[6], 1.2)

    def test_gripper_mode_binary_and_hysteresis(self):
        a = np.zeros(14)
        p = self.mk()   # linear default: half open stays half open
        a[6], a[13] = 0.5, 0.5
        o = p.process(a)
        self.assertAlmostEqual(o[6], 0.5); self.assertAlmostEqual(o[13], 0.5)
        self.assertEqual(p.n_grip_mode_steps, 0)

        b = self.mk(gripper_mode="binary")
        a[6], a[13] = 0.5, 0.9
        o = b.process(a)
        self.assertEqual(o[6], 0.0); self.assertEqual(o[13], 1.0)
        self.assertEqual(b.n_grip_mode_steps, 1)

        h = self.mk(gripper_mode="hysteresis")
        seq = [0.5, 0.1, 0.5, 0.9, 0.85, 0.1]
        out = [h.process(np.array([0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, x]))[6] for x in seq]
        # starts open; drops below 0.15 -> closed; nothing at 0.5 reopens it; above 0.81 -> open
        self.assertEqual(out, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        self.assertEqual(h.n_grip_mode_steps, 6)

    def test_step_limits(self):
        p = self.mk(max_step_rad=0.05, max_step_grip=0.1)
        a = np.zeros(14); a[4] = 0.30; a[6] = 1.0
        o = p.process(a)                      # first command: nothing to clamp against
        self.assertAlmostEqual(o[4], 0.30); self.assertAlmostEqual(o[6], 1.0)
        self.assertEqual(p.n_jump_clip_steps, 0)
        b = np.zeros(14); b[4] = 0.50; b[6] = 0.0
        o = p.process(b)
        self.assertAlmostEqual(o[4], 0.35, places=9)
        self.assertAlmostEqual(o[6], 0.9, places=9)
        self.assertEqual(p.n_jump_clip_steps, 1)
        self.assertEqual(self.mk().cfg["max_step_rad"], None)   # off by default

    def test_bad_gripper_cfg(self):
        with self.assertRaises(ValueError):
            AP.parse_cfg({"gripper_mode": "switch"})
        with self.assertRaises(ValueError):
            AP.parse_cfg({"gripper_hyst": {"lo": 0.8, "hi": 0.2}})
        with self.assertRaises(ValueError):
            AP.parse_cfg({"max_step_rad": 0})
        # binary/hysteresis + per-step gripper limit would turn 0/1 into fractions
        with self.assertRaises(ValueError):
            AP.parse_cfg({"gripper_mode": "binary", "max_step_grip": 0.1})
        with self.assertRaises(ValueError):
            AP.parse_cfg({"gripper_mode": "hysteresis", "max_step_grip": 0.1})

    def test_binary_threshold_is_hi(self):
        # binary uses gripper_hyst.hi; hi 0.5 gives the ordinary 0.5 binarization
        p = self.mk(gripper_mode="binary", gripper_hyst={"hi": 0.5})
        a = np.zeros(14); a[6], a[13] = 0.49, 0.5
        o = p.process(a)
        self.assertEqual(o[6], 0.0); self.assertEqual(o[13], 1.0)

    def test_limits_for_fallback_is_executable(self):
        self.assertEqual(AP.limits_for({}), AP.limits_for({"clip_profile": "executable"}))

    def test_nonfinite(self):
        p = self.mk()
        q = np.arange(14) * 0.01
        q[2], q[9] = -0.02, -0.09   # J3 range is [-2.967, 0]
        a = np.full(14, np.nan)
        o = p.process(a, qpos=q.reshape(1, -1))
        np.testing.assert_allclose(o, q)
        b = np.zeros(14); b[3] = 0.3
        p.process(b)
        c = b.copy(); c[3] = np.inf; c[0] = 0.2
        o = p.process(c)
        self.assertAlmostEqual(o[3], 0.3); self.assertAlmostEqual(o[0], 0.2)
        self.assertEqual(p.n_nonfinite, 2)
        with self.assertRaises(ValueError):
            self.mk().process(np.full(14, np.nan))   # nothing to hold

    def test_bad_dim_and_cfg(self):
        with self.assertRaises(ValueError):
            self.mk().process(np.zeros(16))
        with self.assertRaises(ValueError):
            AP.parse_cfg({"clip_profile": "nope"})
        with self.assertRaises(ValueError):
            AP.parse_cfg({"typo_key": 1})
        with self.assertRaises(ValueError):
            AP.parse_cfg({"custom_limits": [[0, 1]] * 5})

    def test_disabled_passthrough(self):
        p = AP.ActionPost(False)
        a = np.full(14, 9.0)
        self.assertIs(p.process(a), a)

    def test_custom_limits(self):
        p = self.mk(custom_limits=[[-0.1, 0.1]] * 6)
        o = p.process(np.full(14, 0.5))
        self.assertAlmostEqual(o[0], 0.1); self.assertAlmostEqual(o[6], 0.5)

    def test_log_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x" / "ap.jsonl"
            p = AP.ActionPost({}, task="click_bell", log_path=path)
            p.start_episode(seed=7)
            a = np.zeros(14); a[4] = 2.0
            p.process(a); p.process(np.zeros(14))
            s = p.end_episode()
            self.assertEqual(s["arm_clip_steps"], 1); self.assertEqual(s["steps"], 2)
            with open(path) as f:
                recs = [json.loads(l) for l in f]
            self.assertEqual([r["kind"] for r in recs], ["clip", "episode_summary"])
            self.assertEqual(recs[0]["t"], 0); self.assertIn("L_J5", recs[0]["arm"])
            self.assertEqual(recs[1]["seed"], 7)

    def test_torch_roundtrip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch here")
        p = self.mk()
        a = torch.zeros(1, 14, dtype=torch.float32); a[0, 4] = 3.0
        o = p.process(a)
        self.assertIsInstance(o, torch.Tensor); self.assertEqual(o.dtype, torch.float32); self.assertEqual(tuple(o.shape), (1, 14))
        self.assertAlmostEqual(float(o[0, 4]), 1.273820, places=5)   # executable J5 (+72.984513 deg), the default


class TestRouter(unittest.TestCase):
    def args(self, task="click_bell", **kw):
        cfg = _yml()
        cfg["task_name"] = task
        cfg["task_policies"][task] = {"backend": "hold", **kw}
        return cfg

    def test_hold_through_guard(self):
        with tempfile.TemporaryDirectory() as d:
            ua = self.args(hold_chunk=5)
            ua["eval_result_dir"] = d
            off = [0.0] * 14
            off[4] = 2.0       # push L_J5 far beyond the default executable bound (+72.984513 deg)
            ua["hold_offset"] = off
            m = DP.get_model(ua)
            env = FakeEnv()
            DP.reset_model(m)
            obs, info, trunc, times = DP.eval(env, m, env.obs())
            self.assertEqual(len(env.received), 5)
            self.assertEqual(len(times), 1)
            for a in env.received:
                self.assertAlmostEqual(a[4], 1.273820, places=6)
            self.assertEqual(m.post.seed, 1234)
            DP.close_model(m)
            logs = list(Path(d, "click_bell").glob("action_post_*.jsonl"))
            self.assertEqual(len(logs), 1)
            with open(logs[0]) as f:
                kinds = [json.loads(l)["kind"] for l in f]
            self.assertEqual(kinds.count("clip"), 5); self.assertEqual(kinds[-1], "episode_summary")

    def test_episode_loop_like_official(self):
        """Mimics scripts/eval_policy.py: eval() until success or 1000 steps; counters per episode."""
        ua = self.args(hold_chunk=24)
        m = DP.get_model(ua)
        for ep, succ in enumerate((None, 50)):
            env = FakeEnv(max_steps=1000, success_at=succ)
            DP.reset_model(m)
            obs, steps = env.obs(), 0
            while steps < 1000:
                obs, info, trunc, _ = DP.eval(env, m, obs)
                steps = int(info["elapsed_steps"].item())
                if bool(np.asarray(trunc).any()) or env.get_wrapper_attr("is_task_success")():
                    break
            self.assertEqual(steps, 1000 if succ is None else 50)
            self.assertEqual(m.post.t, steps)
            self.assertEqual(m.post.episode, ep)
        DP.close_model(m)
        self.assertEqual(m.post.totals["episodes"], 2)
        self.assertEqual(m.post.totals["steps"], 1050)

    def test_attr_passthrough(self):
        env = FakeEnv()
        g = AP.GuardedEnv(env, AP.ActionPost({}))
        self.assertIs(g.unwrapped, env.unwrapped)
        g.some_flag = 3
        self.assertEqual(env.some_flag, 3)

    def test_undecided_and_unknown(self):
        cfg = _yml()
        cfg["task_name"] = "sample_loading"          # not submitted: backend undecided
        with self.assertRaises(ValueError):
            DP.get_model(dict(cfg))
        cfg["task_name"] = "drawer_open_place"       # OfI, but its pinned checkpoint is not in this machine's cache
        with self.assertRaises((ValueError, RuntimeError)):
            DP.get_model(dict(cfg))
        cfg["task_name"] = "open_pan"
        with self.assertRaises(ValueError):
            DP.get_model(dict(cfg))

    def test_overrides_reach_backend(self):
        ua = self.args(hold_chunk=3, overrides={"hold_chunk": 7})
        m = DP.get_model(ua)
        self.assertEqual(m.inner.chunk, 7)


class TestPackaging(unittest.TestCase):
    def test_yml(self):
        cfg = _yml()
        self.assertEqual(cfg["policy_name"], POLICY_DIR.name)
        self.assertEqual(sorted(cfg["task_policies"]), sorted(TASKS))
        self.assertEqual(cfg["max_steps"], 1000)
        for k in ("policy_name", "task_name", "setting", "model_name", "seed", "max_episodes", "max_steps",
                  "headless", "pytorch_device", "filter_dataset_saving", "eval_video_log",
                  "eval_video_obs_keys", "eval_expert_check"):
            self.assertIn(k, cfg)
        for name, mod in cfg["backends"].items():
            self.assertTrue(mod.startswith(f"policy.{POLICY_DIR.name}."), mod)
            rel = mod.split(".", 2)[2].replace(".", "/") + ".py"
            self.assertTrue((POLICY_DIR / rel).is_file(), rel)

    def test_official_yml_keys_still_present(self):
        repo = _official_repo()
        if repo is None:
            self.skipTest("official repo not found")
        import yaml

        off = yaml.safe_load(open(repo / "policy/Your_Policy/deploy_policy.yml"))
        missing = set(off) - set(_yml())
        self.assertEqual(missing, set(), f"keys of the official template missing in ours: {missing}")

    def test_eval_sh(self):
        r = subprocess.run(["bash", "-n", str(POLICY_DIR / "eval.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.access(POLICY_DIR / "eval.sh", os.X_OK))
        txt = (POLICY_DIR / "eval.sh").read_text()
        self.assertIn(f"POLICY_NAME={POLICY_DIR.name}", txt)

    def test_interface(self):
        pkg = importlib.import_module(f"policy.{POLICY_DIR.name}")
        for fn in ("get_model", "eval", "reset_model"):
            self.assertTrue(callable(getattr(pkg, fn)), fn)

    def test_no_imports_outside_policy(self):
        """The evaluator keeps only policy/. Our code may import the official repo
        (robosynchallenge, embodichain, policy.inference_timing) but nothing of ours outside policy/."""
        bad = []
        for p in POLICY_DIR.rglob("*.py"):
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith(("import ", "from ")) and any(x in s for x in ("rsrec", "real_limits", "final.", "work.")):
                    bad.append(f"{p.relative_to(POLICY_DIR)}: {s}")
        self.assertEqual(bad, [])

    def test_ofi_files_identical_to_preliminary(self):
        repo = Path(os.environ.get("PRELIM_REPO", os.path.expanduser("~/robosyn_ws/mine")))
        if not (repo / ".git").exists():
            self.skipTest("preliminary delivery clone not found")
        for rel_ours, rel_prelim in (("ofi/ofi_backend.py", "ofi_backend.py"), ("ofi/ofi_worker.py", "ofi_worker.py"),
                                     ("ofi/checkpoints.py", "checkpoints.py"),
                                     ("ofi/locks/ofi_worker.txt", "locks/ofi_worker.txt")):
            r = subprocess.run(["git", "-C", str(repo), "show", f"b50f70a:policy/MyPolicy/{rel_prelim}"],
                               capture_output=True)
            self.assertEqual(r.returncode, 0)
            self.assertEqual((POLICY_DIR / rel_ours).read_bytes(), r.stdout, rel_ours)

    def test_ofi_checkpoint_dir(self):
        d = os.environ.get("OFI_CKPT_DIR")
        if not d:
            self.skipTest("OFI_CKPT_DIR not set")
        ck = importlib.import_module(f"policy.{POLICY_DIR.name}.ofi.checkpoints")
        files = ck.verify_ofi_checkpoint(Path(d))
        self.assertIn("weights", files)


if __name__ == "__main__":
    unittest.main()
