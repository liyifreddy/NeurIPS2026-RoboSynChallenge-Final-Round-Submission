"""End-to-end timing of the whole policy/ path, without the simulator.

Drives deploy_policy.get_model / reset_model / eval with a stand-in env (640x480 frames,
14-dim qpos, counts steps, never succeeds) for N steps, the way scripts/eval_policy.py does,
and reports what the official metrics record (the inference times eval() returns) next to
the wall time per env step of the policy side (inference + action post-processing + IPC;
the stand-in env.step costs ~0). Works for every backend: `hold` on the CPU (plumbing),
`ofi` on the GPU (OFI_PYTHON / OFI_SRC_ROOT / MYPOLICY_CKPT_ROOT set as for eval.sh).

  cd <RoboSynChallenge>   # or final/delivery for the hold backend
  python policy/MyPolicy/tools/time_e2e.py --task click_bell --steps 240 [--backend hold] [--out t.json]
"""

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

POLICY = Path(__file__).resolve().parent.parent
ROOT = POLICY.parent.parent
sys.path.insert(0, str(ROOT))


class _Space:
    shape = (14,)


class _Unwrapped:
    def __init__(self):
        self.single_action_space = _Space()
        self.device = "cuda" if _has_cuda() else "cpu"
        self._robosyn_episode_seed = 1


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class StandInEnv:
    def __init__(self, max_steps, seed=0):
        rng = np.random.default_rng(seed)
        self.frames = {c: rng.integers(0, 256, (1, 480, 640, 3), dtype=np.uint8)
                       for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
        self.unwrapped = _Unwrapped()
        self.max_steps = max_steps
        self.t = 0
        self.step_walls = []
        self._last = None

    def obs(self):
        q = np.zeros((1, 14), dtype=np.float32)
        q[0, 6] = q[0, 13] = 1.0
        return {"robot": {"qpos": q}, "sensor": {c: {"color": f} for c, f in self.frames.items()}}

    def step(self, action):
        now = time.perf_counter()
        if self._last is not None:
            self.step_walls.append(now - self._last)
        self._last = now
        self.t += 1
        return (self.obs(), 0.0, np.array([False]), np.array([self.t >= self.max_steps]),
                {"elapsed_steps": np.array([self.t])})

    def get_wrapper_attr(self, name):
        if name == "is_task_success":
            return lambda: False
        raise AttributeError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--backend", default="", help="override the task's backend (e.g. hold)")
    p.add_argument("--policy", default=POLICY.name)
    p.add_argument("--out", default="")
    a = p.parse_args()

    import yaml

    cfg = yaml.safe_load(open(POLICY / "deploy_policy.yml"))
    cfg["task_name"] = a.task
    if a.backend:
        cfg["task_policies"][a.task] = {"backend": a.backend}
    dp = importlib.import_module(f"policy.{a.policy}.deploy_policy")
    t0 = time.perf_counter()
    model = dp.get_model(cfg)
    load_s = time.perf_counter() - t0
    env = StandInEnv(a.steps)
    dp.reset_model(model)
    obs, times, calls = env.obs(), [], 0
    t_start = time.perf_counter()
    while env.t < a.steps:
        obs, info, trunc, new = dp.eval(env, model, obs)
        times.extend(new)
        calls += 1
        if bool(np.asarray(trunc).any()):
            break
    wall = time.perf_counter() - t_start
    if hasattr(dp, "close_model"):
        dp.close_model(model)
    ms = sorted(1000.0 * x for x in times)
    sw = sorted(1000.0 * x for x in env.step_walls)
    res = {"task": a.task, "backend": cfg["task_policies"][a.task]["backend"], "steps": env.t,
           "eval_calls": calls, "inferences": len(times), "load_s": load_s, "wall_s": wall,
           "inference_ms": {"median": statistics.median(ms) if ms else None, "max": ms[-1] if ms else None,
                            "samples": ms},
           "step_wall_ms": {"median": statistics.median(sw) if sw else None, "max": sw[-1] if sw else None},
           "note": "inference_ms = what eval_policy.py averages into the inference-efficiency score; "
                   "step_wall_ms max = the longest pause between two commands (chunk boundary)"}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "inference_ms"} |
                     {"inference_ms_median": res["inference_ms"]["median"],
                      "inference_ms_max": res["inference_ms"]["max"]}))


if __name__ == "__main__":
    main()
