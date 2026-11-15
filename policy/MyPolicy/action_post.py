# ----------------------------------------------------------------------------
# Action post-processing at the policy exit (finals, real robot).
#
# FAQ #16: actions go to the robot through the official AgileX SDK as they
# are, with no interpolation, no limiting, no filtering. So the policy itself
# keeps every command inside the real joint limits before it leaves policy/.
# FAQ #17: rules like this may only read the observation (images,
# proprioception); this module reads only the action (and, for the log, the
# proprioceptive qpos in obs).
#
# 14-dim joint action (FAQ #15, same as the simulator):
#   [left arm J1..J6, left gripper, right arm J1..J6, right gripper]
#   arm joints: absolute targets in rad; gripper: 0 = closed, 1 = open, normalized
#   (real robot: 0-1 x 0.05 m per finger; the simulator's gripper action space is the
#    URDF limit [0, 0.05] m and clamps, so 0-1 behaves like a switch there).
#
# What it does per action (in this order):
#   1. non-finite values -> the last action that went out (or, before the
#      first one, the current qpos from obs); counted as `nonfinite`.
#   2. arm joints clipped to the selected limit profile (deploy_policy.yml
#      `action_post.clip_profile`); every clipped joint is counted with the
#      largest excess.
#   3. gripper mapped from the model's units to 0-1 (`gripper_src_range`,
#      default [0, 1] = identity), optionally turned into a switch
#      (`gripper_mode`: binary / hysteresis; default linear = half-open stays
#      half-open) and clipped to [0, 1]; counted separately (the
#      simulator-trained models often overshoot 1 slightly, which is not a
#      fault; the arm count is the one to watch).
#   4. optional per-step change limit (`max_step_rad`, `max_step_grip`): a
#      replanned chunk can jump; clamped against the previous command, off by
#      default.
#
# Pure numpy, no simulator or torch import: torch tensors are handled by duck
# typing so the same code runs in the simulator, on the real robot, and in the
# CPU unit tests (tests/test_action_post.py).
# ----------------------------------------------------------------------------

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

ACTION_DIM = 14
ARM_IDX = {"left": tuple(range(0, 6)), "right": tuple(range(7, 13))}
GRIP_IDX = {"left": 6, "right": 13}
ARM_ALL = ARM_IDX["left"] + ARM_IDX["right"]
JOINT_NAMES = [f"L_J{i}" for i in range(1, 7)] + ["L_grip"] + [f"R_J{i}" for i in range(1, 7)] + ["R_grip"]

_D = math.radians
_M = math.radians(1.5)  # 0.026179938779914945
# J1-J3 as the official URDF (assets/CobotMagicArm). J2 lower and J3 upper are
# the mechanical zero: the 1.5 deg margin is not applied on those two sides.
_J123 = [(-2.618, 2.618), (0.0, 3.14), (-2.967, 0.0)]
# Per-arm (lo, hi) for J1..J6, rad. Same values for both arms.
PROFILES = {
    # Alternative, kept for explicit reproduction runs only: the real mechanical
    # envelope = the hard-limit URDF copy (final/assets/urdf_real_limits/
    # CobotMagicArm). Not the default and not to be used for the finals.
    "real_hard": _J123 + [(-1.832596, 1.832596), (-1.308997, 1.308997), (-2.988, 2.988)],
    # Former default (finals commander 2026-11-04): real_limits.DEPLOY_CLIP of
    # = the range the real-robot demonstrations used + about 1 deg.
    # NOTE: its J6 +-160 deg is NOT the finals default any more.
    "deploy_margin": _J123 + [(-_D(103.0), _D(103.0)), (-_D(74.5), _D(74.5)), (-_D(160.0), _D(160.0))],
    # The official simulator URDF at the finals pin 45d9b9c (J5 changed 1.22 -> 1.3
    # rad vs a485f91). Comparison runs only.
    "official_sim": _J123 + [(-1.745, 1.745), (-1.3, 1.3), (-2.988, 2.988)],
}

# Finals default: the 45d9b9c official simulator URDF hard limits, every side
# pulled in by 1.5 deg, except the J2 lower / J3 upper mechanical zeros.
# Source: the official simulator URDF at 45d9b9c, read joint by joint.
# Values in rad, computed from the raw URDF limits, not from rounded degrees.
_EXECUTABLE = [
    (-2.591820061220085, 2.591820061220085),      # J1  +-148.500351 deg
    (0.0, 3.113820061220085),                     # J2  0 .. 178.408748 deg
    (-2.940820061220085, 0.0),                    # J3  -168.496578 .. 0 deg
    (-1.7188200612200852, 1.7188200612200852),    # J4  +-98.481135 deg
    (-1.2738200612200852, 1.2738200612200852),    # J5  +-72.984513 deg (1.3 rad URDF)
    (-2.961820061220085, 2.961820061220085),      # J6  +-169.699789 deg
]
PROFILES["executable"] = list(_EXECUTABLE)

DEFAULTS = {
    "enabled": True,
    # The official simulator limits minus 1.5 deg; over-limit commands
    # are clipped, so the SDK never sees an out-of-range target (no dropped command).
    "clip_profile": "executable",
    "custom_limits": None,          # [[lo, hi] x 6] per arm, overrides the profile when set
    "gripper_src_range": [0.0, 1.0],  # model's gripper units -> mapped linearly onto [0, 1]
    "gripper_clip": True,
    # linear: the model's 0-1 command is sent as is (real gripper is linear: half
    # open is genuinely half open; Q1 6.6). binary / hysteresis: switch the
    # command at 0/1, for tasks where the simulator gripper acts like a switch.
    # binary opens at >= gripper_hyst.hi (default 0.81; set hi to 0.5 for an
    # ordinary 0.5 binarization). hysteresis starts each episode OPEN and only
    # closes below lo. Neither may be combined with max_step_grip (checked in
    # parse_cfg), or the switch would be turned into fractions.
    "gripper_mode": "linear",
    # items_handover preliminary hysteresis 15 -> 81 on the 0-100 scale = 0.15 -> 0.81.
    "gripper_hyst": {"lo": 0.15, "hi": 0.81},
    # Optional per-step change limit (rad on arm joints, 0-1 units on grippers).
    # None = off. A replanned chunk can jump; set it to keep the next target near
    # the last one. Applied before the absolute clip.
    "max_step_rad": None,
    "max_step_grip": None,
    "log_dir": "",                  # "" = <eval_result_dir>/<task>/ if known, else no file
    "log_every_event": True,        # one JSON line per clipped step (else only the episode summary)
    "warn_arm_clip_steps": 20,      # print a warning when an episode clips arm joints on more steps than this
}


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def limits_for(cfg):
    """[(lo, hi)] x 14 for the 14-dim action (gripper slots = (0, 1))."""
    if cfg.get("custom_limits"):
        arm = [tuple(map(float, r)) for r in cfg["custom_limits"]]
        if len(arm) != 6 or any(lo >= hi for lo, hi in arm):
            raise ValueError(f"custom_limits must be 6 [lo, hi] pairs with lo < hi, got {cfg['custom_limits']!r}")
    else:
        name = cfg.get("clip_profile") or "executable"
        if name not in PROFILES:
            raise ValueError(f"clip_profile {name!r} unknown (have {sorted(PROFILES)})")
        arm = PROFILES[name]
    return list(arm) + [(0.0, 1.0)] + list(arm) + [(0.0, 1.0)]


def _gripper_hyst(v):
    base = dict(DEFAULTS["gripper_hyst"])
    if v is None:
        return base
    if not isinstance(v, dict):
        raise ValueError(f"gripper_hyst must be a dict, got {v!r}")
    bad = set(v) - set(base)
    if bad:
        raise ValueError(f"gripper_hyst: unknown keys {sorted(bad)}")
    out = dict(base, **{k: float(x) for k, x in v.items()})
    if not 0.0 <= out["lo"] < out["hi"] <= 1.0:
        raise ValueError(f"gripper_hyst needs 0 <= lo < hi <= 1, got {out!r}")
    return out


def _step_limit(v, name):
    if v is None:
        return None
    x = float(v)
    if not x > 0.0:
        raise ValueError(f"{name} must be > 0 (or null/None to turn it off), got {v!r}")
    return x


def parse_cfg(v):
    """deploy_policy.yml `action_post` (dict / None / False) -> full config dict."""
    if v is None or v is True:
        v = {}
    if v is False or (isinstance(v, str) and v.strip().lower() in ("off", "false", "0", "none", "null")):
        return dict(DEFAULTS, enabled=False)
    if not isinstance(v, dict):
        raise ValueError(f"action_post must be a dict, got {v!r}")
    bad = set(v) - set(DEFAULTS)
    if bad:
        raise ValueError(f"action_post: unknown keys {sorted(bad)}")
    cfg = dict(DEFAULTS, **v)
    lo, hi = map(float, cfg["gripper_src_range"])
    if not hi > lo:
        raise ValueError(f"gripper_src_range must be [lo, hi] with hi > lo, got {cfg['gripper_src_range']!r}")
    if cfg["gripper_mode"] not in ("linear", "binary", "hysteresis"):
        raise ValueError(f"gripper_mode must be linear / binary / hysteresis, got {cfg['gripper_mode']!r}")
    cfg["gripper_hyst"] = _gripper_hyst(cfg["gripper_hyst"])
    cfg["max_step_rad"] = _step_limit(cfg["max_step_rad"], "max_step_rad")
    cfg["max_step_grip"] = _step_limit(cfg["max_step_grip"], "max_step_grip")
    if cfg["gripper_mode"] != "linear" and cfg["max_step_grip"] is not None:
        raise ValueError("gripper_mode binary/hysteresis and max_step_grip cannot both be set: "
                         "the step limit would turn the 0/1 command into fractions")
    limits_for(cfg)  # validates the profile
    return cfg


class ActionPost:
    """Clips / maps one 14-dim action at a time and keeps per-episode counters."""

    def __init__(self, cfg=None, task=None, log_path=None):
        self.cfg = parse_cfg(cfg)
        self.task = task
        self.lim = np.asarray(limits_for(self.cfg), dtype=np.float64)
        self.g_lo, self.g_hi = map(float, self.cfg["gripper_src_range"])
        self.hyst_lo = float(self.cfg["gripper_hyst"]["lo"])
        self.hyst_hi = float(self.cfg["gripper_hyst"]["hi"])
        self.max_step_rad = self.cfg["max_step_rad"]
        self.max_step_grip = self.cfg["max_step_grip"]
        self.log_path = Path(log_path) if log_path else None
        self.episode = -1
        self.seed = None
        self.totals = {"episodes": 0, "steps": 0, "arm_clip_steps": 0, "grip_clip_steps": 0,
                       "grip_mode_steps": 0, "jump_clip_steps": 0, "nonfinite_steps": 0}
        self._reset_counters()

    # -- per-episode -------------------------------------------------------
    def _reset_counters(self):
        self.t = 0
        self.last = None
        self.n_arm_clip_steps = 0
        self.n_grip_clip_steps = 0
        self.n_grip_mode_steps = 0
        self.n_jump_clip_steps = 0
        self.n_nonfinite = 0
        # hysteresis starts open, like the preliminary left-gripper hysteresis
        self._grip_open = {"left": True, "right": True}
        self.per_joint = {n: {"n": 0, "max_excess": 0.0} for n in JOINT_NAMES}

    def start_episode(self, seed=None):
        if self.episode >= 0 and self.t > 0:
            self.end_episode()
        self.episode += 1
        self.seed = seed
        self._reset_counters()

    def summary(self):
        return {"kind": "episode_summary", "task": self.task, "episode": self.episode, "seed": self.seed,
                "steps": self.t, "arm_clip_steps": self.n_arm_clip_steps,
                "grip_clip_steps": self.n_grip_clip_steps, "grip_mode_steps": self.n_grip_mode_steps,
                "jump_clip_steps": self.n_jump_clip_steps, "nonfinite_steps": self.n_nonfinite,
                "per_joint": {k: v for k, v in self.per_joint.items() if v["n"]},
                "profile": self.cfg["custom_limits"] and "custom" or self.cfg["clip_profile"],
                "gripper_mode": self.cfg["gripper_mode"]}

    def end_episode(self):
        if self.t == 0:
            return None
        s = self.summary()
        self.totals["episodes"] += 1
        for k_tot, k in (("steps", "steps"), ("arm_clip_steps", "arm_clip_steps"),
                         ("grip_clip_steps", "grip_clip_steps"), ("grip_mode_steps", "grip_mode_steps"),
                         ("jump_clip_steps", "jump_clip_steps"), ("nonfinite_steps", "nonfinite_steps")):
            self.totals[k_tot] += s[k]
        self._write(s)
        msg = (f"[action_post] {self.task} episode={self.episode} steps={self.t} "
               f"arm_clip_steps={self.n_arm_clip_steps} grip_clip_steps={self.n_grip_clip_steps} "
               f"grip_mode={self.cfg['gripper_mode']}({self.n_grip_mode_steps}) "
               f"jump_clip_steps={self.n_jump_clip_steps} nonfinite={self.n_nonfinite}")
        if self.n_arm_clip_steps > int(self.cfg["warn_arm_clip_steps"]) or self.n_nonfinite:
            msg = "WARNING " + msg + " (the policy is driving into the joint limits; do not rely on the clip)"
        print(msg, flush=True)
        self.t = 0  # closed
        return s

    # -- one action --------------------------------------------------------
    def _gripper_switch(self, a, ev):
        """binary / hysteresis gripper mode, in place. linear = no change."""
        mode = self.cfg["gripper_mode"]
        if mode == "linear":
            return
        changed = {}
        for side, gi in GRIP_IDX.items():
            raw = float(a[gi])
            if mode == "binary":
                out = 1.0 if raw >= self.hyst_hi else 0.0
                state = out > 0.5
            else:  # hysteresis: open until the command drops below lo, closed until it rises above hi
                state = self._grip_open[side]
                if state and raw <= self.hyst_lo:
                    state = False
                elif not state and raw >= self.hyst_hi:
                    state = True
                out = 1.0 if state else 0.0
            self._grip_open[side] = state
            if out != raw:
                changed[JOINT_NAMES[gi]] = [round(raw, 6), out]
            a[gi] = out
        if changed:
            self.n_grip_mode_steps += 1
            ev["gripper_mode"] = changed

    def _rate_limit(self, a, ev):
        """Clamp how far one command may move from the previous one. Off by default."""
        if self.last is None or (self.max_step_rad is None and self.max_step_grip is None):
            return
        jumped = {}
        if self.max_step_rad is not None:
            for i in ARM_ALL:
                d = float(a[i] - self.last[i])
                if abs(d) > self.max_step_rad:
                    a[i] = float(self.last[i]) + math.copysign(self.max_step_rad, d)
                    jumped[JOINT_NAMES[i]] = round(abs(d) - self.max_step_rad, 6)
        if self.max_step_grip is not None:
            for i in GRIP_IDX.values():
                d = float(a[i] - self.last[i])
                if abs(d) > self.max_step_grip:
                    a[i] = float(self.last[i]) + math.copysign(self.max_step_grip, d)
                    jumped[JOINT_NAMES[i]] = round(abs(d) - self.max_step_grip, 6)
        if jumped:
            self.n_jump_clip_steps += 1
            ev["jump"] = jumped

    def _write(self, rec):
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:  # logging must never stop an episode
            print(f"[action_post] WARNING: cannot write {self.log_path}: {e}", flush=True)
            self.log_path = None

    def process(self, action, qpos=None):
        """action: [14] or [1, 14] (numpy or torch) -> same type/shape/dtype/device, post-processed."""
        if not self.cfg["enabled"]:
            return action
        a = _to_numpy(action).astype(np.float64)
        shape = a.shape
        a = a.reshape(-1)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"action has {a.shape[0]} values, expected {ACTION_DIM} (shape {shape})")
        ev = {}
        # gripper: model units -> [0, 1] (a NaN stays NaN and is filled below, already in output units)
        for gi in GRIP_IDX.values():
            a[gi] = (a[gi] - self.g_lo) / (self.g_hi - self.g_lo)
        bad = ~np.isfinite(a)
        if bad.any():
            # hold the last action that went out; before the first one, the current qpos
            # (obs qpos: arm in rad, gripper normalized 0-1 = the action's output units)
            fill = self.last if self.last is not None else (None if qpos is None else _to_numpy(qpos).astype(np.float64).reshape(-1)[:ACTION_DIM])
            if fill is None or fill.shape[0] != ACTION_DIM or not np.isfinite(fill).all():
                raise ValueError(f"non-finite action {a.tolist()} and nothing finite to hold")
            a[bad] = fill[bad]
            self.n_nonfinite += 1
            ev["nonfinite"] = [JOINT_NAMES[i] for i in np.flatnonzero(bad)]
        self._gripper_switch(a, ev)
        self._rate_limit(a, ev)
        lo, hi = self.lim[:, 0], self.lim[:, 1]
        clipped = np.clip(a, lo, hi)
        diff = a - clipped
        arm_hit = [i for i in ARM_ALL if diff[i] != 0.0]
        grip_hit = [i for i in GRIP_IDX.values() if diff[i] != 0.0]
        if not self.cfg["gripper_clip"]:
            for i in grip_hit:
                clipped[i] = a[i]
            grip_hit = []
        if arm_hit:
            self.n_arm_clip_steps += 1
            ev["arm"] = {JOINT_NAMES[i]: round(float(diff[i]), 6) for i in arm_hit}
        if grip_hit:
            self.n_grip_clip_steps += 1
            ev["grip"] = {JOINT_NAMES[i]: round(float(diff[i]), 6) for i in grip_hit}
        for i in arm_hit + grip_hit:
            pj = self.per_joint[JOINT_NAMES[i]]
            pj["n"] += 1
            pj["max_excess"] = max(pj["max_excess"], abs(float(diff[i])))
        if ev and self.cfg["log_every_event"]:
            self._write({"kind": "clip", "task": self.task, "episode": self.episode, "t": self.t, **ev})
        self.last = clipped.copy()
        self.t += 1
        out = clipped.reshape(shape)
        if hasattr(action, "detach"):  # torch tensor in, torch tensor out (same dtype / device)
            import torch

            return torch.as_tensor(out, dtype=action.dtype, device=action.device)
        return out.astype(np.asarray(action).dtype if np.asarray(action).dtype.kind == "f" else np.float32)


class GuardedEnv:
    """Env proxy: every env.step(action) goes through ActionPost first.

    Same pattern as the preliminary HookedEnv: attribute reads and writes go to
    the wrapped env, so any backend that calls env.step() / env.unwrapped /
    env.get_wrapper_attr() works unchanged. Only reads `obs` (qpos for the
    non-finite fallback) and the action.
    """

    def __init__(self, env, post, obs=None):
        object.__setattr__(self, "_g_env", env)
        object.__setattr__(self, "_g_post", post)
        object.__setattr__(self, "_g_obs", obs)

    def __getattr__(self, name):
        return getattr(self._g_env, name)

    def __setattr__(self, name, value):
        setattr(self._g_env, name, value)

    def step(self, action):
        q = None
        try:
            q = self._g_obs["robot"]["qpos"] if self._g_obs is not None else None
        except (KeyError, TypeError, IndexError):
            q = None
        out = self._g_env.step(self._g_post.process(action, qpos=q))
        object.__setattr__(self, "_g_obs", out[0])
        return out


def default_log_path(usr_args, task):
    """<log_dir or eval_result_dir/task>/action_post_<time>_<pid>.jsonl, or None."""
    cfg = parse_cfg(usr_args.get("action_post"))
    base = cfg.get("log_dir") or ""
    if not base:
        erd = usr_args.get("eval_result_dir")
        if not erd:
            return None
        base = os.path.join(str(erd), str(task))
    return os.path.join(base, f"action_post_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
