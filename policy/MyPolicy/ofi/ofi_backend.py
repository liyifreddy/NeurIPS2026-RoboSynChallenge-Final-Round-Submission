# ----------------------------------------------------------------------------
# OfI backend: One from Infinity (RoboActualizer @ 85b9dc6) for RoboSynChallenge.
#
# The simulator stays in the official ACT environment; the model runs in
# ofi_worker.py under its own Python 3.10 environment (README Setup step 6,
# OFI_PYTHON). Per env step, the three camera frames are pushed to the worker
# (the model's video clip needs a frame history at every step, as in the
# RoboActualizer RoboTwin policy: record the frame first, then act). When the
# action queue is empty the worker also predicts a 32-step chunk; the first
# `ofi_replan_steps` (24) are queued (the RoboTwin setting). Actions are
# absolute joint targets in the dataset's units and are sent to env.step
# unchanged, as policy/act does. Success is checked after every env.step.
#
# Checkpoint: the task's Hugging Face snapshot (task_policies hf_repo@revision,
# resolved by deploy_policy.py into checkpoint_dir) holds ofi_checkpoint.json,
# which names the weights, the training config, the action/state statistics,
# the instruction-embedding cache and meta/tasks.jsonl, with their sha256; every
# file is checked against it before the worker starts. The frozen V-JEPA 2.1
# encoder comes from `ofi_backbone` (a pinned Hugging Face file). Everything is
# read from the local caches filled by prefetch.py; the worker runs with
# HF_HUB_OFFLINE=1 and downloads nothing.
#
# One optional setting, off by default (null):
#   ofi_left_grip_hyst  hysteresis on the left gripper command. The simulated
#       gripper acts like a switch (from open it only starts closing for
#       commands below about 0.15), and right after the model's left command
#       first drops below that it can bounce back above it, re-opening the
#       gripper. With the setting on, once the left command has dropped below
#       `lo` the left gripper is commanded closed (0) until the model's own
#       command has been >= `hi` for `k_open` consecutive steps. It only starts
#       acting once the left gripper has been open (normalized reading > arm_open)
#       in the episode. It reads only the model's own actions and the
#       proprioceptive gripper reading; arm joints and the right gripper are
#       never changed. "on" uses HYST_DEFAULTS.
#   ofi_pre_place_correction  sample_loading only: before the left hand releases the
#       tube, shift the left arm so the tube bottom sits over the hole estimated
#       from the step-0 camera frames (sl_place_corr.py; estimator files pinned by
#       `ofi_sl_estimator`). "on" uses sl_place_corr.DEFAULTS; a dict overrides keys.
#       At most one of ofi_left_grip_hyst / ofi_pre_place_correction per task.
# Interface: get_model / eval / reset_model / close_model.
# ----------------------------------------------------------------------------

#
# Real-robot serving (finals correction, request real_88e951d11a824c). The station
# delivers one observation per action through a feedback bridge, so the time from an
# observation to the next action is the per-step reply time. Three changes, none of
# which changes an action or the frames the model sees:
#   1. CPU threads of this process and of the worker follow the container CPU quota
#      (cgroup cpu.max / cfs quota and the CPU affinity), instead of the host core count.
#   2. Worker requests run in order on one background thread. A step that does not run
#      an inference returns its already planned action without waiting for the worker to
#      add the frame to its history; a step that infers first waits until every earlier
#      frame is in the history (one pipe, strict request order), so the model input is
#      the same as when every push waited.
#   3. A worker that exits or stops answering raises from eval() within seconds
#      (OFI_STEP_TIMEOUT, OFI_FIRST_INFER_TIMEOUT) instead of blocking.

from __future__ import annotations

import atexit
import concurrent.futures
import json
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .checkpoints import ofi_manifest, ofi_runtime, require_hub_file, switch_value, verify_ofi_checkpoint

CAM_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
WORKER = Path(__file__).resolve().parent / "ofi_worker.py"
GRIP_IDX = (6, 13)  # [left_arm 6, left_gripper, right_arm 6, right_gripper]; gripper in [0, 1], 0 = closed
MAX_CPU_THREADS = 8        # per process, also capped by the container quota (cpu_quota)
STEP_TIMEOUT = 30.0        # s without a worker reply to a push / inference -> eval raises (normal: < 0.1 s)
FIRST_INFER_TIMEOUT = 60.0  # s, the first inference of the worker includes its CUDA warm-up


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame(obs, cam):
    img = _to_numpy(obs["sensor"][cam]["color"])
    if img.ndim == 4:  # [num_envs, H, W, C]
        img = img[0]
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"{cam}: expected HWC image, got {img.shape}")
    img = img[..., :3]
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and float(np.nanmax(img)) <= 1.0:
            raise ValueError(f"{cam}: float image in [0,1]; expected uint8 0-255 like the dataset")
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _qpos(obs):
    q = _to_numpy(obs["robot"]["qpos"]).astype(np.float32)
    if q.ndim == 2:
        q = q[0]
    return q.reshape(-1)


# The official observation normalization ((q - low) / (high - low)) does not clip, so an opening
# gripper reading can overshoot 1 (up to about 1.3 seen). This guard only catches a units error.
GRIP_RANGE = (-0.5, 2.0)


def _grips(obs, idx=GRIP_IDX):
    """(left, right) normalized gripper readings; raises on a gross units error."""
    q = _qpos(obs)
    gl, gr = float(q[idx[0]]), float(q[idx[1]])
    if not (GRIP_RANGE[0] <= gl <= GRIP_RANGE[1] and GRIP_RANGE[0] <= gr <= GRIP_RANGE[1]):
        raise ValueError(f"gripper qpos {gl}, {gr} outside {GRIP_RANGE}: obs gripper is not normalized")
    return gl, gr


def _opt_none(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none"))


THETA = 0.15   # effective switch of the simulated gripper (see header)
HYST_DEFAULTS = {
    "lo": THETA,          # enter "closed" when the raw left command < lo
    "hi": 0.8,            # leave it only when the raw command >= hi ...
    "k_open": 2,          # ... for this many consecutive steps
    "arm_open": 0.84,     # active once the left gripper has been open (> this) in the episode
}
_INT_KEYS = ("k_open",)


def _opt_dict(v, defaults, name):
    """None (off) or a dict of `defaults`. Accepts None / "null" / False / 0 / "off" (off), True / 1 / "on"
    (defaults), a dict, or text "k=v,k=v" / a dict literal (overrides may arrive as text or via ast.literal_eval)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v not in (0, 1):
            raise ValueError(f"{name}: number must be 0 (off) or 1 (on), got {v!r}")
        v = bool(v)
    if _opt_none(v) or v is False or (isinstance(v, str) and v.strip().lower() in ("off", "false", "0")):
        return None
    if v is True or (isinstance(v, str) and v.strip().lower() in ("on", "true", "1", "default")):
        v = {}
    if isinstance(v, str):
        s = v.strip()
        try:
            if s.startswith("{"):
                import ast
                v = ast.literal_eval(s)
            else:
                v = dict(kv.split("=", 1) for kv in s.replace(" ", "").split(",") if kv)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"{name}: cannot parse {s!r} ({e})") from None
    if not isinstance(v, dict):
        raise ValueError(f"{name} must be null, on, a dict or 'k=v,...', got {v!r}")
    bad = set(v) - set(defaults)
    if bad:
        raise ValueError(f"{name}: unknown keys {sorted(bad)}")
    h = dict(defaults)
    for k, x in v.items():
        try:
            f = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {k}={x!r} is not a number") from None
        if k in _INT_KEYS and not f.is_integer():
            raise ValueError(f"{name}: {k}={x!r} must be an integer")
        h[k] = int(f) if k in _INT_KEYS else f
    return h


def _left_grip_hyst(v):
    h = _opt_dict(v, HYST_DEFAULTS, "ofi_left_grip_hyst")
    if h is not None and not (0 < h["lo"] < h["hi"] <= 1 and 1 <= h["k_open"] <= 20 and 0 < h["arm_open"] < 1):
        raise ValueError(f"ofi_left_grip_hyst out of range: {h}")
    return h


def cpu_quota():
    """CPUs this process may use: the CPU affinity, capped by the container CFS quota
    (cgroup v2 cpu.max, else cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quota = None
    try:
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0 and period > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, math.ceil(quota))
    return max(1, n)


def _warn_nonfinite(chunk):
    """A non-finite model output does not stop the episode: action_post holds the last command (before
    the first one, the current qpos) for those values, as it does for any non-finite action."""
    if not np.isfinite(chunk).all():
        print(f"[OfI] WARNING: {int((~np.isfinite(chunk)).sum())} non-finite values in the chunk; "
              "action_post holds the previous command for them", flush=True)


def _env_float(name, default):
    v = os.environ.get(name, "")
    return float(v) if v.strip() else default


def _as_bool(v):
    if isinstance(v, torch.Tensor):
        return bool(v.any().item())
    if isinstance(v, np.ndarray):
        return bool(v.any())
    return bool(v)


class OfIClient:
    def __init__(self, usr_args):
        self.replan_steps = int(usr_args.get("ofi_replan_steps", 24))
        self.act_step = int(usr_args.get("ofi_act_step", self.replan_steps))
        self.state_dim = 14
        self.hyst = _left_grip_hyst(usr_args.get("ofi_left_grip_hyst"))
        ppc = usr_args.get("ofi_pre_place_correction")
        if switch_value(ppc) is None:
            ppc = None           # off (the default): sl_place_corr is not even imported
        else:
            if str(usr_args.get("task_name")) != "sample_loading":
                raise ValueError(f"ofi_pre_place_correction is for sample_loading only (task {usr_args.get('task_name')!r})")
            from . import sl_place_corr
            ppc = sl_place_corr.parse(ppc)
        if ppc is not None and self.hyst is not None:
            raise ValueError("ofi_left_grip_hyst and ofi_pre_place_correction are both on; one control per task")
        self.ppc = None
        if ppc is not None:   # built before the worker so a missing estimator file fails fast
            est = usr_args.get("ofi_sl_estimator") or {}
            need = ("hf_repo", "revision", "weights", "weights_sha256", "mounts", "mounts_sha256")
            if not all(est.get(k) for k in need):
                raise ValueError(f"ofi_pre_place_correction needs ofi_sl_estimator with {need}, got {est!r}")
            w = require_hub_file(est["hf_repo"], str(est["revision"]), est["weights"], "SL rack estimator", est["weights_sha256"])
            m = require_hub_file(est["hf_repo"], str(est["revision"]), est["mounts"], "SL wrist camera mounts", est["mounts_sha256"])
            self.ppc = sl_place_corr.PrePlaceCorrection(ppc, sl_place_corr.RackEstimator(
                w, m, device=f"cuda:{int(usr_args.get('gpu_id', 0))}", mount_max_deg=float(ppc["mount_max_deg"])),
                setting=usr_args.get("setting"))

        ckpt_dir = Path(str(usr_args["checkpoint_dir"]))
        files = verify_ofi_checkpoint(ckpt_dir)   # manifest roles present, every file's sha256 as recorded
        _, sums = ofi_manifest(ckpt_dir)
        verify_sums = {k: sums[str(p.relative_to(ckpt_dir))] for k, p in files.items()}
        bb = usr_args.get("ofi_backbone") or {}
        if not all(bb.get(k) for k in ("hf_repo", "revision", "file", "sha256")):
            raise ValueError(f"ofi_backbone must give hf_repo, revision, file and sha256, got {bb!r}")
        vjepa = require_hub_file(bb["hf_repo"], str(bb["revision"]), bb["file"], "OfI V-JEPA encoder", bb["sha256"])
        # worker python + RoboActualizer / vjepa2 checkouts (a missing vjepa2 tree would make RoboActualizer
        # call torch.hub.list, i.e. go online)
        python, src_root = ofi_runtime(usr_args)
        ra = src_root / "RoboActualizer"

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{ra / 'src'}:{ra}"
        env["TORCH_HOME"] = str(src_root / "torch_home")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # CPU threads follow the container quota, not the host core count (both processes)
        self.cpu_threads = min(cpu_quota(), MAX_CPU_THREADS)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(self.cpu_threads)
        if torch.get_num_threads() > self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        self.step_timeout = _env_float("OFI_STEP_TIMEOUT", STEP_TIMEOUT)
        self.first_infer_timeout = _env_float("OFI_FIRST_INFER_TIMEOUT", FIRST_INFER_TIMEOUT)
        self.async_push = str(usr_args.get("ofi_async_push", True)).strip().lower() not in ("0", "false", "off", "no")
        self.inferred = False      # the worker has answered one inference (first one includes warm-up)
        self.pending = deque()     # pushes sent without waiting, oldest first
        self._io = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ofi-rpc")
        # The simulator keeps all GPUs visible and picks its card by gpu_id (DexSim's Vulkan device
        # ignores CUDA_VISIBLE_DEVICES); the worker only uses CUDA, so it is pinned to that card.
        env["CUDA_VISIBLE_DEVICES"] = str(usr_args.get("ofi_cuda_visible_devices", usr_args.get("gpu_id", 0)))
        cmd = [python, str(WORKER),
               "--train_config", str(files["train_config"]), "--ckpt", str(files["weights"]),
               "--stats", str(files["stats"]), "--text_cache", str(files["text_cache_file"].parent),
               "--dataset_dir", str(files["tasks"].parent.parent), "--vjepa", str(vjepa),
               "--mixed_precision", str(usr_args.get("ofi_mixed_precision", "no")),
               "--num_inference_steps", str(int(usr_args.get("ofi_num_inference_steps", 10)))]
        self.stderr_path = usr_args.get("ofi_worker_log") or None
        self._err = open(self.stderr_path, "a") if self.stderr_path else None
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, env=env)
        atexit.register(self.close)
        self.info = self._rpc({"cmd": "hello"}, timeout=None)   # model loading: no time limit
        if int(self.info.get("chunk_size", 0)) < self.replan_steps:
            raise ValueError(f"replan_steps {self.replan_steps} > chunk {self.info.get('chunk_size')}")
        for key, role in (("ckpt_sha256", "weights"), ("stats_sha256", "stats")):   # the files the worker read
            if self.info.get(key) != verify_sums[role]:
                raise RuntimeError(f"worker {key} {self.info.get(key)} differs from the checkpoint manifest")
        if Path(self.info.get("text_cache", "")).name != files["text_cache_file"].name:
            raise RuntimeError(f"worker read text cache {self.info.get('text_cache')}, manifest names "
                               f"{files['text_cache_file'].name}")
        print("[OfI] worker ready: " + json.dumps({k: self.info[k] for k in
              ("ckpt", "ckpt_sha256", "stats_sha256", "prompt", "text_mask_true", "mixed_precision",
               "accel", "torch", "cuda_visible_devices", "gpu")}), flush=True)
        print("[OfI] strict ckpt check: " + json.dumps(self.info["ckpt_check"]), flush=True)
        print(f"[OfI] backbone: {bb['hf_repo']}@{bb['revision']} {vjepa}", flush=True)
        print("[OfI] inference variant: " + json.dumps({
            "replan_steps": self.replan_steps, "act_step": self.act_step, "left_grip_hyst": self.hyst}), flush=True)
        print("[OfI] serving: " + json.dumps({
            "cpu_threads": self.cpu_threads, "async_push": self.async_push, "step_timeout_s": self.step_timeout,
            "first_infer_timeout_s": self.first_infer_timeout}), flush=True)
        if self.hyst is not None:
            print("[MyPolicy] ofi_left_grip_hyst ON", flush=True)
        if self.ppc is not None:
            print("[MyPolicy] ofi_pre_place_correction ON " + json.dumps(
                {**{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.ppc.cfg.items()},
                 "setting": self.ppc.setting, "active": self.ppc.active}), flush=True)
        self.queue = deque()
        self.need_seed = True
        self.printed_space = False
        self.episode = -1
        self.seed = None
        self._reset_ctl()

    def _reset_ctl(self):
        self.hy_armed, self.hy_closed, self.hy_cnt = False, False, 0
        self.hy_closes = self.hy_forced = self.hy_suppressed = 0
        self.t = 0               # env steps in this episode (ofi_pre_place_correction)
        self.last_left = 1.0     # last executed left gripper command
        if getattr(self, "ppc", None) is not None:
            self.ppc.reset()

    def _flush_ctl(self):
        """One summary line per episode when the hysteresis is on. Safe on a half-built object."""
        if getattr(self, "hyst", None) is not None and hasattr(self, "hy_closes") and self.episode >= 0:
            print(f"[OfI] left_grip_hyst episode={self.episode} seed={self.seed} closes={self.hy_closes} "
                  f"forced={self.hy_forced} suppressed={self.hy_suppressed}", flush=True)
        if getattr(self, "ppc", None) is not None and self.episode >= 0:
            print(f"[OfI] pre_place_correction episode={self.episode} seed={self.seed} " + json.dumps(self.ppc.rec), flush=True)
        self.episode_flushed = True

    def hyst_step(self, obs, raw, exe):
        """ofi_left_grip_hyst: once the raw left command < lo, command closed until it is >= hi for k_open steps."""
        hy, li = self.hyst, GRIP_IDX[0]
        try:
            gl = _grips(obs)[0]
        except ValueError as e:   # an unexpected reading never stops the episode: no hysteresis this step
            if not getattr(self, "hy_warned", False):
                print(f"[OfI] WARNING: left_grip_hyst skipped: {e}", flush=True)
                self.hy_warned = True
            return exe
        if gl > hy["arm_open"]:
            self.hy_armed = True
        if not self.hy_armed:
            return exe
        c = float(raw[li])
        if not self.hy_closed:
            if c >= hy["lo"]:
                return exe
            self.hy_closed, self.hy_cnt = True, 0
            self.hy_closes += 1
        else:
            self.hy_cnt = self.hy_cnt + 1 if c >= hy["hi"] else 0
            if self.hy_cnt >= hy["k_open"]:
                self.hy_closed = False
                return exe
        if exe[li] == 0.0:
            return exe
        if c >= hy["lo"]:
            self.hy_suppressed += 1
        self.hy_forced += 1
        exe = exe.copy()
        exe[li] = 0.0
        return exe

    def _exchange(self, req, payload=b""):
        """One request -> one reply on the worker pipe. Runs only on the self._io thread."""
        if self.proc.poll() is not None:
            raise RuntimeError(f"OfI worker exited with {self.proc.returncode} (log: {self.stderr_path})")
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        if payload:
            self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"OfI worker closed the pipe (rc={self.proc.poll()}, log: {self.stderr_path})")
        rep = json.loads(line)
        if not rep.get("ok"):
            raise RuntimeError(f"OfI worker error: {rep.get('error')}")
        return rep

    def _wait(self, fut, timeout, what, t0=None):
        """Result of one worker request; raises within about 0.2 s of a worker exit and when
        there is no reply `timeout` s after t0 (default: now; then the worker is killed: the
        episode cannot go on)."""
        t0 = time.monotonic() if t0 is None else t0
        while True:
            try:
                return fut.result(timeout=0.2)
            except concurrent.futures.TimeoutError:
                pass
            if self.proc.poll() is not None and not fut.done():
                self._abort()
                raise RuntimeError(f"OfI worker exited with {self.proc.returncode} during {what} "
                                   f"(log: {self.stderr_path})")
            if timeout is not None and time.monotonic() - t0 > timeout and not fut.done():
                self._abort()
                raise TimeoutError(f"OfI worker gave no reply to {what} in {timeout:.0f} s (log: {self.stderr_path})")

    def _abort(self):
        try:
            self.proc.kill()   # unblocks the I/O thread (EOF); the episode fails explicitly
        except Exception:
            pass

    def _drain(self):
        """Wait until every push sent without waiting has been answered, oldest first."""
        while self.pending:
            fut, sent = self.pending[0]
            self._wait(fut, self.step_timeout, "a frame push", t0=sent)
            self.pending.popleft()

    def _check_pending(self):
        """Raise now if an earlier push failed, the worker is gone, or the oldest unanswered push
        was sent more than step_timeout s ago (no waiting)."""
        while self.pending and self.pending[0][0].done():
            self.pending.popleft()[0].result()
        if self.proc.poll() is not None:
            raise RuntimeError(f"OfI worker exited with {self.proc.returncode} (log: {self.stderr_path})")
        if self.pending and time.monotonic() - self.pending[0][1] > self.step_timeout:
            self._abort()
            raise TimeoutError(f"OfI worker gave no reply to a frame push in {self.step_timeout:.0f} s "
                               f"(log: {self.stderr_path})")

    def _rpc(self, req, payload=b"", timeout=-1):
        """Request in strict order after every earlier one; wait for its reply."""
        if timeout == -1:
            timeout = self.step_timeout
        self._drain()
        return self._wait(self._io.submit(self._exchange, req, payload), timeout, req.get("cmd"))

    def push(self, obs, infer, wait=True):
        """Send this step's frames and qpos. infer: also return the next chunk (always waits).
        wait=False (no inference): queue the frame for the history and return None at once."""
        cams = [_frame(obs, c) for c in CAM_ORDER]
        q = _qpos(obs)
        if q.shape[0] != self.state_dim:
            raise ValueError(f"qpos dim {q.shape[0]} != {self.state_dim}")
        req = {"cmd": "push", "infer": bool(infer), "shapes": [list(c.shape) for c in cams],
               "nbytes": int(sum(c.nbytes for c in cams)), "qpos": q.tolist()}
        payload = b"".join(c.tobytes() for c in cams)   # a copy: later changes to obs do not matter
        if infer:
            rep = self._rpc(req, payload, self.step_timeout if self.inferred else self.first_infer_timeout)
            self.inferred = True
            return rep
        if wait:
            return self._rpc(req, payload)
        self._check_pending()
        self.pending.append((self._io.submit(self._exchange, req, payload), time.monotonic()))
        return None

    def reset(self):
        self.queue.clear()
        self.need_seed = True
        if not getattr(self, "episode_flushed", True):
            self._flush_ctl()   # summary of the previous episode
        self.episode += 1
        self.seed = None
        self._reset_ctl()
        self.episode_flushed = False
        self._rpc({"cmd": "reset"})

    def close(self):
        if not getattr(self, "episode_flushed", True):
            self._flush_ctl()
        p = getattr(self, "proc", None)
        if p is not None and p.poll() is None:
            try:
                self._drain()   # the I/O thread is idle afterwards; nothing else writes to the pipe
                self.proc.stdin.write(b'{"cmd": "quit"}\n')
                self.proc.stdin.flush()
                p.wait(timeout=10)
            except Exception:
                p.kill()
        io = getattr(self, "_io", None)
        if io is not None:
            io.shutdown(wait=False, cancel_futures=True)
        err = getattr(self, "_err", None)
        if err is not None:
            err.close()
            self._err = None


def get_model(usr_args):
    return OfIClient(usr_args)


def _episode_seed(env):
    for owner in (env, getattr(env, "unwrapped", None)):
        s = getattr(owner, "_robosyn_episode_seed", None)
        if s is not None:
            return int(s)
    return None


def eval(env, model, obs):
    if model.need_seed:
        s = _episode_seed(env)
        if s is None:
            print("[OfI] WARNING: no _robosyn_episode_seed on env; worker not re-seeded", flush=True)
        else:
            model._rpc({"cmd": "seed", "seed": s})
        model.seed = s
        model.need_seed = False
    if not model.printed_space:
        sp = env.unwrapped.single_action_space
        print(f"[OfI] action space low={np.asarray(sp.low).reshape(-1).tolist()} "
              f"high={np.asarray(sp.high).reshape(-1).tolist()}", flush=True)
        model.printed_space = True

    times = []
    pre = 0.0   # rack-estimate time: added to this call's first inference entry (one entry per inference call)
    if model.ppc is not None and model.t == 0 and "rack_est" not in model.ppc.rec:
        t0 = time.perf_counter()   # policy computation: counted in the inference time
        try:
            model.ppc.estimate(obs, env.unwrapped)   # the step-0 frame, as the estimator was trained
        except Exception as e:   # never stop the episode: run it without the correction
            model.ppc.skip(f"estimate {type(e).__name__}: {e}")
        pre = time.perf_counter() - t0
    final_obs, info, truncated = obs, None, False
    env_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    for _ in range(model.act_step):
        infer = len(model.queue) == 0
        t0 = time.perf_counter()
        # record this frame first, as the RoboTwin policy does; without an inference the frame
        # joins the history in the background (in order, complete before the next inference)
        rep = model.push(final_obs, infer, wait=not model.async_push)
        if infer:
            chunk = np.asarray(rep["chunk"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != env_dim:
                raise ValueError(f"chunk shape {chunk.shape}, env action dim {env_dim}")
            _warn_nonfinite(chunk)
            if model.ppc is not None:   # policy computation: counted in the inference time
                chunk = chunk.copy()
                try:   # the whole chunk confirms the release ramp; only the executed rows are changed
                    model.ppc.apply(chunk, model.t, model.last_left, n_exec=model.replan_steps)
                except Exception as e:
                    chunk = np.asarray(rep["chunk"], dtype=np.float32)   # the model's own chunk, unchanged
                    model.ppc.skip(f"apply {type(e).__name__}: {e}")
            for i in range(model.replan_steps):
                model.queue.append(chunk[i])
            times.append(time.perf_counter() - t0 + pre)
            pre = 0.0
        raw = model.queue.popleft()
        exe = raw
        if model.hyst is not None:
            exe = model.hyst_step(final_obs, raw, exe)
        model.last_left = float(exe[GRIP_IDX[0]])
        model.t += 1
        a = torch.as_tensor(exe, dtype=torch.float32,
                            device=env.unwrapped.device).unsqueeze(0)
        final_obs, reward, terminated, truncated, info = env.step(a)
        if env.get_wrapper_attr("is_task_success")():
            break
        if _as_bool(truncated):
            break
    if pre:   # no inference in this call (cannot happen at t == 0, where the queue is empty): keep the time
        times.append(pre)
    return final_obs, info, truncated, times


def chunkable(model):
    """True when the whole chunk can be fixed at inference time: no pre-place correction (it reads
    each step's observation) and one inference per eval call. With ofi_left_grip_hyst the chunk is
    fixed too: whether the hysteresis is active is read from the observation the chunk is inferred
    on, and within the chunk it follows the model's own commands (plan())."""
    return model.ppc is None and model.act_step == model.replan_steps


def plan(env, model, obs, post):
    """One inference on `obs` and the post-processed actions of the whole chunk, before any of them
    is sent: the first part of eval() with `chunkable(model)`, in the same order (seed, push of this
    frame with the inference, chunk checks); action_post runs here, action by action, in order.
    Returns (actions: list of [1, 14] tensors, inference times)."""
    if not chunkable(model):
        raise ValueError("plan() needs a task whose chunk can be fixed at inference time")
    model.queue.clear()   # the per-step queue is not used on this path
    if model.need_seed:
        s = _episode_seed(env)
        if s is None:
            print("[OfI] WARNING: no _robosyn_episode_seed on env; worker not re-seeded", flush=True)
        else:
            model._rpc({"cmd": "seed", "seed": s})
        model.seed = s
        model.need_seed = False
    if not model.printed_space:
        sp = env.unwrapped.single_action_space
        print(f"[OfI] action space low={np.asarray(sp.low).reshape(-1).tolist()} "
              f"high={np.asarray(sp.high).reshape(-1).tolist()}", flush=True)
        model.printed_space = True
    env_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    t0 = time.perf_counter()
    rep = model.push(obs, True)
    chunk = np.asarray(rep["chunk"], dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != env_dim:
        raise ValueError(f"chunk shape {chunk.shape}, env action dim {env_dim}")
    _warn_nonfinite(chunk)
    times = [time.perf_counter() - t0]
    qpos = obs["robot"]["qpos"] if isinstance(obs, dict) and "robot" in obs else None
    actions = []
    for i in range(model.replan_steps):
        exe = chunk[i]
        if model.hyst is not None:
            # As eval(), except that every step of the chunk reads the gripper of the observation the
            # chunk was inferred on (the later ones are not known yet). The result differs from eval()
            # only if the hysteresis would become active (left gripper reading > arm_open) in the
            # middle of a chunk; once active it depends only on the model's commands.
            exe = model.hyst_step(obs, exe, exe)
        model.last_left = float(exe[GRIP_IDX[0]])
        model.t += 1
        a = torch.as_tensor(exe, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        # qpos is read only to fill a non-finite value before the first action; the chunk is finite
        actions.append(post.process(a, qpos=qpos))
    return actions, times


def feedback(model, obs, more):
    """The observation after one action of a planned chunk joins the frame history, as eval() pushes
    it before the next action; the observation after the last action is pushed with the next inference."""
    if more:
        model.push(obs, False, wait=not model.async_push)


def reset_model(model):
    model.reset()


def close_model(model):
    model.close()
