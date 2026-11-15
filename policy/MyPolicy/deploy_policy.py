# ----------------------------------------------------------------------------
# MyPolicy (finals): per-task router + action post-processing at the exit.
#
# Official adapter interface (scripts/eval_policy.py of the RoboSynChallenge
# final version; FAQ #37):
#   get_model(usr_args) -> model
#   eval(env, model, obs) -> (obs, info, truncated, inference_times)
#   reset_model(model) -> None
#   close_model(model) -> None   (optional)
#
# Which backend runs a task is configuration (deploy_policy.yml: task_policies,
# backends). Every backend calls env.step() itself, as the official adapters do;
# eval() hands it a GuardedEnv, so every action that leaves policy/ goes through
# action_post.ActionPost (real-robot joint limits, gripper 0-1, counted and
# logged). The backend code is not changed for this.
#
# FAQ #12-#14: 10 Hz, 1000 steps per task, inference and execution serial.
# FAQ #16-#17: the SDK sends our actions as they are; rules here read only obs.
# ----------------------------------------------------------------------------

import os

# SimulationManager.destroy() reads this at call time; without it env.close()
# exits before evaluation_metrics.json is written (preliminary lesson).
os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")

import importlib
import sys

from .action_post import ActionPost, GuardedEnv, default_log_path, parse_cfg
from .task_config import task_entry

POLICY_NAME = "MyPolicy"
# Keys of a task entry that are routing information, not backend arguments.
_ROUTE_KEYS = ("backend", "overrides", "helper")


class RoutedModel:
    __slots__ = ("backend", "inner", "task", "backend_name", "post", "seed_pending", "helper")

    def __init__(self, backend, inner, task, backend_name, post, helper=None):
        self.backend = backend
        self.inner = inner
        self.task = task
        self.backend_name = backend_name
        self.post = post
        self.seed_pending = True
        self.helper = helper


def _build_helper(task, entry, log_path):
    """task_policies.<task>.helper: {enabled: true, ...}; absent or enabled false = no helper (unchanged behaviour)."""
    cfg = entry.get("helper") or {}
    if not cfg.get("enabled", False):
        return None
    if task != "click_bell":
        raise ValueError(f"helper is only implemented for click_bell, not {task!r}")
    from .helpers.click_bell.bell_press import BellPress
    from .ofi import checkpoints as _ck

    # head_hf / encoder_hf: {hf_repo, revision, file} pinned like the OfI checkpoints, read from the
    # local Hugging Face cache filled by prefetch.py (the evaluation does not download); the helper
    # then gets plain file paths in head / encoder and checks the head sha256 itself.
    cfg = dict(cfg)
    for key, what in (("head", "bell helper head"), ("encoder", "bell helper encoder")):
        hub = cfg.get(f"{key}_hf")
        if hub:
            cfg[key] = _ck.require_hub_file(hub["hf_repo"], str(hub["revision"]), hub["file"], what)
    log_dir = cfg.get("log_dir") or (os.path.dirname(log_path) if log_path else None)
    h = BellPress(cfg, log_dir=log_dir)
    print(f"[{POLICY_NAME}] helper=bell_press platform={h.c['platform']} z0={h.c['z0']} log_dir={log_dir}", flush=True)
    return h


def _task_entry(usr_args):
    task = usr_args.get("task_name")
    table = usr_args.get("task_policies") or {}
    if task not in table:
        raise ValueError(f"No policy configured for task {task!r} in deploy_policy.yml "
                         f"(task_policies has: {sorted(table)}).")
    return task, task_entry(table, task)


def get_model(usr_args):
    task, entry = _task_entry(usr_args)
    backend_name = entry.get("backend")
    backends = usr_args.get("backends") or {}
    if backend_name not in backends:
        raise ValueError(f"Task {task!r} uses backend {backend_name!r}, which is not listed under "
                         f"backends in deploy_policy.yml (have: {sorted(backends)}).")
    # The task entry's own keys (checkpoint location etc.) and its overrides go to the backend.
    # usr_args is the dict eval_policy.py writes into evaluation_metrics.json, so what ran is recorded.
    for k, v in entry.items():
        if k not in _ROUTE_KEYS:
            usr_args[k] = v
    for k, v in (entry.get("overrides") or {}).items():
        usr_args[k] = v
    post_cfg = parse_cfg(usr_args.get("action_post"))
    log_path = default_log_path(usr_args, task) if post_cfg["enabled"] else None
    post = ActionPost(post_cfg, task=task, log_path=log_path)
    print(f"[{POLICY_NAME}] task={task} backend={backend_name} ({backends[backend_name]}) "
          f"action_post={'on' if post_cfg['enabled'] else 'OFF'} profile={post_cfg['clip_profile']} "
          f"gripper_src_range={post_cfg['gripper_src_range']} log={log_path}", flush=True)
    if not post_cfg["enabled"]:
        print(f"[{POLICY_NAME}] WARNING: action_post is off; nothing keeps the actions inside the "
              "real-robot joint limits (FAQ #16)", file=sys.stderr, flush=True)
    backend = importlib.import_module(backends[backend_name])
    inner = backend.get_model(usr_args)
    helper = _build_helper(task, entry, log_path)
    return RoutedModel(backend, inner, task, backend_name, post, helper)


def _episode_seed(env):
    for owner in (env, getattr(env, "unwrapped", None)):
        s = getattr(owner, "_robosyn_episode_seed", None)
        if s is not None:
            try:
                return int(s)
            except (TypeError, ValueError):
                return None
    return None


def eval(env, model, obs):
    if model.seed_pending:
        model.post.seed = _episode_seed(env)
        model.seed_pending = False
    genv = GuardedEnv(env, model.post, obs)
    if model.helper is not None:
        # outside GuardedEnv: the helper's action still goes through action_post
        from .helpers.click_bell.bell_press import BellHelperEnv

        genv = BellHelperEnv(genv, model.helper, obs)
    return model.backend.eval(genv, model.inner, obs)


def reset_model(model):
    model.post.start_episode()
    model.seed_pending = True
    if model.helper is not None:
        model.helper.end_episode()
        model.helper.start_episode()
    model.backend.reset_model(model.inner)


def close_model(model):
    model.post.end_episode()
    if model.helper is not None:
        model.helper.end_episode()
    close = getattr(model.backend, "close_model", None)
    if close is not None:
        close(model.inner)
