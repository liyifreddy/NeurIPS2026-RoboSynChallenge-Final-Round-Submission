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

import torch

from .action_post import ActionPost, GuardedEnv, default_log_path, parse_cfg
from .task_config import task_entry
from . import wrist_blank as _wb

POLICY_NAME = "MyPolicy"
# Keys of a task entry that are routing information, not backend arguments.
_ROUTE_KEYS = ("backend", "overrides", "helper", "wrist_blank")


class RoutedModel:
    __slots__ = ("backend", "inner", "task", "backend_name", "post", "seed_pending", "helper", "helper_env",
                 "wrist_blank")

    def __init__(self, backend, inner, task, backend_name, post, helper=None, wrist_blank=None):
        self.backend = backend
        self.inner = inner
        self.task = task
        self.backend_name = backend_name
        self.post = post
        self.seed_pending = True
        self.helper, self.helper_env = helper if helper is not None else (None, None)
        self.wrist_blank = wrist_blank


# Per-task helpers live in helpers/<task>/ (each folder self-contained). helper.module / helper.class /
# helper.env_class name the helper module, its helper class (cfg, log_dir=None) with start_episode /
# end_episode / observe / act, and its env proxy class (env, helper, obs); the defaults below apply when
# they are not given.
_HELPER_DEFAULTS = {
    "click_bell": ("bell_press", "BellPress", "BellHelperEnv"),
    "sample_loading": ("sample_hold", "SampleHold", "SampleHelperEnv"),
}


def _build_helper(task, entry, log_path):
    """task_policies.<task>.helper: {enabled: true, ...}; absent or enabled false = no helper (unchanged behaviour).
    Returns (helper, env_class) or None."""
    cfg = entry.get("helper") or {}
    if not cfg.get("enabled", False):
        return None
    d_mod, d_cls, d_env = _HELPER_DEFAULTS.get(task, (None, None, None))
    mod_name, cls_name, env_name = cfg.get("module", d_mod), cfg.get("class", d_cls), cfg.get("env_class", d_env)
    if not (mod_name and cls_name and env_name):
        raise ValueError(f"{task}: helper enabled but helper.module / class / env_class not given")
    import importlib

    from .ofi import checkpoints as _ck

    mod = importlib.import_module(f"{__package__}.helpers.{task}.{mod_name}")
    # head_hf / encoder_hf: {hf_repo, revision, file} pinned like the OfI checkpoints, read from the
    # local Hugging Face cache filled by prefetch.py (the evaluation does not download); the helper
    # then gets plain file paths in head / encoder and checks the head sha256 itself.
    cfg = {k: v for k, v in cfg.items() if k not in ("module", "class", "env_class")}
    for k in [k for k in cfg if k.endswith("_hf")]:   # head_hf, encoder_hf, tip_head_hf, ... -> head, encoder, tip_head
        hub, key = cfg.pop(k), k[:-3]
        if hub:
            cfg[key] = _ck.require_hub_file(hub["hf_repo"], str(hub["revision"]), hub["file"], f"{task} helper {key}")
    log_dir = cfg.get("log_dir") or (os.path.dirname(log_path) if log_path else None)
    h = getattr(mod, cls_name)(cfg, log_dir=log_dir)
    if "." in env_name:   # e.g. sample_hold.SampleHelperEnv: the env proxy lives in another module of the same folder
        env_mod, env_name = env_name.rsplit(".", 1)
        env_cls = getattr(importlib.import_module(f"{__package__}.helpers.{task}.{env_mod}"), env_name)
    else:
        env_cls = getattr(mod, env_name)
    c = getattr(h, "c", {}) or {}
    print(f"[{POLICY_NAME}] helper={mod_name} platform={c.get('platform')} z0={c.get('z0')} log_dir={log_dir}", flush=True)
    return h, env_cls


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
    wb = _wb.enabled(entry)
    if wb:
        print(f"[{POLICY_NAME}] wrist_blank={wb}: the student gets {wb} wrist images (top view unchanged)", flush=True)
    return RoutedModel(backend, inner, task, backend_name, post, helper, wb)


def _episode_seed(env):
    for owner in (env, getattr(env, "unwrapped", None)):
        s = getattr(owner, "_robosyn_episode_seed", None)
        if s is not None:
            try:
                return int(s)
            except (TypeError, ValueError):
                return None
    return None


def _chunked(model):
    """Tasks whose actions are all fixed when the chunk is inferred: no helper, no wrist blanking,
    and a backend that can hand over a whole planned chunk (OfI without hysteresis / correction)."""
    chunkable = getattr(model.backend, "chunkable", None)
    return (model.helper is None and not model.wrist_blank and chunkable is not None
            and chunkable(model.inner))


def eval(env, model, obs):
    # Real station (organizers' rsc_real): one observation in, one chunk of actions out, all executed,
    # each action's real observation fed back (chunks.ChunkBridge rewrites the loop below to declare
    # `actions` first). The actions are post-processed (action_post) before the loop, so the values
    # sent are exactly the declared ones; the step order, the frames the model sees and the actions
    # are those of the per-step path. Tasks whose actions read each step's observation (helpers,
    # gripper hysteresis) keep the per-step path.
    if model.seed_pending:
        model.post.seed = _episode_seed(env)
        model.seed_pending = False
    if not _chunked(model):
        return _eval_per_step(env, model, obs)
    actions, times = model.backend.plan(env, model.inner, obs, model.post)
    info, truncated, n = None, False, 0
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        n += 1
        model.backend.feedback(model.inner, obs, n < len(actions))
        if env.get_wrapper_attr('is_task_success')():
            break
        if torch.as_tensor(truncated).any():
            break
    return obs, info, truncated, times


def _eval_per_step(env, model, obs):
    """Every action goes through GuardedEnv (action_post) as it is sent."""
    genv = GuardedEnv(env, model.post, obs)
    if model.helper is not None:
        # outside GuardedEnv: the helper's action still goes through action_post
        genv = model.helper_env(genv, model.helper, obs)
    if model.wrist_blank:
        # outermost, next to the backend: only the student sees black wrist images
        return model.backend.eval(_wb.BlankWristEnv(genv, model.wrist_blank), model.inner,
                                  _wb.blank_obs(obs, model.wrist_blank))
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
