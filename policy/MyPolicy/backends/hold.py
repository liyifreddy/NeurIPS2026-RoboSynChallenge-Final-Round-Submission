# ----------------------------------------------------------------------------
# hold backend: commands the current joint positions back (the arms stay where
# they are). No model, no GPU. Used to test the plumbing (deploy_policy.py,
# action_post.py, eval.sh) and in the from-scratch rehearsal before the model
# weights are in place. Never routed to a task in a submission.
#
# hold_chunk: actions per eval() call (one "inference" per call, like a chunked model).
# hold_offset: optional [14] added to the held qpos (tests drive it into the limits).
# ----------------------------------------------------------------------------

import time

import numpy as np


class HoldModel:
    def __init__(self, usr_args):
        self.chunk = int(usr_args.get("hold_chunk", 10))
        off = usr_args.get("hold_offset")
        self.offset = None if off is None else np.asarray(off, dtype=np.float64).reshape(-1)
        self.inference_device = None
        self.calls = 0


def _qpos(obs):
    q = obs["robot"]["qpos"]
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float64)
    return q[0] if q.ndim == 2 else q.reshape(-1)


def _done(x):
    if hasattr(x, "any"):
        return bool(x.any())
    return bool(x)


def get_model(usr_args):
    return HoldModel(usr_args)


def eval(env, model, obs):
    t0 = time.perf_counter()
    target = _qpos(obs).copy()
    if model.offset is not None:
        target = target + model.offset
    times = [time.perf_counter() - t0]
    model.calls += 1
    final_obs, info, truncated = obs, None, False
    for _ in range(model.chunk):
        a = target.reshape(1, -1).astype(np.float32)
        try:
            import torch

            a = torch.as_tensor(a, device=getattr(env.unwrapped, "device", "cpu"))
        except ImportError:
            pass
        final_obs, _r, _term, truncated, info = env.step(a)
        if env.get_wrapper_attr("is_task_success")():
            break
        if _done(truncated):
            break
    return final_obs, info, truncated, times


def reset_model(model):
    model.calls = 0
