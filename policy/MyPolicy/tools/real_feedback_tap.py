"""Measurement wrapper of ofi_worker.py for tools/check_real_feedback.py (checks only).

ofi_backend starts `python <WORKER> args...`; the check points WORKER here and sets
TAP_TARGET=<the checkout's ofi_worker.py> and TAP_OUT=<jsonl>. This loads that worker file
unchanged, wraps OfIRunner.push / infer and runs its main(). Per push it logs the time to add the
frame to the history; per inference the sha256 of the model input (frame-history clip, qpos) and
of the returned chunk.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

target = os.environ["TAP_TARGET"]
out = open(os.environ["TAP_OUT"], "a", buffering=1)
sys.path.insert(0, os.path.dirname(target))
spec = importlib.util.spec_from_file_location("ofi_worker", target)
mod = importlib.util.module_from_spec(spec)
sys.modules["ofi_worker"] = mod
spec.loader.exec_module(mod)
R = mod.OfIRunner
_push, _infer = R.push, R.infer
count = {"push": 0, "infer": 0}


def _sha(t):
    import numpy as np
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()


def push(self, cams):
    t0 = time.perf_counter()
    r = _push(self, cams)
    out.write(json.dumps({"ev": "push", "i": count["push"], "s": time.perf_counter() - t0,
                          "frame_sha": _sha(self.frames[-1])}) + "\n")
    count["push"] += 1
    return r


def infer(self, qpos, steps):
    c = self.clip()
    rec = {"ev": "infer", "i": count["infer"], "push_i": count["push"] - 1,
           "clip_sha": _sha(c), "clip_shape": list(c.shape), "qpos_sha": _sha(qpos)}
    r = _infer(self, qpos, steps)
    rec["chunk_sha"] = _sha(r)
    out.write(json.dumps(rec) + "\n")
    count["infer"] += 1
    return r


R.push, R.infer = push, infer
mod.main()
