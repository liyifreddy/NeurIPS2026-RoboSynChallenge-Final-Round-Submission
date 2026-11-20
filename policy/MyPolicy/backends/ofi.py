# ----------------------------------------------------------------------------
# OfI backend for the finals: the preliminary One-from-Infinity backend
# (ofi/ofi_backend.py: b50f70a plus real-station serving and the chunk interface
# plan / feedback, see its header) behind a small adapter that finds the task's
# checkpoint directory.
#
# deploy_policy.yml, per task:
#   ofi_checkpoint_dir: <local dir holding ofi_checkpoint.json>
#   -- or --
#   hf_repo / revision / subfolder                                  (preliminary way: local HF cache, prefetch)
# and, for the frozen V-JEPA 2.1 encoder:
#   ofi_backbone: {path, sha256}              local file, or
#   ofi_backbone: {hf_repo, revision, file, sha256}   (preliminary way)
# Everything the preliminary backend checks (manifest sha256, strict weight load,
# worker python / sources) is unchanged.
# ----------------------------------------------------------------------------

import hashlib
import os
from pathlib import Path

from ..ofi import checkpoints as _ck
from ..ofi import ofi_backend as _ofi


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _local_backbone(bb):
    p = Path(os.path.expandvars(str(bb["path"]))).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"ofi_backbone.path {p} not found")
    if not bb.get("sha256"):
        raise ValueError("ofi_backbone.path needs its sha256 in deploy_policy.yml")
    if _sha256(p) != bb["sha256"]:
        raise RuntimeError(f"ofi_backbone {p}: sha256 differs from deploy_policy.yml")
    return p


def resolve_checkpoint_dir(usr_args):
    d = usr_args.get("ofi_checkpoint_dir")
    if d:
        p = Path(os.path.expandvars(str(d))).expanduser().resolve()
        _ck.verify_ofi_checkpoint(p)  # manifest roles present, sha256 as recorded
        return p
    if usr_args.get("hf_repo") and usr_args.get("revision"):
        return Path(_ck.resolve_checkpoint(usr_args["hf_repo"], str(usr_args["revision"]),
                                           subfolder=usr_args.get("subfolder"))).resolve()
    raise ValueError("OfI task needs ofi_checkpoint_dir, or hf_repo + revision (+ subfolder)")


def get_model(usr_args):
    usr_args["checkpoint_dir"] = str(resolve_checkpoint_dir(usr_args))
    usr_args["checkpoint_path"] = usr_args["checkpoint_dir"]
    bb = usr_args.get("ofi_backbone") or {}
    if bb.get("path"):
        local = _local_backbone(bb)
        # the preliminary backend asks require_hub_file for the encoder; answer with the local file
        _ofi.require_hub_file = lambda *a, **k: local
        usr_args["ofi_backbone"] = {"hf_repo": "local", "revision": "local", "file": str(local),
                                    "sha256": bb["sha256"]}
    return _ofi.get_model(usr_args)


eval = _ofi.eval
reset_model = _ofi.reset_model
close_model = _ofi.close_model
chunkable = _ofi.chunkable
plan = _ofi.plan
feedback = _ofi.feedback
