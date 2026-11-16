#!/usr/bin/env python
"""Required before evaluating (Setup step 5): download everything the evaluation needs.

    python policy/MyPolicy/prefetch.py [--tasks TASK ...]
    python policy/MyPolicy/prefetch.py --check [--tasks TASK ...]   # offline: is everything there?

As in the preliminary delivery: puts the pinned checkpoint of every OfI task in
deploy_policy.yml (task_policies -> configs/<task>.yml: hf_repo + revision + subfolder), the pinned V-JEPA
encoder (ofi_backbone) and, for every task whose helper is enabled, the files it names (every helper.<name>_hf:
head_hf, encoder_hf, tip_head_hf) into the local Hugging Face cache, at exactly the pinned revisions.
This is the only place that downloads: the evaluation only reads the cache and stops with
an error if anything is missing. Every OfI checkpoint is checked against the sha256 in its
ofi_checkpoint.json, the encoder against deploy_policy.yml. Respects HF_HOME, HF_ENDPOINT
and HF_TOKEN (the checkpoint repository is gated: log in with an account that has access).
Safe to rerun; exits non-zero if anything is missing.
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ofi"))
sys.path.insert(0, str(HERE))
from checkpoints import require_hub_file, resolve_checkpoint, verify_ofi_checkpoint  # noqa: E402
from task_config import task_entry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--check", action="store_true", help="offline: report what is missing, download nothing")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / "deploy_policy.yml"))
    table = cfg.get("task_policies") or {}
    tasks = a.tasks or sorted(table)
    todo, bad = [], []
    for t in tasks:
        e = task_entry(table, t, HERE) if t in table else {}
        if e.get("backend") != "ofi":
            print(f"[prefetch] {t}: backend {e.get('backend')!r}, nothing to fetch")
            continue
        todo.append(("checkpoint", t, e))
        h = e.get("helper") or {}
        if not h.get("enabled", False):   # a switched-off helper is not built, so its files are not needed
            h = {}
        for key in [k for k in h if k.endswith("_hf")]:   # head_hf, encoder_hf, tip_head_hf, ...
            if h.get(key):
                todo.append(("file", f"{t} helper {key}", h[key]))
    bb = cfg.get("ofi_backbone") or {}
    if any(k == "checkpoint" for k, _, _ in todo) and bb.get("hf_repo"):
        todo.append(("file", "ofi_backbone", dict(bb)))
    for i, (kind, name, e) in enumerate(todo, 1):
        try:
            if kind == "checkpoint":
                d = resolve_checkpoint(e["hf_repo"], str(e["revision"]), complete=not a.check, subfolder=e.get("subfolder"))
                verify_ofi_checkpoint(Path(d))
            else:
                require_hub_file(e["hf_repo"], str(e["revision"]), e["file"], name, sha256=e.get("sha256"), download=not a.check)
            print(f"[prefetch] {i} / {len(todo)} {name}: ok")
        except Exception as exc:  # report every item, then fail
            bad.append(name)
            print(f"[prefetch] {i} / {len(todo)} {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    mode = "check" if a.check else "done"
    print(f"[prefetch] {mode}: {len(todo) - len(bad)} of {len(todo)}" + (" in the local caches" if a.check else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
