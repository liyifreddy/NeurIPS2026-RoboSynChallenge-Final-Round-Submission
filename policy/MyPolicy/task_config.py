# ----------------------------------------------------------------------------
# Per-task configuration files (configs/<task>.yml).
#
# deploy_policy.yml keeps the official keys and lists each task under
# task_policies as {config: configs/<task>.yml}; that file holds the task's
# backend, checkpoint pin (hf_repo + revision + subfolder), overrides and helper.
# Keys written next to `config:` in deploy_policy.yml override the file.
# An entry without `config:` is used as it is (tests, quick experiments).
# ----------------------------------------------------------------------------
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent


def task_entry(table, task, policy_dir=POLICY_DIR):
    """The resolved task_policies entry of `task` (dict), or KeyError if the task is not listed."""
    entry = dict(table[task] or {})
    ref = entry.pop("config", None)
    if ref is None:
        return entry
    import yaml

    p = (Path(policy_dir) / str(ref)).resolve()
    if Path(policy_dir).resolve() not in p.parents:
        raise ValueError(f"task_policies.{task}.config must point inside policy/MyPolicy, got {ref!r}")
    base = yaml.safe_load(p.read_text()) or {}
    if not isinstance(base, dict):
        raise ValueError(f"{p} is not a mapping")
    base.update(entry)
    return base
