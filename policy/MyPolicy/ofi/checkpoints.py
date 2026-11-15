"""Find Hugging Face checkpoints and model weights in the local caches.

The evaluation never downloads: every checkpoint (including the drawer DPP
files), the SmolVLA backbone, the OfI V-JEPA encoder, the sample_loading rack
estimator and the torchvision
ResNet-18 weights come from the local caches that Setup step 5 (prefetch.py)
fills, and anything missing stops the run with an error naming it, whatever
HF_HUB_OFFLINE says. Only prefetch.py downloads (resolve_checkpoint with
complete=True, require_hub_file with download=True).
"""

import json
import os
from pathlib import Path

REQUIRED_FILES = ("config.json", "model.safetensors")
SMOLVLA_PROCESSORS = ("policy_preprocessor.json", "policy_postprocessor.json")
SETUP_HINT = "run Setup step 5 (policy/MyPolicy/prefetch.py) first"
# Files of a saved tokenizer directory (a checkpoint's tokenizer/) that transformers needs.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
# Files of the SmolVLA backbone repo (SmolVLM2) that the worker loads by name:
# model config and weights, processor and image processor, tokenizer.
BACKBONE_FILES = ("config.json", "processor_config.json", "preprocessor_config.json") + TOKENIZER_FILES


def offline():
    """True if Hugging Face downloads are turned off (HF_HUB_OFFLINE)."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def hub_cache():
    """The Hugging Face hub cache directory huggingface_hub actually uses."""
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


# An OfI (One from Infinity) checkpoint is not a LeRobot directory: its
# ofi_checkpoint.json names the files it consists of, with their sha256
# (see ofi_backend.py). The same parsing serves prefetch.py --check and the backend.
OFI_MANIFEST = "ofi_checkpoint.json"
OFI_FORMAT = "roboactualizer-ofi"
OFI_ROLES = ("weights", "train_config", "stats", "text_cache_file", "tasks")
# Source checkouts under OFI_SRC_ROOT (README Setup step 6): the RoboActualizer package,
# and the vjepa2 tree where torch.hub keeps it (otherwise RoboActualizer calls torch.hub.list).
OFI_SRC_MARKERS = (Path("RoboActualizer") / "src" / "Roboactualizer",
                   Path("torch_home") / "hub" / "facebookresearch_vjepa2_main" / "hubconf.py")


# A sample_loading head-B expert checkpoint (the expert planner + our two
# estimators) is not a LeRobot directory either: its sl_headb_checkpoint.json
# names its files with their sha256. Same parsing serves prefetch.py --check and
# the backend (sample_loading_headb/runtime.py).
SL_HEADB_MANIFEST = "sl_headb_checkpoint.json"
SL_HEADB_FORMAT = "robosyn-sl-headb"
SL_HEADB_ROLES = ("head_a_weights", "head_a_mounts", "head_b_weights",
                  "frozen_constants", "action_config_main", "action_config_fallback")


def sl_headb_manifest(checkpoint_dir):
    """({role: path}, {relative path: sha256}) of a sample_loading head-B checkpoint; raises ValueError on a malformed manifest."""
    d = Path(checkpoint_dir)
    try:
        m = json.loads((d / SL_HEADB_MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{d / SL_HEADB_MANIFEST} is not readable: {exc}") from None
    if not isinstance(m, dict) or m.get("format") != SL_HEADB_FORMAT:
        raise ValueError(f"{d / SL_HEADB_MANIFEST}: format is not {SL_HEADB_FORMAT!r}")
    files, sums = m.get("files") or {}, m.get("sha256") or {}
    missing = [k for k in SL_HEADB_ROLES if k not in files]
    if missing:
        raise ValueError(f"{d / SL_HEADB_MANIFEST}: no entry for {missing}")
    odd = [files[k] for k in SL_HEADB_ROLES
           if not isinstance(files[k], str) or Path(files[k]).as_posix() != files[k] or files[k].startswith(("/", "./", "../"))]
    if odd:
        raise ValueError(f"{d / SL_HEADB_MANIFEST}: file paths must be plain relative paths, got {odd}")
    nosum = [files[k] for k in SL_HEADB_ROLES if files[k] not in sums]
    if nosum:
        raise ValueError(f"{d / SL_HEADB_MANIFEST}: no sha256 for {nosum}")
    return {k: d / files[k] for k in SL_HEADB_ROLES}, sums


def _sl_headb_missing(d):
    try:
        paths, _ = sl_headb_manifest(d)
    except ValueError as exc:
        return [str(exc)]
    return [str(p.relative_to(d)) for p in paths.values() if not p.is_file()]


def verify_sl_headb_checkpoint(checkpoint_dir):
    """Every manifest file present and equal to its recorded sha256; returns {role: path}."""
    paths, sums = sl_headb_manifest(checkpoint_dir)
    bad = []
    for p in paths.values():
        rel = str(p.relative_to(checkpoint_dir))
        if not p.is_file():
            bad.append(f"{rel} missing")
        elif file_sha256(p) != sums[rel]:
            bad.append(f"{rel} sha256 differs from {SL_HEADB_MANIFEST} (delete {checkpoint_dir} and its blobs, then run Setup step 5 again)")
    if bad:
        raise RuntimeError(f"sample_loading head-B checkpoint {checkpoint_dir}: {bad}; {SETUP_HINT}")
    return paths


OFF_VALUES = ("", "null", "none", "off", "false", "0")
ON_VALUES = ("on", "true", "1")


def switch_value(v):
    """Optional setting given as YAML / --overrides value: None when off (null, false, 0, "off", ...),
    {} when simply on (true, 1, "on"), else the dict (a dict, or a str holding a Python / JSON dict)."""
    if v is None or v is False or (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0):
        return None
    if v is True or (isinstance(v, (int, float)) and v == 1):
        return {}
    if isinstance(v, str):
        t = v.strip()
        if t.lower() in OFF_VALUES:
            return None
        if t.lower() in ON_VALUES:
            return {}
        import ast
        try:
            v = ast.literal_eval(t)
        except (ValueError, SyntaxError):
            raise ValueError(f"cannot read setting value {v!r}: use null / off, on, or a dict") from None
    if not isinstance(v, dict):
        raise ValueError(f"setting value must be null / off, on, or a dict, got {v!r}")
    return v


def file_sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def ofi_manifest(checkpoint_dir):
    """({role: path}, {relative path: sha256}) of an OfI checkpoint; raises ValueError on a malformed manifest."""
    d = Path(checkpoint_dir)
    try:
        m = json.loads((d / OFI_MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{d / OFI_MANIFEST} is not readable: {exc}") from None
    if not isinstance(m, dict) or m.get("format") != OFI_FORMAT:
        raise ValueError(f"{d / OFI_MANIFEST}: format is not {OFI_FORMAT!r}")
    files, sums = m.get("files") or {}, m.get("sha256") or {}
    missing = [k for k in OFI_ROLES if k not in files]
    if missing:
        raise ValueError(f"{d / OFI_MANIFEST}: no entry for {missing}")
    odd = [files[k] for k in OFI_ROLES
           if not isinstance(files[k], str) or Path(files[k]).as_posix() != files[k] or files[k].startswith(("/", "./", "../"))]
    if odd:
        raise ValueError(f"{d / OFI_MANIFEST}: file paths must be plain relative paths, got {odd}")
    nosum = [files[k] for k in OFI_ROLES if files[k] not in sums]
    if nosum:
        raise ValueError(f"{d / OFI_MANIFEST}: no sha256 for {nosum}")
    return {k: d / files[k] for k in OFI_ROLES}, sums


def _ofi_missing(d):
    try:
        paths, _ = ofi_manifest(d)
    except ValueError as exc:
        return [str(exc)]
    return [str(p.relative_to(d)) for p in paths.values() if not p.is_file()]


def verify_ofi_checkpoint(checkpoint_dir):
    """Every manifest file present and equal to its recorded sha256; returns {role: path}."""
    paths, sums = ofi_manifest(checkpoint_dir)
    bad = []
    for p in paths.values():
        rel = str(p.relative_to(checkpoint_dir))
        if not p.is_file():
            bad.append(f"{rel} missing")
        elif file_sha256(p) != sums[rel]:
            bad.append(f"{rel} sha256 differs from {OFI_MANIFEST} (a damaged cache file: delete {checkpoint_dir} and its blobs, then run Setup step 5 again)")
    if bad:
        raise RuntimeError(f"OfI checkpoint {checkpoint_dir}: {bad}; {SETUP_HINT}")
    return paths


# An items_handover (ih) delivery checkpoint (head-A estimator + its constants)
# is not a LeRobot directory either: its insert_handover_checkpoint.json names
# its files with their sha256. Same parsing serves prefetch.py --check and the
# backend (insert_handover_dpp/runtime.py).
INSERT_HANDOVER_MANIFEST = "insert_handover_checkpoint.json"
INSERT_HANDOVER_FORMAT = "robosyn-insert-handover-dpp"
INSERT_HANDOVER_ROLES = ("head_a_weights", "head_a_mounts", "pen_roll", "frozen_constants")


def insert_handover_manifest(checkpoint_dir):
    """({role: path}, {relative path: sha256}) of an items_handover checkpoint; raises ValueError on a malformed manifest."""
    d = Path(checkpoint_dir)
    try:
        m = json.loads((d / INSERT_HANDOVER_MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{d / INSERT_HANDOVER_MANIFEST} is not readable: {exc}") from None
    if not isinstance(m, dict) or m.get("format") != INSERT_HANDOVER_FORMAT:
        raise ValueError(f"{d / INSERT_HANDOVER_MANIFEST}: format is not {INSERT_HANDOVER_FORMAT!r}")
    files, sums = m.get("files") or {}, m.get("sha256") or {}
    missing = [k for k in INSERT_HANDOVER_ROLES if k not in files]
    if missing:
        raise ValueError(f"{d / INSERT_HANDOVER_MANIFEST}: no entry for {missing}")
    odd = [files[k] for k in INSERT_HANDOVER_ROLES
           if not isinstance(files[k], str) or Path(files[k]).as_posix() != files[k] or files[k].startswith(("/", "./", "../"))]
    if odd:
        raise ValueError(f"{d / INSERT_HANDOVER_MANIFEST}: file paths must be plain relative paths, got {odd}")
    nosum = [files[k] for k in INSERT_HANDOVER_ROLES if files[k] not in sums]
    if nosum:
        raise ValueError(f"{d / INSERT_HANDOVER_MANIFEST}: no sha256 for {nosum}")
    return {k: d / files[k] for k in INSERT_HANDOVER_ROLES}, sums


def _insert_handover_missing(d):
    try:
        paths, _ = insert_handover_manifest(d)
    except ValueError as exc:
        return [str(exc)]
    return [str(p.relative_to(d)) for p in paths.values() if not p.is_file()]


def verify_insert_handover_checkpoint(checkpoint_dir):
    """Every manifest file present and equal to its recorded sha256; returns {role: path}."""
    paths, sums = insert_handover_manifest(checkpoint_dir)
    bad = []
    for p in paths.values():
        rel = str(p.relative_to(checkpoint_dir))
        if not p.is_file():
            bad.append(f"{rel} missing")
        elif file_sha256(p) != sums[rel]:
            bad.append(f"{rel} sha256 differs from {INSERT_HANDOVER_MANIFEST} (delete {checkpoint_dir} and its blobs, then run Setup step 5 again)")
    if bad:
        raise RuntimeError(f"items_handover checkpoint {checkpoint_dir}: {bad}; {SETUP_HINT}")
    return paths



def ofi_runtime(usr_args):
    """(worker python, source root) for OfI from deploy_policy.yml or OFI_PYTHON / OFI_SRC_ROOT; raises if unusable."""
    out = []
    for key, env_key in (("ofi_python", "OFI_PYTHON"), ("ofi_src_root", "OFI_SRC_ROOT")):
        v = usr_args.get(key)
        if v is None or str(v).strip().lower() in ("", "null", "none"):
            v = os.environ.get(env_key, "")
        if not str(v).strip():
            raise ValueError(f"{key} is not set (deploy_policy.yml / overrides) and {env_key} is empty; "
                             "see README Setup step 6")
        out.append(str(v))
    import shutil

    python, root = shutil.which(out[0]) or out[0], Path(out[1])   # a bare name such as python3.10 is looked up on PATH
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        raise FileNotFoundError(f"OfI worker python {python} is not executable; see README Setup step 6")
    for marker in OFI_SRC_MARKERS:
        if not (root / marker).exists():
            raise FileNotFoundError(f"{root / marker} not found under OFI_SRC_ROOT={root}; see README Setup step 6")
    return python, root


# A DPP checkpoint (drawer_open_place, dpp_backend.py) is not a LeRobot directory either:
# dpp_checkpoint.json names its files (estimator weights, camera mounts, handle link,
# frozen task constants, drawer geometry) with their sha256.
DPP_MANIFEST = "dpp_checkpoint.json"
DPP_FORMAT = "robosyn-drawer-dpp"
DPP_ROLES = ("weights", "wrist_mounts", "handle_link", "constants", "geometry")


def dpp_manifest(checkpoint_dir):
    """({role: path}, {relative path: sha256}) of a DPP checkpoint; raises ValueError on a malformed manifest."""
    d = Path(checkpoint_dir)
    try:
        m = json.loads((d / DPP_MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{d / DPP_MANIFEST} is not readable: {exc}") from None
    if not isinstance(m, dict) or m.get("format") != DPP_FORMAT:
        raise ValueError(f"{d / DPP_MANIFEST}: format is not {DPP_FORMAT!r}")
    files, sums = m.get("files") or {}, m.get("sha256") or {}
    missing = [k for k in DPP_ROLES if k not in files]
    if missing:
        raise ValueError(f"{d / DPP_MANIFEST}: no entry for {missing}")
    odd = [files[k] for k in DPP_ROLES
           if not isinstance(files[k], str) or Path(files[k]).as_posix() != files[k] or files[k].startswith(("/", "./", "../"))
           or ".." in Path(files[k]).parts]
    if odd:
        raise ValueError(f"{d / DPP_MANIFEST}: file paths must be plain relative paths, got {odd}")
    nosum = [files[k] for k in DPP_ROLES if files[k] not in sums]
    if nosum:
        raise ValueError(f"{d / DPP_MANIFEST}: no sha256 for {nosum}")
    return {k: d / files[k] for k in DPP_ROLES}, sums


def _dpp_missing(d):
    try:
        paths, _ = dpp_manifest(d)
    except ValueError as exc:
        return [str(exc)]
    return [str(p.relative_to(d)) for p in paths.values() if not p.is_file()]


def verify_dpp_checkpoint(checkpoint_dir):
    """Every manifest file present and equal to its recorded sha256; returns {role: path}."""
    paths, sums = dpp_manifest(checkpoint_dir)
    bad = []
    for p in paths.values():
        rel = str(p.relative_to(checkpoint_dir))
        if not p.is_file():
            bad.append(f"{rel} missing")
        elif file_sha256(p) != sums[rel]:
            bad.append(f"{rel} sha256 differs from {DPP_MANIFEST} (a damaged cache file: delete {checkpoint_dir} and its blobs, then run Setup step 5 again)")
    if bad:
        raise RuntimeError(f"DPP checkpoint {checkpoint_dir}: {bad}; {SETUP_HINT}")
    return paths


def missing_files(checkpoint_dir):
    """Files a checkpoint directory needs but does not have (empty list = complete).

    OfI checkpoints (ofi_checkpoint.json present) and DPP checkpoints
    (dpp_checkpoint.json present): every file the manifest lists.
    LeRobot checkpoints:

    Always config.json and model.safetensors; for SmolVLA also both processor
    files, every state file they name and every artifact they name by a
    relative path (the tokenizer directory). A download cut off after some
    files leaves a snapshot that is caught here.
    """
    d = Path(checkpoint_dir)
    if (d / OFI_MANIFEST).is_file():
        return _ofi_missing(d)
    if (d / DPP_MANIFEST).is_file():
        return _dpp_missing(d)
    if (d / SL_HEADB_MANIFEST).is_file():
        return _sl_headb_missing(d)
    if (d / INSERT_HANDOVER_MANIFEST).is_file():
        return _insert_handover_missing(d)
    if (d / "pretrained_model").is_dir():
        d = d / "pretrained_model"
    missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
    try:
        is_smolvla = json.loads((d / "config.json").read_text()).get("type") == "smolvla"
    except (OSError, ValueError, AttributeError):  # missing or not JSON: presence check only; the backend reports the rest
        return missing
    processors = SMOLVLA_PROCESSORS if is_smolvla else tuple(p.name for p in sorted(d.glob("policy_*processor.json")))
    for name in processors:
        if not (d / name).is_file():
            missing.append(name)
            continue
        try:
            steps = json.loads((d / name).read_text()).get("steps", [])
        except (OSError, ValueError, AttributeError):
            missing.append(f"{name} (not readable)")
            continue
        for step in steps:
            state = step.get("state_file")
            if state and not (d / state).is_file():
                missing.append(state)
            for key, value in (step.get("artifacts") or {}).items():
                if not isinstance(value, str) or "/" in value:  # a Hub repo id, loaded by name
                    continue
                if not (d / value).exists():
                    missing.append(value)
                elif key == "tokenizer_name":
                    missing += [f"{value}/{f}" for f in TOKENIZER_FILES if not (d / value / f).is_file()]
    return missing


_LISTINGS = {}


def _subfolder(subfolder):
    """None, or a plain relative directory inside the repository (no "..", no leading "/")."""
    if subfolder in (None, ""):
        return None
    p = Path(str(subfolder))
    if p.is_absolute() or ".." in p.parts or str(p) in (".", ""):
        raise ValueError(f"subfolder must be a plain relative directory, got {subfolder!r}")
    return p.as_posix()


def resolve_checkpoint(repo_id, revision, complete=False, subfolder=None):
    """Return a local directory holding `repo_id` at `revision`.

    complete=False (evaluation): use the local Hugging Face cache if it holds a
    complete checkpoint (see missing_files); anything else is an error (the
    evaluation never downloads). complete=True (prefetch.py): list the repo's files at `revision` and
    download every one that is not in the cache yet, file by file (more robust
    than snapshot_download behind mirrors that page listings).

    subfolder: the checkpoint is that directory of the repository (one repository
    holds one directory per task); only its files are downloaded and checked, and
    the returned path is that directory.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if not revision:
        raise ValueError(f"No revision pinned for {repo_id}; refusing to use a moving branch.")
    sub = _subfolder(subfolder)
    where = f"{repo_id}@{revision}" + (f"/{sub}" if sub else "")

    if not complete:
        missing, lookup = list(REQUIRED_FILES), ""
        try:
            cached = snapshot_download(repo_id, revision=revision, local_files_only=True)
            ckpt = str(Path(cached) / sub) if sub else cached
            # one repository holds every task: another task's files can be cached while this
            # task's directory is not there at all; say so instead of naming LeRobot files
            missing = [f"{sub}/ (the whole directory)"] if sub and not Path(ckpt).is_dir() else missing_files(ckpt)
            if not missing:
                return ckpt
        except Exception as exc:  # not cached, or the cache could not be read: say which
            lookup = f"; cache lookup: {type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"{where} is not complete in the local Hugging Face cache "
            f"({hub_cache()}; missing {missing}{lookup}); the evaluation does not download; {SETUP_HINT}."
        )
    if offline():
        raise RuntimeError("prefetch.py downloads; unset HF_HUB_OFFLINE (and TRANSFORMERS_OFFLINE) for it.")

    key = (repo_id, revision)
    if key not in _LISTINGS:   # one listing per repository and revision (all tasks share one repository)
        _LISTINGS[key] = HfApi().list_repo_files(repo_id, revision=revision)
    files = _LISTINGS[key]
    if sub:
        files = [f for f in files if f.startswith(sub + "/")]
    if not files:
        raise RuntimeError(f"{where} has no files.")
    root = None
    for name in files:
        local = Path(hf_hub_download(repo_id, name, revision=revision))
        root = local.parents[len(Path(name).parts) - 1]
    ckpt = root / sub if sub else root
    if complete:  # prefetch.py: what it downloaded must be a complete checkpoint
        missing = missing_files(ckpt)
        if missing:
            raise RuntimeError(f"{where} downloaded but still missing {missing}.")
    return str(ckpt)


def require_hub_snapshot(repo_id, revision):
    """The pinned SmolVLA backbone (loaded by name in the worker) must be in the cache; the evaluation does not download it."""
    from huggingface_hub import snapshot_download

    try:
        cached = Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
    except Exception:
        cached = None
    if cached is None:
        missing = list(BACKBONE_FILES) + ["*.safetensors"]
    else:
        missing = [f for f in BACKBONE_FILES if not (cached / f).is_file()]
        if not any(cached.glob("*.safetensors")):
            missing.append("*.safetensors")
    if missing:
        raise RuntimeError(
            f"{repo_id}@{revision} (SmolVLA backbone) is not complete in the local Hugging Face cache "
            f"({hub_cache()}; missing {missing}); the evaluation does not download; {SETUP_HINT}."
        )
    return str(cached)


def require_hub_file(repo_id, revision, filename, what, sha256=None, download=False):
    """Local path of one pinned Hugging Face file (e.g. the OfI V-JEPA encoder): cache only, or error.

    download=True (prefetch.py only) fetches it when it is not cached. With `sha256`, the file must also
    have that sha256.
    """
    from huggingface_hub import hf_hub_download

    if not revision:
        raise ValueError(f"No revision pinned for {repo_id}; refusing to use a moving branch.")
    try:
        path = hf_hub_download(repo_id, filename, revision=revision, local_files_only=True)
    except Exception:
        if not download:
            raise RuntimeError(
                f"{repo_id}@{revision}/{filename} ({what}) is not in the local Hugging Face cache "
                f"({hub_cache()}); the evaluation does not download; {SETUP_HINT}."
            ) from None
        if offline():
            raise RuntimeError("prefetch.py downloads; unset HF_HUB_OFFLINE (and TRANSFORMERS_OFFLINE) for it.") from None
        path = hf_hub_download(repo_id, filename, revision=revision)
    if sha256 and file_sha256(path) != sha256:
        raise RuntimeError(f"{repo_id}@{revision}/{filename} ({what}) sha256 differs from {sha256}: the cached file "
                           f"{path} is damaged; delete it (and the blob it links to), then {SETUP_HINT}.")
    return path


def torch_backbone(checkpoint_dir):
    """(weights name, cached file) of the torchvision backbone a checkpoint's config.json names, or None.

    LeRobot ACT builds its ResNet backbone with `pretrained_backbone_weights`
    (e.g. ResNet18_Weights.IMAGENET1K_V1); torchvision then loads that file from
    the torch hub cache (TORCH_HOME), downloading it if it is missing.
    """
    d = Path(checkpoint_dir)
    if (d / "pretrained_model").is_dir():
        d = d / "pretrained_model"
    if not (d / "config.json").is_file():  # not a LeRobot checkpoint; the backend reports that itself
        return None
    name = json.loads((d / "config.json").read_text()).get("pretrained_backbone_weights")
    if not name:
        return None
    import torch
    from torchvision.models import get_weight

    return name, Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(get_weight(name).url)


def require_torch_backbone(checkpoint_dir):
    """Refuse to start if torchvision would have to download the backbone weights."""
    found = torch_backbone(checkpoint_dir)
    if found is not None and not found[1].is_file():
        raise RuntimeError(f"{found[1]} is not in the torch hub cache (TORCH_HOME); {SETUP_HINT}.")
    return None if found is None else found[1]
