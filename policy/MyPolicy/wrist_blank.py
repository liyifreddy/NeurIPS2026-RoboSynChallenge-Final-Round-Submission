# ----------------------------------------------------------------------------
# Optional: replace the student's wrist images (cam_left_wrist / cam_right_wrist), top view unchanged.
#
# Off by default. Mode from the environment variable WRIST_BLANK (evaluation runs; it wins when set)
# or from the task's configuration (configs/<task>.yml, `wrist_blank:`):
#   "1" / "black" / true  -> all zeros (black)
#   "gray" / "grey"       -> the training images' normalisation mean: the worker normalises uint8
#                            frames as (x / 255 - 0.5) / 0.5, so 128 (uint8) is ~0 after normalisation
#   "0" / "" / false      -> off
# The replaced copy has the same keys, shapes and dtypes; nothing else in obs changes.
# ----------------------------------------------------------------------------
import os

WRIST_CAMS = ("cam_left_wrist", "cam_right_wrist")
MODES = ("black", "gray")


def _mode(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "black" if v else None
    s = str(v).strip().lower()
    if s in ("", "0", "false", "off", "no"):
        return None
    if s in ("1", "true", "on", "yes", "black"):
        return "black"
    if s in ("gray", "grey"):
        return "gray"
    raise ValueError(f"wrist_blank / WRIST_BLANK must be 0 / 1 / black / gray, got {v!r}")


def enabled(entry):
    """The mode for this task: None (off), 'black' or 'gray'."""
    v = os.environ.get("WRIST_BLANK")
    if v is not None and v.strip() != "":
        return _mode(v)
    return _mode((entry or {}).get("wrist_blank", None))


def _fill_like(x, mode):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if mode == "black":
                return torch.zeros_like(x)
            v = 128 if x.dtype == torch.uint8 else 127.5
            return torch.full_like(x, v)
    except ImportError:
        pass
    if mode == "black":
        return np.zeros_like(x)
    v = 128 if np.asarray(x).dtype == np.uint8 else 127.5
    return np.full_like(x, v)


def blank_obs(obs, mode="black"):
    """A copy of obs whose wrist colour images are replaced (top view and everything else shared)."""
    if not mode or not isinstance(obs, dict) or not isinstance(obs.get("sensor"), dict):
        return obs
    sensor = dict(obs["sensor"])
    for cam in WRIST_CAMS:
        c = sensor.get(cam)
        if isinstance(c, dict) and "color" in c:
            c = dict(c)
            c["color"] = _fill_like(c["color"], mode)
            sensor[cam] = c
    out = dict(obs)
    out["sensor"] = sensor
    return out


class BlankWristEnv:
    """Env proxy for the backend: attribute pass-through; step() returns obs with replaced wrist images."""

    def __init__(self, env, mode="black"):
        object.__setattr__(self, "_b_env", env)
        object.__setattr__(self, "_b_mode", mode)

    def __getattr__(self, name):
        return getattr(self._b_env, name)

    def __setattr__(self, name, value):
        setattr(self._b_env, name, value)

    def step(self, action):
        out = self._b_env.step(action)
        return (blank_obs(out[0], self._b_mode),) + tuple(out[1:])
