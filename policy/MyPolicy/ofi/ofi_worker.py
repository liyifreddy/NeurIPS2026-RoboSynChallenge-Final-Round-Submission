"""One from Infinity (RoboActualizer @ 85b9dc6) inference worker for RoboSynChallenge.

Runs in the OfI worker environment (README Setup step 6, torch 2.7.1+cu128), started by
ofi_backend.py (which runs in the simulator environment). Protocol on stdin/stdout, one request -> one reply:
  request : one JSON line, optionally followed by `nbytes` raw bytes
  reply   : one JSON line
Commands:
  hello                      -> model/ckpt/stats/text info (after strict checkpoint check)
  seed {seed}                -> seeds torch/numpy/random (per episode)
  reset                      -> clears the frame history
  push {infer, shapes, qpos} + raw uint8 frames (cam_high, cam_left_wrist, cam_right_wrist; HWC)
                             -> appends the mosaic to the history; if infer: returns the
                                denormalized action chunk [chunk, 14] (absolute joint targets)
  quit
The pixel / clip / proprio / text path is the official RoboTwin adapter
(experiments/robotwin/robotwin_jpfm_adapter.py) with two changes, both marked [W]:
  1. the instruction embedding is read from the training text cache (the umT5 encoder
     is not loaded; the model only ever saw the dataset's task text);
  2. frames come from RoboSynChallenge sensors (cam_high / cam_left_wrist / cam_right_wrist,
     the dataset's own camera keys) instead of RoboTwin's head/left/right camera names.
"""

import argparse
import hashlib
import json
import os
import sys
import time

# Keep the protocol channel clean: everything printed by libraries goes to stderr.
_PROTO_OUT = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from Roboactualizer.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving  # noqa: E402
from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from Roboactualizer.datasets.robotwin_image import build_mosaic_from_frames  # noqa: E402
from Roboactualizer.utils.config_resolvers import register_default_resolvers  # noqa: E402
from Roboactualizer.utils.pytorch_utils import set_global_seed  # noqa: E402

register_default_resolvers()

CAM_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")  # mosaic: top, bottom-left, bottom-right
_DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def strict_check(model, payload):
    """The RoboActualizer load_checkpoint is strict=False; refuse any missing, unexpected or mis-shaped key."""
    msd = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    csd = payload["model"] if "model" in payload else payload
    missing = sorted(set(msd) - set(csd))
    unexpected = sorted(set(csd) - set(msd))
    shape_bad = sorted(k for k in set(msd) & set(csd) if tuple(msd[k].shape) != tuple(csd[k].shape))
    return csd, {"n_model_keys": len(msd), "n_ckpt_keys": len(csd), "missing": missing[:10],
                 "unexpected": unexpected[:10], "shape_mismatch": shape_bad[:10],
                 "ok": not (missing or unexpected or shape_bad)}


class OfIRunner:
    def __init__(self, a):
        cfg = OmegaConf.load(a.train_config)
        self.cfg = cfg
        self.device = a.device
        self.dtype = _DTYPES[a.mixed_precision]
        mcfg, dcfg = cfg.model, cfg.data.train
        # Backbone weights from the local copy; latent norm stats come from the checkpoint buffers
        # (official sim_robotwin.yaml sets latent_norm_stats_path: null for evaluation).
        mcfg.backbone.ckpt_path = a.vjepa
        mcfg.backbone.cache_dir = os.path.dirname(a.vjepa)
        mcfg.latent_norm_stats_path = None

        model = instantiate(mcfg, model_dtype=self.dtype, device=self.device)
        payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if a.negctl_drop_key:
            sd = dict(payload["model"])
            del sd[sorted(sd)[0]]
            payload = {**payload, "model": sd}
        csd, chk = strict_check(model, payload)
        if not chk["ok"]:
            raise RuntimeError(f"strict checkpoint check failed: {json.dumps(chk)}")
        model.load_state_dict(csd, strict=False)
        after = model.state_dict()
        chk["bitwise_equal"] = sum(int(torch.equal(after[k].cpu(), csd[k].to(after[k].dtype))) for k in csd)
        if chk["bitwise_equal"] != len(csd):
            raise RuntimeError(f"loaded weights differ from checkpoint: {chk['bitwise_equal']}/{len(csd)}")
        chk["step"] = payload.get("step")
        self.model = model.to(self.device).eval()
        self.ckpt_check = chk

        proc = instantiate(dcfg.processor).eval()
        proc.set_normalizer_from_stats(load_dataset_stats_from_json(a.stats))
        self.processor = proc

        # Pixel pipeline = RobotVideoDataset / precompute (val_transforms == train_transforms here).
        self.pixel_dtype = torch.float32 if bool(mcfg.eval_encode_fp32) else self.dtype
        self.clip_len = int(mcfg.clip_len)
        self.frame_interval = int(mcfg.frame_interval)
        self.accel = {k: bool(mcfg[k]) for k in
                      ("use_kv_cache", "use_cuda_graph", "use_adaln_cache", "use_cross_kv_cache")}
        if a.no_accel:
            self.accel = {k: False for k in self.accel}
        self._img_t = proc.val_transforms
        vh, vw = int(dcfg.video_size[0]), int(dcfg.video_size[1])
        self._resize_t = ResizeSmallestSideAspectPreserving(args={"img_w": vw, "img_h": vh})
        self._crop_t = CenterCrop(args={"img_w": vw, "img_h": vh})
        self._norm_t = Normalize(args={"mean": 0.5, "std": 0.5})
        cam_keys = tuple(str(m["key"]) for m in dcfg.shape_meta.images)
        if cam_keys != CAM_ORDER:
            raise RuntimeError(f"training camera order {cam_keys} != {CAM_ORDER}")
        self.frames = []
        self.max_hist = (self.clip_len - 1) * self.frame_interval + 1

        # [W] instruction embedding from the training text cache, for the dataset's task text.
        tasks = [json.loads(l)["task"] for l in open(os.path.join(a.dataset_dir, "meta/tasks.jsonl"))]
        if len(tasks) != 1:
            raise RuntimeError(f"expected one task in the dataset, got {tasks}")
        prompt = DEFAULT_PROMPT.format(task=tasks[0])
        cpath = os.path.join(a.text_cache, hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                             + f".t5_len{int(dcfg.context_len)}.wan22ti2v5b.pt")
        tc = torch.load(cpath, map_location="cpu")
        if tc.get("placeholder_zero", False):
            raise RuntimeError("text cache is the all-zero placeholder")
        ctx, mask = tc["context"], tc["mask"].bool()
        ctx = ctx.clone()
        ctx[~mask] = 0.0
        self.text_context = ctx.unsqueeze(0).to(self.device, dtype=self.dtype)
        self.text_mask = mask.unsqueeze(0).to(self.device)
        self.info = {"prompt": prompt, "text_cache": cpath, "text_mask_true": int(mask.sum()),
                     "ckpt": a.ckpt, "ckpt_sha256": sha256_file(a.ckpt), "stats": a.stats,
                     "stats_sha256": sha256_file(a.stats), "ckpt_check": chk,
                     "chunk_size": int(mcfg.chunk_size), "mixed_precision": a.mixed_precision,
                     "accel": self.accel, "torch": torch.__version__,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def reset(self):
        self.frames = []

    def push(self, cams):
        """cams: list of 3 uint8 HWC arrays in CAM_ORDER."""
        t = torch.stack([torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1).unsqueeze(0) for c in cams], 0)
        m = build_mosaic_from_frames(t, self._img_t, self._resize_t, self._crop_t, self._norm_t)[0]
        self.frames.append(m.to(self.device, dtype=self.pixel_dtype))
        if len(self.frames) > self.max_hist:
            self.frames.pop(0)

    def clip(self):
        """clip_len frames frame_interval apart ending at the current frame (clamped to the first)."""
        n = len(self.frames)
        idxs = [max(0, n - 1 - i * self.frame_interval) for i in range(self.clip_len)]
        return torch.stack([self.frames[i] for i in reversed(idxs)], 0).unsqueeze(0)

    @torch.no_grad()
    def infer(self, qpos, num_inference_steps):
        key_s = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {key_s: torch.as_tensor(qpos, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        proprio = self.processor.normalizer.forward(batch)["state"][key_s]
        with torch.autocast("cuda", dtype=self.dtype):
            act = self.model.generate(self.clip(), self.text_context, self.text_mask, proprio.to(self.device),
                                      num_inference_steps=num_inference_steps, **self.accel)
        key_a = self.processor.shape_meta["action"][0]["key"]
        norm = self.processor.normalizer.normalizers["action"][key_a]
        return norm.backward(act.to(dtype=torch.float32, device="cpu")).numpy()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--text_cache", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--vjepa", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed_precision", default="no")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--no_accel", action="store_true")
    p.add_argument("--negctl_drop_key", action="store_true")
    a = p.parse_args()

    def reply(obj):
        _PROTO_OUT.write(json.dumps(obj) + "\n")
        _PROTO_OUT.flush()

    try:
        runner = OfIRunner(a)
    except Exception as e:  # report the reason, then exit non-zero
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
        raise
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return
        req = json.loads(line)
        cmd = req.get("cmd")
        try:
            if cmd == "hello":
                reply({"ok": True, **runner.info})
            elif cmd == "seed":
                set_global_seed(int(req["seed"]) % (2 ** 32 - 2) + 1, get_worker_init_fn=False)
                reply({"ok": True})
            elif cmd == "reset":
                runner.reset()
                reply({"ok": True})
            elif cmd == "push":
                raw = stdin.read(int(req["nbytes"]))
                cams, off = [], 0
                for shp in req["shapes"]:
                    n = int(np.prod(shp))
                    cams.append(np.frombuffer(raw, dtype=np.uint8, count=n, offset=off).reshape(shp))
                    off += n
                if off != len(raw):
                    raise ValueError(f"frame bytes {len(raw)} != expected {off}")
                runner.push(cams)
                out = {"ok": True, "n_hist": len(runner.frames)}
                if req.get("infer"):
                    t0 = time.perf_counter()
                    chunk = runner.infer(np.asarray(req["qpos"], dtype=np.float32), a.num_inference_steps)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    out["chunk"] = chunk.tolist()
                    out["infer_s"] = time.perf_counter() - t0
                reply(out)
            elif cmd == "quit":
                reply({"ok": True})
                return
            else:
                raise ValueError(f"unknown cmd {cmd!r}")
        except Exception as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
            raise


if __name__ == "__main__":
    main()
