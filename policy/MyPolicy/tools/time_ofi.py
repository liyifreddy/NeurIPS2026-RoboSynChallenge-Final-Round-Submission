"""Inference timing for the OfI model: encoder / denoising / total, per inference.

Runs in the OfI worker environment (OFI_PYTHON; torch + RoboActualizer), on the GPU to be
measured (4090 for the finals estimate). No simulator. Loads the model exactly as the
worker does (ofi/ofi_worker.py OfIRunner: strict checkpoint check, same pixel / text /
proprio path), feeds 640x480 frames, and times with torch.cuda.synchronize():

  push_ms      one frame -> mosaic -> GPU history (paid every env step, not only at inference)
  encoder_ms   backbone.encode_clip_last_group (frozen V-JEPA 2.1 ViT-L, the z0 clip)
  denoise_ms   generate() minus encoder: context builders, KV prefill, num_inference_steps Euler steps
  generate_ms  model.generate() total
  infer_ms     OfIRunner.infer(): state normalisation + generate + action de-normalisation
               (= what the worker reports as infer_s; the backend adds the pipe round trip)

Usage (paths as README step 4/5):
  CUDA_VISIBLE_DEVICES=0 $OFI_PYTHON policy/MyPolicy/tools/time_ofi.py \
      --checkpoint_dir $MYPOLICY_CKPT_ROOT/click_bell \
      --vjepa $MYPOLICY_CKPT_ROOT/shared/vjepa2_1_vitl_384_ema_encoder.pt \
      --ofi_src_root $OFI_SRC_ROOT --warmup 5 --n 50 --out timing_click_bell.json
Optional: --frames frames.npz (arrays cam_high, cam_left_wrist, cam_right_wrist, [T,H,W,3] uint8;
          e.g. real-robot frames) · --num_inference_steps 10 · --no_accel · --mixed_precision no
Output: JSON with per-inference samples and median / p90 / max, GPU name, torch version, settings.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
            "p90": xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))], "min": xs[0], "max": xs[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True, help="dir with ofi_checkpoint.json")
    p.add_argument("--vjepa", required=True)
    p.add_argument("--ofi_src_root", default=os.environ.get("OFI_SRC_ROOT", ""))
    p.add_argument("--frames", default="")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--mixed_precision", default="no")
    p.add_argument("--no_accel", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()

    src = Path(a.ofi_src_root)
    ra = src / "RoboActualizer"
    sys.path[:0] = [str(ra / "src"), str(ra)]
    os.environ.setdefault("TORCH_HOME", str(src / "torch_home"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    ck = _load("mypolicy_ofi_checkpoints", POLICY / "ofi" / "checkpoints.py")
    files = ck.verify_ofi_checkpoint(Path(a.checkpoint_dir))
    worker = _load("mypolicy_ofi_worker", POLICY / "ofi" / "ofi_worker.py")  # note: moves library stdout to stderr

    import numpy as np
    import torch

    wargs = argparse.Namespace(
        train_config=str(files["train_config"]), ckpt=str(files["weights"]), stats=str(files["stats"]),
        text_cache=str(files["text_cache_file"].parent), dataset_dir=str(files["tasks"].parent.parent),
        vjepa=a.vjepa, device="cuda", mixed_precision=a.mixed_precision,
        num_inference_steps=a.num_inference_steps, no_accel=a.no_accel, negctl_drop_key=False)
    t_load = time.perf_counter()
    runner = worker.OfIRunner(wargs)
    load_s = time.perf_counter() - t_load

    def sync():
        torch.cuda.synchronize()

    enc_times = []
    backbone = runner.model.backbone
    orig_enc = backbone.encode_clip_last_group

    def timed_enc(*args, **kw):
        sync()
        t0 = time.perf_counter()
        out = orig_enc(*args, **kw)
        sync()
        enc_times.append(time.perf_counter() - t0)
        return out

    backbone.encode_clip_last_group = timed_enc
    gen_times = []
    orig_gen = runner.model.generate

    def timed_gen(*args, **kw):
        sync()
        t0 = time.perf_counter()
        out = orig_gen(*args, **kw)
        sync()
        gen_times.append(time.perf_counter() - t0)
        return out

    runner.model.generate = timed_gen

    rng = np.random.default_rng(a.seed)
    if a.frames:
        z = np.load(a.frames)
        seq = [z[c] for c in worker.CAM_ORDER]
        T = min(len(s) for s in seq)
        frame_at = lambda i: [s[i % T] for s in seq]  # noqa: E731
    else:
        fixed = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in worker.CAM_ORDER]
        frame_at = lambda i: fixed  # noqa: E731
    qpos = np.zeros(14, dtype=np.float32)
    qpos[6] = qpos[13] = 1.0

    worker.set_global_seed(a.seed + 1, get_worker_init_fn=False)
    runner.reset()
    # fill the clip history as an episode would (clip_len frames, frame_interval apart)
    for i in range(runner.max_hist):
        runner.push(frame_at(i))
    push_s, infer_s = [], []
    total = a.warmup + a.n
    for k in range(total):
        sync()
        t0 = time.perf_counter()
        runner.push(frame_at(runner.max_hist + k))
        sync()
        t1 = time.perf_counter()
        chunk = runner.infer(qpos, a.num_inference_steps)
        sync()
        t2 = time.perf_counter()
        if k >= a.warmup:
            push_s.append(t1 - t0)
            infer_s.append(t2 - t1)
    if not np.isfinite(chunk).all():
        raise RuntimeError("non-finite chunk")
    enc = enc_times[a.warmup:]
    gen = gen_times[a.warmup:]
    den = [g - e for g, e in zip(gen, enc)]
    ms = lambda xs: [1000.0 * x for x in xs]  # noqa: E731
    res = {
        "what": "OfI inference timing (policy/MyPolicy/tools/time_ofi.py)",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_dir": a.checkpoint_dir, "ckpt_sha256": runner.info["ckpt_sha256"],
        "chunk_size": runner.info["chunk_size"], "clip_len": runner.clip_len, "frame_interval": runner.frame_interval,
        "num_inference_steps": a.num_inference_steps, "mixed_precision": a.mixed_precision, "accel": runner.accel,
        "frames": a.frames or "random 640x480", "warmup": a.warmup, "n": a.n, "load_s": load_s,
        "summary_ms": {"push": stats(ms(push_s)), "encoder": stats(ms(enc)), "denoise": stats(ms(den)),
                       "generate": stats(ms(gen)), "infer": stats(ms(infer_s))},
        "samples_ms": {"push": ms(push_s), "encoder": ms(enc), "denoise": ms(den), "generate": ms(gen), "infer": ms(infer_s)},
        "note": "per inference = one chunk; with ofi_replan_steps 24 one inference serves 24 env steps (10 Hz, serial, FAQ #12/#14)",
    }
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    s = res["summary_ms"]
    print(f"[time_ofi] {res['gpu']}: encoder {s['encoder']['median']:.1f} ms, denoise {s['denoise']['median']:.1f} ms, "
          f"generate {s['generate']['median']:.1f} ms, infer {s['infer']['median']:.1f} ms (median of {a.n}); "
          f"push {s['push']['median']:.1f} ms/frame", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
