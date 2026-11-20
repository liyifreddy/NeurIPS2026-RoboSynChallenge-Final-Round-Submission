"""Real-station check (request real_88e951d11a824c): MyPolicy behind the organizers' rsc_real bridge,
fed a recorded real-robot observation sequence.

The sequence is episode N of the official dataset RoboSynChallenge/cobotmagic_Real_<task> (public;
three 480x640 cameras, 10 fps; qpos = state dims 0-6 and 16-22).
--bridge chunk (default): rsc_real.chunks.ChunkBridge + ChunkGuard(fps 10), as the station runs a
  serial model: infer(observation) -> chunk; each action executed through the guard, then the recorded
  observation of the next step goes back with feedback(); the next chunk is asked with the observation
  after the last action. Reports the time to the first chunk and the reply time per chunk.
--bridge serial: rsc_real.bridge.FeedbackBridge, one action per observation. Reports the reply time of
  steps with and without an inference.
Both report the episode wall time (with --period s of emulated robot execution per action).

With --ref <another checkout> (e.g. the frozen commit dffad646) the reference runs the same sequence
with the same seed through the per-step FeedbackBridge, and every action is compared with
np.array_equal; per inference the sha256 of the model input (the frame-history clip and qpos) and of
the returned chunk are compared too (tools/real_feedback_tap.py wraps the worker of each checkout;
the worker file itself is not changed).

  cd <RoboSynChallenge>   # this checkout: policy/MyPolicy
  export OFI_PYTHON=... OFI_SRC_ROOT=...   # as for eval.sh; weights from prefetch.py
  policy/act/.venv/bin/python policy/MyPolicy/tools/check_real_feedback.py \
      --rsc_real <dir holding rsc_real/> --task handle_basket [--ref <frozen checkout>] [--steps 150]

Exit code 0: ran without a policy failure (and, with --ref, bitwise equal).
The dataset download needs network access once (cached under --cache).
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
HERE = Path(__file__).resolve()
TREE = HERE.parents[3]   # <checkout>/policy/MyPolicy/tools/this file


def prepare(task, episode, cache):
    dst = Path(cache) / f"{task}_ep{episode}.npz"
    if dst.exists():
        return dst
    import av
    import pandas as pd
    from huggingface_hub import hf_hub_download
    repo, ep = f"RoboSynChallenge/cobotmagic_Real_{task}", f"episode_{episode:06d}"
    st = pd.read_parquet(hf_hub_download(repo, f"data/chunk-000/{ep}.parquet", repo_type="dataset"))
    st = np.stack(st["observation.state"].to_numpy()).astype(np.float32)
    qpos = np.concatenate([st[:, 0:7], st[:, 16:23]], axis=1)
    vids = []
    for c in CAMS:
        with av.open(hf_hub_download(repo, f"videos/chunk-000/observation.images.{c}/{ep}.mp4",
                                     repo_type="dataset")) as v:
            vids.append(np.stack([f.to_ndarray(format="rgb24") for f in v.decode(video=0)]))
    n = min(len(qpos), *(len(v) for v in vids))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, frames=np.stack([v[:n] for v in vids], axis=1), qpos=qpos[:n])
    return dst


def summ(xs):
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "median_s": round(statistics.median(s), 4),
            "p95_s": round(s[min(len(s) - 1, int(round(.95 * (len(s) - 1))))], 4), "max_s": round(s[-1], 4)}


def run(a):
    """One tree, in this process (the tree is on sys.path as `policy`)."""
    sys.path.insert(0, a.rsc_real)
    sys.path.insert(0, a.tree)
    import yaml
    from rsc_real.bridge import FeedbackBridge
    import policy.MyPolicy.deploy_policy as dp
    from policy.MyPolicy.ofi import ofi_backend as ob
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.hash:
        os.environ.update(TAP_TARGET=str(ob.WORKER), TAP_OUT=str(out / "tap.jsonl"))
        (out / "tap.jsonl").unlink(missing_ok=True)
        ob.WORKER = HERE.parent / "real_feedback_tap.py"
    flags = []
    _push = ob.OfIClient.push

    def push(self, obs, infer, *args, **kw):
        flags.append(bool(infer))
        return _push(self, obs, infer, *args, **kw)
    ob.OfIClient.push = push

    d = np.load(a.obs)
    frames, qpos = d["frames"], d["qpos"]
    T = min(len(frames) if a.steps <= 0 else a.steps, len(frames), 1000)
    usr = yaml.safe_load((Path(a.tree) / "policy/MyPolicy/deploy_policy.yml").read_text())
    usr.update(task_name=a.task, setting="real", eval_result_dir=str(out),
               ofi_python=os.environ["OFI_PYTHON"], ofi_src_root=os.environ["OFI_SRC_ROOT"])
    usr["action_post"] = dict(usr["action_post"], log_dir=str(out))
    t0 = time.perf_counter()
    model = dp.get_model(usr)
    load_s = time.perf_counter() - t0
    if a.bridge == "serial":
        bridge = FeedbackBridge(dp, model, a.task)
        if a.seed is not None:
            bridge._robosyn_episode_seed = a.seed
    if a.bridge == "chunk":
        return run_chunk(a, dp, model, load_s, frames, qpos, T, out)
    lat, acts, err = [], [], None
    t_ep = time.perf_counter()
    for t in range(T):
        obs = {"sensor": {c: {"color": frames[t, k][None]} for k, c in enumerate(CAMS)},
               "robot": {"qpos": qpos[t][None]}}
        t1 = time.perf_counter()
        try:
            act = bridge.infer(obs)
        except Exception as e:
            err = f"step {t}: {type(e).__name__}: {e}"
            break
        lat.append(time.perf_counter() - t1)
        acts.append(np.asarray(act, np.float32).copy())
        if a.period > 0:
            time.sleep(a.period)
    ep_s = time.perf_counter() - t_ep
    try:
        bridge.close()
    except Exception as e:
        err = (err or "") + f" | close: {e}"
    dp.close_model(model)
    n = len(lat)
    inf = flags[:n]
    res = {"tree": a.tree, "task": a.task, "bridge": "serial", "steps": n, "seed": a.seed, "period_s": a.period,
           "error": err, "load_s": round(load_s, 3), "first_action_s": round(lat[0], 4) if lat else None,
           "episode_wall_s": round(ep_s, 2), "n_inference_steps": int(sum(inf)),
           "reply_inference_step": summ([lat[i] for i in range(1, n) if inf[i]]),
           "reply_other_step": summ([lat[i] for i in range(1, n) if not inf[i]])}
    np.savez(out / "actions.npz", actions=np.concatenate(acts) if acts else np.zeros((0, 14), np.float32),
             infer=np.array(inf, bool))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def run_chunk(a, dp, model, load_s, frames, qpos, T, out):
    """The station side of rsc_real.chunks: infer(observation) -> chunk; every action executed through
    ChunkGuard(fps 10); after each action the recorded observation of the next step goes back with
    feedback(); the next chunk is asked with the observation after the last action."""
    from rsc_real.chunks import ChunkBridge, ChunkGuard

    def payload(t):
        t = min(t, T - 1)   # past the recording (end of the last chunk): repeat the last frame
        p = {"observation.state": qpos[t]}
        for k, c in enumerate(CAMS):
            p["observation.images." + c] = frames[t, k]
        return p

    bridge = ChunkBridge(dp, model, a.task)   # raises if eval has no reviewable action loop
    if a.seed is not None:
        bridge._robosyn_episode_seed = a.seed
    lo, hi = np.full(14, -10.0), np.full(14, 10.0)   # interface bounds for the check; grippers [0, 1]
    lo[[6, 13]], hi[[6, 13]] = 0.0, 1.0
    guard = ChunkGuard("check", lo, hi, fps=10)
    acts, flags, chunk_s, err, t, seq = [], [], [], None, 0, 0
    t_ep = time.perf_counter()
    try:
        while t < T:
            t1 = time.perf_counter()
            chunk = bridge.infer(payload(t))
            chunk_s.append(time.perf_counter() - t1)
            values = guard.accept({"attempt_id": "check", "seq": seq, "actions": chunk}, time.monotonic())
            seq += 1
            for i, v in enumerate(values):
                guard.execute(v, lambda command: None)
                acts.append(np.asarray(v, np.float32)[None].copy())
                flags.append(i == 0)
                if a.period > 0:
                    time.sleep(a.period)
                t += 1
                bridge.feedback(payload(t))
        bridge.finish()
    except Exception as e:
        err = f"step {t}: {type(e).__name__}: {e}"
        try:
            bridge.close()
        except Exception as e2:
            err += f" | close: {e2}"
    ep_s = time.perf_counter() - t_ep
    dp.close_model(model)
    n = min(len(acts), T)
    res = {"tree": a.tree, "task": a.task, "bridge": "chunk", "steps": n, "chunks": len(chunk_s), "seed": a.seed,
           "period_s": a.period, "error": err, "load_s": round(load_s, 3),
           "first_chunk_s": round(chunk_s[0], 4) if chunk_s else None, "chunk_reply": summ(chunk_s[1:]),
           "episode_wall_s": round(ep_s, 2), "n_inference_steps": int(sum(flags[:n]))}
    np.savez(out / "actions.npz", actions=np.concatenate(acts[:n]) if acts else np.zeros((0, 14), np.float32),
             infer=np.array(flags[:n], bool))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def compare(ra, rb):
    a, b = np.load(ra / "actions.npz"), np.load(rb / "actions.npz")
    ta = [x for x in map(json.loads, open(ra / "tap.jsonl")) if x["ev"] == "infer"]
    tb = [x for x in map(json.loads, open(rb / "tap.jsonl")) if x["ev"] == "infer"]
    n = min(len(a["actions"]), len(b["actions"]))
    bad = [t for t in range(n) if not np.array_equal(a["actions"][t], b["actions"][t])]
    keys = ("clip_sha", "qpos_sha", "chunk_sha", "clip_shape", "push_i")
    bad_inf = [i for i in range(min(len(ta), len(tb))) if any(ta[i].get(k) != tb[i].get(k) for k in keys)]
    ok = (len(a["actions"]) == len(b["actions"]) > 0 and not bad and len(ta) == len(tb) > 0 and not bad_inf
          and np.array_equal(a["infer"], b["infer"]))
    return {"steps": [len(a["actions"]), len(b["actions"])], "inferences": [len(ta), len(tb)],
            "action_mismatch_steps": bad[:10], "inference_mismatch": bad_inf[:10], "bitwise_equal": bool(ok)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rsc_real", required=True, help="directory holding the organizers' rsc_real/ package")
    p.add_argument("--task", required=True)
    p.add_argument("--bridge", choices=("chunk", "serial"), default="chunk",
                   help="chunk: ChunkBridge + ChunkGuard (tasks without helper / hysteresis); serial: FeedbackBridge")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=int, default=0, help="0 = whole recorded episode (max 1000)")
    p.add_argument("--period", type=float, default=0.1, help="emulated robot execution after each action, s")
    p.add_argument("--seed", type=int, default=0, help="episode seed given to the policy (as in simulation)")
    p.add_argument("--no_seed", action="store_true", help="no episode seed, as the real bridge (not reproducible)")
    p.add_argument("--ref", default="", help="another checkout to compare against bit for bit (needs a seed)")
    p.add_argument("--cache", default=os.path.expanduser("~/.cache/rsc_real_check"))
    p.add_argument("--out", default="")
    p.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--tree", default=str(TREE), help=argparse.SUPPRESS)
    p.add_argument("--obs", default="", help=argparse.SUPPRESS)
    p.add_argument("--hash", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.no_seed:
        a.seed = None
    if a._run:
        return run(a)
    if a.ref and a.seed is None:
        p.error("--ref needs a seed: without one the diffusion noise differs from run to run")
    obs = prepare(a.task, a.episode, a.cache)
    out = Path(a.out or f"check_real_{a.task}_{time.strftime('%Y%m%d_%H%M%S')}")
    trees = [("this", str(TREE))] + ([("ref", str(Path(a.ref).resolve()))] if a.ref else [])
    base = [sys.executable, str(HERE), "--_run", "--rsc_real", str(Path(a.rsc_real).resolve()), "--task", a.task,
            "--obs", str(obs), "--steps", str(a.steps), "--period", str(a.period)]
    base += ["--no_seed"] if a.seed is None else ["--seed", str(a.seed)]
    results = {}
    for name, tree in trees:
        bridge = a.bridge if name == "this" else "serial"   # the reference: the per-step path it was built for
        cmd = base + ["--tree", tree, "--out", str(out / name), "--bridge", bridge] + (["--hash"] if a.ref else [])
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{name}.log", "w") as log:
            rc = subprocess.call(cmd, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
        rp = out / name / "result.json"
        results[name] = json.loads(rp.read_text()) if rp.exists() else {"error": f"rc {rc}, see {out / (name + '.log')}"}
    summary = {"task": a.task, "episode": a.episode, "dataset": f"RoboSynChallenge/cobotmagic_Real_{a.task}",
               "obs_sha256": hashlib.sha256(Path(obs).read_bytes()).hexdigest(), "results": results}
    ok = all(not r.get("error") for r in results.values())
    if a.ref:
        summary["compare"] = compare(out / "this", out / "ref")
        ok = ok and summary["compare"]["bitwise_equal"]
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
