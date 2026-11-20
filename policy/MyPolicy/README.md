# MyPolicy (finals)

Everything we add is inside `policy/MyPolicy/` (FAQ #37). It is based on the final
version of RoboSynChallenge (pinned `45d9b9c`; tested so far on `a485f91`, which
differs only in the J5 URDF limit) and uses the official adapter interface: `eval.sh`,
`deploy_policy.yml`, `deploy_policy.py` (`get_model`, `eval`, `reset_model`). Outputs
are 14 absolute joint targets per step (`[left arm 6, left gripper, right arm 6, right
gripper]`, FAQ #15).

## What the policy does on every step

- `deploy_policy.py` picks the backend configured for the task (`task_policies`) and
  hands it the environment wrapped by `action_post.GuardedEnv`, so **every action that
  leaves `policy/` is post-processed** (`action_post.py`, settings in `deploy_policy.yml`
  under `action_post`):
  - arm joints are clipped to the real-robot limits. Default profile `executable` = the
    official simulator URDF at pin `45d9b9c`, every joint side pulled in by 1.5°
    (J2 lower / J3 upper are the mechanical zero and keep margin 0):
    J1 ±148.500351°, J2 0…178.408748°, J3 −168.496578…0°, J4 ±98.481135°,
    J5 ±72.984513°, J6 ±169.699789°. Over-limit commands are clipped, so the SDK never
    receives an out-of-target command and never drops one. (`deploy_margin`, `real_hard`
    and `official_sim` remain as explicit alternative profiles; the old `deploy_margin`
    J6 ±160° is not the default.)
  - grippers are sent in [0, 1] (0 = closed, 1 = open), the same normalized unit as the
    real-robot recordings (preliminary real 60: 0–1 continuous). The real side converts
    0–1 × 0.05 m per finger (organizers). In the official simulator the gripper action
    space is the URDF joint limit [0, 0.05] m and `set_qpos` clamps, so on sim a value
    ≥0.05 is fully open — i.e. the sim gripper acts like a switch (the same behaviour the
    preliminary runs showed). Observations are normalized back to 0–1
    (`normalize_robot_joint_data`, joint ids 6/13). Send normalized 0–1: do **not** send
    metres (20× too closed) or 0–100 (100× too open).
    `gripper_mode` can turn the command into a switch (`binary`, `hysteresis` 0.15→0.81,
    the preliminary handover 15→81; hysteresis starts each episode open; binary opens at
    `hi`, so set `hi: 0.5` for a plain 0.5 threshold). Neither switch mode can be combined
    with `max_step_grip` (the rate limit would turn 0/1 into fractions; the config loader
    rejects it). Default `linear` keeps the model's value, so half-open stays half-open on
    the real robot. `gripper_clip: false` is an explicit opt-out and lets values leave [0, 1].
  - `max_step_rad` / `max_step_grip` optionally limit how far one command may move from
    the previous one (off by default; set when a replanned chunk jumps).
  - a non-finite value is replaced by the previous command;
  - every clipped step is counted and written to
    `<eval_result_dir>/<task>/action_post_<time>_<pid>.jsonl` with a per-episode summary.
- These rules read only the action and the observation (FAQ #17). Inference and
  execution are serial (FAQ #14); one inference returns a chunk that is executed over
  the next steps.

## Requirements

- Linux x86-64, one NVIDIA GPU (driver for CUDA 12.8), `uv`, `git`.
- About 50 GB free disk (uv package cache) plus the weights (Setup step 5).
- Two Python environments: the official one for the simulator / adapter
  (Python 3.11, `policy/act/.venv`, pins in `locks/act_env.txt`) and our model worker
  (Python 3.10, pins in `ofi/locks/ofi_worker.txt`).

## Setup (run from one working directory, one shell)

**1. Official environment** at RoboSynChallenge `45d9b9c` and EmbodiChain
`v0.2.4.post1`, following the official installation guide. This directory then
contains `RoboSynChallenge/` (and `EmbodiChain/`).

**2. Our policy.** Copy this repository's whole `policy/` tree into `RoboSynChallenge/policy/`:
`policy/MyPolicy` (ours) and `policy/act` (the official ACT adapter, identical to RoboSynChallenge
`45d9b9c`). We do not use the ACT model; we only reuse the Python environment that `policy/act` sets up.

**3. Python environment for the simulator and our policy**: the official evaluation environment set up in `policy/act`, with CUDA 12.8 PyTorch. The ACT model itself is not used. Use uv's own
Python (`UV_PYTHON_PREFERENCE=only-managed`): a system Python without development headers cannot build
`toppra` from source.
```bash
export UV_PYTHON_PREFERENCE=only-managed
(cd RoboSynChallenge/policy/act && uv sync)
uv pip install --python RoboSynChallenge/policy/act/.venv/bin/python \
    -c RoboSynChallenge/policy/MyPolicy/locks/act_env.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128
```

**4. Model worker environment and sources:**
```bash
uv venv --python 3.10 ofi_env
uv pip install --python ofi_env/bin/python --no-deps \
    -r RoboSynChallenge/policy/MyPolicy/ofi/locks/ofi_worker.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
git clone https://github.com/7mare/RoboActualizer.git ofi_src/RoboActualizer
git -C ofi_src/RoboActualizer checkout 85b9dc6d38d9d5dab3fe52e157d5fd69dfc50806
git clone https://github.com/facebookresearch/vjepa2.git ofi_src/torch_home/hub/facebookresearch_vjepa2_main
git -C ofi_src/torch_home/hub/facebookresearch_vjepa2_main checkout 204698b45b3712590f06245fbfba32d3be539812
export OFI_PYTHON=$PWD/ofi_env/bin/python
export OFI_SRC_ROOT=$PWD/ofi_src
```

**5. Weights** (as in the preliminary delivery). Every task runs the checkpoint pinned for
it in `deploy_policy.yml` (`task_policies`: Hugging Face repository + 40-character revision +
subfolder), read from the local Hugging Face cache; the evaluation downloads nothing.
Fill the cache once with an account that has access to the gated repository
`liyifreddy/RSC-finals-checkpoints`:
```bash
hf auth login
(cd RoboSynChallenge && policy/act/.venv/bin/python policy/MyPolicy/prefetch.py)
```
It ends with `[prefetch] done: N of N`; if it is interrupted, run it again. Every OfI
checkpoint is checked against the sha256 in its `ofi_checkpoint.json`, the V-JEPA encoder
(`ofi_backbone`) against `deploy_policy.yml`. Offline check (no download):
```bash
(cd RoboSynChallenge && HF_HUB_OFFLINE=1 policy/act/.venv/bin/python policy/MyPolicy/prefetch.py --check)
```

**6. Self-check** (CPU, no simulator, a few seconds):
```bash
(cd RoboSynChallenge && policy/act/.venv/bin/python policy/MyPolicy/tests/test_skeleton.py)
(cd RoboSynChallenge && policy/act/.venv/bin/python policy/MyPolicy/tests/test_executable_profile.py)
```

## Evaluate: one command per task

Setting `random`, GPU 0 (for `clear`, replace `random`):
```bash
cd RoboSynChallenge
bash policy/MyPolicy/eval.sh click_bell random auto 0
bash policy/MyPolicy/eval.sh drawer_open_place random auto 0
bash policy/MyPolicy/eval.sh handle_basket random auto 0
bash policy/MyPolicy/eval.sh item_assembly random auto 0
bash policy/MyPolicy/eval.sh items_handover random auto 0
bash policy/MyPolicy/eval.sh manipulate_pipette random auto 0
bash policy/MyPolicy/eval.sh mixer_operating random auto 0
bash policy/MyPolicy/eval.sh sample_loading random auto 0
bash policy/MyPolicy/eval.sh table_rearrangement random auto 0
bash policy/MyPolicy/eval.sh water_pouring random auto 0
```
Further options (e.g. `--max_episodes 20`) go to the official `scripts/eval_policy.py`.
We evaluated with `eval_reset_sync_steps = 0` (the default of `scripts/eval_policy.py`; our
`deploy_policy.yml` does not set it).
The evaluation downloads nothing.

## Timing

The official evaluator already records inference timing in `evaluation_metrics.json`
(`average_inference_time_seconds`, per-episode totals). Two extra scripts give the
breakdown before a full run:

```bash
# model only, inside the OfI worker environment (encoder / denoising / generate)
$OFI_PYTHON policy/MyPolicy/tools/time_ofi.py --checkpoint-dir mypolicy_ckpt/click_bell
# whole policy path with a stand-in env, no simulator (chunk boundary stalls)
policy/act/.venv/bin/python policy/MyPolicy/tools/time_e2e.py --steps 200
```
Both need a GPU for a meaningful number; CPU numbers only prove the path runs.

### Real station

The real station runs a serial model: one observation in, one chunk of actions out, the whole chunk
executed, the real observation after each action fed back to the policy while the robot moves
(organizers' `rsc_real/chunks.py`, `ChunkBridge`). `deploy_policy.eval` has the reviewed loop
shape for this: for a task whose actions are all fixed when the chunk is inferred (every task
without a helper and without gripper hysteresis), it infers once, post-processes the whole chunk
with `action_post` in order, then `for action in actions: ... = env.step(action)` and feeds each
returned observation into the model's frame history (the model reads frames t, t-4, t-8, t-12).
Worker requests, frames in the history and actions are the same as on the per-step path. Tasks
whose actions read each step's observation (`click_bell` and `sample_loading` helpers,
`items_handover` gripper hysteresis) keep the per-step path (`rsc_real/bridge.py`,
`FeedbackBridge`).

Serving details in `ofi/ofi_backend.py` (no action or model input changes):

- CPU threads of the adapter process and of the OfI worker follow the container CPU quota
  (cgroup `cpu.max` / `cpu.cfs_quota_us` and the CPU affinity, at most 8), not the host core count.
- Worker requests run in strict order on one background thread. A step without an inference
  returns its already planned action at once while its frame joins the model's history in the
  background; a step with an inference first waits until every earlier frame is in the history.
  `ofi_async_push: false` switches this off (every push waits, as before).
- Only a broken model process stops an episode: a worker that exits makes `eval()` raise at once,
  one that gives no reply raises after `OFI_STEP_TIMEOUT` (default 30 s; normal replies take under
  0.1 s) or, for the first inference with its CUDA warm-up, `OFI_FIRST_INFER_TIMEOUT` (default 60 s).
  A non-finite model output does not stop the episode (`action_post` holds the previous command for
  those values), nor does an unexpected gripper reading in the hysteresis (skipped for that step).

`tools/check_real_feedback.py` reproduces this: it wraps the policy in the organizers'
`ChunkBridge` with `ChunkGuard` (fps 10) (`--bridge serial`: `FeedbackBridge`), feeds a recorded
episode of the official real-robot dataset (`RoboSynChallenge/cobotmagic_Real_<task>`, three
480×640 cameras) as the observations and feedback, and reports the time to the first chunk, the
reply time per chunk (serial: per step, with and without an inference) and the episode wall time.
With `--ref <another checkout>` it runs the reference on the same observations with the same seed
through the per-step `FeedbackBridge` and compares every action (`np.array_equal`) and, per
inference, the model input (frame-history clip, qpos) and output chunk by sha256:

```bash
cd RoboSynChallenge   # OFI_PYTHON, OFI_SRC_ROOT and the weights as for eval.sh
HF_HUB_OFFLINE=0 policy/act/.venv/bin/python policy/MyPolicy/tools/check_real_feedback.py \
    --rsc_real <directory holding rsc_real/> --task handle_basket --ref <checkout of dffad646>
```
Without an episode seed (as on the real bridge) the diffusion noise differs from run to run in
any version, so the bitwise comparison uses a seed (default 0).

## Licenses

- RoboActualizer (the OfI model code): MIT (`LICENSE` in the upstream checkout).
- V-JEPA 2 / 2.1 (encoder code): MIT; the checkpoint itself has no separate license file
  in the upstream repo — confirm the checkpoint terms before any redistribution.
- This policy directory contains only our own code plus the preliminary `ofi/` client,
  which is ours.

## Files

| File | What |
|---|---|
| `eval.sh` | entry point (official argument order) |
| `deploy_policy.yml` | official keys + `action_post`, `backends`, `task_policies`, backend parameters |
| `deploy_policy.py` | `get_model` / `eval` / `reset_model` / `close_model`: routing + post-processing |
| `action_post.py` | joint-limit clip, gripper 0–1 + optional switch, optional step limit, counters and log (numpy only) |
| `backends/ofi.py` | One-from-Infinity model backend (finds the task checkpoint) |
| `backends/hold.py` | holds the current pose; tests only, not routed to any task |
| `ofi/` | the model client, worker and checkpoint checks (preliminary submission; the client's serving changed for the real station, see Timing) |
| `tools/time_ofi.py` | inference timing: encoder / denoising / total |
| `tools/time_e2e.py` | timing of the whole `policy/` path without the simulator |
| `tools/check_real_feedback.py`, `tools/real_feedback_tap.py` | real-station check: organizers' feedback bridge, recorded real observations, reply times, bitwise comparison with another checkout |
| `tests/test_skeleton.py` | CPU self-check |
| `locks/` | pinned package versions |
