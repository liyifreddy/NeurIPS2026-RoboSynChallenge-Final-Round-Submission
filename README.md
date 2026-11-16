# RoboSynChallenge 2026 — real-robot finals submission (team OC-Manip)

Thank you to the RoboSynChallenge organizers for the benchmark, the simulator, the released data and
the real-robot evaluation.

This repository holds our policy for the real-robot finals. Everything the evaluation needs is under
`policy/`:

- `policy/MyPolicy/`: our policy. Entry `eval.sh`, `deploy_policy.yml`, `deploy_policy.py`; one
  configuration file per task (`configs/<task>.yml`); per-task helpers (`helpers/<task>/`); the model
  client (`ofi/`). Full setup and file list: [`policy/MyPolicy/README.md`](policy/MyPolicy/README.md).
- `policy/act/`: the official ACT adapter, unchanged from RoboSynChallenge `45d9b9c`. Our policy runs
  in the official ACT environment.

## Method

Each task runs its own OfI (One from Infinity, RoboActualizer) checkpoint. A frozen V-JEPA 2.1
ViT-L/16-384 encoder reads the top camera and the two wrist cameras. A diffusion action head predicts
32-step chunks of 14 absolute joint targets (`[left arm 6, left gripper, right arm 6, right gripper]`).
We execute 24 steps per inference (32 for `click_bell`) at 10 Hz. Before an action leaves `policy/`,
arm joints are clipped to the real-robot limits. Grippers are sent in [0, 1] (0 = closed).

**Training data:** expert demonstrations we synthesized in the official simulator with the official
assets (randomized appearance, object sizes and joint zero offsets), and the officially released
real-robot datasets. No other real-robot data.

## Checkpoints per task

All weights are in one Hugging Face repository. Each task pins a 40-character revision and a
subfolder in `policy/MyPolicy/configs/<task>.yml`. The evaluation reads exactly these files.

| Task | Hugging Face checkpoint | Subfolder | Revision | Notes |
|---|---|---|---|---|
| `click_bell` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/click_bell/student) | `click_bell/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `drawer_open_place` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/drawer_open_place/student) | `drawer_open_place/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `handle_basket` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/handle_basket/student) | `handle_basket/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `item_assembly` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/item_assembly/student) | `item_assembly/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `items_handover` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/items_handover/student) | `items_handover/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `manipulate_pipette` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/manipulate_pipette/student) | `manipulate_pipette/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `mixer_operating` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/mixer_operating/student) | `mixer_operating/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `sample_loading` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/sample_loading/student) | `sample_loading/student` | `63c52ab2064d60584c679933460448f1a493da3b` | helper on (helpers/sample_loading/) |
| `table_rearrangement` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/table_rearrangement/student) | `table_rearrangement/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |
| `water_pouring` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/63c52ab2064d60584c679933460448f1a493da3b/water_pouring/student) | `water_pouring/student` | `63c52ab2064d60584c679933460448f1a493da3b` | — |

All tasks also use the shared V-JEPA 2.1 encoder (`shared/`, `ofi_backbone` in
[`policy/MyPolicy/deploy_policy.yml`](policy/MyPolicy/deploy_policy.yml)).

## Helpers

A helper is a small rule-based controller for one task. It is switched on or off in that task's
configuration file (`helper.enabled`). Enabled in this submission:

- `sample_loading` (`helpers/sample_loading/sample_search.py`): when the left gripper holds the tube above the rack and the policy starts to open it, the helper takes the left arm, lowers the tube until it meets resistance, searches for the hole in a small spiral, pushes the tube in, opens the gripper slowly and hands the arm back. It acts at most once per episode and always hands back before the episode ends.

Helpers read only the observation the evaluation provides: the joint readings (`robot/qpos`) and
camera images. The `sample_loading` helper uses the top camera (`cam_high`) once at the start to
locate the rack, with a fixed camera calibration, and the joint readings afterwards. Helpers do not
read simulator state. Their actions go through the same joint-limit clipping as the policy's.

## Setup

Full commands: [`policy/MyPolicy/README.md`](policy/MyPolicy/README.md), Setup 1–6.

0. System packages for the simulator renderer (Ubuntu 22.04):
   `apt-get install -y libvulkan1 vulkan-tools libxext6 libx11-6 libgl1 libegl1`.
   `vulkaninfo --summary` should list the NVIDIA GPU.
1. Official RoboSynChallenge at `45d9b9c` and EmbodiChain `v0.2.4.post1`
   ([installation guide](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html)).
2. Copy this repository's whole `policy/` tree (`policy/MyPolicy` and `policy/act`) into
   `RoboSynChallenge/policy/`.
3. Official ACT environment: `export UV_PYTHON_PREFERENCE=only-managed`, then
   `(cd RoboSynChallenge/policy/act && uv sync)`. Then install PyTorch 2.7.1+cu128 with
   `policy/MyPolicy/locks/act_env.txt`. The CUDA 12.8 build supports the RTX 5090 (`sm_120`).
4. Model worker: a Python 3.10 environment from `policy/MyPolicy/ofi/locks/ofi_worker.txt`
   (PyTorch 2.7.1+cu128, `--no-deps`), plus RoboActualizer `85b9dc6` and vjepa2 `204698b`.
   Set `OFI_PYTHON` and `OFI_SRC_ROOT`.
5. Weights: run `policy/MyPolicy/prefetch.py` once. It downloads every pinned file into the local
   Hugging Face cache and checks its sha256. `--check` verifies offline. The evaluation downloads
   nothing.
6. Self-check: `policy/MyPolicy/tests/*.py` (CPU, a few seconds).

## Running

From the RoboSynChallenge root, one command per task (setting `random`, GPU 0):

```bash
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

Other options (for example `--max_episodes 20`) go to the official `scripts/eval_policy.py`.
We evaluated with `eval_reset_sync_steps = 0`, the default of `scripts/eval_policy.py`. Our
`deploy_policy.yml` does not set it.

## Requirements

Linux x86-64 and one NVIDIA GPU with a driver for CUDA 12.8 (≥ 570). Python 3.11 (simulator and
adapter) and Python 3.10 (model worker), PyTorch 2.7.1+cu128, `uv`, `git`. Tested on a single RTX 5090
(32 GB) through the official entry for all ten tasks. Peak GPU memory is about 6.4 GB per task with the
simulator, and about 7.4 GB with a helper on.

## Licenses

Our code: see the file headers. RoboActualizer (OfI model code) and V-JEPA 2 / 2.1 (encoder code): MIT,
fetched from their upstream repositories at the pinned commits.

## Contact

Yi Li (team OC-Manip). Please open an issue in this repository.


