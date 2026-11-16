# RoboSynChallenge 2026 — real-robot finals submission (team OC-Manip)

Policy and evaluation entry point for our RoboSynChallenge (NeurIPS 2026) real-robot finals
submission. Everything the evaluation needs is under `policy/`:

- `policy/MyPolicy/` — our policy: `eval.sh`, `deploy_policy.yml`, `deploy_policy.py`, one
  configuration file per task (`configs/<task>.yml`), the action post-processing, and the OfI
  model client (`ofi/`). Detailed setup, file list and timing tools:
  [`policy/MyPolicy/README.md`](policy/MyPolicy/README.md).
- `policy/act/` — the official ACT adapter, unchanged from RoboSynChallenge `45d9b9c`. Our policy
  runs in the official ACT environment and imports nothing else from outside `policy/`.

## Method

Every task runs its own OfI (One from Infinity, RoboActualizer) world-action checkpoint: a frozen
V-JEPA 2.1 ViT-L/16-384 encoder over the top and the two wrist cameras plus a diffusion action head
that predicts 32-step chunks of 14 absolute joint targets (`[left arm 6, left gripper, right arm 6,
right gripper]`), executed 24 steps per inference (32 for `click_bell`) at 10 Hz. Before any action
leaves `policy/`, arm joints are clipped to the real-robot limits and grippers are sent in [0, 1]
(0 = closed). The policy reads only the observation the evaluation provides.

**Training data:** expert demonstrations we synthesized in the official simulator with the official
assets (randomized appearance, object sizes and joint zero offsets), together with the officially
released real-robot datasets. No other real-robot data.

## Per-task checkpoints

| Task | Hugging Face checkpoint | Subfolder | Revision | Notes |
|---|---|---|---|---|
| `click_bell` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/click_bell/student) | `click_bell/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `drawer_open_place` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/drawer_open_place/student) | `drawer_open_place/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `handle_basket` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/handle_basket/student) | `handle_basket/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `item_assembly` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/item_assembly/student) | `item_assembly/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `items_handover` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/items_handover/student) | `items_handover/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `manipulate_pipette` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/manipulate_pipette/student) | `manipulate_pipette/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `mixer_operating` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/mixer_operating/student) | `mixer_operating/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `sample_loading` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/sample_loading/student) | `sample_loading/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | helper on (helpers/sample_loading/) |
| `table_rearrangement` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/table_rearrangement/student) | `table_rearrangement/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |
| `water_pouring` | [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints/tree/3150e66085c0df112ed2dd91aa5635dd7ee04717/water_pouring/student) | `water_pouring/student` | `3150e66085c0df112ed2dd91aa5635dd7ee04717` | — |

All tasks also use the shared V-JEPA 2.1 encoder (`ofi_backbone` in
[`policy/MyPolicy/deploy_policy.yml`](policy/MyPolicy/deploy_policy.yml)). The table above is what
`policy/MyPolicy/configs/<task>.yml` pins; those files are what the evaluation reads.

## Setup (summary)

Full commands: [`policy/MyPolicy/README.md`](policy/MyPolicy/README.md), Setup 1–6.

1. Official RoboSynChallenge at `45d9b9c` and EmbodiChain `v0.2.4.post1`
   ([installation guide](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html)).
2. Copy this repository's whole `policy/` tree (`policy/MyPolicy` and `policy/act`) into
   `RoboSynChallenge/policy/`.
3. Official ACT environment (`policy/act`, `uv sync` with `UV_PYTHON_PREFERENCE=only-managed`), then PyTorch 2.7.1+cu128 from
   `policy/MyPolicy/locks/act_env.txt` — the CUDA 12.8 build runs on the RTX 5090 (`sm_120`).
4. Model worker: a Python 3.10 environment from `policy/MyPolicy/ofi/locks/ofi_worker.txt`
   (PyTorch 2.7.1+cu128, `--no-deps`), RoboActualizer `85b9dc6` and vjepa2 `204698b` checkouts;
   `OFI_PYTHON` and `OFI_SRC_ROOT` point to them.
5. Weights: `policy/MyPolicy/prefetch.py` downloads every pinned file into the local Hugging Face
   cache once and checks its sha256 (`--check` verifies offline). The evaluation downloads nothing.
6. Self-check: `policy/MyPolicy/tests/*.py` (CPU, a few seconds).

## Running an evaluation

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

Further options (for example `--max_episodes 20`) go to the official `scripts/eval_policy.py`.
We evaluated with `eval_reset_sync_steps = 0` (the default of `scripts/eval_policy.py`; our
`deploy_policy.yml` does not set it).

## Requirements

Linux x86-64, one NVIDIA GPU with a driver for CUDA 12.8 (≥ 570), `uv`, `git`. Python 3.11
(simulator / adapter) and Python 3.10 (model worker), PyTorch 2.7.1+cu128. Tested on a single
RTX 5090 (32 GB) through the official entry for all ten tasks.

## Licenses

Our code: see the file headers. RoboActualizer (OfI model code) and V-JEPA 2 / 2.1 (encoder code):
MIT, fetched from their upstream repositories at the pinned commits.


