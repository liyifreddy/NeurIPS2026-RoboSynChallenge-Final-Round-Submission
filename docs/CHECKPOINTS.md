# Checkpoint repository

All weights are in one Hugging Face repository, [liyifreddy/RSC-finals-checkpoints](https://huggingface.co/liyifreddy/RSC-finals-checkpoints), at revision
`2a395d9680be8466a0a554a8fb888b2a82aada1b`. Every task pins this revision in `policy/MyPolicy/configs/<task>.yml`, and
`policy/MyPolicy/prefetch.py` downloads exactly these files.

| Task | Policy | Training step | Weights sha256 (first 8) | Simulator success / episodes | Helper files |
|---|---|---|---|---|---|
| Click Bell | `click_bell/student/` | 160000 | `02a2ed93` | 7 / 20 | `click_bell/helper/head.pt`, `click_bell/helper/tip_head.pt` (off) |
| Drawer Open Place | `drawer_open_place/student/` | 160000 | `91619ff7` | 2 / 34 | none |
| Handle Basket | `handle_basket/student/` | 160000 | `6c6f1e52` | 0 / 25 | none |
| Item Assembly | `item_assembly/student/` | 160000 | `1db9a74c` | 4 / 20 | none |
| Items Handover | `items_handover/student/` | 160000 | `ca4b805f` | 2 / 28 | none |
| Manipulate Pipette | `manipulate_pipette/student/` | 120000 | `0222dd54` | 6 / 36 | none |
| Mixer Operating | `mixer_operating/student/` | 180000 | `733f5d97` | 3 / 43 | none |
| Sample Loading | `sample_loading/student/` | 160000 | `917d9285` | 3 / 49 | `sample_loading/helper/head.pt` (on) |
| Table Rearrangement | `table_rearrangement/student/` | 160000 | `ae7a3079` | 46 / 50 | none |
| Water Pouring | `water_pouring/student/` | 160000 | `c7b98d4a` | 12 / 39 | none |

Simulator numbers are from our own evaluation in the official simulator with the wrist cameras at the
real-robot mounting angle; the real robot was not tested. The shared encoder is
`shared/vjepa2_1_vitl_384_ema_encoder.pt` (sha256 `49231ae6951961715ccdf993aadd755b8d5f27864e130b1129bef0305e0edcf0`, about 1.2 GB).

## Why the weights are on Hugging Face

GitHub holds the code only. The organizers asked for models on Hugging Face, and the encoder alone is
above GitHub's file size limit. Weights are gated; authenticated access is granted to the
RoboSynChallenge account. The evaluation itself downloads nothing; run `prefetch.py` once before it.
