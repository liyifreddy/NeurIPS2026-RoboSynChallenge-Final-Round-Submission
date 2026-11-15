# RoboSynChallenge 2026 — real-robot finals policy (team OC-Manip)

Everything the evaluation needs is under `policy/`, as the finals rules ask:

- `policy/MyPolicy/` — our policy: `eval.sh`, `deploy_policy.yml`, `deploy_policy.py`, one
  configuration file per task (`configs/<task>.yml`), the click_bell helper
  (`helpers/click_bell/`) and the OfI model client (`ofi/`). Setup, weights and the command
  for each task: [`policy/MyPolicy/README.md`](policy/MyPolicy/README.md).
- `policy/act/` — the official ACT adapter, unchanged from RoboSynChallenge (our policy runs in
  the official ACT environment).

Weights: Hugging Face `liyifreddy/RSC-finals-checkpoints` (one folder per task, shared encoder in
`shared/`), pinned per task in `policy/MyPolicy/configs/<task>.yml`.
