# Third-party code in this repository

The helpers load their keypoint models without extra installs, so each helper folder carries copies of
the following code (`helpers/click_bell/bell_kp/` and `helpers/sample_loading/kp/`). Each copy keeps its
original license file.

| Code | Where | License |
|---|---|---|
| RoboActualizer (`src/`) | `code/RoboActualizer/` | MIT (`LICENSE`) |
| V-JEPA 2 | `code/vjepa2/` | MIT (`LICENSE`) |
| timm 1.0.31 | `pylib/timm/` | Apache-2.0 (`pylib/timm-1.0.31.dist-info/licenses/LICENSE`) |
| OmegaConf 2.4.0 | `pylib/omegaconf/` | BSD-3-Clause (`pylib/omegaconf-2.4.0.dist-info/licenses/LICENSE`) |
| einops 0.8.2 | `pylib/einops/` | MIT (`pylib/einops-0.8.2.dist-info/licenses/LICENSE`) |

`policy/act/` is the official RoboSynChallenge ACT adapter, unchanged. The model worker installs
RoboActualizer (`85b9dc6`) and vjepa2 (`204698b`) from their upstream repositories (README, Setup 4).
