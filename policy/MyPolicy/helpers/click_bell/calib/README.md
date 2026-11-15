# cam_high calibration used by helpers/bell_press.py

Cameras of the three real-robot platforms (official finals platform calibration data, 2026-11-15),
expressed in the simulator world frame (left arm base + (0.233, 0.3, 0.835) m, same axes).
Fields: `cam_high.K`, `cam_high.R_world_cam_gl` / `cam_high.t_world_cam` (OpenGL camera axes),
`cam_high.T_world_from_cam_opencv` (same pose, OpenCV axes).

| file | sha256 (first 16) | content |
|---|---|---|
| platform_1.json | 8980e4ca71e34a45 | platform 1 |
| platform_2.json | e3951af3037f50ca | platform 2 |
| platform_3.json | 19a51051beea4c0c | platform 3 |
| platform_avg.json | e1cd3a4b0717e109 | mean of the three: translation arithmetic mean, rotation quaternion mean renormalised, K element-wise mean |

These are the original calibration files with only descriptive text fields removed (numbers unchanged;
originals: platform_1 49930ba598a52fd0, platform_2 baf7b621c26dca06, platform_3 1862ecab5e8351d9, avg f98262d180fa7981).
The helper reads `platform_avg.json` (deploy_policy.yml `helper.platform: avg`, sha checked at start): the platform
is not known at run time. On the cover-top plane the average camera is 5-15 mm away from each platform's own calibration.
