# ----------------------------------------------------------------------------
# Bell "last press" helper (click_bell only; off unless task_policies.click_bell.helper.enabled).
#
# The student policy runs unchanged. This wraps the env the backend steps:
# it reads only obs (cam_high image + robot qpos) and, once per episode, when
# the pressing hand has stalled just above the bell cover, replaces that arm's
# six joint targets with a short scripted press: straight down until the arm
# stops following (cover bottomed) or a depth cap, hold, lift, then hands the
# arm back to the student. Nothing else in the action is changed.
#
# Perception: a frozen V-JEPA 2.1 ViT-L encoder (the OfI backbone file) + a
# small heat-map head finds the bell cover top in cam_high; the pixel is cast
# to the plane z = z0 with the cam_high intrinsics / extrinsics read from a
# calibration file in helpers/calib/. Hand position = forward kinematics of the
# measured joints on the robot URDF in helpers/robot/ (TCP = link6 + 0.143 m
# along its z axis). No simulator state is read.
#
# Gates (no action unless all hold): cover seen in K consecutive calls whose
# world points agree within spread_mm; the estimate is frozen when a hand comes
# within lock_mm (hands occlude the cover); the hand is within trigger_mm of the
# cover axis, 5-60 mm above z0, and has moved < stall_mm over stall_n steps.
# One press per episode. Every episode writes one jsonl line (platform, z0,
# estimate, events) to helper.log_dir (or the action_post log directory).
# ----------------------------------------------------------------------------
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"left": slice(0, 6), "right": slice(7, 13)}      # qpos / action: [left J1-J6, left grip, right J1-J6, right grip]
TCP_OFF = 0.143                                          # link6 -> tool point along link6 z (m)
BASE_Z = 0.835                                           # robot base_link in the calibration / simulator world frame (m)


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _qmat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Kin:
    """Forward kinematics + damped least-squares IK on the robot URDF (pytorch_kinematics)."""

    def __init__(self, urdf):
        import torch
        import pytorch_kinematics as pk
        self.torch = torch
        xml = open(urdf, "rb").read()
        self.ch = {a: pk.build_serial_chain_from_urdf(xml, f"{a}_link6").to(dtype=torch.float64) for a in ARMS}
        self.base = np.eye(4); self.base[2, 3] = BASE_Z

    def tcp(self, arm, q):
        T = self.ch[arm].forward_kinematics(self.torch.tensor([list(map(float, q))], dtype=self.torch.float64)).get_matrix()[0].numpy()
        T = self.base @ T
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, TCP_OFF])
        return T

    @staticmethod
    def _err(T, Td):
        dp = Td[:3, 3] - T[:3, 3]; Re = Td[:3, :3] @ T[:3, :3].T
        return np.r_[dp, 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])]

    def ik_step(self, arm, q, Td, iters=2, h=1e-4, lam=1e-4):
        q = np.asarray(q, float).copy()
        for _ in range(iters):
            e = self._err(self.tcp(arm, q), Td); J = np.zeros((6, 6))
            for i in range(6):
                dq = np.zeros(6); dq[i] = h
                J[:, i] = (e - self._err(self.tcp(arm, q + dq), Td)) / h
            q = q + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        return q


def load_calib(path):
    c = json.load(open(path))["cam_high"]
    return dict(R=np.asarray(c["R_world_cam_gl"], float), t=np.asarray(c["t_world_cam"], float), K=np.asarray(c["K"], float),
                width=c.get("width"), height=c.get("height"))


def ray_z(uv, R, t, K, z0):
    d = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]); d = R @ np.array([d[0], -d[1], -d[2]])
    return t + (z0 - t[2]) / d[2] * d


class KP:
    """The bell cover keypoint head (code in helpers/bell_kp, weights from the checkpoint root)."""

    def __init__(self, head_path, encoder_path, rate_hz, dev="cuda", half=True, top_peak_only=True, enc_obj=None, n_out=1, taus=None):
        root = os.path.join(HERE, "bell_kp")
        sys.path.insert(0, root)
        import bell_kp_online as B
        import torch
        self.B = B
        sys.path.insert(0, os.path.join(root, "code/RoboActualizer/src")); sys.path.append(os.path.join(root, "pylib"))   # pylib (timm, omegaconf) only where the env lacks them
        from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone
        m = B.BellKP.__new__(B.BellKP)
        m.dev, m.half = dev, half and str(dev).startswith("cuda")
        m.enc = enc_obj if enc_obj is not None else VJEPABackbone(encoder_path, torch_hub_dir=os.path.join(root, "code/vjepa2"), image_size=(B.H_IN, B.W_IN),
                                                                  num_cameras=1, device=dev, dtype=torch.float32)   # 决SF：指尖头共用铃盖头那份编码器
        m.head = B.Head(n_out).to(dev); m.head.load_state_dict(torch.load(head_path, map_location=dev)); m.head.eval()
        self.n_out = n_out; self.peak_only = top_peak_only
        # 多通道头（决LB 11-16 TargetKP：0 左手指尖、1 右手指尖、2 铃盖顶）：每个通道一个 cam_high 跟踪器，门线 taus[ch]
        self.trs = [B._Track(float(taus[c]), int(round(B.MEM_S * rate_hz))) for c in range(n_out)] if taus is not None else None
        G = json.load(open(os.path.join(root, "v2.json")))["gates"]
        mem_max = int(round(B.MEM_S * rate_hz))
        m.tr = {c: B._Track(G.get(f"{c}|button.cover_top", {}).get("tau"), mem_max) for c in B.CAMS}
        if top_peak_only:   # cam_high: peak gate only (visibility score unreliable on some episodes; offline 848 frames: p90 23.5 vs 24.2 px)
            f = m.tr["cam_high"].step; m.tr["cam_high"].step = lambda p, vp: f(p, 1.0)
        self.m = m

    def reset(self):
        self.m.reset()
        for t in (self.trs or []):
            t.reset()

    def track_feat_ch(self, z, ch):
        """Channel `ch` of a multi-output head on shared features z (one cam_high frame)."""
        import torch, torch.nn.functional as F
        B, m = self.B, self.m
        with torch.no_grad():
            f = z.transpose(1, 2).reshape(1, -1, B.GH, B.GW).half().float()
            h = m.head.a(f); hm = m.head.hm(F.interpolate(h, size=(96, 128), mode="bilinear", align_corners=False))
            P = hm.flatten(2).softmax(-1).view(hm.shape)[0, ch].cpu().numpy(); V = m.head.vis(h.mean((2, 3))).sigmoid()[0, ch].item()
        px, kind, conf = self.trs[ch].step(P, 1.0 if self.peak_only else float(V))   # 总-592②：多通道路径同样只看峰值
        return dict(px=px, kind=kind, conf=float(conf), passed=kind in ("glob", "loc", "jump"), p_vis=float(V), peak=float(P.max()))

    def track_feat(self, z, n):
        """Run this head + its cam_high tracker on encoder output z already computed for the same cam_high frame (shared with the other head)."""
        import torch, torch.nn.functional as F
        B, m = self.B, self.m
        with torch.no_grad():
            f = z.transpose(1, 2).reshape(n, -1, B.GH, B.GW).half().float()
            h = m.head.a(f); hm = m.head.hm(F.interpolate(h, size=(96, 128), mode="bilinear", align_corners=False))
            P = hm.flatten(2).softmax(-1).view(hm.shape)[:, 0].cpu().numpy(); V = m.head.vis(h.mean((2, 3))).sigmoid()[:, 0].cpu().numpy()
        px, kind, conf = m.tr["cam_high"].step(P[0], float(V[0]))
        return dict(px=px, kind=kind, conf=float(conf), passed=kind in ("glob", "loc", "jump"), p_vis=float(V[0]), peak=float(P[0].max()))

    def encode(self, img):
        """cam_high frame -> encoder features (once per frame; both heads read them)."""
        import torch, torch.nn.functional as F
        B, m = self.B, self.m
        img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        if img.shape[:2] != (480, 640):
            raise ValueError(f"cam_high {img.shape[:2]} is not 480 x 640")
        with torch.no_grad():
            x = torch.from_numpy(img[None]).permute(0, 3, 1, 2).float().to(m.dev)
            x = F.interpolate(x, size=(B.H_IN, B.W_IN), mode="bilinear", align_corners=False).round().clamp(0, 255).div(255)
            clip = (x * 2 - 1).unsqueeze(1).repeat(1, 2, 1, 1, 1)
            with torch.autocast("cuda", dtype=torch.float16, enabled=m.half):
                return m.enc.encode_clip_last_group(clip).float()

    def top(self, img):
        img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        if img.shape[:2] != (480, 640):     # the head's pixel grid is 640 x 480
            raise ValueError(f"cam_high {img.shape[:2]} is not 480 x 640")
        return self.m.step({"cam_high": img})["cam_high"]


class BellHelperEnv:
    """Env proxy for the backend: same attribute pass-through as GuardedEnv; step() may rewrite one arm."""

    def __init__(self, env, helper, obs):
        object.__setattr__(self, "_h_env", env)
        object.__setattr__(self, "_h", helper)
        helper.env = env
        helper.observe(obs)

    def __getattr__(self, name):
        return getattr(self._h_env, name)

    def __setattr__(self, name, value):
        setattr(self._h_env, name, value)

    def step(self, action):
        out = self._h_env.step(self._h.act(action))
        self._h.observe(out[0])
        return out


class BellPress:
    def __init__(self, cfg, ckpt_root=None, log_dir=None):
        g = lambda k, d: cfg.get(k, d)
        self.c = dict(platform=str(g("platform", "avg")), z0=float(g("z0", 0.861)), kp_every=int(g("kp_every", 4)), kp_k=int(g("kp_k", 3)),
                      spread=float(g("spread_mm", 10)) / 1000, lock=float(g("lock_mm", 60)) / 1000, trig=float(g("trigger_mm", 30)) / 1000,
                      h_lo=float(g("h_lo_mm", 5)) / 1000, h_hi=float(g("h_hi_mm", 60)) / 1000, stall_n=int(g("stall_n", 8)),
                      stall=float(g("stall_mm", 5)) / 1000, desc=float(g("desc_mm", 1.9)) / 1000, max_desc=float(g("max_desc_mm", 60)) / 1000,
                      lag=float(g("lag_mm", 3)) / 1000, slow=float(g("slow_frac", 0.5)),
                      floor=float(g("floor_below_z0_mm", 23)) / 1000,   # z0 - (1.6 x 5 mm travel + 15 mm z0 uncertainty) is the lowest target
                      min_desc=float(g("min_desc_mm", 5)) / 1000, dwell=int(g("dwell", 14)),
                      lift=float(g("lift_mm", 2.5)) / 1000, lift_off=float(g("lift_off_mm", 40)) / 1000, max_rad=float(g("max_rad", 0.05)),
                      test_bias=float(g("test_bias_mm", 0)) / 1000,
                      camera=str(g("camera", "file")),
                      mode=str(g("mode", "takeover")),   # takeover (delivered 11-15) | servo (总-580②: compare fingertip and cover in the same cam_high image; no z0, no world target)
                      trig_px=float(g("servo_trigger_px", 35)), tol_px=float(g("servo_tol_px", 3)), gain=float(g("servo_gain", 0.6)),
                      xy_step=float(g("servo_xy_step_mm", 2.0)) / 1000, servo_desc_max=float(g("servo_max_desc_mm", 80)) / 1000,
                      lock_px=float(g("servo_lock_px", 70)), cap_spread_px=float(g("servo_cap_spread_px", 8)),
                      blend=int(g("handback_steps", 10)), blend_max=int(g("handback_max_steps", 40)), tk_xy_step=float(g("takeover_xy_step_mm", 2.0)) / 1000,
                      tk_xy_tol=float(g("takeover_xy_tol_mm", 3.0)) / 1000, servo_max_steps=int(g("servo_max_steps", 150)),
                      servo_max_lost=int(g("servo_max_lost", 5)), servo_max_drift=float(g("servo_max_drift_mm", 40)) / 1000,
                      tip_med=int(g("servo_tip_median", 1)),   # 决SF：指尖点逐帧抖得厉害时取最近 N 次的中位（1 = 不取）
                      align_mm=float(g("servo_align_mm", 2.3)) / 1000, tol_mm=float(g("servo_tol_mm", 1.0)) / 1000)   # 决NO：门按 mm 设（铃盖处约 1 mm/px；≤ 最小盖半径 4.7 mm 的一半）   # 决SF 审查：交还要平滑、接手后每步水平位移要有上限   # file = helpers/calib (deployment); sim = read cam_high from the simulator (simulation checks only)   # robustness test only: shifts the frozen estimate by this much along +x (0 = off)
        import hashlib
        cp = os.path.join(HERE, "calib", f"platform_{self.c['platform']}.json")
        want_c = str(g("calib_sha256", ""))
        got_c = hashlib.sha256(open(cp, "rb").read()).hexdigest()
        if want_c and not got_c.startswith(want_c):
            raise ValueError(f"bell helper calibration {cp}: sha256 {got_c[:16]} != {want_c}")
        self.cal = load_calib(cp)
        self.kin = Kin(os.path.join(HERE, "robot", "CobotMagicWithGripperV100.urdf"))
        enc = os.path.expandvars(str(g("encoder", "${MYPOLICY_CKPT_ROOT}/shared/vjepa2_1_vitl_384_ema_encoder.pt")))
        head = os.path.expandvars(str(g("head", "${MYPOLICY_CKPT_ROOT}/finals_1115/bell_kp/head_vjepa.pt")))
        want = str(g("head_sha256", "6630020352c8821d"))     # first 16 hex (or full) of the head file; checked at start
        import hashlib
        got = hashlib.sha256(open(head, "rb").read()).hexdigest()
        if not got.startswith(want):
            raise ValueError(f"bell helper head {head}: sha256 {got[:16]} != {want}")
        self.kp = KP(head, enc, rate_hz=10.0 / self.c["kp_every"], dev=g("device", "cuda"))
        self.tip = None   # 总-592②：指尖头只看峰值（真机 vis 头几乎都判看不见：两道门 12.9%，峰值 99.6%）
        if self.c["mode"] == "servo":   # fingertip (two-finger midpoint) head in cam_high; same encoder file, its own heat-map head
            if g("tip_tau", None) is None:   # 决SF：伺服模式下指尖门线必须显式给，没给就不起伺服
                raise ValueError("bell helper mode servo needs tip_tau")
            tip_head = os.path.expandvars(str(g("tip_head", "${MYPOLICY_CKPT_ROOT}/finals_1115/bell_kp/head_tip.pt")))
            want_t = str(g("tip_head_sha256", ""))   # 决SF：指尖头开局也核 sha
            if not want_t:
                raise ValueError("bell helper mode servo needs tip_head_sha256")
            got_t = hashlib.sha256(open(tip_head, "rb").read()).hexdigest()
            if not got_t.startswith(want_t):
                raise ValueError(f"bell helper tip head {tip_head}: sha256 {got_t[:16]} != {want_t}")
            chs = str(g("tip_channels", ""))   # 决LB TargetKP 多通道头：如 "left:0,right:1"（铃盖顶是第 2 路）；空 = 单通道头
            self.tip_ch = {k: int(v) for k, v in (x.split(":") for x in chs.split(",") if x)} if chs else None
            if self.tip_ch:
                taus = [float(x) for x in str(g("tip_tau", "")).split(",")]   # 每通道一个门线，按通道号排
                self.tip = KP(tip_head, enc, rate_hz=10.0, dev=g("device", "cuda"), top_peak_only=True, enc_obj=self.kp.m.enc, n_out=len(taus), taus=taus)
            else:
                self.tip = KP(tip_head, enc, rate_hz=10.0, dev=g("device", "cuda"), top_peak_only=True, enc_obj=self.kp.m.enc)
                self.tip.m.tr["cam_high"].tau = float(g("tip_tau", None))
        self.log_dir = log_dir; self.ep = 0; self.S = None; self.env = None

    # --- episode bookkeeping -------------------------------------------------
    def start_episode(self):
        self.kp.reset()
        if self.tip is not None:
            self.tip.reset()
        self.S = dict(phase="watch", t=0, obs=None, q=None, kq=[], far=[], est=None, locked=False, hist=[], ev=[], kp_calls=0, kp_ok=0)

    def end_episode(self):
        if self.S is None:
            return
        rec = dict(kind="bell_helper", episode=self.ep, platform=self.c["platform"], camera=self.c["camera"], z0=self.c["z0"], phase_end=self.S["phase"],
                   est=None if self.S["est"] is None else np.round(self.S["est"], 4).tolist(), kp_calls=self.S["kp_calls"], kp_ok=self.S["kp_ok"],
                   events=self.S["ev"], cfg=self.c)
        print("[bell_helper] " + json.dumps({k: rec[k] for k in ("episode", "platform", "camera", "z0", "phase_end", "est", "kp_calls", "kp_ok")}, default=str)
              + " events=" + json.dumps(rec["events"][:12], default=str), flush=True)
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, f"bell_helper_{os.getpid()}.jsonl"), "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass
        self.ep += 1; self.S = None

    def _ev(self, **k):
        self.S["ev"].append(dict(t=self.S["t"], **k))

    # --- per step ------------------------------------------------------------
    def observe(self, obs):
        if self.S is None:
            self.start_episode()
        self.S["obs"] = obs

    def _q(self):
        q = _to_np(self.S["obs"]["robot"]["qpos"]).astype(float).reshape(-1)
        return {a: q[s] for a, s in ARMS.items()}

    def _cam(self):
        """cam_high R (OpenGL axes) / t / K. camera: file (delivered default) = helpers/calib. camera: sim reads the simulator's
        privileged camera object (env.unwrapped.sim) and exists only for simulation checks of the helper logic; the real-robot
        path never takes this branch (deploy_policy.yml sets camera: file)."""
        if self.c["camera"] != "sim":
            return self.cal
        if self.S.get("simcam") is None:
            u = getattr(self.env, "unwrapped", self.env); sim = getattr(u, "sim", None); cam = sim.get_sensor("cam_high")
            P = None
            for name in ("get_world_pose", "get_local_pose", "get_pose"):
                f = getattr(cam, name, None)
                if callable(f):
                    v = _to_np(f()).astype(float).reshape(-1)
                    if v.size in (7, 16):
                        P = v; break
            if P.size == 16:
                M = P.reshape(4, 4); R, t = M[:3, :3], M[:3, 3]
            else:
                R, t = _qmat(P[3:7]), P[:3]
            k = cam.get_intrinsics(); k = k[0] if isinstance(k, tuple) else k
            self.S["simcam"] = dict(R=R, t=t, K=_to_np(k).astype(float).reshape(-1, 3, 3)[0])
            self._ev(kind="sim camera", t_cam=np.round(t, 4).tolist(), fx=round(float(self.S["simcam"]["K"][0, 0]), 1))
        return self.S["simcam"]

    def _hz(self, p):
        v = p - self.S["est"]; return float(np.hypot(v[0], v[1])), float(v[2])

    def act(self, action):
        a = self._act(action)
        S = self.S
        if S is not None and S["phase"] == "release" and S.get("arm") is not None and S.get("q") is not None:
            # hand-back blend (决SF): over `blend` steps go from the helper's last joint targets to the student's, each joint <= max_rad per step
            k = S.get("hb", 0)
            if not S.get("hb_done"):
                sl = ARMS[S["arm"]]; qs = np.asarray(_to_np(a), float).reshape(-1)[sl]
                if float(np.max(np.abs(qs - S["q"]))) < self.c["max_rad"] or k >= self.c["blend_max"]:   # 决SF 复审：追到每关节差 < max_rad 才交还；blend_max 步兜底
                    S["hb_done"] = True; self._ev(kind="handed back", steps=k)
                    if k < self.c["blend_max"]:
                        return a
                qb = S["q"] + np.clip(qs - S["q"], -self.c["max_rad"], self.c["max_rad"])
                S["q"] = qb; S["hb"] = k + 1
                return self._set(a, sl, qb)
        return a

    def _act(self, action):
        S = self.S; t = S["t"]; S["t"] += 1
        try:
            q = self._q(); T = {a: self.kin.tcp(a, q[a]) for a in ARMS}
            if self.c["mode"] == "servo":
                return self._servo(action, t, q, T)
            if S["phase"] == "watch":
                if S["est"] is not None and not S["locked"]:
                    dn = min(self._hz(T[a][:3, 3])[0] for a in ARMS)
                    if dn < self.c["lock"]:
                        S["locked"] = True
                        if self.c["test_bias"]:
                            S["est"] = S["est"] + np.array([self.c["test_bias"], 0.0, 0.0])
                        self._ev(kind="lock", n_far=len(S["far"]), hand_mm=round(dn * 1000, 1), est=np.round(S["est"], 4).tolist())
                if not S["locked"] and t % self.c["kp_every"] == 0:
                    o = self.kp.top(_to_np(S["obs"]["sensor"]["cam_high"]["color"]).reshape(480, 640, -1)); S["kp_calls"] += 1
                    if o.get("passed") and o.get("kind") in ("glob", "loc", "jump"):
                        S["kp_ok"] += 1; cm = self._cam(); S["kq"].append(ray_z(o["px"], cm["R"], cm["t"], cm["K"], self.c["z0"])); S["kq"] = S["kq"][-self.c["kp_k"]:]
                    else:
                        S["kq"] = []
                    if len(S["kq"]) >= self.c["kp_k"]:
                        Q = np.stack(S["kq"])
                        if max(float(np.linalg.norm(a - b)) for a in Q for b in Q) <= self.c["spread"]:
                            S["far"].append(np.median(Q, axis=0)); S["est"] = np.median(np.stack(S["far"]), axis=0)
                if S["locked"]:
                    arm = min(ARMS, key=lambda a: self._hz(T[a][:3, 3])[0]); p = T[arm][:3, 3]
                    S["hist"].append((arm, p)); S["hist"] = S["hist"][-(self.c["stall_n"] + 1):]
                    still = len(S["hist"]) > self.c["stall_n"] and all(a == arm for a, _ in S["hist"]) and np.linalg.norm(S["hist"][-1][1] - S["hist"][0][1]) < self.c["stall"]
                    hz, h = self._hz(p)
                    if hz <= self.c["trig"] and self.c["h_lo"] < h <= self.c["h_hi"] and still:
                        S.update(phase="down", arm=arm, q=q[arm].copy(), T0=T[arm].copy(), z=float(p[2]), z_start=float(p[2]))
                        self._ev(kind="take", arm=arm, hz_mm=round(hz * 1000, 1), h_mm=round(h * 1000, 1))
            if S["phase"] in ("down", "dwell", "lift"):
                arm = S["arm"]; zm = float(T[arm][:3, 3][2])
                if S["phase"] == "down":
                    lag = zm - S["z"]; done = S["z_start"] - S["z"]
                    S.setdefault("zh", []).append(zm); zh = S["zh"][-3:]
                    slow = len(zh) == 3 and (zh[0] - zh[-1]) < self.c["slow"] * 2 * self.c["desc"]   # last two steps moved < half the command
                    if lag > self.c["lag"] or (slow and done >= self.c["min_desc"]):   # the arm stopped following: cover bottomed (or something in the way)
                        if done < self.c["min_desc"]:
                            S["phase"] = "release"; self._ev(kind="blocked early, release", desc_mm=round(done * 1000, 1))
                        else:
                            S["phase"] = "dwell"; S["dwell"] = 0; S["z_contact"] = zm; S["z"] = zm - 0.001   # hold just below where it stopped, do not push deeper
                            self._ev(kind="bottomed", desc_mm=round(done * 1000, 1), lag_mm=round(lag * 1000, 1), slow=bool(slow), below_z0_mm=round((self.c["z0"] - zm) * 1000, 1))
                    elif done >= self.c["max_desc"] or S["z"] - self.c["desc"] < self.c["z0"] - self.c["floor"]:
                        S["phase"] = "dwell"; S["dwell"] = 0; S["z_contact"] = zm; S["z"] = zm - 0.001; self._ev(kind="depth cap", desc_mm=round(done * 1000, 1), below_z0_mm=round((self.c["z0"] - zm) * 1000, 1))
                    else:
                        if float(np.linalg.norm(S.get("xy_cmd", S["est"][:2]) - S["est"][:2])) <= self.c["tk_xy_tol"]:   # 决SF：xy 差 ≤ 3 mm 再往下压
                            S["z"] -= self.c["desc"]
                elif S["phase"] == "dwell":
                    S["dwell"] += 1
                    if S["dwell"] >= self.c["dwell"]:
                        S["phase"] = "lift"; S["z"] = zm; self._ev(kind="lift")
                elif S["phase"] == "lift":
                    S["z"] += self.c["lift"]
                    if S["z"] >= S["z_contact"] + self.c["lift_off"]:
                        S["phase"] = "release"; self._ev(kind="release")
                if S["phase"] in ("down", "dwell", "lift"):
                    cur = S.setdefault("xy_cmd", S["T0"][:2, 3].copy()); dxy = S["est"][:2] - cur; dn = float(np.linalg.norm(dxy))
                    if dn > self.c["tk_xy_step"]:
                        dxy = dxy * self.c["tk_xy_step"] / dn
                    S["xy_cmd"] = cur + dxy                                  # 决SF：接手后水平目标每步最多走 takeover_xy_step_mm
                    Td = S["T0"].copy(); Td[:3, 3] = [S["xy_cmd"][0], S["xy_cmd"][1], S["z"]]
                    qn = self.kin.ik_step(arm, S["q"], Td); S["q"] = S["q"] + np.clip(qn - S["q"], -self.c["max_rad"], self.c["max_rad"])
                    return self._set(action, ARMS[arm], S["q"])
        except Exception as ex:  # noqa: BLE001  (the helper never stops the episode; it just stops helping)
            self._ev(error=f"{type(ex).__name__}: {ex}"); S["phase"] = "release" if S["phase"] != "watch" else "off"
        return action

    def _both(self, img):
        """Cover + fingertip from ONE encoder pass on the same cam_high frame (决NO：延时减半)."""
        if hasattr(self.kp, "encode") and hasattr(self.tip, "track_feat"):
            z = self.kp.encode(img); S = self.S
            if getattr(self, "tip_ch", None):
                if S.get("phase") == "servo":   # 接手后只跟按压手那一路
                    return self.kp.track_feat(z, 1), {S["arm"]: self.tip.track_feat_ch(z, self.tip_ch[S["arm"]])}
                return self.kp.track_feat(z, 1), {a: self.tip.track_feat_ch(z, c) for a, c in self.tip_ch.items()}
            return self.kp.track_feat(z, 1), {"any": self.tip.track_feat(z, 1)}
        return self.kp.top(img), {"any": self.tip.top(img)}

    def _img(self):
        return _to_np(self.S["obs"]["sensor"]["cam_high"]["color"]).reshape(480, 640, -1)

    def _px2dxy(self, px, dpx, z):
        """Pixel step -> world xy step on the plane z (local linearisation with the calibration; only direction / scale matter, the loop closes in pixels)."""
        cm = self._cam(); p0 = ray_z(px, cm["R"], cm["t"], cm["K"], z)
        J = np.stack([ray_z((px[0] + 1, px[1]), cm["R"], cm["t"], cm["K"], z)[:2] - p0[:2], ray_z((px[0], px[1] + 1), cm["R"], cm["t"], cm["K"], z)[:2] - p0[:2]], axis=1)
        return J @ np.asarray(dpx, float)

    def _servo(self, action, t, q, T):
        """总-580②: lock the cover pixel before the hand gets near, then align the fingertip pixel to it in the same image
        (no calibration-to-world error), descending while aligned until the arm stops following, hold, lift, hand back."""
        S = self.S
        if S["phase"] == "watch":
            if t % self.c["kp_every"] != 0:
                return action
            img = self._img(); o, ots = self._both(img); S["kp_calls"] += 1
            cand = {a: np.asarray(x["px"], float) for a, x in ots.items() if x.get("passed") and x.get("px") is not None}
            ref = S.get("cap_px")
            tarm = (min(cand, key=lambda a: float(np.linalg.norm(cand[a] - ref))) if ref is not None else next(iter(cand))) if cand else None
            tp = cand[tarm] if tarm is not None else None
            if not S.get("cap_px_locked"):
                if o.get("passed") and o.get("kind") in ("glob", "loc", "jump"):
                    S["kp_ok"] += 1; S.setdefault("cq", []).append(np.asarray(o["px"], float)); S["cq"] = S["cq"][-self.c["kp_k"]:]
                else:
                    S["cq"] = []
                if len(S.get("cq", [])) >= self.c["kp_k"]:
                    Q = np.stack(S["cq"])
                    if max(float(np.linalg.norm(a - b)) for a in Q for b in Q) <= self.c["cap_spread_px"]:
                        S.setdefault("cfar", []).append(np.median(Q, axis=0)); S["cap_px"] = np.median(np.stack(S["cfar"]), axis=0)
                if S.get("cap_px") is not None and tp is not None and float(np.linalg.norm(tp - S["cap_px"])) < self.c["lock_px"]:
                    S["cap_px_locked"] = True; self._ev(kind="lock px", cap_px=np.round(S["cap_px"], 1).tolist(), tip_px=np.round(tp, 1).tolist(), n=len(S.get("cfar", [])))
                return action
            # locked: trigger when the fingertip is near the cover in the image and the hand has stalled
            if tp is None:
                return action
            # the pressing arm = the one whose TCP moved least relative to the cover pixel test: use the arm whose projected TCP is closest to the tip pixel
            cm = self._cam()
            def proj(P):
                v = cm["R"].T @ (P - cm["t"]); u = cm["K"] @ np.array([v[0], -v[1], -v[2]]); return u[:2] / u[2]
            arm = tarm if tarm in ARMS else min(ARMS, key=lambda a: float(np.linalg.norm(proj(T[a][:3, 3]) - tp)))   # 多通道头直接给出是哪只手
            p = T[arm][:3, 3]; S["hist"].append((arm, p)); S["hist"] = S["hist"][-(self.c["stall_n"] // self.c["kp_every"] + 2):]
            still = len(S["hist"]) >= 3 and all(a == arm for a, _ in S["hist"]) and np.linalg.norm(S["hist"][-1][1] - S["hist"][0][1]) < self.c["stall"]
            e = S["cap_px"] - tp
            if float(np.linalg.norm(e)) <= self.c["trig_px"] and still:
                S.update(phase="servo", arm=arm, q=q[arm].copy(), T0=T[arm].copy(), xy=T[arm][:2, 3].copy(), z=float(p[2]), z_start=float(p[2]), zh=[], lost=0, tip_px=tp)
                self._ev(kind="take", arm=arm, err_px=np.round(e, 1).tolist())
            return action
        if S["phase"] in ("servo", "dwell", "lift"):
            arm = S["arm"]; zm = float(T[arm][:3, 3][2])
            if S["phase"] == "servo":
                _, ots = self._both(self._img()); ot = ots.get(arm) or ots.get("any") or {}
                if ot.get("passed") and ot.get("px") is not None:
                    S.setdefault("tq", []).append(np.asarray(ot["px"], float)); S["tq"] = S["tq"][-self.c["tip_med"]:]
                    S["tip_px"] = np.median(np.stack(S["tq"]), axis=0); S["lost"] = 0
                else:
                    S["lost"] += 1
                S["sv_n"] = S.get("sv_n", 0) + 1
                drift = float(np.linalg.norm(S["xy"] - S["T0"][:2, 3]))
                if S["sv_n"] > self.c["servo_max_steps"] or S["lost"] > self.c["servo_max_lost"] or drift > self.c["servo_max_drift"]:   # 决SF：步数 / 丢指尖 / 横向漂移上限 → 交还
                    S["phase"] = "release"; self._ev(kind="servo limit, release", steps=S["sv_n"], lost=S["lost"], drift_mm=round(drift * 1000, 1), aligned_frac=round(S.get("al_n", 0) / max(1, S["sv_n"]), 2))
                    return action
                e = S["cap_px"] - S["tip_px"]; en = float(np.linalg.norm(e))
                en_mm = float(np.linalg.norm(self._px2dxy(S["tip_px"], e, zm)))   # 像素差换成当前高度上的 mm（只用标定的局部尺度，不用它的绝对位置）
                if S["lost"] == 0 and en_mm > self.c["tol_mm"]:
                    d = self._px2dxy(S["tip_px"], e * self.c["gain"], zm); dn = float(np.linalg.norm(d))
                    S["xy"] = S["xy"] + (d * min(1.0, self.c["xy_step"] / dn) if dn > 0 else d)
                aligned = en_mm <= self.c["align_mm"] and S["lost"] < 3
                S["al_n"] = S.get("al_n", 0) + int(aligned)   # 决SF：伺服阶段对准步数占比   # 假臂「一直对不准」：原 15 px 门太松；决NO：改按 mm，≤ 2.3 mm 才下压
                lag = zm - S["z"]; done = S["z_start"] - S["z"]
                S["zh"].append(zm); zh = S["zh"][-3:]
                slow = aligned and len(zh) == 3 and done > 0.004 and (zh[0] - zh[-1]) < self.c["slow"] * 2 * self.c["desc"]
                if (lag > self.c["lag"] or slow) and done >= self.c["min_desc"]:
                    S["phase"] = "dwell"; S["dwell"] = 0; S["z_contact"] = zm; S["z"] = zm - 0.001
                    self._ev(kind="bottomed", desc_mm=round(done * 1000, 1), err_px=round(en, 1), err_mm=round(en_mm * 1000, 1), aligned_frac=round(S.get("al_n", 0) / max(1, S.get("sv_n", 1)), 2))
                elif (lag > self.c["lag"]) and 0.002 < done < self.c["min_desc"]:   # only once we have actually started down
                    S["phase"] = "release"; self._ev(kind="blocked early, release", desc_mm=round(done * 1000, 1))
                elif done >= self.c["servo_desc_max"]:
                    S["phase"] = "release"; self._ev(kind="no contact within max descent, release", desc_mm=round(done * 1000, 1))
                elif aligned:
                    S["z"] -= self.c["desc"]
            elif S["phase"] == "dwell":
                S["dwell"] += 1
                if S["dwell"] >= self.c["dwell"]:
                    S["phase"] = "lift"; S["z"] = zm; self._ev(kind="lift")
            elif S["phase"] == "lift":
                S["z"] += self.c["lift"]
                if S["z"] >= S["z_contact"] + self.c["lift_off"]:
                    S["phase"] = "release"; self._ev(kind="release")
            if S["phase"] in ("servo", "dwell", "lift"):
                Td = S["T0"].copy(); Td[:3, 3] = [S["xy"][0], S["xy"][1], S["z"]]
                qn = self.kin.ik_step(arm, S["q"], Td); S["q"] = S["q"] + np.clip(qn - S["q"], -self.c["max_rad"], self.c["max_rad"])
                return self._set(action, ARMS[arm], S["q"])
        return action

    @staticmethod
    def _set(action, sl, vals):
        if hasattr(action, "clone"):
            a = action.clone(); a.reshape(-1)[sl] = a.new_tensor(np.asarray(vals, float)); return a
        a = np.array(action, dtype=float, copy=True); a.reshape(-1)[sl] = vals
        return a.astype(np.asarray(action).dtype if np.asarray(action).dtype.kind == "f" else np.float32)
