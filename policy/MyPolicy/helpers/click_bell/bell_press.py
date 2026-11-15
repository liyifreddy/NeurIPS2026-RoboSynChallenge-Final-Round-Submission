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

    def __init__(self, head_path, encoder_path, rate_hz, dev="cuda", half=True, top_peak_only=True):
        root = os.path.join(HERE, "bell_kp")
        sys.path.insert(0, root)
        import bell_kp_online as B
        import torch
        self.B = B
        sys.path.insert(0, os.path.join(root, "code/RoboActualizer/src")); sys.path.append(os.path.join(root, "pylib"))   # pylib (timm, omegaconf) only where the env lacks them
        from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone
        m = B.BellKP.__new__(B.BellKP)
        m.dev, m.half = dev, half and str(dev).startswith("cuda")
        m.enc = VJEPABackbone(encoder_path, torch_hub_dir=os.path.join(root, "code/vjepa2"), image_size=(B.H_IN, B.W_IN), num_cameras=1,
                              device=dev, dtype=torch.float32)
        m.head = B.Head(1).to(dev); m.head.load_state_dict(torch.load(head_path, map_location=dev)); m.head.eval()
        G = json.load(open(os.path.join(root, "v2.json")))["gates"]
        mem_max = int(round(B.MEM_S * rate_hz))
        m.tr = {c: B._Track(G.get(f"{c}|button.cover_top", {}).get("tau"), mem_max) for c in B.CAMS}
        if top_peak_only:   # cam_high: peak gate only (visibility score unreliable on some episodes; offline 848 frames: p90 23.5 vs 24.2 px)
            f = m.tr["cam_high"].step; m.tr["cam_high"].step = lambda p, vp: f(p, 1.0)
        self.m = m

    def reset(self):
        self.m.reset()

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
                      camera=str(g("camera", "file")))   # file = helpers/calib (deployment); sim = read cam_high from the simulator (simulation checks only)   # robustness test only: shifts the frozen estimate by this much along +x (0 = off)
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
        self.log_dir = log_dir; self.ep = 0; self.S = None; self.env = None

    # --- episode bookkeeping -------------------------------------------------
    def start_episode(self):
        self.kp.reset()
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
        S = self.S; t = S["t"]; S["t"] += 1
        try:
            q = self._q(); T = {a: self.kin.tcp(a, q[a]) for a in ARMS}
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
                    Td = S["T0"].copy(); Td[:3, 3] = [S["est"][0], S["est"][1], S["z"]]
                    qn = self.kin.ik_step(arm, S["q"], Td); S["q"] = S["q"] + np.clip(qn - S["q"], -self.c["max_rad"], self.c["max_rad"])
                    return self._set(action, ARMS[arm], S["q"])
        except Exception as ex:  # noqa: BLE001  (the helper never stops the episode; it just stops helping)
            self._ev(error=f"{type(ex).__name__}: {ex}"); S["phase"] = "release" if S["phase"] != "watch" else "off"
        return action

    @staticmethod
    def _set(action, sl, vals):
        if hasattr(action, "clone"):
            a = action.clone(); a.reshape(-1)[sl] = a.new_tensor(np.asarray(vals, float)); return a
        a = np.array(action, dtype=float, copy=True); a.reshape(-1)[sl] = vals
        return a.astype(np.asarray(action).dtype if np.asarray(action).dtype.kind == "f" else np.float32)
