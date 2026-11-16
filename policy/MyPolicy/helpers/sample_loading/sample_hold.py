# ----------------------------------------------------------------------------
# helpers/sample_loading/sample_hold.py -- sample-loading "no early release above the rack" helper
# (sample_loading only; off unless task_policies.sample_loading.helper.enabled). Self-contained folder
# (总-508②): V-JEPA code / pylib, calibration and URDF are copies inside helpers/sample_loading/.
#
# The student policy runs unchanged. This wraps the env the backend steps and
# reads only obs (cam_high image + robot qpos). It never moves an arm: at most
# it replaces the student's LEFT gripper command with the "holding" command
# while the left hand holds the tube above the rack and the tube has not been
# lowered into a hole yet (failure analysis: most sample-loading failures are
# the student opening the gripper while the tube still rests on the rack top).
#
# Perception: a frozen V-JEPA 2.1 ViT-L encoder (the OfI backbone file) + a
# small heat-map head finds the rack hole centre (rack._holes_mean) in cam_high
# during the first frames of the episode (the rack does not move; hands occlude
# it later); pixels are cast to the plane z = rack_z with the cam_high
# calibration file in helpers/calib/, median -> rack centre. Hand position =
# forward kinematics of the measured joints on the robot URDF (copy of the bell helper's Kin).
# Units (决MV 08:4x): obs qpos[6] / action[6] = left gripper, 0-1 normalised, 0 = closed, 1 = open (0.05 m per finger).
#
# Block (all must hold): rack centre known; left gripper reading inside the
# holding window; right TCP >= free_mm from left TCP; left TCP within
# zone_mm (horizontal) of the rack centre; left TCP higher than rack_z +
# release_dz_mm; the student commands the left gripper more open than the
# running minimum of its commands in this zone by > open_tol.
# Release (stop blocking for the rest of the episode): left TCP below the
# release line, leaves the zone, gripper leaves the window, the left TCP rises
# > lift_rel_mm above where blocking started (the student is pulling back), or
# max_block steps. One jsonl line per episode to helper.log_dir.
# ----------------------------------------------------------------------------
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"left": slice(0, 6), "right": slice(7, 13)}      # qpos / action: [left J1-J6, left grip, right J1-J6, right grip]
TCP_OFF = 0.143                                          # link6 -> tool point along link6 z (m)   (same as the bell helper)
BASE_Z = 0.835                                           # robot base_link in the calibration / simulator world frame (m)


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class Kin:
    """Forward kinematics on the robot URDF (pytorch_kinematics); copy of the bell helper's Kin (FK only)."""

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


def load_calib(path):
    c = json.load(open(path))["cam_high"]
    return dict(R=np.asarray(c["R_world_cam_gl"], float), t=np.asarray(c["t_world_cam"], float), K=np.asarray(c["K"], float))


def ray_z(uv, R, t, K, z0):
    d = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]); d = R @ np.array([d[0], -d[1], -d[2]])
    return t + (z0 - t[2]) / d[2] * d
GRIP_L = 6                     # qpos / action index of the left gripper


class RackKP:
    """rack._holes_mean head (code shared with helpers/bell_kp; names / gates in helpers/sample_rack; weights from the checkpoint root)."""

    def __init__(self, head_path, encoder_path, dev="cuda", half=True, rate_hz=5.0):
        kp_root = os.path.join(HERE, "kp"); rk = os.path.join(HERE, "sample_rack")   # kp/ = copy of the bell helper code/pylib (总-508②: no cross-task imports)
        sys.path.insert(0, os.path.join(kp_root, "code/RoboActualizer/src")); sys.path.append(os.path.join(kp_root, "pylib"))
        sys.path.insert(0, rk)
        import torch
        import target_kp_online as T
        from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone
        m = T.TargetKP.__new__(T.TargetKP)
        m.dev, m.half = dev, half and str(dev).startswith("cuda")
        m.enc = VJEPABackbone(encoder_path, torch_hub_dir=os.path.join(kp_root, "code/vjepa2"), image_size=(T.H_IN, T.W_IN), num_cameras=1,
                              device=dev, dtype=torch.float32)
        m.names = json.load(open(os.path.join(rk, "kp_names.json")))
        m.head = T.Head(len(m.names)).to(dev); m.head.load_state_dict(torch.load(head_path, map_location=dev)); m.head.eval()
        G = json.load(open(os.path.join(rk, "gates.json")))["gates"]
        mem_max = int(round(T.MEM_S * rate_hz))
        m.tr = {c: {n: T._Track(G.get(f"{c}|{n}", {}).get("tau"), mem_max) for n in m.names} for c in T.CAMS}
        self.m = m

    def reset(self):
        self.m.reset()

    def centre(self, img):
        img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
        if img.shape[:2] != (480, 640):
            raise ValueError(f"cam_high {img.shape[:2]} is not 480 x 640")
        return self.m.step({"cam_high": img})["cam_high"].get("rack._holes_mean") or {}


class SampleHelperEnv:
    """Env proxy for the backend (same pass-through as BellHelperEnv); step() may rewrite the left gripper command only."""

    def __init__(self, env, helper, obs):
        object.__setattr__(self, "_h_env", env)
        object.__setattr__(self, "_h", helper)
        helper.observe(obs)

    def __getattr__(self, name):
        return getattr(self._h_env, name)

    def __setattr__(self, name, value):
        setattr(self._h_env, name, value)

    def step(self, action):
        out = self._h_env.step(self._h.act(action))
        self._h.observe(out[0])
        return out


class SampleHold:
    def __init__(self, cfg, log_dir=None):
        g = lambda k, d: cfg.get(k, d)
        self.c = dict(platform=str(g("platform", "avg")), rack_z=float(g("rack_z", 0.899)), rack_steps=int(g("rack_steps", 30)), kp_every=int(g("kp_every", 2)),
                      grip_lo=float(g("grip_held_lo", 0.04)), grip_hi=float(g("grip_held_hi", 0.8)),       # left gripper READING window meaning "holding" (0-1; = 0.002-0.04 m in the sim shell)
                      open_tol=float(g("open_tol", 0.04)),                                                   # left gripper COMMAND more open than the running minimum by this much (0-1; = 0.002 m)
                      free=float(g("free_mm", 80)) / 1000, zone=float(g("zone_mm", 120)) / 1000, rel_dz=float(g("release_dz_mm", 75)) / 1000,
                      lift_rel=float(g("lift_rel_mm", 15)) / 1000, max_block=int(g("max_block", 80)))
        cp = os.path.join(HERE, "calib", f"platform_{self.c['platform']}.json")
        want_c = str(g("calib_sha256", ""))
        got_c = hashlib.sha256(open(cp, "rb").read()).hexdigest()
        if want_c and not got_c.startswith(want_c):
            raise ValueError(f"sample helper calibration {cp}: sha256 {got_c[:16]} != {want_c}")
        self.cal = load_calib(cp)
        self.kin = Kin(os.path.join(HERE, "robot", "CobotMagicWithGripperV100.urdf"))
        enc = os.path.expandvars(str(g("encoder", "${MYPOLICY_CKPT_ROOT}/shared/vjepa2_1_vitl_384_ema_encoder.pt")))
        head = os.path.expandvars(str(g("head", "${MYPOLICY_CKPT_ROOT}/finals_1115/sample_rack/head_vjepa.pt")))
        want = str(g("head_sha256", "c45cbdbc81f19a1a"))
        got = hashlib.sha256(open(head, "rb").read()).hexdigest()
        if not got.startswith(want):
            raise ValueError(f"sample helper head {head}: sha256 {got[:16]} != {want}")
        self.kp = RackKP(head, enc, dev=g("device", "cuda"), rate_hz=10.0 / self.c["kp_every"])
        self.log_dir = log_dir; self.ep = 0; self.S = None

    def start_episode(self):
        self.kp.reset()
        self.S = dict(t=0, obs=None, ests=[], rc=None, done=False, hcmd=None, run=0, z0=None, ev=[], kp_calls=0, kp_ok=0, blocked=0, release=None, first_open=None, hany=None)

    def end_episode(self):
        if self.S is None:
            return
        rec = dict(kind="sample_helper", episode=self.ep, platform=self.c["platform"], rack_c=None if self.S["rc"] is None else np.round(self.S["rc"], 4).tolist(),
                   kp_calls=self.S["kp_calls"], kp_ok=self.S["kp_ok"], blocked=self.S["blocked"], release=self.S["release"], first_open=self.S["first_open"],
                   events=self.S["ev"], cfg=self.c)
        print("[sample_helper] " + json.dumps({k: rec[k] for k in ("episode", "rack_c", "kp_ok", "blocked", "release", "first_open")}, default=str), flush=True)
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, f"sample_helper_{os.getpid()}.jsonl"), "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass
        self.ep += 1; self.S = None

    def _ev(self, **k):
        self.S["ev"].append(dict(t=self.S["t"], **k))

    def observe(self, obs):
        if self.S is None:
            self.start_episode()
        self.S["obs"] = obs

    def _rack(self, t):
        S = self.S
        if S["rc"] is not None or S["done"]:
            return
        if t > self.c["rack_steps"]:
            if S["ests"]:
                S["rc"] = np.median(np.stack(S["ests"]), axis=0); self._ev(kind="rack", n=len(S["ests"]), rc=np.round(S["rc"], 4).tolist())
            else:
                S["done"] = True; self._ev(kind="rack not seen, helper off this episode")
            return
        if t % self.c["kp_every"]:
            return
        o = self.kp.centre(_to_np(S["obs"]["sensor"]["cam_high"]["color"]).reshape(480, 640, -1)); S["kp_calls"] += 1
        if o.get("passed") and o.get("px") is not None:
            S["kp_ok"] += 1; S["ests"].append(ray_z(o["px"], self.cal["R"], self.cal["t"], self.cal["K"], self.c["rack_z"]))

    def act(self, action):
        S = self.S; t = S["t"]; S["t"] += 1
        try:
            self._rack(t)
            if S["done"] or S["rc"] is None:
                return action
            q = _to_np(S["obs"]["robot"]["qpos"]).astype(float).reshape(-1)
            pl = self.kin.tcp("left", q[ARMS["left"]])[:3, 3]; pr = self.kin.tcp("right", q[ARMS["right"]])[:3, 3]
            g = float(q[GRIP_L]); req = float(_to_np(action).reshape(-1)[GRIP_L])
            held = self.c["grip_lo"] < g < self.c["grip_hi"] and float(np.linalg.norm(pr - pl)) >= self.c["free"]
            xy = float(np.hypot(*(pl - S["rc"])[:2])); above = float(pl[2] - (S["rc"][2] + self.c["rel_dz"]))
            if held and S["first_open"] is None:
                S["hany"] = req if S["hany"] is None else min(S["hany"], req)
                if req > S["hany"] + self.c["open_tol"]:
                    S["first_open"] = dict(t=t, xy_mm=round(xy * 1000, 1), above_mm=round(above * 1000, 1), in_zone=bool(xy <= self.c["zone"] and above > 0))
            if not held or xy > self.c["zone"]:
                if S["run"]:
                    S["done"] = True; S["release"] = "left the zone / not holding"
                S["hcmd"] = None; return action
            if above <= 0:
                if S["run"] or S["hcmd"] is not None:
                    S["done"] = True; S["release"] = "below the release line"
                return action
            S["hcmd"] = req if S["hcmd"] is None else min(S["hcmd"], req)
            if req > S["hcmd"] + self.c["open_tol"]:
                if S["z0"] is None:
                    S["z0"] = float(pl[2])
                elif float(pl[2]) - S["z0"] > self.c["lift_rel"]:
                    S["done"] = True; S["release"] = "student lifting while blocked"; return action
                S["run"] += 1; S["blocked"] += 1
                if S["run"] > self.c["max_block"]:
                    S["done"] = True; S["release"] = "max_block"; return action
                return self._set(action, GRIP_L, S["hcmd"])
        except Exception as ex:  # noqa: BLE001  (the helper never stops the episode; it just stops helping)
            self._ev(error=f"{type(ex).__name__}: {ex}"); S["done"] = True
        return action

    @staticmethod
    def _set(action, i, v):
        if hasattr(action, "clone"):
            a = action.clone(); a.reshape(-1)[i] = float(v); return a
        a = np.array(action, dtype=float, copy=True); a.reshape(-1)[i] = v
        return a.astype(np.asarray(action).dtype if np.asarray(action).dtype.kind == "f" else np.float32)
