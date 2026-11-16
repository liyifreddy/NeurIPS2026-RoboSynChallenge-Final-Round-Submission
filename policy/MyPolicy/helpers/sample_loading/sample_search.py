# ----------------------------------------------------------------------------
# helpers/sample_loading/sample_search.py -- sample-loading "find the hole and push in" helper (Yi's method, 总-593①).
# Off unless task_policies.sample_loading.helper.enabled (mode: search). Same algorithm and parameters as the
# simulation plugin plugin_sample_search.py (决NR2); reads only obs robot/qpos + cam_high (rack centre once at
# the start, same RackKP / calibration / FK as sample_hold.py in this folder).
#
# Trigger (only after step t_min and after a real grasp was seen: the left gripper read >= grip_open_seen (open) earlier
# in the episode and has since closed into the holding window -- a gripper closed at the start does not count; 决NO 10:0x,
# V16 smoke took over at step 48): left gripper reading in the holding window, right TCP >= free_mm away, left TCP within zone_mm of the
# rack centre and band_lo..band_hi above it, and the student commands the left gripper open (more open than the
# running minimum by open_tol) -> take the LEFT arm (orientation frozen at that moment, gripper kept at the
# holding command).
# down    : go straight down 1 mm / step until resistance (commanded TCP below measured by > res_mm for 3 steps,
#           or < 2 mm descent over 15 steps) -> h0. If it went down >= direct_mm first, the tube was already in a
#           hole -> release.
# search  : light press (command z = measured z - press_mm), Archimedean spiral around the contact point
#           (pitch spiral_mm / turn, step arc_mm, radius <= spiral_max_mm). Slipped in when measured z < h0 - slip_mm;
#           laterally blocked (|cmd - measured| xy > lat_mm for 5 steps) -> treat as in a hole.
# insert  : straight down 1 mm / step until resistance or ins_max_mm below h0.
# release : hold 5 steps, open the gripper linearly over open_steps, then lift lift_mm, hand back with a <= 20-step
#           blend toward the student's left-arm command (<= 0.05 rad / step, same as the simulation shell's HELP_BLEND).
# Bail out (release in place): spiral radius reached, search_max steps, total_max steps, dragged > drag_mm,
#           pushed up > 5 mm while searching. One press per episode; one jsonl line per episode.
# Units: obs qpos[6] / action[6] = left gripper 0-1 (0 closed); joints rad; TCP m.
# ----------------------------------------------------------------------------
import json
import math
import os

import numpy as np

from .sample_hold import ARMS, GRIP_L, HERE, Kin, RackKP, _to_np, load_calib, ray_z

TCP_STEP = 0.003        # commanded TCP may move at most 3 mm / step (same as the simulation shell)
MAX_RAD = 0.05          # per-joint change per step


def _rotlog(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _ik_step(kin, arm, q, Td, iters=2, h=1e-4, lam=1e-4):
    q = np.asarray(q, float).copy()
    def err(T):
        return np.r_[Td[:3, 3] - T[:3, 3], _rotlog(Td[:3, :3] @ T[:3, :3].T)]
    for _ in range(iters):
        e = err(kin.tcp(arm, q)); J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6); dq[i] = h
            J[:, i] = (e - err(kin.tcp(arm, q + dq))) / h
        q = q + np.clip(J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e), -0.1, 0.1)
    return q


class SampleSearch:
    def __init__(self, cfg, log_dir=None):
        g = lambda k, d: cfg.get(k, d)
        f = lambda k, d: float(g(k, d))
        self.c = dict(platform=str(g("platform", "avg")), rack_z=f("rack_z", 0.899), rack_steps=int(g("rack_steps", 30)), kp_every=int(g("kp_every", 2)),
                      grip_lo=f("grip_held_lo", 0.10), grip_hi=f("grip_held_hi", 0.8), open_tol=f("open_tol", 0.04),
                      free=f("free_mm", 80) / 1000, zone=f("zone_mm", 120) / 1000, band_lo=f("band_lo_mm", 0) / 1000, band_hi=f("band_hi_mm", 250) / 1000,
                      dstep=f("down_step_mm", 1) / 1000, res=f("res_mm", 5) / 1000, press=f("press_mm", 3) / 1000,
                      sp=f("spiral_mm", 2.5) / 1000, sp_max=f("spiral_max_mm", 12) / 1000, arc=f("arc_mm", 1.5) / 1000,
                      slip=f("slip_mm", 8) / 1000, lat=f("lat_mm", 4) / 1000, drag=f("drag_mm", 10) / 1000, ins_max=f("ins_max_mm", 75) / 1000,
                      direct=f("direct_mm", 20) / 1000, open_n=int(g("open_steps", 12)), lift=f("lift_mm", 60) / 1000,
                      search_max=int(g("search_max", 200)), total_max=int(g("total_max", 360)),
                      t_min=int(g("t_min", 150)), grip_open_seen=f("grip_open_seen", 0.8), debug_trace=bool(g("debug_trace", False)), require_grasp=bool(g("require_grasp", True)),
                      ep_max=int(g("ep_max", 500)), reserve=int(g("reserve_steps", 60)), min_left=int(g("min_left_steps", 100)),
                      hold_cmd=(None if g("hold_cmd", None) is None else float(g("hold_cmd", None))))   # C: after take-over pin the left gripper command to this (0-1; 0 = fully closed); None = the student's holding command
        import hashlib
        cp = os.path.join(HERE, "calib", f"platform_{self.c['platform']}.json")
        want_c = str(g("calib_sha256", ""))
        if want_c and not hashlib.sha256(open(cp, "rb").read()).hexdigest().startswith(want_c):
            raise ValueError(f"sample helper calibration {cp}: sha256 mismatch")
        self.cal = load_calib(cp)
        self.kin = Kin(os.path.join(HERE, "robot", "CobotMagicWithGripperV100.urdf"))
        enc = os.path.expandvars(str(g("encoder", "${MYPOLICY_CKPT_ROOT}/shared/vjepa2_1_vitl_384_ema_encoder.pt")))
        head = os.path.expandvars(str(g("head", "${MYPOLICY_CKPT_ROOT}/finals_1115/sample_rack/head_vjepa.pt")))
        want = str(g("head_sha256", "c45cbdbc81f19a1a"))
        if not hashlib.sha256(open(head, "rb").read()).hexdigest().startswith(want):
            raise ValueError(f"sample helper head {head}: sha256 mismatch")
        self.kp = RackKP(head, enc, dev=g("device", "cuda"), rate_hz=10.0 / self.c["kp_every"])
        self.log_dir = log_dir; self.ep = 0; self.S = None

    # --- episode bookkeeping
    def start_episode(self):
        self.kp.reset()
        self.S = dict(t=0, obs=None, ests=[], rc=None, off=False, hmin=None, ph="watch", ev=[], rec={}, seen_open=False, grasped=False)

    def end_episode(self):
        if self.S is None:
            return
        rec = dict(kind="sample_search", episode=self.ep, rack_c=None if self.S["rc"] is None else np.round(self.S["rc"], 4).tolist(),
                   phase_end=self.S["ph"], search=self.S["rec"], events=self.S["ev"], cfg=self.c)
        print("[sample_search] " + json.dumps({k: rec[k] for k in ("episode", "rack_c", "phase_end", "search")}, default=str), flush=True)
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, f"sample_search_{os.getpid()}.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass
        self.ep += 1; self.S = None

    def _ev(self, **k):
        e = dict(t=self.S["t"], **k); self.S["ev"].append(e)
        # append each event immediately (the official eval may never call end_episode / close_model -- 决MV V16 1116g)
        line = dict(type="sample_search_event", episode=self.ep, rec=self.S["rec"], **e)
        print("[sample_search] " + json.dumps(line, default=str), flush=True)
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, f"sample_search_{os.getpid()}.jsonl"), "a") as fh:
                    fh.write(json.dumps(line, default=str) + "\n")
            except OSError:
                pass

    def observe(self, obs):
        if self.S is None:
            self.start_episode()
        self.S["obs"] = obs

    # --- helpers
    def _rack(self, t):
        S = self.S
        if S["rc"] is not None or S["off"]:
            return
        if t > self.c["rack_steps"]:
            if S["ests"]:
                S["rc"] = np.median(np.stack(S["ests"]), axis=0)
            else:
                S["off"] = True; self._ev(kind="rack not seen, helper off this episode")
            return
        if t % self.c["kp_every"]:
            return
        o = self.kp.centre(_to_np(S["obs"]["sensor"]["cam_high"]["color"]).reshape(480, 640, -1))
        if o.get("passed") and o.get("px") is not None:
            S["ests"].append(ray_z(o["px"], self.cal["R"], self.cal["t"], self.cal["K"], self.c["rack_z"]))

    def _drive(self, action, Td):
        """Move the commanded left q one step toward Td (TCP step <= 3 mm, joints <= 0.05 rad); return the action with the left arm and gripper set."""
        S = self.S; Tc = self.kin.tcp("left", S["qc"]); Td = np.asarray(Td, float).copy()
        dv = Td[:3, 3] - Tc[:3, 3]; n = float(np.linalg.norm(dv))
        if n > TCP_STEP:
            Td[:3, 3] = Tc[:3, 3] + dv * (TCP_STEP / n)
        qn = _ik_step(self.kin, "left", S["qc"], Td)
        S["qc"] = S["qc"] + np.clip(qn - S["qc"], -MAX_RAD, MAX_RAD)
        return self._put(action, S["qc"], S["grip"])

    @staticmethod
    def _put(action, qc, grip):
        """Write the left arm + left gripper into a copy of the action, keeping its type / shape / dtype
        (the eval chain passes a torch tensor [1, 14]; env.step accepts only tensors -- 决MV V16 smoke)."""
        if hasattr(action, "clone"):
            a = action.clone(); flat = a.reshape(-1)
            flat[ARMS["left"]] = a.new_tensor(np.asarray(qc, float)); flat[GRIP_L] = float(grip)
            return a
        x = np.asarray(action); a = np.array(x, dtype=float, copy=True); flat = a.reshape(-1)
        flat[ARMS["left"]] = qc; flat[GRIP_L] = grip
        return a.astype(x.dtype if x.dtype.kind == "f" else np.float32)

    def _T(self, xy, z):
        T = np.eye(4); T[:3, :3] = self.S["R"]; T[:3, 3] = [xy[0], xy[1], z]; return T

    def _contact(self, za):
        S = self.S; zc = float(self.kin.tcp("left", S["qc"])[2, 3])
        S["zh"].append(za); S["zh"] = S["zh"][-16:]
        S["rn"] = S["rn"] + 1 if (za - zc) > self.c["res"] else 0
        return S["rn"] >= 3 or (len(S["zh"]) >= 16 and (S["zh"][0] - S["zh"][-1]) < 0.002)

    def _rel(self, Ta, why):
        S = self.S; S.update(ph="release", rel_k=0, rel_xy=Ta[:2, 3].copy(), rel_z=float(Ta[2, 3])); S["rec"]["end"] = why

    def act(self, action):
        S = self.S; t = S["t"]; S["t"] += 1
        try:
            self._rack(t)
            if S["off"] or S["rc"] is None or S["ph"] == "done":
                return action
            q = _to_np(S["obs"]["robot"]["qpos"]).astype(float).reshape(-1)
            Ta = self.kin.tcp("left", q[ARMS["left"]]); pl = Ta[:3, 3]; pr = self.kin.tcp("right", q[ARMS["right"]])[:3, 3]
            if self.c["debug_trace"] and self.log_dir and t < 2000:     # 调试用（缺省关）：每步一行，看各道门
                try:
                    a_ = _to_np(action).reshape(-1)
                    with open(os.path.join(self.log_dir, f"sample_search_trace_{os.getpid()}.jsonl"), "a") as fh:
                        fh.write(json.dumps(dict(ep=self.ep, t=t, ph=S["ph"], g=round(float(q[GRIP_L]), 4), gr=round(float(q[13]), 4), req=round(float(a_[GRIP_L]), 4),
                                                 xy_mm=round(float(np.hypot(*(pl - S["rc"])[:2])) * 1000, 1), dz_mm=round(float(pl[2] - S["rc"][2]) * 1000, 1),
                                                 free_mm=round(float(np.linalg.norm(pr - pl)) * 1000, 1), seen_open=S["seen_open"], grasped=S["grasped"], hmin=S["hmin"])) + "\n")
                except Exception:  # noqa: BLE001
                    pass
            if S["ph"] == "watch":
                req = float(_to_np(action).reshape(-1)[GRIP_L]); g = float(q[GRIP_L])
                if g >= self.c["grip_open_seen"]:
                    S["seen_open"] = True
                elif S["seen_open"] and self.c["grip_lo"] < g < self.c["grip_hi"]:
                    S["grasped"] = True
                if not self.c["require_grasp"]:
                    S["grasped"] = True              # 调试用：关掉「见过张开再夹住」那道门
                if t < self.c["t_min"] or not S["grasped"] or t > self.c["ep_max"] - self.c["min_left"]:
                    return action                 # too early / never grasped / too little of the episode left to finish (官方上样一集 500 步，总-604①)
                held = self.c["grip_lo"] < g < self.c["grip_hi"] and float(np.linalg.norm(pr - pl)) >= self.c["free"]
                dz = float(pl[2] - S["rc"][2]); inz = float(np.hypot(*(pl - S["rc"])[:2])) <= self.c["zone"] and self.c["band_lo"] < dz < self.c["band_hi"]
                if not (held and inz):
                    S["hmin"] = None; return action
                S["hmin"] = req if S["hmin"] is None else min(S["hmin"], req)
                if req <= S["hmin"] + self.c["open_tol"]:
                    return action
                if self.c["hold_cmd"] is not None:
                    S["hmin"] = self.c["hold_cmd"]      # C 版（总-615①）：接手后夹紧，管不在爪里往上滑；慢松也从这个值张开
                S.update(ph="down", n=0, k=0, grip=S["hmin"], R=Ta[:3, :3].copy(), qc=q[ARMS["left"]].copy(), cz=float(pl[2]), z_take=float(pl[2]), xy=pl[:2].copy(), zh=[], rn=0, ln=0, th=0.0)
                S["rec"].update(t_take=t, dz_mm=round(dz * 1000, 1)); self._ev(kind="take")
            S["n"] += 1; S["k"] += 1; za = float(pl[2]); c = self.c
            if S["n"] > c["total_max"] and S["ph"] in ("down", "search", "insert"):
                self._rel(Ta, "total_max")
            elif t >= c["ep_max"] - c["reserve"] and S["ph"] in ("down", "search", "insert"):
                self._rel(Ta, "episode time up")     # leave reserve_steps (release 37 + blend <= 20) so the hand-back finishes inside the episode
            if S["ph"] == "down":
                if self._contact(za):
                    S.update(h0=za, c=pl[:2].copy(), th=0.0, ln=0, k=0, ph="search"); S["rec"].update(h0=round(za, 4), down_mm=round((S["z_take"] - za) * 1000, 1))
                    if S["z_take"] - za >= c["direct"]:
                        S["rec"]["slip"] = "already in a hole"; self._rel(Ta, "bottomed right away")
                else:
                    S["cz"] -= c["dstep"]; return self._drive(action, self._T(S["xy"], S["cz"]))
            if S["ph"] == "search":
                Tc = self.kin.tcp("left", S["qc"]); dl = float(np.linalg.norm(Tc[:2, 3] - pl[:2])); S["ln"] = S["ln"] + 1 if dl > c["lat"] else 0
                r_now = c["sp"] * S["th"] / (2 * math.pi)
                if za > S["h0"] + 0.005:
                    self._rel(Ta, "pushed up while searching")
                elif za < S["h0"] - c["slip"]:
                    S.update(ph="insert", cz=za, zh=[], rn=0); S["rec"].update(slip="slipped in", search_steps=S["k"])
                elif dl > c["drag"]:
                    self._rel(Ta, "dragged")
                elif r_now >= c["sp_max"] or S["k"] > c["search_max"]:
                    S["rec"]["search_steps"] = S["k"]; self._rel(Ta, "spiral done, no hole")
                elif S["ln"] >= 5:
                    S.update(ph="insert", cz=za, zh=[], rn=0); S["rec"].update(slip="laterally blocked = in a hole", search_steps=S["k"])
                else:
                    S["th"] += c["arc"] / max(r_now, 0.002); r = min(c["sp_max"], c["sp"] * S["th"] / (2 * math.pi))
                    xy = S["c"] + r * np.array([math.cos(S["th"]), math.sin(S["th"])])
                    return self._drive(action, self._T(xy, za - c["press"]))
            if S["ph"] == "insert":
                if self._contact(za) or (S["h0"] - za) >= c["ins_max"]:
                    S["rec"]["ins_mm"] = round((S["h0"] - za) * 1000, 1); self._rel(Ta, "pushed in")
                else:
                    S["cz"] -= c["dstep"]; return self._drive(action, self._T(pl[:2], S["cz"]))
            if S["ph"] == "release":
                S["rel_k"] += 1; k = S["rel_k"]; hold = S["hmin"]
                S["grip"] = hold if k <= 5 else (hold + (1.0 - hold) * min(1.0, (k - 5) / c["open_n"]))
                z = S["rel_z"] + max(0, k - 5 - c["open_n"]) * 0.003
                out = self._drive(action, self._T(S["rel_xy"], min(z, S["rel_z"] + c["lift"])))
                if z >= S["rel_z"] + c["lift"]:
                    S["ph"] = "blend"; S["bk"] = 0; S["rec"]["rel_steps"] = k; self._ev(kind="hand back (blending)")
                return out
            if S["ph"] == "blend":          # same as the simulation shell's HELP_BLEND: follow the student's left arm at <= 0.05 rad / step, <= 20 steps
                a_stu = _to_np(action).reshape(-1).astype(float); d = a_stu[ARMS["left"]] - S["qc"]; S["bk"] += 1
                if float(np.max(np.abs(d))) <= MAX_RAD or S["bk"] > 20:
                    S["ph"] = "done"; self._ev(kind="hand back done", steps=S["bk"]); return action
                S["qc"] = S["qc"] + np.clip(d, -MAX_RAD, MAX_RAD)
                return self._put(action, S["qc"], float(a_stu[GRIP_L]))
        except Exception as ex:  # noqa: BLE001  (the helper never stops the episode; it just stops helping)
            self._ev(error=f"{type(ex).__name__}: {ex}"); S["ph"] = "done"
        return action
