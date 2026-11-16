"""决LB 11-15（总-481 ②，给决NR / 决NO）：通用目标点 V-JEPA 关键点头在线版（按铃 bell_kp_online.py 的多点版；搅拌开关 / 上样插孔 / 抽屉把手）。
root 目录里要有：head_vjepa.pt、gates.json（gate_camkp.py 出的，「相机|点名」门线）、kp_names.json（点名列表），以及 code/、pylib/、vjepa/（同 bell_kp 包）。
  kp = TargetKP(root, dev="cuda", rate_hz=10)；kp.reset()；out = kp.step({"cam_high": img, ...}) → out[cam][点名] = dict(px, kind, conf, passed, p_vis, peak)
  r = kp.detect(img, cam, prev_uv={点名: (u, v)} 或 None) → r[点名] = dict(u, v, vis, peak, tau, passed)
以下是按铃版原说明（规则相同，只是点从 1 个变成多个）：
按铃铃盖顶 V-JEPA 关键点头，在线版（每个控制步喂一次三路画面）。
和离线 kp_v2.py「跟踪 + 记忆」同一套规则，只把逐帧循环改成逐步调用：
  - 头：冻住的 V-JEPA 2.1 ViT-L（512×384，同一帧复制成 2 帧取最后一组）+ 小热图头 head_vjepa.pt（训在 kp_bell 仿真标签，11-14 04:46）。
  - 过线：可见性 ≥ 0.5 且热图峰值 ≥ τ（按相机：顶视 0.371 / 左腕 0.434 / 右腕 0.405，v2.json；仿真留出门后 中位 6.7 / p90 14.7 px）。
  - 跟踪：有上一帧位置 ⇒ 只在它附近 R = 60 px 里找；远处更强的峰置信度对折、且要连续 2 步都在那里才跳过去；
    找不到 ⇒ 保留旧位置当记忆，置信度 3 s 内线性降到 0（kind = "mem"），超过 3 s 回全图找。
用法：
  from bell_kp_online import BellKP
  kp = BellKP(root="<解包目录>", dev="cuda", rate_hz=10, half=True)   # rate_hz = 喂帧频率（记忆 3 s 按它换算成步数）；half = 编码器 fp16
  kp.reset()                                                # 每集开头
  out = kp.step({"cam_high": img, "cam_left_wrist": img, "cam_right_wrist": img})   # img: uint8 [480,640,3] RGB，可只给其中几路
  # out[cam] = dict(px=(u, v) 或 None, kind="glob|loc|jump|mem|none", conf=float, passed=bool, p_vis=float, peak=float)
  # passed = 这一步给出且不是记忆（kind ∈ glob / loc / jump）；mem 的 px 是被挡前的旧位置
  BellKP.ray_z(px, R, t, K, z0)   # 像素 → 世界平面 z0 上的点（GL 约定：R / t = 相机在世界里的朝向 / 位置，录制头 cameras 那套）
自测：python bell_kp_online.py <解包目录> <按铃 frames npz>（决LW 学生画面那种，键 cam_high / cam_left_wrist / cam_right_wrist）"""
import json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

H_IN, W_IN, GH, GW = 384, 512, 24, 32
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
R_PX, PERSIST, MEM_S = 60.0, 2, 3.0
XS = (np.arange(128) + 0.5) * 5.0; YS = (np.arange(96) + 0.5) * 5.0; GX, GY = np.meshgrid(XS, YS)


class Head(nn.Module):
    def __init__(self, K, D=1024):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(D, 256, 1), nn.GroupNorm(32, 256), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.hm = nn.Conv2d(256, K, 3, padding=1); self.vis = nn.Linear(256, K)


def softpos(p, mask=None):
    q = p if mask is None else p * mask; s = q.sum()
    return (float((q * GX).sum() / s), float((q * GY).sum() / s)) if s > 0 else (None, None)


class _Track:
    def __init__(self, tau, mem_max):
        self.tau, self.mem_max = tau, mem_max; self.reset()

    def reset(self):
        self.prev, self.conf_prev, self.age, self.cand, self.cand_n = None, 0.0, 0, None, 0

    def step(self, p, vp):
        tau = self.tau
        if tau is None:
            return None, "off", 0.0
        iy, ix = np.unravel_index(p.argmax(), p.shape); gpos = (XS[ix], YS[iy]); gpk = float(p.max())
        if self.prev is not None:
            pv = self.prev; m = ((GX - pv[0]) ** 2 + (GY - pv[1]) ** 2) <= R_PX * R_PX
            pk_loc = float((p * m).max()); lpos = softpos(p, m); far = np.hypot(gpos[0] - pv[0], gpos[1] - pv[1]) > R_PX
            if far and gpk > pk_loc and vp >= 0.5 and 0.5 * gpk >= tau:
                self.cand_n = self.cand_n + 1 if (self.cand is not None and np.hypot(gpos[0] - self.cand[0], gpos[1] - self.cand[1]) <= R_PX) else 1; self.cand = gpos
                if self.cand_n >= PERSIST:
                    self.prev = softpos(p, ((GX - gpos[0]) ** 2 + (GY - gpos[1]) ** 2) <= R_PX * R_PX); self.conf_prev, self.age, self.cand, self.cand_n = gpk, 0, None, 0
                    return self.prev, "jump", gpk
            else:
                self.cand, self.cand_n = None, 0
            if vp >= 0.5 and pk_loc >= tau and lpos[0] is not None:
                self.prev, self.conf_prev, self.age = lpos, pk_loc, 0; return lpos, "loc", pk_loc
            self.age += 1
            if self.age <= self.mem_max:
                return self.prev, "mem", self.conf_prev * (1 - self.age / (self.mem_max + 1))
            self.prev = None
        if vp >= 0.5 and gpk >= tau:
            self.prev, self.conf_prev, self.age = softpos(p), gpk, 0; return self.prev, "glob", gpk
        return None, "none", 0.0


class TargetKP:
    def __init__(self, root, dev="cuda", rate_hz=10.0, half=True):
        self.dev, self.half = dev, half and str(dev).startswith("cuda")
        sys.path.insert(0, os.path.join(root, "code/RoboActualizer/src")); sys.path.insert(0, os.path.join(root, "pylib"))
        from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone
        self.enc = VJEPABackbone(os.path.join(root, "vjepa/vjepa2_1_vitl_384_ema_encoder.pt"), torch_hub_dir=os.path.join(root, "code/vjepa2"),
                                 image_size=(H_IN, W_IN), num_cameras=1, device=dev, dtype=torch.float32)
        self.names = json.load(open(os.path.join(root, "kp_names.json")))
        self.head = Head(len(self.names)).to(dev); self.head.load_state_dict(torch.load(os.path.join(root, "head_vjepa.pt"), map_location=dev)); self.head.eval()
        G = json.load(open(os.path.join(root, "gates.json")))["gates"]
        mem_max = int(round(MEM_S * rate_hz))
        self.tr = {c: {n: _Track(G.get(f"{c}|{n}", {}).get("tau"), mem_max) for n in self.names} for c in CAMS}

    def reset(self):
        for d in self.tr.values():
            for t in d.values():
                t.reset()

    @torch.no_grad()
    def _heat(self, imgs):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(self.dev)
        x = F.interpolate(x, size=(H_IN, W_IN), mode="bilinear", align_corners=False).round().clamp(0, 255).div(255)
        clip = (x * 2 - 1).unsqueeze(1).repeat(1, 2, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):   # 编码器 fp16 自动混合（half=False 回到训练时的 fp32）
            z = self.enc.encode_clip_last_group(clip).float()
        f = z.transpose(1, 2).reshape(len(imgs), -1, GH, GW).half().float()
        h = self.head.a(f); hm = self.head.hm(F.interpolate(h, size=(96, 128), mode="bilinear", align_corners=False))
        return hm.flatten(2).softmax(-1).view(hm.shape).cpu().numpy(), self.head.vis(h.mean((2, 3))).sigmoid().cpu().numpy()   # [B,K,96,128], [B,K]

    def detect(self, img, cam, prev_uv=None):
        """无状态单张：prev_uv = {点名: (u, v)} 给了的点只在附近 60 px 里取峰。返回 {点名: dict(u, v, vis, peak, tau, passed)}。"""
        P, V = self._heat([np.asarray(img, np.uint8)]); out = {}
        for k, n in enumerate(self.names):
            p, vp, tau = P[0, k], float(V[0, k]), self.tr[cam][n].tau; pv = (prev_uv or {}).get(n)
            if pv is not None:
                m = ((GX - pv[0]) ** 2 + (GY - pv[1]) ** 2) <= R_PX * R_PX; pk = float((p * m).max()); u, v = softpos(p, m)
            else:
                pk = float(p.max()); u, v = softpos(p)
            out[n] = dict(u=u, v=v, vis=vp, peak=pk, tau=tau, passed=bool(tau is not None and vp >= 0.5 and pk >= tau and u is not None))
        return out

    def step(self, frames):
        cams = [c for c in CAMS if c in frames]
        P, V = self._heat([np.asarray(frames[c], np.uint8) for c in cams]); out = {}
        for i, c in enumerate(cams):
            out[c] = {}
            for k, n in enumerate(self.names):
                px, kind, conf = self.tr[c][n].step(P[i, k], float(V[i, k]))
                out[c][n] = dict(px=px, kind=kind, conf=float(conf), passed=kind in ("glob", "loc", "jump"), p_vis=float(V[i, k]), peak=float(P[i, k].max()))
        return out

    @staticmethod
    def ray_z(uv, R, t, K, z0):
        R, t, K = np.asarray(R, float), np.asarray(t, float), np.asarray(K, float)
        d = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]); d = R @ np.array([d[0], -d[1], -d[2]]); s = (z0 - t[2]) / d[2]; return t + s * d


if __name__ == "__main__":
    import time
    root, npz = sys.argv[1:3]; Z = np.load(npz); kp = TargetKP(root); kp.reset(); t0 = time.time(); n = 0
    for j in range(min(10, len(Z["t"]))):
        o = kp.step({c: Z[c][j] for c in CAMS if c in Z.files}); n += 1
    print("步数", n, "每步", round((time.time() - t0) / n * 1000), "ms", {c: {k: v["kind"] for k, v in d.items()} for c, d in o.items()})
