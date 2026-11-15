"""Bell cover-top keypoint head, online use (one call per control step, any subset of the three cameras).
  - frozen V-JEPA 2.1 ViT-L encoder (512 x 384, the frame repeated as a 2-frame clip, last group) + a small heat-map head (head_vjepa.pt).
  - gate: visibility >= 0.5 and heat-map peak >= tau per camera (v2.json).
  - tracking: with a previous position, search only within 60 px of it; a stronger far peak must persist 2 calls before jumping;
    if nothing is found, keep the old position as memory (kind = "mem", confidence decays to 0 over 3 s), then search the whole image.
Usage:
  kp = BellKP(root, dev="cuda", rate_hz=10, half=True); kp.reset()
  out = kp.step({"cam_high": img})   # img uint8 [480, 640, 3] RGB
  # out[cam] = dict(px=(u, v) or None, kind="glob|loc|jump|mem|none", conf, passed, p_vis, peak)
  BellKP.ray_z(px, R, t, K, z0)      # pixel -> point on the world plane z = z0 (OpenGL camera axes)"""
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


class BellKP:
    def __init__(self, root, dev="cuda", rate_hz=10.0, half=True):
        self.dev, self.half = dev, half and str(dev).startswith("cuda")
        sys.path.insert(0, os.path.join(root, "code/RoboActualizer/src")); sys.path.insert(0, os.path.join(root, "pylib"))
        from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone
        self.enc = VJEPABackbone(os.path.join(root, "vjepa/vjepa2_1_vitl_384_ema_encoder.pt"), torch_hub_dir=os.path.join(root, "code/vjepa2"),
                                 image_size=(H_IN, W_IN), num_cameras=1, device=dev, dtype=torch.float32)
        self.head = Head(1).to(dev); self.head.load_state_dict(torch.load(os.path.join(root, "head_vjepa.pt"), map_location=dev)); self.head.eval()
        G = json.load(open(os.path.join(root, "v2.json")))["gates"]
        mem_max = int(round(MEM_S * rate_hz))
        self.tr = {c: _Track(G.get(f"{c}|button.cover_top", {}).get("tau"), mem_max) for c in CAMS}

    def reset(self):
        for t in self.tr.values():
            t.reset()

    @torch.no_grad()
    def _heat(self, imgs):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(self.dev)
        x = F.interpolate(x, size=(H_IN, W_IN), mode="bilinear", align_corners=False).round().clamp(0, 255).div(255)
        clip = (x * 2 - 1).unsqueeze(1).repeat(1, 2, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):   # encoder in fp16 autocast (half=False = fp32 as trained)
            z = self.enc.encode_clip_last_group(clip).float()
        f = z.transpose(1, 2).reshape(len(imgs), -1, GH, GW).half().float()
        h = self.head.a(f); hm = self.head.hm(F.interpolate(h, size=(96, 128), mode="bilinear", align_corners=False))
        return hm.flatten(2).softmax(-1).view(hm.shape)[:, 0].cpu().numpy(), self.head.vis(h.mean((2, 3))).sigmoid()[:, 0].cpu().numpy()

    def detect(self, img, cam, prev_uv=None):
        """Stateless single image: img uint8 [480, 640, 3]. With prev_uv, take the peak within 60 px of it (local soft-argmax), else the whole image.
        Returns dict(u, v, vis, peak, tau, passed); passed = vis >= 0.5 and peak >= the camera's gate. No jump / memory logic (that is step())."""
        P, V = self._heat([np.asarray(img, np.uint8)]); p, vp = P[0], float(V[0]); tau = self.tr[cam].tau
        if prev_uv is not None:
            m = ((GX - prev_uv[0]) ** 2 + (GY - prev_uv[1]) ** 2) <= R_PX * R_PX; pk = float((p * m).max()); u, v = softpos(p, m)
        else:
            pk = float(p.max()); u, v = softpos(p)
        return dict(u=u, v=v, vis=vp, peak=pk, tau=tau, passed=bool(tau is not None and vp >= 0.5 and pk >= tau and u is not None))

    def step(self, frames):
        cams = [c for c in CAMS if c in frames]
        P, V = self._heat([np.asarray(frames[c], np.uint8) for c in cams]); out = {}
        for i, c in enumerate(cams):
            px, kind, conf = self.tr[c].step(P[i], float(V[i]))
            out[c] = dict(px=px, kind=kind, conf=float(conf), passed=kind in ("glob", "loc", "jump"), p_vis=float(V[i]), peak=float(P[i].max()))
        return out

    @staticmethod
    def ray_z(uv, R, t, K, z0):
        R, t, K = np.asarray(R, float), np.asarray(t, float), np.asarray(K, float)
        d = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]); d = R @ np.array([d[0], -d[1], -d[2]]); s = (z0 - t[2]) / d[2]; return t + s * d


if __name__ == "__main__":
    import time
    root, npz = sys.argv[1:3]; Z = np.load(npz); kp = BellKP(root); kp.reset(); n = {c: 0 for c in CAMS}; t0 = time.time()   # self-test on a frames npz
    for j in range(len(Z["t"])):
        o = kp.step({c: Z[c][j] for c in CAMS})
        for c in CAMS:
            n[c] += o[c]["passed"]
    print("steps", len(Z["t"]), "ms/step", round((time.time() - t0) / len(Z["t"]) * 1000), "passed", n)
