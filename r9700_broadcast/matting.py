"""Matting engines: turn a camera frame into (alpha, straight foreground colour) with no green screen.

Everything here works on CUDA/HIP tensors shaped (1, C, H, W), float16 or float32, range [0, 1].

Engines
-------
RVMEngine        Robust Video Matting. Recurrent (temporally stable), ~7 ms @1080p on one R9700.
                 Predicts alpha *and* the un-premultiplied foreground colour, like CorridorKey does
                 for green screen plates.
BiRefNetEngine   BiRefNet-matting (MIT). Per-frame, sharper hair/edge detail, ~30-60 ms. Output is
                 upsampled with a guided filter and foreground colour recovered with blur-fusion.
PlateAssist      Optional clean-plate helper (shoot the empty room once, then step in). Uses a
                 difference matte to rescue held objects the AI ignores and to kill false positives
                 such as an empty chair. Classic VFX trick, no green screen needed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("R9700_BROADCAST_MODELS", os.path.join(ROOT, "models"))


@dataclass
class Matte:
    alpha: torch.Tensor          # (1,1,H,W)
    fgr: torch.Tensor | None     # (1,3,H,W) straight (un-premultiplied) foreground colour
    src: torch.Tensor | None = None  # the frame this matte belongs to, when it is not the latest


# --------------------------------------------------------------------------------------------
# Image filters
# --------------------------------------------------------------------------------------------

def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window using cumulative sums (O(1) in r), edge-normalised."""
    if r <= 0:
        return x
    def _1d(t, dim):
        n = t.shape[dim]
        c = torch.cumsum(t.float(), dim=dim)
        idx_hi = torch.clamp(torch.arange(n, device=t.device) + r, max=n - 1)
        idx_lo = torch.arange(n, device=t.device) - r - 1
        hi = c.index_select(dim, idx_hi)
        lo = c.index_select(dim, torch.clamp(idx_lo, min=0)) * (idx_lo >= 0).view(
            *[1] * dim, n, *[1] * (t.dim() - dim - 1)).to(c.dtype)
        cnt = (idx_hi - torch.clamp(idx_lo, min=-1)).to(c.dtype).view(*[1] * dim, n, *[1] * (t.dim() - dim - 1))
        return ((hi - lo) / cnt).to(t.dtype)
    return _1d(_1d(x, 2), 3)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Fast guided filter (He et al.): low-res coefficients, applied to full-res guide.

    guide: (1,3,H,W) full res; src: (1,1,h,w) any res. Returns (1,1,H,W).
    """
    H, W = guide.shape[-2:]
    h, w = src.shape[-2:]
    g_lo = F.interpolate(guide, size=(h, w), mode="bilinear", align_corners=False).float()
    s = src.float()
    gray = g_lo.mean(1, keepdim=True)
    mean_g = box_filter(gray, r)
    mean_s = box_filter(s, r)
    cov = box_filter(gray * s, r) - mean_g * mean_s
    var = box_filter(gray * gray, r) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    a = F.interpolate(box_filter(a, r), size=(H, W), mode="bilinear", align_corners=False)
    b = F.interpolate(box_filter(b, r), size=(H, W), mode="bilinear", align_corners=False)
    out = a * guide.float().mean(1, keepdim=True) + b
    return out.clamp_(0, 1).to(guide.dtype)


def blur_fusion_foreground(img: torch.Tensor, alpha: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """Approximate fast foreground colour estimation (Forte & Pitie 2021, 'blur fusion' x2).

    Removes background colour bleeding into semi-transparent edges (hair), giving a straight
    foreground suitable for compositing over any new background.
    """
    H, W = img.shape[-2:]
    sz = (max(8, int(H * scale)), max(8, int(W * scale)))
    I = F.interpolate(img.float(), size=sz, mode="bilinear", align_corners=False)
    a = F.interpolate(alpha.float(), size=sz, mode="bilinear", align_corners=False)

    def step(Fg, Bg, r):
        ba = box_filter(a, r)
        bF = box_filter(Fg * a, r) / (ba + 1e-5)
        bB = box_filter(Bg * (1 - a), r) / ((1 - ba) + 1e-5)
        Fg = bF + a * (I - a * bF - (1 - a) * bB)
        return Fg.clamp(0, 1), bB

    r1 = max(2, int(90 * scale * min(H, W) / 1080))
    Fg, Bg = step(I, I, r1)
    Fg, Bg = step(Fg, Bg, max(1, r1 // 15))
    # Recombine: low-res colour correction applied as a delta onto full-res detail.
    delta = F.interpolate(Fg - I, size=(H, W), mode="bilinear", align_corners=False)
    return (img.float() + delta).clamp_(0, 1).to(img.dtype)


# --------------------------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------------------------

class RVMEngine:
    name = "rvm"

    def __init__(self, device="cuda:0", variant="resnet50", dtype=torch.float16):
        sys.path.insert(0, os.path.join(ROOT, "third_party", "RobustVideoMatting"))
        from model import MattingNetwork  # GPL-3.0, see third_party/RobustVideoMatting/LICENSE
        self.device, self.dtype = torch.device(device), dtype
        net = MattingNetwork(variant).eval()
        net.load_state_dict(torch.load(os.path.join(MODELS, f"rvm_{variant}.pth"), map_location="cpu"))
        self.net = net.to(self.device, dtype)
        self.rec = [None] * 4
        self.downsample_ratio = None  # None -> auto
        # HIP graph replay removes ~3.5 ms/frame of CPU kernel-launch overhead at 1080p
        self.use_graph = os.environ.get("R9700_BROADCAST_GRAPHS", "1") == "1"
        self._graph = None
        self._graph_shape = None
        self._eager_frames = 0

    def reset(self):
        self.rec = [None] * 4
        self._graph = None
        self._eager_frames = 0

    def _capture(self, src, ratio):
        self.static_src = src.to(self.dtype).clone()
        self.static_rec = [r.clone() for r in self.rec]
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm the allocator on the side stream
                self.net(self.static_src, *self.static_rec, downsample_ratio=ratio)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        # thread_local: other threads (e.g. the semantic net on the 2nd GPU) may keep launching work
        # while we capture; "global" mode would invalidate the capture / fail their launches
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            self.g_fgr, self.g_pha, *self.g_rec = self.net(self.static_src, *self.static_rec,
                                                           downsample_ratio=ratio)
        self._graph = g
        self._graph_shape = (tuple(src.shape), ratio)

    @torch.no_grad()
    def __call__(self, src: torch.Tensor) -> Matte:
        H, W = src.shape[-2:]
        ratio = self.downsample_ratio or min(1.0, 512 / max(H, W))
        key = (tuple(src.shape), ratio)
        if self.use_graph and self._graph is not None and self._graph_shape == key:
            self.static_src.copy_(src)
            self._graph.replay()
            for s_, o in zip(self.static_rec, self.g_rec):
                s_.copy_(o)
            return Matte(self.g_pha.clone(), self.g_fgr.clone())
        fgr, pha, *self.rec = self.net(src.to(self.dtype), *self.rec, downsample_ratio=ratio)
        self._eager_frames += 1
        if self.use_graph and self._eager_frames >= 3 and self.rec[0] is not None:
            try:
                self._capture(src, ratio)
            except Exception as e:  # noqa: BLE001 - graphs are an optimisation only
                print(f"[rvm] graph capture failed, staying eager: {e}")
                self.use_graph = False
                self._graph = None
        return Matte(pha, fgr)


def _patch_torchvision_ops():
    from .ops import deform_conv2d, torchvision_has_ops
    if not torchvision_has_ops():
        import torchvision.ops
        torchvision.ops.deform_conv2d = deform_conv2d


class BiRefNetEngine:
    name = "birefnet"
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, device="cuda:0", model="birefnet_lite_matting", size=(576, 1024),
                 dtype=torch.float16, decontaminate=True):
        _patch_torchvision_ops()
        from transformers import AutoModelForImageSegmentation
        self.device, self.dtype, self.size = torch.device(device), dtype, size
        self.decontaminate = decontaminate
        net = AutoModelForImageSegmentation.from_pretrained(os.path.join(MODELS, model), trust_remote_code=True)
        self.net = net.eval().to(self.device, dtype)
        self.mean = torch.tensor(self.MEAN, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=self.device, dtype=dtype).view(1, 3, 1, 1)

    def reset(self):
        pass

    def prep(self, src: torch.Tensor) -> torch.Tensor:
        return F.interpolate(src.to(self.device, self.dtype), size=self.size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def forward_small(self, x: torch.Tensor) -> torch.Tensor:
        """Network only: (1,3,h,w) resized frame -> (1,1,h,w) alpha in [0,1]."""
        out = self.net((x.to(self.device, self.dtype) - self.mean) / self.std)
        return torch.sigmoid(out[-1].float())

    @torch.no_grad()
    def finish(self, src: torch.Tensor, lo: torch.Tensor) -> Matte:
        """Full-res edge-aware upsample + straight-colour foreground recovery."""
        alpha = guided_filter(src, lo.to(src.device), r=2, eps=1e-4)
        fgr = blur_fusion_foreground(src, alpha) if self.decontaminate else None
        return Matte(alpha.to(src.dtype), fgr)

    @torch.no_grad()
    def __call__(self, src: torch.Tensor) -> Matte:
        src = src.to(self.device, self.dtype)
        return self.finish(src, self.forward_small(self.prep(src)))


def _gpu_worker(device, model, size, inq, outq):
    import torch as _t
    eng = BiRefNetEngine(device, model, size)
    outq.put("ready")
    while True:
        item = inq.get()
        if item is None:
            break
        x = item.to(device)
        outq.put(eng.forward_small(x).half().cpu())


class MultiGPU:
    """Spread a per-frame engine over several GPUs, one *process* per extra GPU (threads in one
    process serialise on the GIL / HIP runtime; separate processes scale ~1.5x on 2x R9700).

    The main process keeps one engine on its own GPU and does the full-res finishing. Calls are
    pipelined: the returned Matte carries `src`, the frame it belongs to (one frame behind)."""

    def __init__(self, devices, model, size=(576, 1024)):
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.local = BiRefNetEngine(devices[0], model, size)
        self.device = self.local.device
        self.name = f"birefnet x{len(devices)}"
        self.workers = []
        for d in devices[1:]:
            iq, oq = ctx.Queue(2), ctx.Queue(2)
            p = ctx.Process(target=_gpu_worker, args=(d, model, size, iq, oq), daemon=True)
            p.start()
            self.workers.append((p, iq, oq))
        for _, _, oq in self.workers:
            assert oq.get(timeout=600) == "ready"
        self.n = 1 + len(self.workers)
        self.i = 0
        self.pending = []   # list of (src, worker_index or None, local_result)

    def reset(self):
        self._drain()

    def _drain(self):
        while self.pending:
            src, wi, res = self.pending.pop(0)
            if wi is not None:
                self.workers[wi][2].get()

    def close(self):
        self._drain()
        for p, iq, _ in self.workers:
            iq.put(None)
            p.join(timeout=5)

    def __call__(self, src: torch.Tensor) -> Matte:
        slot = self.i % self.n
        self.i += 1
        small = self.local.prep(src)
        if slot == 0:
            self.pending.append((src, None, self.local.forward_small(small)))
        else:
            self.workers[slot - 1][1].put(small.cpu())
            self.pending.append((src, slot - 1, None))
        if len(self.pending) < self.n:
            # warm-up: answer synchronously while leaving the entry in flight to fill the pipeline
            m = self.local(src)
            m.src = src
            return m
        s0, wi, res = self.pending.pop(0)
        lo = res if wi is None else self.workers[wi][2].get().to(self.device)
        m = self.local.finish(s0, lo)
        m.src = s0
        return m


class TemporalStabilizer:
    """Motion-adaptive alpha smoothing for per-frame engines: where the image did not change,
    blend toward the previous alpha (kills edge shimmer); where it moved, trust the new alpha."""

    def __init__(self, strength=0.6):
        self.strength = strength
        self.prev_a = None
        self.prev_img = None

    def reset(self):
        self.prev_a = self.prev_img = None

    def __call__(self, img, alpha):
        if self.strength <= 0 or self.prev_a is None or self.prev_a.shape != alpha.shape:
            self.prev_a, self.prev_img = alpha, img
            return alpha
        small = lambda t: F.avg_pool2d(t.float(), 4)
        motion = (small(img) - small(self.prev_img)).abs().amax(1, keepdim=True)
        still = (1 - (motion / 0.06).clamp(0, 1))
        k = F.interpolate(still * self.strength, size=alpha.shape[-2:], mode="bilinear", align_corners=False)
        out = (alpha.float() * (1 - k) + self.prev_a.float() * k).to(alpha.dtype)
        self.prev_a, self.prev_img = out, img
        return out


class PlateAssist:
    """Clean-plate difference matte fused with the AI matte.

    capture(frame) a few times with nobody in shot; then each frame:
      rescue  - pixels that differ strongly from the plate *near* the subject become foreground
                (a mug, a guitar, a product you hold up to camera)
      suppress- pixels the AI marked foreground but that match the plate are removed
                (empty chair, poster with a face on it)
    Robust to webcam auto-exposure (per-channel gain fitted on background pixels only) and to
    sensor noise / flicker (per-pixel noise map measured while capturing the plate).
    """

    LO = 4  # work at 1/4 resolution
    full = None
    chair_mask = None

    def __init__(self):
        self.plate = None      # (1,3,h,w) low-res median plate
        self.noise = None      # (1,1,h,w) per-pixel temporal std
        self.full_shape = None
        self._acc = []
        self.rescue = True
        self.suppress = True
        self.threshold = 0.10   # colour distance (above noise) where a pixel counts as 'changed'
        self.softness = 0.05
        self.reach = 0.20       # how far (fraction of width) from the subject rescue may extend
        self.furniture = 0.0    # 0 = safest; 1 = also remove plate-matching stuff right next to you

    @property
    def ready(self):
        return self.plate is not None

    def clear(self):
        self.plate = self.noise = None
        self.full = None
        self.chair_mask = None
        self._acc = []

    def add_sample(self, img: torch.Tensor, n_target=30):
        if len(self._acc) % 6 == 0:            # a few full-res frames for the chair cut-out
            self._full_acc = ([] if not self._acc else getattr(self, "_full_acc", [])) + [img.detach().half()]
        self._acc.append(F.avg_pool2d(img.detach().float(), self.LO))
        if len(self._acc) >= n_target:
            stack = torch.cat(self._acc)
            self.plate = stack.median(0, keepdim=True).values
            self.full = torch.cat(self._full_acc).float().mean(0, keepdim=True).half()
            self._full_acc = []
            self.chair_mask = None
            self.noise = stack.std(0, keepdim=True).mean(1, keepdim=True)
            self.noise = F.max_pool2d(self.noise, 5, 1, 2)  # flicker spreads a little
            self.full_shape = img.shape
            self._acc = []
            return True
        return False

    @torch.no_grad()
    def compute_chair_mask(self, semantic: "SemanticGuard", device_for_birefnet: str):
        """One-off: cut the chair out of the *empty-room* plate. With nobody in shot the chair is the
        salient object, so BiRefNet gives a crisp matte; DeepLab's chair class gates it so other
        salient things (a poster, a lamp) are not mistaken for the chair."""
        if self.full is None:
            return None
        eng = BiRefNetEngine(device_for_birefnet, "birefnet_lite_matting", decontaminate=False)
        sal = eng(self.full.to(eng.device)).alpha.float().to(self.full.device)
        del eng
        torch.cuda.empty_cache()
        x = F.interpolate(self.full, size=semantic.SIZE, mode="bilinear", align_corners=False)
        probs = semantic._run_net(x.to(semantic.net_device)).to(self.full.device)
        chair = F.interpolate(probs[:, 1:2], size=self.full.shape[-2:], mode="bilinear", align_corners=False)
        gate = F.max_pool2d(F.avg_pool2d((chair > 0.2).float(), 8), 9, 1, 4)
        gate = F.interpolate(gate, size=self.full.shape[-2:], mode="bilinear", align_corners=False)
        self.chair_mask = (sal * gate).clamp(0, 1).half()
        return self.chair_mask

    def changed_full(self, img: torch.Tensor) -> torch.Tensor:
        """Full-res soft map of pixels that differ from the plate (e.g. you, in front of the chair)."""
        a_lo = torch.zeros_like(F.avg_pool2d(img[:, :1].float(), self.LO))
        d = self.difference(img, a_lo)
        ch = ((d - self.threshold) / self.softness + 0.5).clamp(0, 1)
        ch = F.max_pool2d(ch, 3, 1, 1)
        return F.interpolate(ch, size=img.shape[-2:], mode="bilinear", align_corners=False)

    def apply_chair(self, img: torch.Tensor, alpha: torch.Tensor, mode: str) -> torch.Tensor:
        """keep: the chair (as cut from the plate) is foreground. remove: chair pixels that still
        look like the empty chair go transparent - your headset/hair in front of it differ from the
        plate, so they stay (no 60 px safety guard needed here)."""
        if self.chair_mask is None or self.chair_mask.shape[-2:] != alpha.shape[-2:]:
            return alpha
        cm = self.chair_mask.float()
        a = alpha.float()
        if mode == "keep":
            a = torch.maximum(a, cm)
        elif mode == "remove":
            a = a * (1 - cm * (1 - self.changed_full(img)))
        return a.clamp(0, 1).to(alpha.dtype)

    def difference(self, img: torch.Tensor, alpha_lo: torch.Tensor) -> torch.Tensor:
        """Soft 'changed vs plate' map at low res, exposure- and noise-compensated."""
        i_s = F.avg_pool2d(img.float(), self.LO)
        bg = (alpha_lo < 0.05).float()
        if bg.sum() > 100:
            ratio = (i_s + 0.02) / (self.plate + 0.02)
            sel = bg.expand_as(ratio) > 0
            gain = torch.stack([ratio[:, c][sel[:, c]].median() for c in range(3)]).clamp(0.5, 2.0)
        else:
            gain = torch.ones(3, device=img.device)
        d = (i_s - self.plate * gain.view(1, 3, 1, 1)).abs().mean(1, keepdim=True)
        d = F.avg_pool2d(d, 3, 1, 1)
        return (d - 3 * self.noise).clamp(min=0)

    def __call__(self, img: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if self.plate is None or self.full_shape != img.shape:
            return alpha
        a_s = F.avg_pool2d(alpha.float(), self.LO)
        dist = self.difference(img, a_s)
        diff_a = ((dist - self.threshold) / self.softness + 0.5).clamp(0, 1)
        out = a_s
        if self.rescue:
            r = max(1, int(self.reach * a_s.shape[-1]))
            near = F.max_pool2d((a_s > 0.5).float(), 2 * r + 1, 1, r)
            # close small holes in the rescued object so solid things stay solid
            grown = F.max_pool2d(diff_a, 5, 1, 2)
            diff_a = -F.max_pool2d(-grown, 5, 1, 2)
            out = torch.maximum(out, diff_a * near)
        if self.suppress:
            keep = ((dist - self.threshold * 0.3) / self.softness + 0.5).clamp(0, 1)
            # Only remove *large* regions that match the plate everywhere (an empty chair), never
            # small patches inside the subject whose colour happens to match what is behind them.
            # guard radius: ~60 px at 1080p (safe) down to ~10 px (removes a chair right behind you)
            frac = 1 / 32 + (1 / 200 - 1 / 32) * float(self.furniture)
            r = max(1, int(a_s.shape[-1] * frac))
            keep = F.max_pool2d(keep, 2 * r + 1, 1, r)
            out = out * keep
        # Only apply the correction delta, so the AI's fine edge detail survives.
        delta = F.interpolate(out - a_s, size=alpha.shape[-2:], mode="bilinear", align_corners=False)
        return (alpha.float() + delta).clamp_(0, 1).to(alpha.dtype)


class SemanticGuard:
    """Person/chair semantics (torchvision DeepLabV3-MobileNetV3, BSD, COCO->VOC classes) fused with
    the matte.

    Matting nets judge by colour/edges, so a black headset in front of a black gaming chair turns
    semi-transparent ("ghosting"), and the chair itself is often kept. A semantic net knows that
    headphones on a head belong to the *person* and the headrest is a *chair*:

      fill   - person pixels the matte dropped (headphones, mic, glasses arms) are filled back in,
               snapped to real image edges with a guided filter, only near the existing matte
      chair  - "auto": leave as matted | "remove": chair/sofa pixels not covered by the person -> 0 |
               "keep": chair pixels become foreground (the chair stays in shot over a new background)
    """

    PERSON, CHAIR, SOFA = 15, 9, 18
    SIZE = (384, 672)          # ~16:9; ~4-7 ms on an R9700

    def __init__(self, device, net_device=None):
        """device: where the matte lives. net_device: where the segmentation net runs - a second
        GPU makes the guard almost free for the main GPU (runs asynchronously in a thread)."""
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        self.device = torch.device(device)
        self.net_device = torch.device(net_device or device)
        net = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True, num_classes=21)
        net.load_state_dict(torch.load(os.path.join(MODELS, "torchvision",
                                                    "deeplabv3_mobilenet_v3_large-fc3c493d.pth"), map_location="cpu"))
        self.net = net.eval().to(self.net_device, torch.float16)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.net_device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.net_device, dtype=torch.float16).view(1, 3, 1, 1)
        self.fill = True
        self.chair = "auto"       # auto | remove | keep
        self.every = 1            # same-GPU mode: run the net every N frames
        self._i = 0
        self._probs = None
        self._probs_frame = None   # the (small) frame the current probabilities were computed from
        self._maps = None          # cached edge-snapped maps for the current probabilities
        self.async_mode = self.net_device != self.device
        if self.async_mode:
            import queue
            import threading
            self._q = queue.Queue(maxsize=1)
            self._latest_slot = 0
            self._latest_id = 0
            self._fetched_id = 0
            self._init_async()
            self._t = threading.Thread(target=self._worker, daemon=True)
            self._t.start()
            import atexit
            atexit.register(self.close)

    def close(self):
        """Stop the 2nd-GPU worker (a daemon thread still inside HIP at interpreter exit aborts)."""
        if getattr(self, "_t", None) is not None and self._t.is_alive():
            try:
                while True:
                    self._q.get_nowait()
            except Exception:  # noqa: BLE001
                pass
            self._q.put(None)
            self._t.join(timeout=3)

    @property
    def active(self):
        return self.fill or self.chair != "auto"

    @torch.no_grad()
    def _run_net(self, x):
        logits = self.net((x - self.mean) / self.std)["out"].float()
        p = logits.softmax(1)
        return torch.cat([p[:, self.PERSON:self.PERSON + 1],
                          p[:, self.CHAIR:self.CHAIR + 1] + p[:, self.SOFA:self.SOFA + 1]], 1)

    # ---- 2-GPU mode: the GPUs never touch each other's memory or streams; frames and results
    # cross through small pinned host buffers. (A worker issuing work on the main GPU while RVM
    # captures its HIP graph there corrupts the graph -> GPU memory fault on replay.)
    def _init_async(self):
        h, w = self.SIZE
        self._in = [torch.empty((1, 3, h, w), dtype=torch.float16, pin_memory=True) for _ in range(2)]
        self._in_ev = [None, None]
        self._out = [torch.empty((1, 2, h, w), dtype=torch.float16, pin_memory=True) for _ in range(3)]
        self._in_i = 0
        self._in_copy = [None, None]

    def _worker(self):
        stream = torch.cuda.Stream(self.net_device)
        graph = None
        h, w = self.SIZE
        with torch.cuda.device(self.net_device), torch.cuda.stream(stream):
            static_x = torch.zeros((1, 3, h, w), dtype=torch.float16, device=self.net_device)
            try:
                # one graph replay per frame instead of hundreds of kernel launches: the worker then
                # barely needs the Python GIL, so it keeps up while the main thread is busy
                for _ in range(2):
                    self._run_net(static_x)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                    static_out = self._run_net(static_x).half()
            except Exception as e:  # noqa: BLE001
                print(f"[semantic] graph capture failed, eager mode: {e}", flush=True)
                graph = None
        while True:
            item = self._q.get()
            if item is None:
                return
            i = item
            try:
                self._in_ev[i].synchronize()          # main GPU finished writing this pinned frame
                with torch.cuda.device(self.net_device), torch.cuda.stream(stream):
                    slot = (self._latest_id + 1) % 3
                    if graph is not None:
                        static_x.copy_(self._in[i], non_blocking=True)
                        graph.replay()
                        self._out[slot].copy_(static_out, non_blocking=True)
                    else:
                        probs = self._run_net(self._in[i].to(self.net_device, non_blocking=True)).half()
                        self._out[slot].copy_(probs, non_blocking=True)
                    stream.synchronize()
                self._latest_slot = slot
                self._latest_src = i
                self._latest_id += 1
            except Exception as e:  # noqa: BLE001 - keep the guard alive; next frame retries
                print(f"[semantic] {type(e).__name__}: {e}", flush=True)

    @torch.no_grad()
    def _segment(self, src):
        x = F.interpolate(src.to(torch.float16), size=self.SIZE, mode="bilinear", align_corners=False)
        if not self.async_mode:
            self._probs = self._run_net(x)
            return
        if self._probs is None:                # very first frame: synchronous, through the host
            first = self._run_net(x.cpu().to(self.net_device)).cpu()
            self._probs = first.to(self.device).float()
            self._probs_frame = x
        if self._q.empty():                    # worker idle: give it the newest frame; never wait
            i = self._in_i
            self._in_i ^= 1
            self._in[i].copy_(x, non_blocking=True)
            self._in_copy[i] = x                   # GPU copy of the frame being segmented
            ev = torch.cuda.Event()
            ev.record()
            self._in_ev[i] = ev
            try:
                self._q.put_nowait(i)
            except Exception:  # noqa: BLE001
                pass
        if self._latest_id != self._fetched_id:    # new finished result: async upload from pinned
            self._fetched_id = self._latest_id
            self._probs = self._out[self._latest_slot].to(self.device, non_blocking=True).float()
            self._probs_frame = self._in_copy[self._latest_src]


    @torch.no_grad()
    def __call__(self, src: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return alpha
        before = self._probs
        if self._probs is None or self.async_mode or self._i % max(1, self.every) == 0:
            self._segment(src)
            if not self.async_mode:
                self._probs_frame = None
        self._i += 1
        H, W = alpha.shape[-2:]
        # all semantic work at <=540p; only the *correction* is upsampled and added, so the matte's
        # fine edges (hair) are untouched where nothing needs fixing
        k = 4 if H >= 1000 else 2
        h2, w2 = H // k, W // k
        r_gf = max(2, int(12 / k * H / 1080))
        # edge-snapped semantic maps: recomputed only when a new segmentation arrives (the guided
        # filter is the expensive part; at 100/120 fps results arrive every 1-3 frames)
        key = (id(self._probs), self.fill, self.chair, H, W)
        if self._probs is not before or self._maps is None or self._maps[0] != key:
            person_lo, chair_lo = self._probs[:, 0:1], self._probs[:, 1:2]
            guide = F.avg_pool2d(src.float(), k)
            person_hi = ((guided_filter(guide, person_lo, r=r_gf, eps=2e-3).float() - 0.45) / 0.3).clamp(0, 1) \
                if self.fill else None
            person_up = F.interpolate(person_lo, size=(h2, w2), mode="bilinear", align_corners=False)
            chair_map = None
            if self.chair == "remove":
                chair_up = F.interpolate(chair_lo, size=(h2, w2), mode="bilinear", align_corners=False)
                chair_map = ((chair_up - person_up - 0.1) / 0.3).clamp(0, 1)
            elif self.chair == "keep":
                # a kept chair must be *solid*: hard threshold, close holes, then a thin soft edge
                c = guided_filter(guide, chair_lo, r=r_gf, eps=2e-3).float()
                c = ((c - 0.3) / 0.12).clamp(0, 1)
                rc = max(1, int(8 / k * H / 1080))
                c = -F.max_pool2d(-F.max_pool2d(c, 2 * rc + 1, 1, rc), 2 * rc + 1, 1, rc)   # close
                chair_map = F.avg_pool2d(c, 3, 1, 1)
            self._maps = (key, person_hi, chair_map)
        _, person_hi, chair_map = self._maps
        # motion check: where the picture changed since the segmented frame, a (stale) semantic map
        # must not add or remove anything - otherwise it would paint where your head *was*
        trust = None
        if self._probs_frame is not None:
            cur = F.interpolate(src.to(torch.float16), size=self.SIZE, mode="bilinear", align_corners=False)
            d = (cur.float() - self._probs_frame.float()).abs().mean(1, keepdim=True)
            d = F.max_pool2d(F.avg_pool2d(d, 3, 1, 1), 9, 1, 4)
            trust = F.interpolate((1 - (d - 0.03) / 0.05).clamp(0, 1), size=(h2, w2), mode="bilinear",
                                  align_corners=False)
        a_full = alpha.float()
        a = F.avg_pool2d(a_full, k)
        out = a
        if person_hi is not None:
            r = max(4, int(30 / k * H / 1080))
            near = F.max_pool2d(F.avg_pool2d((a > 0.5).float(), 4), 2 * (r // 4) + 1, 1, r // 4)
            near = F.interpolate(near, size=(h2, w2), mode="bilinear", align_corners=False)
            fill = person_hi * near
            out = torch.maximum(out, fill if trust is None else fill * trust)
        if chair_map is not None:
            cm = chair_map if trust is None else chair_map * trust
            out = out * (1 - cm) if self.chair == "remove" else torch.maximum(out, cm)
        delta = F.interpolate(out - a, size=(H, W), mode="bilinear", align_corners=False)
        return (a_full + delta).clamp(0, 1).to(alpha.dtype)


ENGINES = {
    "rvm": "Realtime - RVM ResNet50 (temporal, 60 fps+)",
    "rvm-mobile": "Realtime - RVM MobileNetV3 (lowest power)",
    "birefnet-lite": "Studio - BiRefNet-lite matting (sharp hair)",
    "birefnet": "Studio Max - BiRefNet matting (best edges, multi-GPU)",
}


def build_engine(kind: str, devices: list[str]):
    if kind == "rvm":
        return RVMEngine(devices[0], "resnet50")
    if kind == "rvm-mobile":
        return RVMEngine(devices[0], "mobilenetv3")
    if kind in ("birefnet-lite", "birefnet"):
        model = "birefnet_lite_matting" if kind == "birefnet-lite" else "birefnet_matting"
        return BiRefNetEngine(devices[0], model) if len(devices) == 1 else MultiGPU(devices, model)
    raise ValueError(kind)
