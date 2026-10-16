"""Subject core tracking with MatAnyone 2 (Yang et al., CVPR 2026) + core/edge fusion.

Why
---
Per-frame and recurrent matting nets (RVM, BiRefNet) decide "person or background" from colour and
edges. Where the subject touches a similar-looking background - a black/red headset against a
black/red gaming-chair headrest - they turn the subject semi-transparent ("ghosting"), and leave a
see-through ghost of the chair. A semantic net alone is too coarse at the edges and lags when you turn.

MatAnyone 2 is a memory-propagation video matter (built on Cutie): it is told once *which object is
the subject* and then tracks that object's matte from frame to frame. Its paper's key idea is that the
**core** of a subject should be semantically stable while the **boundary** keeps fine detail. We use it
exactly that way:

  core  - MatAnyone 2 at ~384p on the second GPU (separate process) says what is certainly subject
          (incl. headset, mic, glasses, the side of your head mid-turn) and what is certainly
          background (the chair, the room)
  edge  - RVM keeps the full-resolution hair / beard detail inside the thin uncertain band

Seeding is automatic: a DeepLabV3 person mask (chairs/sofas excluded) is the first-frame target; the
tracker re-seeds itself when it loses the subject (you leave and come back) or on request.

MatAnyone 2 is under the NTU S-Lab License 1.0 (non-commercial use). It is optional: without its
weights the app falls back to RVM + the semantic guard.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .matting import MODELS, ROOT
from .seeding import complete_seed
MA2_DIR = os.path.join(ROOT, "third_party", "MatAnyone2")
MA2_WEIGHTS = os.path.join(MODELS, "matanyone2", "matanyone2.pth")
DEEPLAB_WEIGHTS = os.path.join(MODELS, "torchvision", "deeplabv3_mobilenet_v3_large-fc3c493d.pth")


def available() -> bool:
    return os.path.isfile(MA2_WEIGHTS) and os.path.isdir(os.path.join(MA2_DIR, "matanyone2"))


def tracker_size(h: int, w: int, short: int) -> tuple[int, int]:
    """Tracker working size: short side `short`, both sides multiples of 16."""
    if h <= w:
        return short, int(round(w * short / h / 16)) * 16
    return int(round(h * short / w / 16)) * 16, short


class CoreTracker:
    """MatAnyone 2 + automatic subject seeding (Grounded SAM 2, see seeding.py).

    step() takes a mid-resolution frame (e.g. 720p, (1,3,H,W) float [0,1]): the seeder uses it at
    that resolution, MatAnyone runs at `short` px on the short side."""

    PERSON, CHAIR, SOFA = 15, 9, 18
    WARMUP = 8              # first-frame refinement steps after (re)seeding
    CHECK_EVERY = 30        # frames between seed health checks

    def __init__(self, device="cuda:0", short=384, seeder=True):
        self.device = torch.device(device)
        self.short = short
        torch.cuda.set_device(self.device)
        if MA2_DIR not in sys.path:
            sys.path.insert(0, MA2_DIR)
        from omegaconf import OmegaConf

        from matanyone2.inference.inference_core import InferenceCore
        from matanyone2.model.matanyone2 import MatAnyone2
        # compose the config by hand: going through Hydra would clash with SAM 2's global Hydra state
        cdir = os.path.join(MA2_DIR, "matanyone2", "config")
        cfg = OmegaConf.load(os.path.join(cdir, "eval_matanyone_config.yaml"))
        for k in ("defaults", "hydra"):
            cfg.pop(k, None)
        cfg.model = OmegaConf.load(os.path.join(cdir, "model", "base.yaml"))
        cfg.weights = MA2_WEIGHTS
        cfg.model.pretrained_resnet = False      # weights come from the checkpoint; no ImageNet download
        net = MatAnyone2(cfg, single_object=True).to(self.device).eval()
        net.load_weights(torch.load(MA2_WEIGHTS, map_location=self.device, weights_only=True))
        self._InferenceCore = InferenceCore
        self.net, self.cfg = net, cfg
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        dl = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True, num_classes=21)
        dl.load_state_dict(torch.load(DEEPLAB_WEIGHTS, map_location="cpu", weights_only=True))
        self.seg = dl.eval().to(self.device, torch.float16)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.seeder = None
        if seeder:
            from . import seeding
            if seeding.available():
                self.seeder = seeding.SubjectSeeder(self.device)
        self.reset()

    def reset(self):
        self.proc = self._InferenceCore(self.net, cfg=self.cfg, device=self.device)
        self.seeded = False
        self.warm = 0
        self.t = 0
        self.bad_checks = 0
        self.last = None
        self.last_seed = None
        self._cands = []
        self.nobody = False

    def request_reseed(self):
        self.seeded = False
        self._cands = []

    @torch.no_grad()
    def semantics(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft (person, chair+sofa) probabilities, same size as img."""
        h, w = img.shape[-2:]
        x = F.interpolate(img.float(), size=(384, int(round(384 * w / h / 8)) * 8), mode="bilinear",
                          align_corners=False)
        p = self.seg(((x - self.mean) / self.std).half())["out"].float().softmax(1)
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
        return p[:, self.PERSON:self.PERSON + 1], p[:, self.CHAIR:self.CHAIR + 1] + p[:, self.SOFA:self.SOFA + 1]

    def person_mask(self, img):
        return self.semantics(img)[0]

    @torch.no_grad()
    def seed_mask(self, img: torch.Tensor, hint: torch.Tensor | None) -> torch.Tensor | None:
        """First-frame target (1,1,H,W) float {0,1}, or None if nobody is in shot."""
        person, chair = self.semantics(img)
        if (person > 0.5).float().mean() < 0.01:
            return None
        if self.seeder is not None:
            rgb = (img[0].permute(1, 2, 0).float().clamp(0, 1) * 255).byte().cpu().numpy()
            m = self.seeder(rgb, person[0, 0].float().cpu().numpy(), chair[0, 0].float().cpu().numpy())
            if m is None:
                return None
            m = complete_seed(self.seeder.last_body, self.seeder.last_accessories)
            return torch.from_numpy(m.astype(np.float32)).to(self.device)[None, None]
        # fallback without SAM 2: semantic person, sharpened by the edge matte, minus the chair
        m = (person > 0.5).float()
        if hint is not None:
            m = m * (F.interpolate(hint.float(), size=m.shape[-2:], mode="bilinear") > 0.6).float()
        m = m * (chair < 0.35).float()
        r = max(1, img.shape[-2] // 128)
        return F.max_pool2d(-F.max_pool2d(-m, 2 * r + 1, 1, r), 2 * r + 1, 1, r)

    def _small(self, img):
        h, w = tracker_size(*img.shape[-2:], self.short)
        if img.shape[-2:] == (h, w):
            return img.float()
        return F.interpolate(img.float(), size=(h, w), mode="bilinear", align_corners=False)

    SEED_CANDIDATES = 4     # Grounded-SAM seeds gathered (one per tracker step, ~0.5 s apart)
    seed_stride = 1         # frames between candidates (live: seeding itself takes ~0.5 s)

    @torch.no_grad()
    def _seed(self, img: torch.Tensor, small: torch.Tensor, hint: torch.Tensor | None = None) -> bool:
        """Gather candidate seeds; once enough, start tracking from the best one.

        Detection boxes jitter from frame to frame, and a loose box makes SAM include part of the chair
        behind the headset. The true accessories are the same in every candidate while the chair
        spill-over comes and goes, so the tightest candidate that still has the full accessory set
        (>= 60 % of the median area: no missing ear cup) is the clean one."""
        if self._cands and self.t % self.seed_stride:
            return False                 # offline use: space candidates out like the live tracker does
        m = self.seed_mask(img, hint)
        self.nobody = m is None
        if m is None:
            self._cands = []
            return False
        m = (F.interpolate(m, size=small.shape[-2:], mode="bilinear", align_corners=False) > 0.5).float()
        if self.seeder is None:
            best = (m, small)
        else:
            self._cands.append((self._candidate_stats(hint, img.shape[-2:]), m, small))
            if len(self._cands) < self.SEED_CANDIDATES:
                return False
            best = self._pick_candidate()
            self._cands = []
        m, seed_frame = best
        self.last_seed = m
        self.proc = self._InferenceCore(self.net, cfg=self.cfg, device=self.device)
        self.proc.step(seed_frame[0], m[0, 0] * 255, objects=[1])
        self.seeded = True
        self.warm = self.WARMUP
        self.bad_checks = 0
        return True

    def _candidate_stats(self, hint, size):
        """(accessory area, two-sided?, edge-matte agreement) for the seeder's last accessories."""
        acc = self.seeder.last_accessories
        body = self.seeder.last_body
        if acc is None or not acc.any():
            return (0.0, False, 0.0)
        ys, xs = np.nonzero(body)
        axis = np.median(xs)
        left = (acc[:, :int(axis)]).sum()
        two_sided = min(left, acc.sum() - left) > 0.15 * acc.sum()
        agree = 0.5
        if hint is not None:
            h = F.interpolate(hint.float(), size=acc.shape, mode="bilinear", align_corners=False)[0, 0]
            agree = float(h[torch.from_numpy(acc).to(h.device)].mean())
        return (float(acc.sum()), bool(two_sided), agree)

    def _pick_candidate(self):
        """Seeds jitter with the detector's box: a loose box makes SAM add a slice of the chair wing
        behind the ear cup. The edge matte (RVM) mostly rejects the chair wing, and the true accessory
        set is the same size every time while spill-over only adds area, so: among candidates that
        have accessories on both sides of the head (both ear cups), take the best
        agreement-minus-extra-area."""
        cands = [c for c in self._cands if c[0][0] > 0]
        if not cands:
            return self._cands[-1][1], self._cands[-1][2]
        two = [c for c in cands if c[0][1]] or cands
        amin = min(c[0][0] for c in two)
        _, m, small = max(two, key=lambda c: c[0][2] - 0.5 * (c[0][0] / amin - 1))
        return m, small

    @torch.no_grad()
    def step(self, img: torch.Tensor, hint: torch.Tensor | None = None) -> torch.Tensor | None:
        """img (1,3,H,W) [0,1] -> subject alpha (1,1,h,w) float32 at tracker size, or None while no
        subject is locked yet. hint: optional edge matte (only used by the fallback seeder)."""
        self.t += 1
        small = self._small(img)
        if not self.seeded and not self._seed(img, small, hint):
            self.last = None
            return None
        with torch.amp.autocast("cuda"):
            if self.warm > 0:
                prob = self.proc.step(small[0], first_frame_pred=True)
                self.warm -= 1
            else:
                prob = self.proc.step(small[0])
            a = self.proc.output_prob_to_mask(prob).float().clamp(0, 1)[None, None]
        if self.t % self.CHECK_EVERY == 0:
            self._health(small, a)
        self.last = a
        return a

    def _health(self, img, a):
        """Re-seed when the tracker and the person detector disagree for ~1.5 s (subject left and came
        back, someone new stepped in, or the track drifted)."""
        p = self.person_mask(img) > 0.5
        m = a > 0.5
        inter = (p & m).sum().item()
        union = (p | m).sum().item()
        iou = inter / union if union else 1.0
        self.bad_checks = self.bad_checks + 1 if iou < 0.4 else 0
        if self.bad_checks >= 3 or (m.float().mean() < 0.002 and p.float().mean() > 0.02):
            self.seeded = False


# ---------------------------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------------------------

def _maxpool_sep(x: torch.Tensor, r: int) -> torch.Tensor:
    """Max over a (2r+1)^2 square as two 1-D passes (much cheaper than one 2-D window)."""
    x = F.max_pool2d(x, (2 * r + 1, 1), 1, (r, 0))
    return F.max_pool2d(x, (1, 2 * r + 1), 1, (0, r))


def _fill_holes(m: torch.Tensor) -> torch.Tensor:
    """Fill background fully enclosed by the mask (a hole in the tracker's subject). This is an
    exact, connectivity-aware flood from the frame border: a naive GPU max-pool flood jumps thin
    barriers, which would fuse a hole to the outside and leave it transparent. The subject's outer
    boundary is untouched, so boundary detail (ears, hair) is preserved."""
    from scipy import ndimage
    hard = (m > 0.5)
    arr = hard.detach().cpu().numpy()
    filled = np.empty_like(arr)
    it = np.nditer(arr[..., 0, 0], flags=["multi_index"])
    for _ in it:
        b, c = it.multi_index
        filled[b, c] = ndimage.binary_fill_holes(arr[b, c])
    return torch.from_numpy(filled.astype(np.float32)).to(m.device)


def _envelope(core: torch.Tensor, motion: torch.Tensor | None, band_px: float, motion_px: float,
              H: int, W: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The tracker's (hi, lo) envelope at full resolution: a thin band around the tracked core
    (filled), widened by motion_px where the picture moved. Also returns hi_static, the band
    without any motion widening, so the caller can tell the two zones apart."""
    # the envelope comes from a ~384p matte: build it at 1/2 (720p) or 1/4 (1080p+) resolution and
    # upsample; only the final clamp runs at full resolution
    k = 4 if H >= 1000 else 2
    h2, w2 = H // k, W // k
    c = F.interpolate(core.float(), size=(h2, w2), mode="bilinear", align_corners=False)
    c = _fill_holes(c)          # a gap in the tracked core must never punch a see-through hole
    r = max(1, int(round(band_px * H / 1080 / k)))
    hi_s = _maxpool_sep(c, r)
    hi = hi_s
    lo = -_maxpool_sep(-c, r)
    if motion is not None:
        rm = max(r + 1, int(round((band_px + motion_px) * H / 1080 / k)))
        m = F.interpolate(motion.float(), size=(h2, w2), mode="bilinear", align_corners=False)
        # widen fully or stay closed: a semi-open band would keep the edge matte at its own value
        # out there, and a confident-but-wrong matte (the dark chair wing right behind the headset)
        # shows up as a translucent fringe by the ears. Slight change (webcam noise, a hand near
        # the face) is below the gate; a real move saturates it.
        gate = ((m - 0.3) / 0.2).clamp_(0, 1)
        hi = torch.lerp(hi, _maxpool_sep(c, rm), gate)
        lo = torch.lerp(lo, -_maxpool_sep(-c, rm), gate)
    hi = F.interpolate(hi, size=(H, W), mode="bilinear", align_corners=False)
    lo = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)
    hi_s = F.interpolate(hi_s, size=(H, W), mode="bilinear", align_corners=False)
    return hi, lo, hi_s


@torch.no_grad()
def fuse_core(alpha: torch.Tensor, core: torch.Tensor, motion: torch.Tensor | None = None,
              band_px: float = 8.0, motion_px: float = 48.0) -> torch.Tensor:
    """Edge matte constrained to the tracker's envelope.

    alpha   (1,1,H,W) edge matte (RVM) of the current frame - fine hair, but colour-confused
    core    (1,1,h,w) tracker alpha (MatAnyone 2) - knows *what* is subject, coarser edges
    motion  (1,1,h',w') optional 0..1: where the picture changed since the tracker's frame

        out = clamp(alpha, erode_r(core), dilate_r(core))

    Deep inside the subject erode(core)=1, so the edge matte cannot turn a headset or temple
    see-through; well outside it dilate(core)=0, so it cannot keep the chair; within r px of the
    tracker's boundary the edge matte's own detail (hair strands, beard) is kept. Where the picture
    is really moving (the tracker is 1-2 frames behind) the band widens by motion_px, so a fast
    hand or a turning head is never clipped, while static furniture stays strictly outside. The
    widening is all-or-nothing: a half-open band would show a confident-but-wrong edge matte as a
    translucent fringe (the dark chair wing right behind the ears). And inside the motion-only
    ring (outside the static band) only a confident edge matte is admitted: mid-confidence matte
    out there is the furniture the subject just uncovered, while the subject itself reads high.
    """
    H, W = alpha.shape[-2:]
    hi, lo, hi_s = _envelope(core, motion, band_px, motion_px, H, W)
    a = alpha.float()
    # any motion contribution at all (even just above the knee, where the band is only ajar):
    # a turning head sweeps the earcup fast while the tracker lags, and RVM overshoots the cup
    # edge at low confidence there. The static feather (hi == hi_s) is untouched.
    opened = (hi - hi_s) > 0.05
    if motion is not None and bool(opened.any()):
        conf = ((a - 0.55) / 0.15).clamp(0, 1)
        hi = torch.where(opened, hi * conf, hi)
    return torch.minimum(torch.maximum(a, lo), hi).to(alpha.dtype)


# ---------------------------------------------------------------------------------------------
# Live: tracker in its own process (second GPU when there is one)
# ---------------------------------------------------------------------------------------------

def _register(t: torch.Tensor):
    """Page-lock a shared-memory tensor so GPU copies to/from it are truly asynchronous."""
    try:
        torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
    except Exception:  # noqa: BLE001 - only a speed-up
        pass


def _worker(device, frames, hints, outs, inq, outq, short):
    import r9700_broadcast  # noqa: F401 - BLAS/allocator env before any GPU work
    torch.cuda.set_device(device)
    for t in (*frames, *hints, *outs):
        _register(t)
    try:
        tr = CoreTracker(device, short)
    except Exception as e:  # noqa: BLE001
        outq.put(("error", f"{type(e).__name__}: {e}"))
        return
    outq.put(("ready", tr.seeder is not None))
    oslot = 0
    while True:
        msg = inq.get()      # the main process keeps at most one frame in flight: no backlog to drop
        kind = msg[0]
        if kind == "stop":
            return
        if kind == "reseed":
            tr.request_reseed()
            continue
        _, slot, fid = msg
        try:
            with torch.no_grad():
                img = frames[slot].to(device, non_blocking=True).permute(2, 0, 1)[None].float().div_(255)
                hint = hints[slot].to(device, non_blocking=True)[None, None].float().div_(255)
                a = tr.step(img, hint)
                if a is None:
                    outq.put(("none", fid, bool(tr._cands), tr.nobody))
                    continue
                outs[oslot].copy_((a[0, 0] * 255 + 0.5).clamp(0, 255).byte(), non_blocking=True)
                torch.cuda.synchronize(device)
            outq.put(("core", oslot, fid))
            oslot = (oslot + 1) % len(outs)
        except Exception as e:  # noqa: BLE001 - keep tracking; report once per failure
            outq.put(("error", f"{type(e).__name__}: {e}"))


class SubjectLock:
    """Main-process side of the live core tracker.

    Per frame: the frame (720p) and the RVM matte are copied asynchronously into shared memory and
    handed to the tracker process; the newest finished tracker matte is fused with the current RVM
    matte (fuse_core). The tracker lags 1-2 frames; its matte is shifted by the subject's motion
    since then (centroid of the RVM matte) and the fusion band is widened slightly, so nothing
    snaps or ghosts during fast moves. Never blocks the GPU thread."""

    MID = 720
    NOBODY_AFTER = 15       # tracker results (~0.5 s) with no person before the output empties

    def __init__(self, main_device, net_device, src_shape, short=384):
        import multiprocessing as mp
        import threading
        self.device = torch.device(main_device)
        H, W = src_shape[-2:]
        self.mid = (self.MID, int(round(W * self.MID / H / 16)) * 16) if H <= W else \
            (int(round(H * self.MID / W / 16)) * 16, self.MID)
        self.small = tracker_size(*self.mid, short)
        mh, mw = self.mid
        sh, sw = self.small
        self.frames = [torch.zeros((mh, mw, 3), dtype=torch.uint8).share_memory_() for _ in range(2)]
        self.hints = [torch.zeros((mh, mw), dtype=torch.uint8).share_memory_() for _ in range(2)]
        self.outs = [torch.zeros((sh, sw), dtype=torch.uint8).share_memory_() for _ in range(3)]
        for t in (*self.frames, *self.hints, *self.outs):
            _register(t)
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(str(net_device), self.frames, self.hints, self.outs,
                                                      self.inq, self.outq, short), daemon=True)
        self.proc.start()
        self.ready = False
        self.has_seeder = False
        self.error = None
        self.status = "starting"
        self._busy = False
        self._fid = 0
        self._slot = 0
        self._pending = {}          # fid -> (event, slot)
        self._sub_centroid = {}     # fid -> RVM-matte centroid at submit time (GPU tensor)
        self._sub_luma = {}         # fid -> 1/8-res luma at submit time (motion since then)
        self._nobody_n = 0          # consecutive tracker results with no person in shot
        self._empty = 0.0           # 0..1 fade of the "empty room" output
        self._core_luma = None
        self._latest = None         # (out slot, fid) of the newest finished tracker matte
        self._shown = None
        self._core = None
        self._core_c = None
        self._lock = threading.Lock()
        self._alive = True
        self._rx = threading.Thread(target=self._receiver, daemon=True)
        self._rx.start()
        self._tx_q = __import__("queue").Queue()
        self._tx = threading.Thread(target=self._sender, daemon=True)
        self._tx.start()
        self.mid_for = None
        import atexit
        atexit.register(self.close)

    # -- threads (host only: they never touch the GPU) -------------------------------------------
    def _sender(self):
        while self._alive:
            item = self._tx_q.get()
            if item is None:
                return
            ev, slot, fid = item
            ev.synchronize()             # host wait: the async copy into shared memory finished
            self.inq.put(("frame", slot, fid))

    def _receiver(self):
        while self._alive:
            try:
                msg = self.outq.get(timeout=0.5)
            except Exception:  # noqa: BLE001
                if not self.proc.is_alive() and self._alive:
                    self.error = self.error or "subject tracker stopped"
                    self.status = "stopped"
                    return
                continue
            kind = msg[0]
            if kind == "ready":
                self.ready, self.has_seeder = True, msg[1]
                self.status = "locking on"
            elif kind == "error":
                self.error = msg[1]
                print(f"[subject-lock] {msg[1]}", flush=True)
                self._busy = False
            elif kind == "none":
                with self._lock:
                    self._latest = None
                    self.status = "locking on" if msg[2] else "looking for subject"
                    self._nobody_n = self._nobody_n + 1 if msg[3] else 0
                self._busy = False
            elif kind == "core":
                with self._lock:
                    self._latest = (msg[1], msg[2])
                    self.status = "locked"
                self._busy = False

    # -- control ---------------------------------------------------------------------------------
    def reseed(self):
        self.inq.put(("reseed",))
        with self._lock:
            self._latest = None
        self._core = None
        self.status = "locking on"

    def close(self):
        if not self._alive:
            return
        self._alive = False
        try:
            self.inq.put(("stop",))
            self._tx_q.put(None)
        except Exception:  # noqa: BLE001
            pass
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.kill()

    # -- per frame (GPU thread) ------------------------------------------------------------------
    @staticmethod
    def _centroid(a_lo: torch.Tensor) -> torch.Tensor:
        """(x, y) centroid of a low-res matte in normalised [-1, 1] coords, as a GPU tensor."""
        h, w = a_lo.shape[-2:]
        m = a_lo[0, 0].float()
        tot = m.sum().clamp_min(1e-3)
        xs = torch.linspace(-1, 1, w, device=m.device)
        ys = torch.linspace(-1, 1, h, device=m.device)
        return torch.stack([(m.sum(0) * xs).sum() / tot, (m.sum(1) * ys).sum() / tot])

    @torch.no_grad()
    def warmup(self, alpha: torch.Tensor):
        """Compile the fusion kernels now, not on the frame the lock first engages."""
        h, w = self.small
        core = torch.rand((1, 1, h, w), device=alpha.device)
        theta = torch.zeros((1, 2, 3), device=alpha.device)
        theta[0, 0, 0] = theta[0, 1, 1] = 1
        grid = F.affine_grid(theta, [1, 1, h, w], align_corners=False)
        core = F.grid_sample(core, grid, mode="bilinear", padding_mode="border", align_corners=False)
        fuse_core(alpha, core, torch.rand_like(F.avg_pool2d(alpha.float(), 8)), band_px=10.0)
        self._centroid(F.avg_pool2d(alpha.float(), 8))

    @torch.no_grad()
    def __call__(self, src: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        a_lo = F.avg_pool2d(alpha.float(), 8)
        cur_c = self._centroid(a_lo)
        luma = F.avg_pool2d(src.float().mean(1, keepdim=True), 8)
        if self.ready and not self._busy:
            self._busy = True
            self._fid += 1
            slot = self._slot
            self._slot ^= 1
            mid = F.interpolate(src.float(), size=self.mid, mode="bilinear", align_corners=False)
            hint = F.interpolate(alpha.float(), size=self.mid, mode="bilinear", align_corners=False)
            self.frames[slot].copy_((mid[0].permute(1, 2, 0) * 255 + 0.5).clamp(0, 255).byte(), non_blocking=True)
            self.hints[slot].copy_((hint[0, 0] * 255 + 0.5).clamp(0, 255).byte(), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self._sub_centroid[self._fid] = cur_c
            self._sub_luma[self._fid] = luma
            for k in [k for k in self._sub_centroid if k < self._fid - 8]:
                del self._sub_centroid[k]
                self._sub_luma.pop(k, None)
            self._tx_q.put((ev, slot, self._fid))
        with self._lock:
            latest = self._latest
            nobody = self._nobody_n >= self.NOBODY_AFTER
        if latest is None:
            self._core = None
            # nobody in shot (for ~0.5 s): fade out whatever the edge matte still keeps - an empty
            # chair, a coat on a hook - instead of showing furniture as "you"
            self._empty = min(1.0, self._empty + 0.1) if nobody else 0.0
            if self._empty > 0:
                return (alpha.float() * (1 - self._empty)).to(alpha.dtype)
            return alpha
        self._empty = 0.0
        if latest != self._shown:
            oslot, fid = latest
            self._shown = latest
            self._core = self.outs[oslot].to(self.device, non_blocking=True)[None, None].float().div_(255)
            self._core_c = self._sub_centroid.get(fid)
            self._core_luma = self._sub_luma.get(fid)
        core = self._core
        if self._core_c is not None:
            # follow the subject: shift the (1-2 frame old) tracker matte by the RVM matte's motion
            d = cur_c - self._core_c
            theta = torch.zeros((1, 2, 3), device=core.device)
            theta[0, 0, 0] = 1
            theta[0, 1, 1] = 1
            theta[0, :, 2] = -d
            grid = F.affine_grid(theta, list(core.shape), align_corners=False)
            core = F.grid_sample(core, grid, mode="bilinear", padding_mode="border", align_corners=False)
        motion = None
        if self._core_luma is not None:
            d = (luma - self._core_luma).abs()
            motion = ((F.max_pool2d(d, 5, 1, 2) - 0.03) / 0.05).clamp(0, 1)
            motion = F.avg_pool2d(motion, 5, 1, 2)
        return fuse_core(alpha, core, motion, band_px=10.0)
