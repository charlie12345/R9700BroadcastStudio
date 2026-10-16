"""Studio: the per-frame processing graph shared by the GUI, the headless daemon and the file CLI.

    camera frame (uint8 BGR, CPU)
      -> upload + normalise (GPU)
      -> matting engine (RVM / BiRefNet / multi-GPU)        alpha + straight foreground
      -> temporal stabiliser (per-frame engines only)
      -> clean-plate assist (optional)
      -> compositor (blur / replace / green / matte, light wrap, relight, auto-frame)
      -> uint8 RGB (CPU) for the virtual camera / recorder / preview
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .effects import Compositor, Settings
from .matting import ENGINES, PlateAssist, SemanticGuard, TemporalStabilizer, build_engine


def visible_devices() -> list[str]:
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]


class Studio:
    def __init__(self, engine="rvm", devices: list[str] | None = None, settings: Settings | None = None):
        self.devices = devices or visible_devices()[:1]
        self.device = torch.device(self.devices[0])
        self.dtype = torch.float16
        self.settings = settings or Settings()
        self.compositor = Compositor()
        self.plate = PlateAssist()
        self.plate.threshold = self.settings.plate_sensitivity
        self.plate.furniture = self.settings.plate_furniture
        self.stabilizer = TemporalStabilizer(self.settings.temporal_smooth)
        self.semantic = None   # created lazily (only if the option is used)
        self.subject = None    # SubjectLock (MatAnyone 2 tracker process), created lazily
        self.lock = threading.RLock()
        self._warmed: set = set()
        self.semantic_every = 1   # LivePipeline sets 2 for 100/120 fps presets
        self.engine_kind = None
        self.engine = None
        self.capturing_plate = False
        self.stats = {"ms_total": 0.0, "ms_matte": 0.0, "fps": 0.0}
        self._last_t = None
        self.set_engine(engine)

    # -- control -----------------------------------------------------------------------------
    def set_engine(self, kind: str, devices: list[str] | None = None):
        assert kind in ENGINES, kind
        with self.lock:
            if devices:
                self.devices = devices
                self.device = torch.device(devices[0])
            if kind == self.engine_kind and not devices:
                return
            self.engine = None
            torch.cuda.empty_cache()
            devs = self.devices if kind.startswith("birefnet") else self.devices[:1]
            self.engine = build_engine(kind, devs)
            self.engine_kind = kind
            self.stabilizer.reset()
            self.compositor.framer.reset()

    def _other_gpu(self):
        n = torch.cuda.device_count()
        return next((f"cuda:{i}" for i in range(n) if torch.device(f"cuda:{i}") != self.device), None)

    def _subject_ok(self) -> bool:
        if not hasattr(self, "_subject_avail"):
            from . import coretrack
            self._subject_avail = coretrack.available()
        return self._subject_avail

    def relock_subject(self):
        if self.subject is not None:
            self.subject.reseed()

    def subject_status(self) -> str:
        if not self.settings.subject_lock:
            return "off"
        if not self._subject_ok():
            return "unavailable"
        if self.subject is None:
            return "starting"
        if self.subject.error:
            return "error: " + self.subject.error
        return self.subject.status

    def close(self):
        for o in (self.subject, self.semantic):
            if o is not None:
                try:
                    o.close()
                except Exception:  # noqa: BLE001
                    pass
        self.subject = None

    def start_plate_capture(self):
        with self.lock:
            self.plate.clear()
            self.capturing_plate = True

    def set_background_image(self, rgb: np.ndarray | None):
        with self.lock:
            self.compositor.set_background(None if rgb is None else self.upload(rgb, bgr=False))

    # -- tensors -----------------------------------------------------------------------------
    def upload(self, frame: np.ndarray, bgr=True) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(frame)).pin_memory().to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0)
        if bgr:
            t = t.flip(1)
        return t.to(self.dtype) / 255

    @staticmethod
    def download(t: torch.Tensor) -> np.ndarray:
        return (t[0].float() * 255 + 0.5).clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

    # -- per frame ---------------------------------------------------------------------------
    @torch.no_grad()
    def warmup(self, src: torch.Tensor):
        """Run every effect path once at this resolution so GPU kernels are compiled up front.
        Otherwise the first use of e.g. auto-frame mid-stream stalls ~0.75 s (dozens of dropped frames)."""
        from dataclasses import replace
        key = tuple(src.shape)
        if key in self._warmed:
            return
        self._warmed.add(key)
        a = torch.zeros_like(src[:, :1])
        a[..., src.shape[-2] // 4:, src.shape[-1] // 3: 2 * src.shape[-1] // 3] = 1
        saved_framer = self.compositor.framer
        from .effects import AutoFramer, MODES
        self.compositor.framer = AutoFramer()
        for mode in MODES:
            for extra in ({}, {"auto_frame": True, "feather": 2.0, "choke": 1.0, "vignette": 0.3,
                              "relight": 0.3, "denoise": 0.3, "bg_blur": 0.5, "mirror": True},
                          {"choke": -1.0, "feather": 6.0}):
                self.compositor(src, a, src, replace(self.settings, mode=mode, **extra))
        self.plate.difference(src, F.avg_pool2d(a.float(), self.plate.LO)) if self.plate.ready else None
        torch.cuda.synchronize(self.device)
        self.compositor.framer = saved_framer
        self.compositor.denoiser.reset()

    def process_tensor(self, src: torch.Tensor):
        """Returns (composited, alpha, fgr, src_used) tensors on GPU."""
        s = self.settings
        if tuple(src.shape) not in self._warmed:
            self.warmup(src)
        with self.lock:
            if self.capturing_plate:
                if self.plate.add_sample(src):
                    self.capturing_plate = False
            if s.mode == "off":
                return self.compositor(src, None, None, s), None, None, src
            t0 = time.perf_counter()
            m = self.engine(src)
            used = m.src if m.src is not None else src
            alpha = m.alpha
            if not self.engine_kind.startswith("rvm"):
                alpha = self.stabilizer(used, alpha)
            locked = False
            if s.subject_lock and self._subject_ok():
                if self.subject is not None and self.subject.mid_for != tuple(used.shape[-2:]):
                    self.subject.close()
                    self.subject = None
                if self.subject is None:
                    from .coretrack import SubjectLock
                    self.subject = SubjectLock(self.device, self._other_gpu() or self.device, used.shape)
                    self.subject.mid_for = tuple(used.shape[-2:])
                    self.subject.warmup(alpha)
                alpha = self.subject(used, alpha)
                locked = self.subject._core is not None
            elif self.subject is not None:
                self.subject.close()
                self.subject = None
            # the tracker already keeps worn accessories and drops the chair; the semantic guard is
            # the fallback while it locks on / when MatAnyone 2 is not installed
            # "keep chair" is plate-only (a semantic-only chair is a blob of room clutter)
            if (s.keep_accessories and not locked) or (s.chair == "remove" and not locked):
                if self.semantic is None:
                    # put the segmentation net on a second GPU when there is one (async, ~free)
                    n = torch.cuda.device_count()
                    other = next((f"cuda:{i}" for i in range(n) if torch.device(f"cuda:{i}") != self.device), None)
                    self.semantic = SemanticGuard(self.device, other)
                self.semantic.fill = s.keep_accessories
                self.semantic.chair = "remove" if s.chair == "remove" else "auto"
                self.semantic.every = self.semantic_every
                alpha = self.semantic(used, alpha)
            if self.plate.ready:
                alpha = self.plate(used, alpha)
                if s.chair != "auto" and self.semantic is not None:
                    if self.plate.chair_mask is None and self.plate.full is not None:
                        n = torch.cuda.device_count()
                        other = next((f"cuda:{i}" for i in range(n) if torch.device(f"cuda:{i}") != self.device),
                                     str(self.device))
                        self.plate.compute_chair_mask(self.semantic, other)
                    alpha = self.plate.apply_chair(used, alpha, s.chair)
            self.stats["ms_matte"] = (time.perf_counter() - t0) * 1000  # CPU-side; GPU is async
            out = self.compositor(used, alpha, m.fgr, s)
            return out, alpha, m.fgr, used

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        src = self.upload(frame_bgr)
        out, *_ = self.process_tensor(src)
        rgb = self.download(out)
        now = time.perf_counter()
        self.stats["ms_total"] = (now - t0) * 1000
        if self._last_t:
            dt = now - self._last_t
            self.stats["fps"] = 0.9 * self.stats["fps"] + 0.1 * (1 / max(dt, 1e-6))
        self._last_t = now
        return rgb
