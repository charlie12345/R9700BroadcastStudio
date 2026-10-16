"""GPU compositor: matte refinement + background effects + auto-frame + relight.

All tensors are (1,C,H,W) on the GPU in [0,1]. Nothing here round-trips to the CPU except the
auto-frame bounding box (4 numbers per frame).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

CHROMA_GREEN = (0.0, 177 / 255, 64 / 255)   # broadcast chroma green
CHROMA_BLUE = (0.0, 71 / 255, 187 / 255)

MODES = {
    "off": "Camera passthrough",
    "blur": "Background blur",
    "replace": "Replace background (image/video)",
    "remove": "Remove background (solid colour)",
    "green": "Virtual green screen (for OBS chroma key)",
    "blue": "Virtual blue screen",
    "matte": "Alpha matte (black/white, for luma key / debugging)",
    "transparent": "Transparent - real alpha channel (OBS via PipeWire camera)",
}


@dataclass
class Settings:
    mode: str = "blur"
    # matte
    choke: float = 0.0          # px at 1080p; + shrinks, - grows
    feather: float = 0.0        # px at 1080p
    clip_black: float = 0.02    # alpha below -> 0
    clip_white: float = 0.98    # alpha above -> 1
    decontaminate: bool = True  # use predicted straight foreground colour on edges
    temporal_smooth: float = 0.5  # stabiliser blend for per-frame engines 0..1 (RVM is recurrent already)
    # clean plate
    plate_sensitivity: float = 0.10  # colour distance counting as 'changed' vs the empty room
    plate_furniture: float = 0.0     # 0 = safest; 1 = also remove plate-matching stuff right next to you
    # background
    blur_strength: float = 0.6  # 0..1
    bg_color: tuple = (0.12, 0.12, 0.14)
    bg_dim: float = 0.0         # darken background 0..1
    bg_blur: float = 0.0        # blur the replacement image/video (depth of field) 0..1
    bg_fit: str = "cover"       # cover (crop to fill) | contain (letterbox) | stretch
    light_wrap: float = 0.25    # bleed background light over subject edges
    # subject
    relight: float = 0.0        # virtual key light: subject exposure lift 0..1
    vignette: float = 0.0
    # framing
    auto_frame: bool = False
    frame_zoom: float = 1.6     # how tight auto-frame crops (1 = no zoom)
    frame_speed: float = 0.08   # smoothing; lower = calmer camera
    mirror: bool = False
    denoise: float = 0.0        # video noise removal (motion-adaptive temporal) 0..1
    subject_lock: bool = True       # MatAnyone 2 core tracker (Grounded-SAM 2 seeded) fused with RVM
    keep_accessories: bool = True   # semantic guard: headphones / mic / glasses stay solid
    chair: str = "auto"             # auto | remove | keep
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------

def _box_cumsum(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """Edge-normalised running mean of width 2r+1 along `dim`, via cumulative sums. Same kernels for
    every radius: no per-size MIOpen search/compile, so dragging a slider can't hitch or fault."""
    n = x.shape[dim]
    r = min(r, n - 1)
    c = torch.cumsum(x, dim=dim)
    zero = torch.zeros_like(c.narrow(dim, 0, 1))
    c = torch.cat([zero, c], dim=dim)                       # c[i] = sum x[:i]
    idx = torch.arange(n, device=x.device)
    hi = torch.clamp(idx + r + 1, max=n)
    lo = torch.clamp(idx - r, min=0)
    shape = [1] * x.dim()
    shape[dim] = n
    s = c.index_select(dim, hi) - c.index_select(dim, lo)
    return s / (hi - lo).to(x.dtype).view(shape)


def _box_radii(sigma: float, n: int = 3) -> list[int]:
    """Kutskir 'boxes for gauss': n box widths (two adjacent odd sizes) whose summed variance == sigma^2."""
    wl = int(math.floor(math.sqrt(12 * sigma * sigma / n + 1)))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 1)
    wu = wl + 2
    m = round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4))
    return [(wl - 1) // 2 if i < m else (wu - 1) // 2 for i in range(n)]


def gaussian_blur(x: torch.Tensor, sigma: float, full_res: bool = True, max_level_sigma: float = 3.0):
    """Gaussian approximation: area-downsample to a pyramid level, then 3 box passes per axis
    (central-limit theorem), then bilinear upsample. Works for any sigma with a fixed kernel set."""
    if sigma < 0.5:
        return x
    H, W = x.shape[-2:]
    down = 1
    while sigma / down > max_level_sigma and min(H, W) // (down * 2) >= 16:
        down *= 2
    y = F.avg_pool2d(x, down) if down > 1 else x
    y = y.float()
    s = sigma / down
    for r in _box_radii(s):
        if r > 0:
            y = _box_cumsum(y, r, 3)
            y = _box_cumsum(y, r, 2)
    y = y.to(x.dtype)
    if down > 1 and full_res:
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    return y


def refine_alpha(a: torch.Tensor, s: Settings, scale: float) -> torch.Tensor:
    a = a.float()
    lo, hi = s.clip_black, max(s.clip_white, s.clip_black + 1e-3)
    a = ((a - lo) / (hi - lo)).clamp(0, 1)
    choke = s.choke * scale
    if abs(choke) >= 0.5:
        r = int(round(abs(choke)))
        if choke > 0:   # erode
            a = -F.max_pool2d(-a, 2 * r + 1, 1, r)
        else:           # dilate
            a = F.max_pool2d(a, 2 * r + 1, 1, r)
    if s.feather * scale >= 0.5:
        a = gaussian_blur(a, s.feather * scale)
    return a


class AutoFramer:
    """Smoothly tracks the subject bounding box from the matte and crops like a camera operator.

    Sync-free: the bounding box is reduced on the GPU and copied to pinned memory asynchronously;
    each update uses the newest box that has *already* arrived (typically one frame old). Calling
    .item()/.nonzero() here used to stall the whole pipeline every frame (dropped frames)."""

    def __init__(self):
        self.box = None  # cx, cy, w (normalised)
        self._host = None
        self._event = None
        self._pending = False
        self._basis = None

    def reset(self):
        self.box = None
        self._pending = False

    def _measure_async(self, alpha):
        a = F.avg_pool2d(alpha.float(), 8)[0, 0]
        h, w = a.shape
        m = a > 0.5
        rows = m.any(1).to(torch.int32)
        cols = m.any(0).to(torch.int32)
        stats = torch.stack([
            m.sum().float(),
            rows.argmax().float() / h, (h - 1 - rows.flip(0).argmax()).float() / h,
            cols.argmax().float() / w, (w - 1 - cols.flip(0).argmax()).float() / w])
        if self._host is None:
            self._host = torch.empty(5, dtype=torch.float32, pin_memory=True)
            self._event = torch.cuda.Event()
        self._host.copy_(stats, non_blocking=True)
        self._event.record()
        self._pending = True

    def update(self, alpha: torch.Tensor, zoom: float, speed: float):
        target = None
        if self._pending and self._event.query():
            count, y0, y1, x0, x1 = self._host.tolist()
            self._pending = False
            if count < 20:
                target = (0.5, 0.5, 1.0)
            else:
                cx = (x0 + x1) / 2      # frame head-and-shoulders: centre on the upper subject
                head_y = y0 + min(0.18, (y1 - y0) * 0.25)
                width = min(1.0, max((x1 - x0) * 1.15, 1.0 / zoom))
                target = (cx, head_y + width * 0.5 * 0.42, width)
        if not self._pending:
            self._measure_async(alpha)
        if target is not None:
            if self.box is None:
                self.box = target
            else:
                dz = 0.015    # dead-zone so the 'camera' does not micro-jitter
                b = list(self.box)
                for i, t in enumerate(target):
                    d = t - b[i]
                    if abs(d) > dz:
                        b[i] += (d - math.copysign(dz, d)) * speed
                self.box = tuple(b)
        return self.box or (0.5, 0.5, 1.0)

    def crop(self, img: torch.Tensor, box) -> torch.Tensor:
        H, W = img.shape[-2:]
        cx, cy, w = box
        w = min(1.0, max(0.2, w))
        h = w  # keep aspect: crop height fraction == width fraction
        x0 = min(max(cx - w / 2, 0.0), 1.0 - w)
        y0 = min(max(cy - h / 2, 0.0), 1.0 - h)
        # theta = w*Sx + h*Sy + tx*Tx + ty*Ty from cached GPU basis tensors. Python floats used as
        # arithmetic operands become kernel arguments; assigning them into a CUDA tensor
        # (theta[0,0,0] = w) or torch.tensor(..., device=cuda) does a *blocking* H2D copy each.
        if self._basis is None or self._basis.device != img.device:
            b = torch.zeros((4, 1, 2, 3), dtype=torch.float32, device=img.device)
            b[0, 0, 0, 0] = 1
            b[1, 0, 1, 1] = 1
            b[2, 0, 0, 2] = 1
            b[3, 0, 1, 2] = 1
            self._basis = b
        bs = self._basis
        theta = bs[0] * w + bs[1] * h + bs[2] * ((x0 + w / 2) * 2 - 1) + bs[3] * ((y0 + h / 2) * 2 - 1)
        grid = F.affine_grid(theta, (1, img.shape[1], H, W), align_corners=False)
        return F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border",
                             align_corners=False).to(img.dtype)


class TemporalDenoiser:
    """Motion-adaptive recursive filter: averages over time where the pixel is still, passes
    through where it moves. Removes low-light webcam grain without smearing motion."""

    def __init__(self):
        self.acc = None

    def reset(self):
        self.acc = None

    def __call__(self, img: torch.Tensor, strength: float) -> torch.Tensor:
        if strength <= 0 or self.acc is None or self.acc.shape != img.shape:
            self.acc = img.float()
            return img
        diff = (img.float() - self.acc).abs().mean(1, keepdim=True)
        diff = F.avg_pool2d(diff, 5, 1, 2)
        noise = 0.015 + 0.04 * strength
        k = (1 - diff / noise).clamp(0, 1) * (0.5 + 0.4 * strength)   # weight of history
        self.acc = self.acc * k + img.float() * (1 - k)
        return self.acc.to(img.dtype)


class Compositor:
    def __init__(self):
        self.framer = AutoFramer()
        self.denoiser = TemporalDenoiser()
        self.background: torch.Tensor | None = None   # (1,3,H,W) replacement background (image)
        self.background_src = None                     # callable(h,w,device)->tensor for video bgs
        self.out_alpha: torch.Tensor | None = None     # alpha of the last output (transparent mode)
        self._const: dict = {}
        self._bg_cache = None                          # (key, tensor) blurred static background

    def set_background(self, img: torch.Tensor | None):
        self.background = img
        self._bg_cache = None

    def _color(self, rgb, ref):
        key = ("c", tuple(round(float(v), 4) for v in rgb), ref.device)
        t = self._const.get(key)
        if t is None:
            t = torch.tensor(rgb, device=ref.device, dtype=torch.float32).view(1, 3, 1, 1)
            self._const[key] = t
        return t.expand(1, 3, *ref.shape[-2:])

    def _grid(self, H, W, device):
        key = ("g", H, W, device)
        g = self._const.get(key)
        if g is None:
            yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
            xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
            key_light = (1.2 - ((xx - 0.35) ** 2 + (yy - 0.25) ** 2).sqrt()).clamp(0, 1)
            radial = (((xx * 2 - 1) ** 2 + (yy * 2 - 1) ** 2) / 2).clamp(0, 1) ** 1.5
            g = self._const[key] = (key_light, radial)
        return g

    def _replacement(self, src, s: Settings):
        H, W = src.shape[-2:]
        sigma = 40 * s.bg_blur * H / 1080.0
        if self.background_src is not None:
            bg = self.background_src(H, W, src.device)
            if bg is not None:
                bg = bg.float()
                return gaussian_blur(bg, sigma) if sigma >= 0.5 else bg
        if self.background is not None:
            key = (H, W, src.device, round(sigma, 1), id(self.background))
            if self._bg_cache is None or self._bg_cache[0] != key:
                bg = F.interpolate(self.background.to(src.device).float(), size=(H, W),
                                   mode="bilinear", align_corners=False)
                if sigma >= 0.5:
                    bg = gaussian_blur(bg, sigma)
                self._bg_cache = (key, bg)
            return self._bg_cache[1]
        return self._color(s.bg_color, src)

    @torch.no_grad()
    def __call__(self, src: torch.Tensor, alpha: torch.Tensor | None, fgr: torch.Tensor | None,
                 s: Settings) -> torch.Tensor:
        H, W = src.shape[-2:]
        scale = H / 1080.0
        self.out_alpha = None
        if s.denoise > 0:
            clean = self.denoiser(src, s.denoise)
            if fgr is not None:
                # keep the engine's edge decontamination as a delta on top of the clean frame
                fgr = (clean.float() + (fgr.float() - src.float())).clamp(0, 1)
            src = clean
        if s.mode == "off" or alpha is None:
            out = src.float()
        else:
            a = refine_alpha(alpha, s, scale)
            fg = (fgr if (s.decontaminate and fgr is not None) else src).float()
            img = src.float()
            if s.relight > 0:
                key_light, _ = self._grid(H, W, src.device)
                fg = 1 - (1 - fg) ** (1 + s.relight * (0.6 + 0.8 * key_light))

            if s.mode == "matte":
                out = a.expand(1, 3, H, W)
            elif s.mode == "transparent":
                out = fg
                self.out_alpha = a
            else:
                if s.mode == "blur":
                    sigma = 4 + 60 * s.blur_strength * scale
                    # normalised convolution in ONE pass: blur [img*(1-a), (1-a)] together, so the
                    # subject never bleeds into the blurred background (no halo)
                    inv = 1 - a
                    both = gaussian_blur(torch.cat([img * inv, inv, img], 1), sigma)
                    w = both[:, 3:4]
                    # where the subject covers the whole blur window fall back to the plain blur
                    t = (w / 2e-3).clamp(0, 1)
                    bg = both[:, :3] / (w + 1e-4) * t + both[:, 4:7] * (1 - t)
                elif s.mode == "green":
                    bg = self._color(CHROMA_GREEN, src)
                elif s.mode == "blue":
                    bg = self._color(CHROMA_BLUE, src)
                elif s.mode == "replace":
                    bg = self._replacement(src, s)
                else:  # remove
                    bg = self._color(s.bg_color, src)

                if s.bg_dim > 0 and s.mode in ("blur", "replace"):
                    bg = bg * (1 - s.bg_dim)

                if s.light_wrap > 0 and s.mode in ("blur", "replace"):
                    # computed on a low-res level: it is a smooth glow anyway
                    lo = 4
                    a_lo = F.avg_pool2d(a, lo)
                    edge_lo = (1 - gaussian_blur(a_lo, 10 * scale / lo)) * a_lo
                    glow_lo = gaussian_blur(F.avg_pool2d(bg, lo), 12 * scale / lo)
                    wrap = F.interpolate(torch.cat([glow_lo, edge_lo], 1), size=(H, W),
                                         mode="bilinear", align_corners=False)
                    fg = fg + (wrap[:, :3] - fg) * wrap[:, 3:4] * s.light_wrap

                out = fg * a + bg * (1 - a)

            if s.vignette > 0 and s.mode not in ("matte", "transparent"):
                _, radial = self._grid(H, W, src.device)
                out = out * (1 - s.vignette * radial)

            if s.auto_frame:
                box = self.framer.update(a, s.frame_zoom, s.frame_speed)
                out = self.framer.crop(out, box)
                if self.out_alpha is not None:
                    self.out_alpha = self.framer.crop(self.out_alpha, box)

        if s.mirror:
            out = out.flip(-1)
            if self.out_alpha is not None:
                self.out_alpha = self.out_alpha.flip(-1)
        return out.clamp(0, 1)
