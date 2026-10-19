"""Fun props: face-tracked comical overlays (googly eyes, sunglasses, hats, ears,
light bulb, speech bubble).

Tracking: MediaPipe BlazeFace (Tasks API, CPU, ~3 ms) on a small frame every few video
frames, EMA-smoothed with short coasting so props don't jitter or vanish for a frame.
Drawing: procedural RGBA sprites (no art assets to ship) composited in torch on the GPU.
Positions come from the face box + eye/nose keypoints, so props ride the head through
turns; hats anchor above the face box, on the hair.
"""
from __future__ import annotations

import math
import os
import threading

import cv2
import numpy as np
import torch

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")
MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                                           "mediapipe", "blaze_face_short_range.tflite"))

PROPS = ("googly_eyes", "sunglasses", "propeller_hat", "santa_hat", "elf_hat",
         "rabbit_ears", "light_bulb", "speech_bubble")

LABELS = {"googly_eyes": "Googly eyes", "sunglasses": "Funny sunglasses",
          "propeller_hat": "Propeller hat", "santa_hat": "Santa hat", "elf_hat": "Elf hat",
          "rabbit_ears": "Rabbit ears", "light_bulb": "Cartoon light bulb",
          "speech_bubble": "Speech bubble (animated bob)"}

_sprites: dict = {}
_sprites_lock = threading.Lock()


# ---------------------------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------------------------

class Face:
    """One smoothed face, pixel coords of the frame being drawn."""
    __slots__ = ("box", "leye", "reye", "nose", "mouth", "score")

    def __init__(self, box, leye, reye, nose, mouth, score):
        self.box = box            # (x, y, w, h) top-left origin
        self.leye = leye          # (x, y) each
        self.reye = reye
        self.nose = nose
        self.mouth = mouth
        self.score = score

    def scaled(self, sx, sy):
        (x, y, w, h) = self.box
        f = lambda p: (p[0] * sx, p[1] * sy)
        return Face((x * sx, y * sy, w * sx, h * sy), f(self.leye), f(self.reye),
                    f(self.nose), f(self.mouth), self.score)

    def mirrored(self, width):
        """The same face in a horizontally flipped frame of this width."""
        (x, y, w, h) = self.box
        f = lambda p: (width - p[0], p[1])
        return Face((width - x - w, y, w, h), f(self.reye), f(self.leye), f(self.nose), f(self.mouth),
                    self.score)


def _ema_face(prev: Face | None, new: Face, a: float) -> Face:
    if prev is None:
        return new
    mix = lambda p, q: (a * p[0] + (1 - a) * q[0], a * p[1] + (1 - a) * q[1])
    (x0, y0, w0, h0), (x1, y1, w1, h1) = prev.box, new.box
    box = (a * x1 + (1 - a) * x0, a * y1 + (1 - a) * y0,
           a * w1 + (1 - a) * w0, a * h1 + (1 - a) * h0)
    return Face(box, mix(new.leye, prev.leye), mix(new.reye, prev.reye), mix(new.nose, prev.nose),
                mix(new.mouth, prev.mouth), new.score)


class FaceTracker:
    """BlazeFace, throttled + smoothed. detect_bgr() is the raw model call; update() holds the
    EMA/coast state (unit-testable without a model); __call__ does both, throttled."""

    def __init__(self, model_path: str = MODEL_PATH, every: int = 3, ema: float = 0.6,
                 coast: int = 10, min_score: float = 0.5):
        self.model_path = model_path
        self.every, self._ema, self._coast, self.min_score = every, ema, coast, min_score
        self._det = None
        self._failed = False
        self._i = 0
        self._last: Face | None = None
        self._miss = 0

    def _ensure(self):
        if self._det is None and not self._failed:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision
                if not os.path.isfile(self.model_path):
                    self._failed = True
                    return None
                opts = vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path))
                self._det = vision.FaceDetector.create_from_options(opts)
            except Exception:  # noqa: BLE001 - props simply stay off
                self._failed = True
        return self._det

    def detect(self, img_np: np.ndarray, bgr: bool = True) -> list[Face]:
        """Raw detections in input-pixel coords, best first. Empty without a model."""
        if bgr:
            return self.detect_bgr(img_np)
        det = self._ensure()
        if det is None:
            return []
        return self._detect_rgb(det, img_np)

    def detect_bgr(self, img_bgr: np.ndarray) -> list[Face]:
        """Raw detections in image-pixel coords, best first. Empty without a model."""
        det = self._ensure()
        if det is None:
            return []
        h, w = img_bgr.shape[:2]
        scale = 320 / max(h, w) if max(h, w) > 320 else 1.0   # BlazeFace short-range wants small input
        small = img_bgr if scale == 1.0 else cv2.resize(img_bgr, (max(1, int(w * scale)),
                                                                 max(1, int(h * scale))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return self._detect_rgb(det, rgb, 1.0 / scale if scale != 1.0 else 1.0)

    def _detect_rgb(self, det, rgb: np.ndarray, back: float = 1.0) -> list[Face]:
        import mediapipe as mp
        h, w = rgb.shape[:2]
        try:
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        except Exception:  # noqa: BLE001
            return []
        out = []
        for d in (res.detections or []):
            score = float(sum(c.score for c in d.categories) / max(1, len(d.categories)))
            bb = d.bounding_box
            kp = [(k.x * w * back, k.y * h * back) for k in d.keypoints]
            if len(kp) < 4:
                continue
            out.append(Face((bb.origin_x * back, bb.origin_y * back,
                               bb.width * back, bb.height * back),
                            (kp[1][0], kp[1][1]), (kp[0][0], kp[0][1]),
                            (kp[2][0], kp[2][1]), (kp[3][0], kp[3][1]), score))
        return sorted(out, key=lambda f: -f.score)

    def update(self, dets: list[Face]) -> Face | None:
        """Fold new detections into the smoothed track (pure state machine)."""
        good = [d for d in dets if d.score >= self.min_score]
        if good:
            best = max(good, key=lambda d: d.score)
            self._last = _ema_face(self._last, best, self._ema)
            self._miss = 0
        else:
            self._miss += 1
            if self._miss > self._coast:
                self._last = None
        return self._last

    def __call__(self, img_np: np.ndarray, bgr: bool = True) -> Face | None:
        """Detect (throttled) in input-pixel coords, smoothed; coasts briefly on dropout."""
        self._i += 1
        if (self._i - 1) % self.every:
            return self._last
        return self.update(self.detect(img_np, bgr=bgr))

    def close(self):
        det, self._det = self._det, None
        if det is not None:
            try:
                det.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------

def anchors(face: Face) -> dict:
    """Named pixel anchors for prop placement from a pixel-coords face."""
    (x, y, w, h) = face.box
    cx = x + w / 2
    return {"leye": face.leye, "reye": face.reye, "nose": face.nose, "mouth": face.mouth,
            "face_cx": cx, "face_cy": y + h / 2, "face_w": w, "face_h": h,
            "head_top": (cx, y - 0.22 * h),          # hairline, where hats sit
            "head_l": x, "head_r": x + w}


def stamp(out: torch.Tensor, sprite_rgba: np.ndarray, cx: float, cy: float, device) -> None:
    """Alpha-composite an RGBA sprite centred at (cx, cy) onto a (1,3,H,W) float frame, or onto a
    (1,4,H,W) RGBA layer (straight colour), whose alpha then grows to cover the sprite."""
    sh, sw = sprite_rgba.shape[:2]
    H, W = out.shape[-2:]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    dx0, dy0 = max(0, -x0), max(0, -y0)
    dx1, dy1 = min(sw, W - x0), min(sh, H - y0)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sp = torch.from_numpy(sprite_rgba[dy0:dy1, dx0:dx1].astype(np.float32) / 255).to(device)
    rgb, a = sp[..., :3].permute(2, 0, 1), sp[..., 3:4].permute(2, 0, 1)
    tgt = out[0, :3, y0 + dy0:y0 + dy1, x0 + dx0:x0 + dx1]
    tgt.mul_(1 - a).add_(rgb * a)
    if out.shape[1] == 4:
        ta = out[0, 3:4, y0 + dy0:y0 + dy1, x0 + dx0:x0 + dx1]
        ta.mul_(1 - a).add_(a)


# ---------------------------------------------------------------------------------------------
# Sprites (built with cv2 in BGRA, cached by key)
# ---------------------------------------------------------------------------------------------

def _rgba(h, w):
    return np.zeros((h, w, 4), np.uint8)


def _ellipse(img, cx, cy, rx, ry, color, angle=0):
    cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, color, -1,
                lineType=cv2.LINE_AA)


def _round_rect(img, x0, y0, x1, y1, r, color):
    cv2.rectangle(img, (int(x0 + r), int(y0)), (int(x1 - r), int(y1)), color, -1)
    cv2.rectangle(img, (int(x0), int(y0 + r)), (int(x1), int(y1 - r)), color, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (int(cx), int(cy)), int(r), color, -1, lineType=cv2.LINE_AA)


def _C(c):
    """RGB(A) palette tuple -> cv2 BGRA order."""
    return (c[2], c[1], c[0], c[3])


def _cached(key, build):
    with _sprites_lock:
        sp = _sprites.get(key)
    if sp is None:
        sp = build()
        with _sprites_lock:
            _sprites[key] = sp
    return sp


def _to_rgba(bgra: np.ndarray) -> np.ndarray:
    return bgra[:, :, [2, 1, 0, 3]]


def _sunglasses_sprite(s: int) -> np.ndarray:
    sp = _rgba(int(0.42 * s), int(1.5 * s))
    h, w = sp.shape[:2]
    lw, lh = int(0.62 * s), int(0.34 * s)
    for ex in (int(0.37 * s), int(1.13 * s)):
        _round_rect(sp, ex - lw / 2, h * 0.5 - lh / 2, ex + lw / 2, h * 0.5 + lh / 2,
                    int(0.1 * s), _C((10, 10, 10, 255)))
        cv2.line(sp, (int(ex - lw * 0.3), int(h * 0.5 - lh * 0.25)),
                 (int(ex - lw * 0.05), int(h * 0.5 + lh * 0.15)), _C((255, 255, 255, 200)),
                 max(2, int(0.035 * s)), cv2.LINE_AA)          # shine slash
    cv2.rectangle(sp, (int(0.37 * s + lw / 2 - 2), int(h * 0.5 - 0.05 * s)),
                  (int(1.13 * s - lw / 2 + 2), int(h * 0.5 + 0.05 * s)), _C((10, 10, 10, 255)),
                  -1)
    return _to_rgba(sp)


def _cap_sprite(s: int, color=(60, 120, 255, 255)) -> np.ndarray:
    sp = _rgba(int(0.34 * s), int(1.3 * s))
    h, w = sp.shape[:2]
    _ellipse(sp, w / 2, h * 0.95, w * 0.48, h * 0.42, _C(color))   # dome
    cv2.ellipse(sp, (int(w / 2), int(h * 0.95)), (int(w * 0.62), int(h * 0.16)), 0, 0, 360,
                _C((30, 30, 30, 255)), -1, lineType=cv2.LINE_AA)   # brim
    return _to_rgba(sp)


def _cone_hat_sprite(s: int, body=(210, 40, 40, 255), flop=0.35) -> np.ndarray:
    sp = _rgba(int(0.66 * s), int(1.1 * s))
    h, w = sp.shape[:2]
    apex = (w * (0.5 + flop), h * 0.06)
    cv2.fillPoly(sp, [np.array([(w * 0.08, h * 0.86), apex, (w * 0.92, h * 0.86)], np.int32)],
                 _C(body), lineType=cv2.LINE_AA)
    cv2.rectangle(sp, (int(w * 0.02), int(h * 0.80)), (int(w * 0.98), int(h * 0.98)),
                  _C((245, 245, 245, 255)), -1)                    # brim
    cv2.circle(sp, (int(apex[0]), int(apex[1])), int(0.07 * s), _C((245, 245, 245, 255)), -1,
               lineType=cv2.LINE_AA)                               # pom
    return _to_rgba(sp)


def _bulb_sprite(s: int) -> np.ndarray:
    sp = _rgba(int(0.75 * s), int(0.9 * s))
    h, w = sp.shape[:2]
    cx, cy, r = w / 2, h * 0.40, 0.20 * s
    for i in range(8):                                         # rays
        a = math.pi / 4 * i
        cv2.line(sp, (int(cx + math.cos(a) * r * 1.45), int(cy + math.sin(a) * r * 1.45)),
                 (int(cx + math.cos(a) * r * 1.9), int(cy + math.sin(a) * r * 1.9)),
                 _C((255, 210, 60, 255)), max(2, int(0.03 * s)), cv2.LINE_AA)
    _ellipse(sp, cx, cy, r, r, _C((255, 220, 90, 255)))        # glass
    _ellipse(sp, cx - r * 0.3, cy - r * 0.3, r * 0.28, r * 0.28, _C((255, 255, 255, 230)))
    cv2.rectangle(sp, (int(cx - r * 0.45), int(cy + r * 0.85)),
                  (int(cx + r * 0.45), int(cy + r * 1.35)), _C((120, 120, 120, 255)), -1)
    return _to_rgba(sp)


def _bubble_sprite(s: int, text: str) -> np.ndarray:
    scale = max(0.6, s / 300)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, max(2, int(3 * scale)))
    pad = int(0.12 * s)
    w, h = tw + 2 * pad, th + 2 * pad + int(0.16 * s)
    sp = _rgba(h + int(0.2 * s), w)
    _round_rect(sp, 0, 0, w, h, int(0.12 * s), _C((255, 255, 255, 255)))
    _round_rect(sp, 3, 3, w - 3, h - 3, int(0.1 * s), _C((30, 30, 30, 255)))   # dark outline
    _round_rect(sp, 6, 6, w - 6, h - 6, int(0.09 * s), _C((255, 255, 255, 255)))
    cv2.putText(sp, text, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale,
                _C((200, 30, 30, 255)), max(2, int(3 * scale)), cv2.LINE_AA)
    tx = int(w * 0.3)                                                        # tail
    cv2.fillPoly(sp, [np.array([(tx, h - 4), (tx + int(0.14 * s), h - 4),
                                (tx - int(0.02 * s), h + int(0.18 * s))], np.int32)],
                 _C((255, 255, 255, 255)), lineType=cv2.LINE_AA)
    return _to_rgba(sp)


def _blades_sprite(s: int, step: int, steps: int = 8) -> np.ndarray:
    sp = _rgba(int(0.5 * s), int(1.3 * s))
    h, w = sp.shape[:2]
    cx, cy = w / 2, h / 2
    cv2.circle(sp, (int(cx), int(cy)), max(2, int(0.035 * s)), (80, 80, 80, 255), -1)
    a = 2 * math.pi * step / steps
    L = 0.58 * s
    for k in (0, 1):
        ang = a + k * math.pi
        foreshort = abs(math.cos(ang))                     # spin squashes the far blade
    cv2.ellipse(sp, (int(cx), int(cy)), (int(L / 2), max(2, int(0.075 * s * (0.3 + foreshort)))),
                int(math.degrees(ang) % 180), 0, 360, _C((255, 200, 60, 255)), -1,
                lineType=cv2.LINE_AA)
    return _to_rgba(sp)


# ---------------------------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------------------------

def draw_props(out: torch.Tensor, face: Face | None, props, t: float,
               bubble_text: str = "HELLO!") -> torch.Tensor:
    """Composite the enabled props onto a (1,3,H,W) float frame or (1,4,H,W) RGBA layer
    (no-op without a face)."""
    if face is None or not props:
        return out
    dev = str(out.device)
    H, W = out.shape[-2:]
    (fx, fy, fw, fh) = face.box
    s = max(40.0, fw)                                     # prop scale unit = face width
    bob = math.sin(t * 2.2) * 0.02 * s                    # gentle float for worn things
    leye = (face.leye[0], face.leye[1] + bob)
    reye = (face.reye[0], face.reye[1] + bob)
    hairline = fy - 0.22 * fh                             # where hats sit: top of the hair
    top = (fx + fw / 2, hairline + bob)

    if "googly_eyes" in props:
        for i, (ex, ey) in enumerate((leye, reye)):
            rx, ry = 0.17 * s, 0.20 * s
            white = _rgba(int(2 * ry) + 4, int(2 * rx) + 4)
            _ellipse(white, white.shape[1] / 2, white.shape[0] / 2, rx, ry, (255, 255, 255, 255))
            stamp(out, _to_rgba(white), ex, ey, dev)
            wob = 0.05 * s
            px = ex + wob * math.sin(t * 3.1 + i * 2.4)
            py = ey + wob * 0.8 * math.cos(t * 2.3 + i * 1.7)
            pr = 0.075 * s
            pupil = _rgba(int(2 * pr) + 4, int(2 * pr) + 4)
            _ellipse(pupil, pupil.shape[1] / 2, pupil.shape[0] / 2, pr, pr, (15, 15, 15, 255))
            stamp(out, _to_rgba(pupil), px, py, dev)
            glint = _rgba(10, 10)
            _ellipse(glint, 5, 5, 2.5, 2.5, (255, 255, 255, 255))
            stamp(out, _to_rgba(glint), px - pr * 0.3, py - pr * 0.3, dev)

    if "sunglasses" in props:
        sp = _cached(("shades", int(s)), lambda: _sunglasses_sprite(int(s)))
        mx, my = (leye[0] + reye[0]) / 2, (leye[1] + reye[1]) / 2
        stamp(out, sp, mx, my, dev)

    if "propeller_hat" in props:
        cap = _cached(("cap", int(s)), lambda: _cap_sprite(int(s)))
        stamp(out, cap, top[0], top[1] - 0.02 * s, dev)
        step = int(t * 9) % 8
        blades = _cached(("blades", int(s), step),
                         lambda: _blades_sprite(int(s), step))
        stamp(out, blades, top[0], top[1] - 0.30 * s, dev)

    if "santa_hat" in props:
        sp = _cached(("santa", int(s)), lambda: _cone_hat_sprite(int(s), flop=0.35))
        stamp(out, sp, top[0] + 0.05 * s, top[1] - 0.18 * s, dev)

    if "elf_hat" in props:
        sp = _cached(("elf", int(s)), lambda: _cone_hat_sprite(
            int(s), body=(40, 170, 80, 255), flop=-0.4))
        stamp(out, sp, top[0] - 0.05 * s, top[1] - 0.18 * s, dev)

    if "rabbit_ears" in props:
        for i, ex in enumerate((top[0] - 0.20 * s, top[0] + 0.20 * s)):
            tilt = int((-12 if i == 0 else 12) + 4 * math.sin(t * 2.6 + i * 3.1))
            ear = _rgba(int(0.62 * s), int(0.34 * s))
            _ellipse(ear, ear.shape[1] / 2, ear.shape[0] / 2, 0.15 * s, 0.27 * s,
                     _C((235, 235, 240, 255)), angle=tilt)
            _ellipse(ear, ear.shape[1] / 2, ear.shape[0] / 2 + int(0.04 * s), 0.07 * s, 0.17 * s,
                     _C((255, 150, 180, 255)), angle=tilt)
            stamp(out, _to_rgba(ear), ex, top[1] - 0.18 * s, dev)

    if "light_bulb" in props:
        sp = _cached(("bulb", int(s)), lambda: _bulb_sprite(int(s)))
        pulse = 0.72 + 0.28 * math.sin(t * 4.0)
        by = top[1] - 0.16 * s + bob * 0.5
        flick = sp.copy()          # pulse the sprite's alpha; never mutate the cache
        flick[..., 3] = (flick[..., 3].astype(np.float32) * pulse).astype(np.uint8)
        stamp(out, flick, top[0], by, dev)

    if "speech_bubble" in props:
        sp = _cached(("bubble", int(s), bubble_text),
                     lambda: _bubble_sprite(int(s), bubble_text or "HELLO!"))
        bx = min(max(top[0] + 0.75 * s, sp.shape[1] / 2 + 4), W - sp.shape[1] / 2 - 4)
        by = max(top[1] - 0.30 * s, sp.shape[0] / 2 + 4) + bob * 0.5
        stamp(out, sp, bx, by, dev)

    return out
