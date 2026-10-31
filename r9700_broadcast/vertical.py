"""Vertical (9:16) recording template: a crop window over the landscape output.

The crop is a full-height 9:16 window (zoom > 1 makes it smaller and lets it move vertically too).
Position is live - it can be panned while recording; the crop *size* is fixed when a recording
starts, because the encoder's input size cannot change mid-file.

Social apps cover parts of a vertical video with their own UI. SAFE_ZONES are the regions to keep
clear of faces and text (fractions of the 9:16 frame, union of TikTok / Reels / Shorts layouts).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

ASPECT = 9 / 16

# (x, y, w, h) fractions of the vertical frame that the apps' UI covers
SAFE_ZONES = {
    "top": (0.0, 0.0, 1.0, 0.10),        # status bar, "Following | For You"
    "bottom": (0.0, 0.72, 1.0, 0.28),    # caption, user name, sound, progress bar
    "right": (0.86, 0.30, 0.14, 0.42),   # like / comment / share column
}

OUTPUT_SIZES = {
    "1080x1920": (1080, 1920),   # what TikTok / Reels / Shorts expect (upscaled from the crop)
    "native": None,              # the crop's own pixels, no scaling
}

REC_MODES = {"long": "Long form only", "vertical": "Vertical only", "both": "Long form + Vertical"}


@dataclass
class VerticalFrame:
    x: float = 0.5        # crop centre, fraction of the source width
    y: float = 0.5        # crop centre, fraction of the source height (matters only when zoom > 1)
    zoom: float = 1.0     # 1 = full height; 2 = half height (tighter on the subject)

    def size(self, W: int, H: int) -> tuple[int, int]:
        """Crop size in source pixels: even numbers (yuv420 encoders need them), never wider than W."""
        h = min(H, int(round(H / max(1.0, self.zoom))))
        w = int(round(h * ASPECT))
        if w > W:                                    # very tall/narrow sources
            w = W
            h = min(H, int(round(w / ASPECT)))
        return max(2, w & ~1), max(2, h & ~1)

    def rect(self, W: int, H: int, size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
        """(x0, y0, w, h) of the crop in source pixels, clamped inside the frame."""
        w, h = size or self.size(W, H)
        x0 = int(round(self.x * W - w / 2))
        y0 = int(round(self.y * H - h / 2))
        return min(max(x0, 0), W - w), min(max(y0, 0), H - h), w, h

    def clamp(self, W: int, H: int):
        """Snap x/y back so the stored centre matches where the crop actually is."""
        x0, y0, w, h = self.rect(W, H)
        self.x, self.y = (x0 + w / 2) / W, (y0 + h / 2) / H

    def crop(self, frame: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        H, W = frame.shape[:2]
        x0, y0, w, h = self.rect(W, H, size)
        return frame[y0:y0 + h, x0:x0 + w]


def output_size(key: str, crop_size: tuple[int, int]) -> tuple[int, int]:
    return OUTPUT_SIZES.get(key) or crop_size


def unique_path(path: str) -> str:
    """'name.mp4' -> 'name (2).mp4' if it exists - a recording never overwrites another."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{base} ({n}){ext}"):
        n += 1
    return f"{base} ({n}){ext}"


def safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_.()[]" else "_" for c in name.strip())
    return keep.strip(" .")


def recording_paths(directory: str, mode: str, ext: str, name: str = "",
                    stamp: str | None = None) -> dict[str, str]:
    """{'long': .../Longform_<name>.ext, 'vertical': .../Vertical_<name>.ext} for the chosen mode.
    <name> is the user's name for the take, or the date and time when blank. Both files of one take
    share the same <name> (and the same ' (n)' suffix if a take of that name already exists)."""
    base = safe_name(name) or (stamp or time.strftime("%Y-%m-%d %H-%M-%S"))
    kinds = {"long": ["long"], "vertical": ["vertical"]}.get(mode, ["long", "vertical"])
    prefix = {"long": "Longform_", "vertical": "Vertical_"}
    n = 1
    while True:
        suffix = "" if n == 1 else f" ({n})"
        paths = {k: os.path.join(directory, f"{prefix[k]}{base}{suffix}.{ext}") for k in kinds}
        if not any(os.path.exists(p) for p in paths.values()):
            return paths
        n += 1
