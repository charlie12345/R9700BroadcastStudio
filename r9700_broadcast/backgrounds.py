"""Background library: images and looping videos in (almost) any format.

* Images  - png jpg jpeg webp bmp tif tiff (OpenCV), anything else ffmpeg can decode (avif, heic,
            jxl, ...) via an ffmpeg single-frame fallback.
* Videos  - mp4 mov mkv webm avi m4v mpg mpeg wmv flv ts gif apng ... streamed by an ffmpeg
            subprocess that loops forever at the clip's native rate and scales to the output size
            (cover / contain / stretch), so a 10-minute 4K clip costs no RAM.
* Library - built-in backgrounds (assets/backgrounds) + user items listed in
            ~/.local/share/r9700-broadcast/library.json (we never delete user files).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTIN_DIR = os.path.join(HERE, "assets", "backgrounds")
DATA_DIR = os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "r9700-broadcast")
LIBRARY_JSON = os.path.join(DATA_DIR, "library.json")
THUMB_DIR = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "r9700-broadcast", "thumbs")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".jpe", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".heic",
             ".heif", ".jxl", ".tga", ".exr", ".hdr", ".ppm", ".pgm", ".jp2"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv",
             ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".gif", ".apng", ".mxf"}
FILE_FILTER = ("Backgrounds (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXT | VIDEO_EXT)) + ");;"
               "Images (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXT)) + ");;"
               "Videos (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXT)) + ");;All files (*)")
FITS = {"cover": "Fill (crop)", "contain": "Fit (letterbox)", "stretch": "Stretch"}


def kind_of(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    # unknown extension: ask ffprobe
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                              "-show_entries", "stream=codec_type,nb_read_packets", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    if not out.startswith("video"):
        return None
    n = out.split(",")[-1]
    return "video" if n.isdigit() and int(n) > 1 else "image"


def _ffmpeg_frame(path: str) -> np.ndarray | None:
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "image2pipe",
                            "-vcodec", "png", "-"], capture_output=True, timeout=20)
        if r.returncode == 0 and r.stdout:
            return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        pass
    return None


def load_image_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = _ffmpeg_frame(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_image(img: np.ndarray, w: int, h: int, fit: str = "cover") -> np.ndarray:
    ih, iw = img.shape[:2]
    if fit == "stretch":
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = max(w / iw, h / ih) if fit == "cover" else min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if fit == "cover":
        x0, y0 = (nw - w) // 2, (nh - h) // 2
        return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])
    out = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def _vf(w, h, fit):
    if fit == "stretch":
        return f"scale={w}:{h}"
    if fit == "contain":
        return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


class VideoBackground:
    """Looping, natively-paced ffmpeg decode of any video/GIF; exposes the newest RGB frame."""

    def __init__(self, path: str, width: int, height: int, fit: str = "cover"):
        self.path, self.size, self.fit = path, (width, height), fit
        self._frame = None
        self.seq = 0
        self._stop = False
        self._proc = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _cmd(self):
        w, h = self.size
        pre = ["-ignore_loop", "0"] if self.path.lower().endswith(".gif") else []
        return ["ffmpeg", "-nostdin", "-v", "error", "-stream_loop", "-1", "-re", *pre, "-i", self.path,
                "-an", "-sn", "-vf", _vf(w, h, self.fit) + ",format=rgb24",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    def _run(self):
        w, h = self.size
        n = w * h * 3
        while not self._stop:
            self._proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          bufsize=n * 2)
            got_any = False
            while not self._stop:
                buf = self._proc.stdout.read(n)
                if len(buf) < n:
                    break
                got_any = True
                self._frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                self.seq += 1
            self._proc.kill()
            self._proc.wait()
            if not got_any:
                time.sleep(1.0)  # undecodable file: don't spin

    def latest(self):
        return self.seq, self._frame

    def close(self):
        self._stop = True
        if self._proc:
            self._proc.kill()
        self._t.join(timeout=2)


class BackgroundProvider:
    """GPU-side provider plugged into Compositor.background_src. Uploads only new frames and
    rebuilds the decoder when the output size or fit mode changes."""

    def __init__(self, path: str, fit: str = "cover"):
        self.path, self.fit = path, fit
        self.kind = kind_of(path)
        self._video: VideoBackground | None = None
        self._image = load_image_rgb(path) if self.kind == "image" else None
        self._cache = None   # (key, tensor)
        self._last_seq = -1
        if self.kind is None or (self.kind == "image" and self._image is None):
            raise ValueError(f"cannot decode {path}")

    def __call__(self, h, w, device):
        import torch
        if self.kind == "image":
            key = (h, w, self.fit, str(device))
            if self._cache is None or self._cache[0] != key:
                rgb = fit_image(self._image, w, h, self.fit)
                t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0).half() / 255
                self._cache = (key, t)
            return self._cache[1]
        if self._video is None or self._video.size != (w, h) or self._video.fit != self.fit:
            if self._video:
                self._video.close()
            self._video = VideoBackground(self.path, w, h, self.fit)
            self._last_seq = -1
            self._cache = None
        seq, frame = self._video.latest()
        if frame is None:
            return self._cache[1] if self._cache else None
        if seq != self._last_seq:
            self._last_seq = seq
            t = torch.from_numpy(frame).pin_memory().to(device, non_blocking=True)
            self._cache = (seq, t.permute(2, 0, 1).unsqueeze(0).half() / 255)
        return self._cache[1]

    def close(self):
        if self._video:
            self._video.close()


# ---------------------------------------------------------------------------------------------
# Library + thumbnails
# ---------------------------------------------------------------------------------------------

def thumbnail(path: str, w=192, h=108) -> np.ndarray | None:
    """RGB thumbnail (cached on disk). Videos use a frame 1 s in (or the first frame)."""
    os.makedirs(THUMB_DIR, exist_ok=True)
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime}|{w}x{h}".encode()).hexdigest()
    tp = os.path.join(THUMB_DIR, key + ".jpg")
    if os.path.exists(tp):
        img = cv2.imread(tp)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = None
    if kind_of(path) == "image":
        img = load_image_rgb(path)
    else:
        for ss in ("1", "0"):
            r = subprocess.run(["ffmpeg", "-v", "error", "-ss", ss, "-i", path, "-frames:v", "1",
                                "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, timeout=20)
            if r.stdout:
                img = cv2.cvtColor(cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB)
                break
    if img is None:
        return None
    t = fit_image(img, w, h, "cover")
    cv2.imwrite(tp, cv2.cvtColor(t, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return t


class Library:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.user: list[str] = []
        try:
            with open(LIBRARY_JSON) as f:
                self.user = [p for p in json.load(f).get("items", []) if os.path.exists(p)]
        except Exception:  # noqa: BLE001
            pass

    def builtin(self) -> list[str]:
        if not os.path.isdir(BUILTIN_DIR):
            return []
        return sorted(os.path.join(BUILTIN_DIR, f) for f in os.listdir(BUILTIN_DIR)
                      if kind_of(os.path.join(BUILTIN_DIR, f)))

    def items(self) -> list[tuple[str, bool]]:
        """[(path, is_builtin)] user items first (newest first), then built-ins."""
        return [(p, False) for p in reversed(self.user)] + [(p, True) for p in self.builtin()]

    def add(self, paths) -> list[str]:
        added = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isfile(p) and kind_of(p) and p not in self.user:
                self.user.append(p)
                added.append(p)
        self.save()
        return added

    def remove(self, path):
        if path in self.user:
            self.user.remove(path)
            self.save()

    def save(self):
        tmp = LIBRARY_JSON + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"items": self.user}, f, indent=1)
        shutil.move(tmp, LIBRARY_JSON)
