"""PipeWire virtual camera (no kernel module, no root). Carries alpha into OBS Studio.

Frames are written as BGRA into a shared-memory triple buffer that the native helper
`bin/r9700-pw-camera` publishes as a PipeWire Video/Source node with media.role=Camera.
"""
from __future__ import annotations

import mmap
import os
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(HERE, "bin", "r9700-pw-camera")
MAGIC = 0x52393742
HDR = 64
NODE_NAME = "r9700_broadcast_camera"


class PipeWireCamera:
    def __init__(self, width: int, height: int, fps: int, description="R9700 Broadcast Camera"):
        if not os.path.exists(HELPER):
            raise RuntimeError("bin/r9700-pw-camera missing - run: make -C native")
        self.size = (width, height)
        self.fps = fps
        self.slot_bytes = width * height * 4
        run = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
        self.path = os.path.join(run, f"r9700-broadcast-cam-{os.getpid()}.shm")
        total = HDR + 3 * self.slot_bytes
        with open(self.path, "wb") as f:
            f.truncate(total)
        self._f = open(self.path, "r+b")
        self.mm = mmap.mmap(self._f.fileno(), total)
        struct.pack_into("<6I", self.mm, 0, MAGIC, 1, width, height, self.slot_bytes, fps)
        struct.pack_into("<QI", self.mm, 24, 0, 0)
        self.frames = np.frombuffer(self.mm, dtype=np.uint8, count=3 * self.slot_bytes, offset=HDR) \
            .reshape(3, height, width, 4)
        self.seq = 0
        wrap = os.environ.get("R9700_PWCAM_WRAP", "").split()
        self.proc = subprocess.Popen(wrap + [HELPER, self.path, str(width), str(height), str(fps), description],
                                     stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.device = f"PipeWire: {description}"

    @property
    def alive(self):
        return self.proc.poll() is None

    def send_bgra(self, bgra: np.ndarray):
        slot = (self.seq + 1) % 3
        self.frames[slot] = bgra
        self.seq += 1
        struct.pack_into("<QI", self.mm, 24, self.seq, slot)
        try:
            self.proc.stdin.write(b"\x01")
        except (BrokenPipeError, OSError):
            pass

    def send(self, rgb: np.ndarray, alpha: np.ndarray | None = None):
        """rgb: HxWx3 uint8; alpha: HxW uint8 (255 = opaque) or None."""
        h, w = rgb.shape[:2]
        if (w, h) != self.size:
            import cv2
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, self.size, interpolation=cv2.INTER_AREA)
        slot = (self.seq + 1) % 3
        dst = self.frames[slot]
        dst[..., 0] = rgb[..., 2]
        dst[..., 1] = rgb[..., 1]
        dst[..., 2] = rgb[..., 0]
        dst[..., 3] = 255 if alpha is None else alpha
        self.seq += 1
        struct.pack_into("<QI", self.mm, 24, self.seq, slot)
        try:
            self.proc.stdin.write(b"\x01")
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        del self.frames
        self.mm.close()
        self._f.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass
