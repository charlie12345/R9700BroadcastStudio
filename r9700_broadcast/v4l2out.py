"""V4L2 loopback output written with raw ioctls (no pyvirtualcam).

Two devices are created by scripts/setup-virtualcam.sh:
  /dev/video42 "R9700 Broadcast Camera"       YUYV   - Zoom, Discord, Teams, browsers
  /dev/video43 "R9700 Broadcast Alpha (OBS)"  ABGR32 - OBS "Video Capture Device (V4L2)" maps
                                                       ABGR32 -> VIDEO_FORMAT_BGRA, alpha intact
ABGR32 ('AR24') is B,G,R,A in memory: exactly the BGRA frames the pipeline already produces, so the
alpha device is a straight write() with no conversion.

Struct layouts / ioctl numbers below were checked against <linux/videodev2.h> (tests/test_v4l2out.py).
"""
from __future__ import annotations

import errno
import fcntl
import glob
import os
import struct

import cv2
import numpy as np

VIDIOC_QUERYCAP = 0x80685600
VIDIOC_G_FMT = 0xC0D05604
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_S_PARM = 0xC0CC5616
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8
FOURCC = {"ABGR32": 0x34325241, "YUYV": 0x56595559}
BPP = {"ABGR32": 4, "YUYV": 2}
SIZEOF_FORMAT = 208     # struct v4l2_format (fmt union at offset 8)
SIZEOF_STREAMPARM = 204  # struct v4l2_streamparm (timeperframe at offset 12)

LABEL_CHAT = "R9700 Broadcast Camera"
LABEL_ALPHA = "R9700 Broadcast Alpha"


def pack_format(width: int, height: int, fmt: str) -> bytearray:
    buf = bytearray(SIZEOF_FORMAT)
    bpl = width * BPP[fmt]
    struct.pack_into("<I", buf, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)
    # v4l2_pix_format: width height pixelformat field bytesperline sizeimage colorspace priv
    struct.pack_into("<8I", buf, 8, width, height, FOURCC[fmt], V4L2_FIELD_NONE, bpl, bpl * height,
                     V4L2_COLORSPACE_SRGB, 0)
    return buf


def pack_streamparm(fps: int) -> bytearray:
    buf = bytearray(SIZEOF_STREAMPARM)
    struct.pack_into("<I", buf, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)
    struct.pack_into("<2I", buf, 12, 1, fps)   # timeperframe = 1/fps
    return buf


def device_name(dev: str) -> str:
    try:
        with open(f"/sys/class/video4linux/{os.path.basename(dev)}/name") as f:
            return f.read().strip().strip('"')   # tolerate quotes left over by modprobe label parsing
    except OSError:
        return ""


def find_device(label: str) -> str | None:
    for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[10:]) if p[10:].isdigit() else 0):
        if device_name(dev).startswith(label):
            return dev
    return None


class V4L2Camera:
    """Write frames to a v4l2loopback device in ABGR32 (alpha) or YUYV."""

    def __init__(self, width: int, height: int, fps: int, fmt: str = "YUYV", device: str | None = None):
        if fmt not in FOURCC:
            raise ValueError(fmt)
        label = LABEL_ALPHA if fmt == "ABGR32" else LABEL_CHAT
        device = device or find_device(label)
        if device is None:
            raise RuntimeError(f"No “{label}” v4l2loopback device. Run once: sudo scripts/setup-virtualcam.sh")
        self.device, self.fmt, self.fps = device, fmt, fps
        self.size = (width, height)
        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        try:
            buf = pack_format(width, height, fmt)
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, buf)
            w, h, four = struct.unpack_from("<3I", buf, 8)
            if (w, h, four) != (width, height, FOURCC[fmt]):
                raise RuntimeError(f"{device} refused {fmt} {width}x{height} (got {w}x{h} {four:#x})")
            self.frame_bytes = struct.unpack_from("<I", buf, 8 + 20)[0] or width * height * BPP[fmt]
            try:
                fcntl.ioctl(self.fd, VIDIOC_S_PARM, pack_streamparm(fps))
            except OSError:
                pass  # older v4l2loopback: rate is then taken from the write cadence
        except Exception:
            os.close(self.fd)
            raise
        self._yuyv = np.empty((height, width, 2), np.uint8) if fmt == "YUYV" else None
        self.dropped = 0

    def send_bgra(self, bgra: np.ndarray):
        h, w = bgra.shape[:2]
        if (w, h) != self.size:
            bgra = cv2.resize(bgra, self.size, interpolation=cv2.INTER_AREA)
        if self.fmt == "YUYV":
            cv2.cvtColor(bgra, cv2.COLOR_BGRA2YUV_YUY2, dst=self._yuyv)
            data = self._yuyv
        else:
            data = np.ascontiguousarray(bgra)
        try:
            os.write(self.fd, memoryview(data).cast("B"))
        except BlockingIOError:
            self.dropped += 1        # no reader keeping up: drop rather than stall the pipeline
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.ENOSPC):
                raise

    def send(self, rgb: np.ndarray):
        self.send_bgra(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA))

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
