"""V4L2 loopback writer: ABI constants vs <linux/videodev2.h>, packing, YUYV colour, and (when the
v4l2loopback devices exist) a real write -> read round trip including the alpha channel."""
import os, shutil, struct, subprocess, time
import numpy as np
import pytest

from r9700_broadcast import v4l2out as V

ABI_C = r"""
#include <stdio.h>
#include <stddef.h>
#include <linux/videodev2.h>
int main(void){
 printf("%zu %zu %zu %zu %lu %lu %lu %lu %u %u\n", sizeof(struct v4l2_format), offsetof(struct v4l2_format, fmt),
   sizeof(struct v4l2_streamparm), offsetof(struct v4l2_streamparm, parm.output.timeperframe),
   (unsigned long)VIDIOC_QUERYCAP, (unsigned long)VIDIOC_G_FMT, (unsigned long)VIDIOC_S_FMT,
   (unsigned long)VIDIOC_S_PARM, V4L2_PIX_FMT_ABGR32, V4L2_PIX_FMT_YUYV);
 printf("%zu %zu %zu\n", offsetof(struct v4l2_pix_format, pixelformat), offsetof(struct v4l2_pix_format, sizeimage),
   offsetof(struct v4l2_pix_format, colorspace));
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc + kernel headers")
def test_abi_matches_kernel_headers(tmp_path):
    src = tmp_path / "abi.c"
    src.write_text(ABI_C)
    subprocess.run(["gcc", "-o", str(tmp_path / "abi"), str(src)], check=True)
    l1, l2 = subprocess.run([str(tmp_path / "abi")], capture_output=True, text=True).stdout.split("\n")[:2]
    got = [int(x) for x in l1.split()]
    assert got == [V.SIZEOF_FORMAT, 8, V.SIZEOF_STREAMPARM, 12, V.VIDIOC_QUERYCAP, V.VIDIOC_G_FMT,
                   V.VIDIOC_S_FMT, V.VIDIOC_S_PARM, V.FOURCC["ABGR32"], V.FOURCC["YUYV"]]
    assert [int(x) for x in l2.split()] == [8, 20, 24]   # field offsets used by pack_format


def test_pack_format_and_streamparm():
    b = V.pack_format(1920, 1080, "ABGR32")
    assert len(b) == 208
    t, = struct.unpack_from("<I", b, 0)
    w, h, four, field, bpl, size, cs = struct.unpack_from("<7I", b, 8)
    assert (t, w, h, four, field, bpl, size, cs) == (2, 1920, 1080, 0x34325241, 1, 7680, 7680 * 1080, 8)
    y = V.pack_format(1280, 720, "YUYV")
    assert struct.unpack_from("<I", y, 8 + 16)[0] == 2560
    p = V.pack_streamparm(120)
    assert struct.unpack_from("<I2I", p, 0)[0] == 2 and struct.unpack_from("<2I", p, 12) == (1, 120)


def test_yuyv_colour():
    import cv2
    bgra = np.zeros((4, 4, 4), np.uint8)
    bgra[..., 2] = 255; bgra[..., 3] = 255            # pure red
    yuy2 = cv2.cvtColor(bgra, cv2.COLOR_BGRA2YUV_YUY2)
    back = cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2)
    assert back[..., 2].min() > 230 and back[..., 0].max() < 30


def _read_frames(dev, n=3, timeout=10):
    out = subprocess.run(["v4l2-ctl", "-d", dev, "--stream-mmap", f"--stream-count={n}", "--stream-to=-"],
                         capture_output=True, timeout=timeout)
    return out.stdout


needs_dev = pytest.mark.skipif(V.find_device(V.LABEL_ALPHA) is None or V.find_device(V.LABEL_CHAT) is None,
                               reason="v4l2loopback devices not set up (sudo scripts/setup-virtualcam.sh)")


@needs_dev
@pytest.mark.parametrize("fmt", ["ABGR32", "YUYV"])
def test_live_roundtrip(fmt):
    import threading
    W, H = 640, 360
    cam = V.V4L2Camera(W, H, 60, fmt)
    frame = np.zeros((H, W, 4), np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 40, 160, 220     # B, G, R
    frame[:, : W // 2, 3] = 0                                        # left half transparent
    frame[:, W // 2:, 3] = 255                                       # right half opaque
    stop = False

    def pump():
        while not stop:
            cam.send_bgra(frame)
            time.sleep(1 / 60)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    time.sleep(0.3)
    try:
        data = _read_frames(cam.device)
    finally:
        stop = True
        t.join()
        cam.close()
    fb = W * H * V.BPP[fmt]
    assert len(data) >= fb
    f = np.frombuffer(data[-fb:], np.uint8)
    if fmt == "ABGR32":
        f = f.reshape(H, W, 4)
        assert (f[..., :3] == frame[..., :3]).all()
        assert (f[:, : W // 2, 3] == 0).all() and (f[:, W // 2:, 3] == 255).all()   # alpha survived
    else:
        import cv2
        bgr = cv2.cvtColor(f.reshape(H, W, 2), cv2.COLOR_YUV2BGR_YUY2)
        assert np.abs(bgr.astype(int) - frame[..., :3]).mean() < 6
