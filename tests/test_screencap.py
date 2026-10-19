"""Screen share background: capture commands, the frame reader and the GPU provider.
The portal handshake itself needs a person to click the desktop's picker; the PipeWire reader is
tested against a gst test source, and the X11 grabber against Xvfb, when those tools exist."""
import os
import shutil
import subprocess
import time

import numpy as np
import pytest
import torch

from r9700_broadcast import screencap as sc

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_commands():
    g = sc.gst_command(42, fd=7, w=640, h=360)
    assert g[:2] == ["gst-launch-1.0", "-q"] and "fd=7" in g and "path=42" in g
    assert "video/x-raw,format=BGRx,width=640,height=360,pixel-aspect-ratio=1/1" in g
    f = sc.ffmpeg_command("x11", (1920, 0, 2560, 1440), 30, 640, 360)
    assert "x11grab" in f and f[f.index("-i") + 1].endswith("+1920,0") and "2560x1440" in f
    w = sc.ffmpeg_command("gdi", (0, 0, 1920, 1080), 30)
    assert "gdigrab" in w and "desktop" in w


def test_provider_letterboxes_to_output_size():
    class Cap:
        seq, frame, error = 1, np.zeros((360, 640, 4), np.uint8), None

        def close(self):
            pass
    Cap.frame[..., 2] = 255                    # BGRx red
    prov = sc.ScreenBackground(sc.ScreenShare(Cap()))
    t = prov(270, 480, torch.device(DEV))
    assert t.shape == (1, 3, 270, 480)
    assert t[0, 0].mean() > 0.95 and t[0, 2].mean() < 0.05     # RGB order
    assert prov(270, 480, torch.device(DEV)) is t               # cached until a new frame


def _ffmpeg_has(dev):
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-devices"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False
    return dev in out


@pytest.mark.skipif(not (shutil.which("Xvfb") and _ffmpeg_has("x11grab")), reason="needs Xvfb + ffmpeg x11grab")
def test_x11_grabber_on_xvfb():
    xv = subprocess.Popen(["Xvfb", ":97", "-screen", "0", "1280x720x24"], stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        env_disp = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = ":97"
        try:
            cap = sc.ScreenCapture(sc.ffmpeg_command("x11", (0, 0, 1280, 720), 15, 640, 360), w=640, h=360)
        finally:
            if env_disp is None:
                os.environ.pop("DISPLAY")
            else:
                os.environ["DISPLAY"] = env_disp
        assert cap.wait_first_frame(8), cap.error
        assert cap.frame.shape == (360, 640, 4)
        cap.close()
    finally:
        xv.terminate()
        xv.wait()


@pytest.mark.skipif(not (shutil.which("gst-launch-1.0") and shutil.which("pw-cli")),
                    reason="needs GStreamer + PipeWire")
def test_pipewire_reader_against_test_source():
    name = f"r9700_test_{os.getpid()}"
    src = subprocess.Popen(["gst-launch-1.0", "-q", "videotestsrc", "is-live=true", "!",
                            "video/x-raw,width=320,height=240,framerate=30/1", "!", "pipewiresink",
                            f"stream-properties=props,media.class=Video/Source,node.name={name}",
                            "mode=provide"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        node = None
        for _ in range(30):
            out = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True).stdout
            blocks = out.split("\tid ")
            node = next((b.split(",")[0] for b in blocks if f'node.name = "{name}"' in b), None)
            if node:
                break
            time.sleep(0.1)
        if node is None:
            pytest.skip("PipeWire not running")
        cap = sc.ScreenCapture(sc.gst_command(node, w=640, h=360), w=640, h=360)
        assert cap.wait_first_frame(8), cap.error
        n = cap.seq
        time.sleep(0.5)
        assert cap.seq > n                      # keeps streaming
        cap.close()
    finally:
        src.terminate()
        src.wait()
