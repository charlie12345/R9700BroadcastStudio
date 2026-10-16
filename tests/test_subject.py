import os
import shutil
import subprocess
import time

import numpy as np
import pytest
import torch

from r9700_broadcast.coretrack import fuse_core, tracker_size
from r9700_broadcast.seeding import SubjectSeeder

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_tracker_size_multiple_of_16():
    assert tracker_size(1080, 1920, 384) == (384, 688)
    h, w = tracker_size(1920, 1080, 384)
    assert w == 384 and h % 16 == 0


def test_fuse_core_envelope():
    """Inside the tracker's subject the edge matte cannot go see-through; outside it cannot keep
    furniture; near the boundary the edge matte's own detail survives."""
    H, W = 1080, 1920
    core = torch.zeros(1, 1, 270, 480, device=DEV)
    core[..., 60:240, 140:340] = 1                      # subject block (x4 = 240..960 rows, 560..1360 cols)
    alpha = torch.zeros(1, 1, H, W, device=DEV)
    alpha[..., 300:900, 700:1200] = 0.3                 # "ghosted" headset inside the subject
    alpha[..., 100:200, 100:300] = 1.0                  # chair far outside the subject
    alpha[..., 238:242, 800:900] = 0.5                  # hair strand on the boundary
    out = fuse_core(alpha, core, band_px=8.0)
    assert out[..., 400:800, 800:1100].min() > 0.99     # ghost filled
    assert out[..., 100:200, 100:300].max() < 0.01      # chair gone
    assert 0.3 < out[..., 240, 850].item() < 0.7        # boundary detail kept


def test_best_boxes_nms_prefers_confident_tight_box():
    dets = [("headphones", 0.34, [460, 142, 830, 505]), ("headset", 0.34, [460, 141, 830, 511]),
            ("headphones", 0.37, [515, 157, 829, 507]), ("microphone", 0.4, [900, 600, 950, 700])]
    kept = SubjectSeeder._best_boxes(dets)
    assert kept[0] == [900, 600, 950, 700] or kept[0] == [515, 157, 829, 507]
    assert [515, 157, 829, 507] in kept and [460, 142, 830, 505] not in kept and len(kept) == 2


def test_thumbnail_render():
    pytest.importorskip("PyQt6")
    from PyQt6.QtGui import QColor, QGuiApplication, QImage
    app = QGuiApplication.instance() or QGuiApplication(["x", "-platform", "offscreen"])  # noqa: F841
    from r9700_broadcast.thumbnail import render_thumbnail
    img = QImage(1920, 1080, QImage.Format.Format_RGB32)
    img.fill(QColor("#336699"))
    t = render_thumbnail(img, "a very long title that must be shrunk to fit the thumbnail width", 25,
                         QColor("#fff"), QColor("#000"), "bottom", 0.5)
    assert (t.width(), t.height()) == (1280, 720)
    # text was drawn: some pixels are now white
    assert any(t.pixelColor(x, y).lightness() > 240 for x in range(0, 1280, 7) for y in range(560, 700, 7))


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("pactl") or not shutil.which("pw-record"),
                    reason="needs ffmpeg + PipeWire")
def test_recorder_has_sound(tmp_path):
    from r9700_broadcast.media import Recorder
    sink = "r9700_pytest_sink"
    mod = subprocess.run(["pactl", "load-module", "module-null-sink", f"sink_name={sink}"],
                         capture_output=True, text=True).stdout.strip()
    try:
        time.sleep(0.3)
        out = str(tmp_path / "rec.mp4")
        rec = Recorder(out, 320, 180, 30, "h264", audio=f"{sink}.monitor")
        f = np.zeros((180, 320, 4), np.uint8)
        t0 = time.perf_counter()
        worst = 0.0
        for i in range(60):
            t = time.perf_counter()
            rec.send_bgra(f)
            worst = max(worst, time.perf_counter() - t)
            time.sleep(max(0, t0 + (i + 1) / 30 - time.perf_counter()))
        rec.close()
    finally:
        subprocess.run(["pactl", "unload-module", mod])
    s = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0",
                        out], capture_output=True, text=True).stdout.split()
    kinds = {x.split(",")[0]: float(x.split(",")[1]) for x in s}
    assert "audio" in kinds and "video" in kinds
    assert abs(kinds["audio"] - kinds["video"]) < 0.25       # same length: in sync
    assert worst < 0.02                                       # never blocks the caller


def test_fuse_core_motion_keeps_fast_hand_but_not_static_chair():
    H, W = 1080, 1920
    core = torch.zeros(1, 1, 270, 480, device=DEV)
    core[..., 60:240, 140:340] = 1                       # lagged tracker: subject at cols 560..1360
    alpha = torch.zeros(1, 1, H, W, device=DEV)
    alpha[..., 500:600, 1380:1400] = 1.0                 # hand that moved 20-40 px past the lagged outline
    alpha[..., 500:600, 500:520] = 1.0                   # static chair edge 40-60 px outside, other side
    motion = torch.zeros(1, 1, 135, 240, device=DEV)
    motion[..., 55:85, 165:185] = 1                      # only around the hand
    out = fuse_core(alpha, core, motion, band_px=8.0)
    assert out[..., 520:580, 1385:1395].min() > 0.99     # hand kept
    assert out[..., 520:580, 500:515].max() < 0.01       # chair still removed
