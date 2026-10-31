"""Vertical (9:16) recording: crop geometry, Longform_/Vertical_ file naming, real encodes."""
import glob
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from r9700_broadcast import vertical as VT


@pytest.mark.parametrize("W,H", [(1920, 1080), (1280, 720), (3840, 2160), (640, 360)])
def test_crop_is_even_9x16_full_height(W, H):
    w, h = VT.VerticalFrame().size(W, H)
    assert h == H - (H % 2) and w % 2 == 0 and h % 2 == 0
    assert abs(w / h - 9 / 16) < 0.01


def test_zoom_and_clamp():
    vf = VT.VerticalFrame(x=0.0, y=0.0, zoom=2.0)
    x0, y0, w, h = vf.rect(1920, 1080)
    assert (x0, y0) == (0, 0) and h == 540
    vf.x, vf.y = 1.0, 1.0
    x0, y0, w, h = vf.rect(1920, 1080)
    assert x0 + w == 1920 and y0 + h == 1080
    vf.clamp(1920, 1080)
    assert vf.rect(1920, 1080) == (x0, y0, w, h)


def test_locked_size_survives_zoom_change():
    vf = VT.VerticalFrame()
    locked = vf.size(1920, 1080)
    vf.zoom = 2.0
    assert vf.rect(1920, 1080, locked)[2:] == locked
    frame = np.zeros((1080, 1920, 4), np.uint8)
    assert vf.crop(frame, locked).shape[:2] == (locked[1], locked[0])


def test_crop_follows_position():
    frame = np.zeros((720, 1280, 4), np.uint8)
    frame[:, :640] = 255                      # left half white
    vf = VT.VerticalFrame(x=0.1)
    assert vf.crop(frame).min() == 255
    vf.x = 0.9
    assert vf.crop(frame).max() == 0


def test_recording_paths(tmp_path):
    d = str(tmp_path)
    p = VT.recording_paths(d, "both", "mp4", "My Take", stamp="x")
    assert sorted(os.path.basename(v) for v in p.values()) == ["Longform_My Take.mp4", "Vertical_My Take.mp4"]
    assert list(VT.recording_paths(d, "vertical", "mp4", "", stamp="2026-10-31 10-00-00").values()) == \
        [os.path.join(d, "Vertical_2026-10-31 10-00-00.mp4")]
    assert list(VT.recording_paths(d, "long", "mov", "a/b:c").values()) == [os.path.join(d, "Longform_a_b_c.mov")]
    # an existing file of either kind bumps BOTH so a take's pair keeps matching names
    open(os.path.join(d, "Vertical_My Take.mp4"), "w").close()
    p = VT.recording_paths(d, "both", "mp4", "My Take")
    assert sorted(os.path.basename(v) for v in p.values()) == \
        ["Longform_My Take (2).mp4", "Vertical_My Take (2).mp4"]


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], int(s["nb_read_frames"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("codec", ["x264", "hevc"])
@pytest.mark.parametrize("size_key,expect", [("1080x1920", (1080, 1920)), ("native", (404, 720))])
def test_vertical_encode(tmp_path, codec, size_key, expect):
    if codec == "hevc" and not glob.glob("/dev/dri/renderD*"):
        pytest.skip("no VAAPI device")
    from r9700_broadcast.media import Recorder
    W, H = 1280, 720
    vf = VT.VerticalFrame(x=0.3)
    crop = vf.size(W, H)
    assert crop == (404, 720)
    path = str(tmp_path / "v.mp4")
    rec = Recorder(path, W, H, 30, codec, crop=vf, out_size=VT.output_size(size_key, crop))
    frame = np.zeros((H, W, 4), np.uint8)
    import time
    for i in range(30):
        frame[..., 1] = i * 8
        vf.x = 0.3 + i * 0.01                 # panning while recording must not break the stream
        rec.send_bgra(frame)
        time.sleep(1 / 30)
    rec.close()
    w, h, n = _probe(path)
    assert (w, h) == expect and n >= 20
