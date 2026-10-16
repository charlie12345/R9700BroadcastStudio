"""Every advertised format decodes through BackgroundProvider at the requested output size."""
import os, subprocess, time
import numpy as np
import pytest
import torch

from r9700_broadcast.backgrounds import BackgroundProvider, Library, fit_image, kind_of, thumbnail

SRC = "testsrc2=s=640x360:r=30:d=2"
IMAGES = ["png", "jpg", "webp", "bmp", "tiff", "avif"]
VIDEOS = {"mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "mov": ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
          "mkv": ["-c:v", "libx264"], "webm": ["-c:v", "libvpx-vp9", "-b:v", "1M"], "avi": ["-c:v", "mpeg4"],
          "gif": [], "m4v": ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "flv": ["-c:v", "flv"],
          "ts": ["-c:v", "mpeg2video"], "ogv": ["-c:v", "libtheora"], "wmv": ["-c:v", "wmv2"],
          "mpg": ["-c:v", "mpeg2video"]}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("bg")
    out = {}
    for ext in IMAGES:
        p = str(d / f"img.{ext}")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", SRC, "-frames:v", "1", p])
        if r.returncode == 0 and os.path.getsize(p) > 0:
            out[ext] = p
    for ext, args in VIDEOS.items():
        p = str(d / f"vid.{ext}")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", SRC, *args, p])
        if r.returncode == 0:
            out[ext] = p
    return out


@pytest.mark.parametrize("ext", IMAGES + list(VIDEOS))
def test_format_decodes(media, ext):
    if ext not in media:
        pytest.skip(f"this ffmpeg cannot encode .{ext} test media")
    p = media[ext]
    assert kind_of(p) == ("image" if ext in IMAGES else "video")
    prov = BackgroundProvider(p, "cover")
    try:
        t = None
        for _ in range(100):           # videos: wait for the decoder thread
            t = prov(270, 480, DEV)
            if t is not None:
                break
            time.sleep(0.05)
        assert t is not None and t.shape == (1, 3, 270, 480)
        assert 0.05 < t.float().mean().item() < 0.95       # real picture, not black
        if prov.kind == "video":
            s0 = prov._video.seq
            time.sleep(0.3)
            assert prov._video.seq > s0                    # it is playing / looping
    finally:
        prov.close()


def test_fit_modes():
    img = np.full((100, 400, 3), 200, np.uint8)            # very wide
    assert fit_image(img, 160, 90, "cover").shape == (90, 160, 3)
    c = fit_image(img, 160, 90, "contain")
    assert c.shape == (90, 160, 3) and c[0, 80].sum() == 0 and c[45, 80].sum() > 0  # letterboxed
    assert fit_image(img, 160, 90, "stretch").shape == (90, 160, 3)


def test_thumbnail_and_library(media, tmp_path, monkeypatch):
    import r9700_broadcast.backgrounds as B
    monkeypatch.setattr(B, "LIBRARY_JSON", str(tmp_path / "lib.json"))
    monkeypatch.setattr(B, "THUMB_DIR", str(tmp_path / "thumbs"))
    lib = Library()
    added = lib.add([media["mp4"], media["png"], media["mp4"], "/nonexistent.mp4"])
    assert added == [os.path.abspath(media["mp4"]), os.path.abspath(media["png"])]
    assert Library().user == added                          # persisted
    assert len(lib.builtin()) >= 8
    th = thumbnail(media["mp4"])
    assert th.shape == (108, 192, 3)
    lib.remove(added[0])
    assert Library().user == [added[1]]
