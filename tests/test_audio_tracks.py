"""Mic + desktop audio as two separate tracks, and the live level meter.

Uses a private PipeWire null sink: test tones are inaudible and nothing real is recorded."""
import json
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest

from r9700_broadcast.media import Recorder, pw_target_args, resolve_desktop

HAVE_PW = all(shutil.which(t) for t in ("pactl", "pw-record", "pw-cat", "ffmpeg"))
SINK = "r9700_tracks_test"


def test_pw_target_args():
    assert pw_target_args("@default") == []
    assert pw_target_args("alsa_input.x") == ["--target", "alsa_input.x"]
    assert pw_target_args("alsa_output.y.monitor") == ["--target", "alsa_output.y", "-P",
                                                       "{ stream.capture.sink = true }"]
    assert resolve_desktop("none") is None
    assert resolve_desktop("alsa_output.y") == "alsa_output.y.monitor"


@pytest.fixture
def null_sink():
    if not HAVE_PW:
        pytest.skip("needs PipeWire tools + ffmpeg")
    r = subprocess.run(["pactl", "load-module", "module-null-sink", f"sink_name={SINK}",
                        "sink_properties=device.description=r9700-tracks-test"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip("cannot create a null sink")
    time.sleep(0.5)
    tone_stop = threading.Event()
    player = subprocess.Popen(["pw-cat", "--playback", "--target", SINK, "--raw", "--rate", "48000",
                               "--channels", "1", "--format", "s16", "--latency", "20ms", "-"],
                              stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def tone():
        t = np.arange(960) / 48000
        block = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()   # 20 ms, -6 dBFS
        try:
            while not tone_stop.is_set():
                player.stdin.write(block)
                player.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
    th = threading.Thread(target=tone, daemon=True)
    th.start()
    yield SINK
    tone_stop.set()
    player.terminate()
    player.wait(timeout=3)
    subprocess.run(["pactl", "unload-module", r.stdout.strip()])


def test_two_tracks_with_titles(tmp_path, null_sink):
    path = str(tmp_path / "t.mp4")
    rec = Recorder(path, 320, 180, 30, "x264", audio=f"{null_sink}.monitor", desktop=f"{null_sink}.monitor")
    f = np.zeros((180, 320, 4), np.uint8)
    for _ in range(60):
        rec.send_bgra(f)
        time.sleep(1 / 30)
    rec.close()
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
                                     capture_output=True, text=True, check=True).stdout)
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert [s["channels"] for s in a] == [1, 2]
    assert [s["tags"].get("title") or s["tags"].get("handler_name") for s in a] == ["Microphone", "Desktop audio"]
    # both tracks carry the tone (not silence)
    for i in range(2):
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:a:{i}", "-f", "s16le", "-ac", "1",
                              "-"], capture_output=True, check=True).stdout
        assert np.abs(np.frombuffer(pcm, np.int16)).max() > 4000


def test_missing_device_still_closes(tmp_path):
    if not HAVE_PW:
        pytest.skip("needs PipeWire tools + ffmpeg")
    path = str(tmp_path / "m.mp4")
    rec = Recorder(path, 320, 180, 30, "x264", audio="no_such_device_xyz", desktop="no_such_sink.monitor")
    f = np.zeros((180, 320, 4), np.uint8)
    for _ in range(30):
        rec.send_bgra(f)
        time.sleep(1 / 30)
    t0 = time.time()
    rec.close()
    assert time.time() - t0 < 15
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of",
                          "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
    kinds = [o.split(",")[0] for o in out]
    assert kinds.count("video") == 1 and kinds.count("audio") == 2
    # never falls back to the default mic: the tracks are pure silence, as long as the video
    for i in range(2):
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:a:{i}", "-f", "s16le", "-ac", "1",
                              "-"], capture_output=True, check=True).stdout
        a = np.frombuffer(pcm, np.int16)
        assert a.size > 48000 * 0.7 and np.abs(a).max() == 0


def test_target_exists(null_sink):
    from r9700_broadcast.media import target_exists
    assert target_exists(f"{null_sink}.monitor")
    assert not target_exists("no_such_device_xyz")
    assert not target_exists("no_such_sink.monitor")
    assert target_exists("@default")


def test_level_meter_missing_device_is_silent():
    if not HAVE_PW:
        pytest.skip("needs PipeWire tools")
    from r9700_broadcast.audio import LevelMeter
    m = LevelMeter("no_such_device_xyz")
    time.sleep(0.5)
    assert m.missing and m.alive and m.peak == 0.0
    m.stop()


def test_level_meter(null_sink):
    from r9700_broadcast.audio import LevelMeter
    m = LevelMeter(f"{null_sink}.monitor")
    try:
        deadline = time.time() + 5
        while time.time() < deadline and m.peak < 0.1:
            time.sleep(0.05)
        # -6 dBFS tone; PipeWire's mono->stereo->mono mixing on the test sink lowers it ~10 dB
        assert 0.1 < m.peak < 0.7
    finally:
        m.stop()
    assert not m.alive
