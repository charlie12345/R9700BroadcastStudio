"""Sources (camera / file) and sinks (virtual camera / recorder)."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------

def list_cameras() -> list[tuple[str, str]]:
    """[(path, label)] for real capture devices; skips our own loopback outputs."""
    out = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[10:])):
        name_f = f"/sys/class/video4linux/{os.path.basename(dev)}/name"
        name = open(name_f).read().strip() if os.path.exists(name_f) else dev
        if "R9700 Broadcast" in name or "loopback" in name.lower():
            continue
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            out.append((dev, f"{name} ({dev})"))
    return out


class FrameSource:
    """Background reader that always exposes the newest frame (drops stale ones = low latency)."""

    def __init__(self, uri: str, width=1920, height=1080, fps=60, loop=True):
        self.uri, self.loop = uri, loop
        self.is_file = os.path.isfile(uri)
        if self.is_file:
            self.cap = cv2.VideoCapture(uri)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        else:
            self.cap = cv2.VideoCapture(uri, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            # a few driver buffers absorb GIL/GPU jitter at 100-120 fps; the reader thread always
            # drains them immediately, so this adds no steady-state latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {uri}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame = None
        self._seq = 0
        self._cv = threading.Condition()
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self._cv:
                self._frame = f
                self._seq += 1
                self._cv.notify_all()
            if self.is_file:  # pace files at their native rate
                nxt += period
                time.sleep(max(0.0, nxt - time.perf_counter()))
                if time.perf_counter() - nxt > 0.5:
                    nxt = time.perf_counter()

    def read(self, last_seq=-1, timeout=1.0):
        """Block until a frame newer than last_seq exists. Returns (seq, frame) or (last_seq, None)."""
        with self._cv:
            self._cv.wait_for(lambda: self._seq != last_seq or self._stop, timeout)
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._frame

    def close(self):
        self._stop = True
        self._t.join(timeout=2)
        self.cap.release()


# ---------------------------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------------------------

def find_loopback(label: str = "R9700 Broadcast Camera") -> str | None:
    from .v4l2out import find_device
    return find_device(label)


class VirtualCamera:
    """Chat-app virtual camera: /dev/video42 'R9700 Broadcast Camera', YUYV (Zoom/Discord/Teams)."""

    def __new__(cls, width, height, fps, device=None, fmt="YUYV"):
        from .v4l2out import V4L2Camera
        return V4L2Camera(width, height, fps, fmt=fmt, device=device)


STUDIO_MIC_NODE = "r9700_broadcast_mic"


def studio_mic_running() -> bool:
    try:
        out = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return False
    return f'node.name = "{STUDIO_MIC_NODE}"' in out


def resolve_audio(choice: str | None) -> str | None:
    """'auto' | 'studio' | 'default' | 'none' | a PipeWire node name -> pw-record target (or None
    for no audio). 'auto' = the AI noise-removed mic when it is running, else the default mic."""
    if not choice or choice == "none" or shutil.which("pw-record") is None:
        return None
    if choice == "auto":
        return STUDIO_MIC_NODE if studio_mic_running() else "@default"
    if choice == "studio":
        return STUDIO_MIC_NODE
    if choice == "default":
        return "@default"
    return choice


class Recorder:
    """ffmpeg recorder with sound. 'h264'/'hevc' use the R9700 VCN (VAAPI); 'prores4444' keeps the
    alpha channel for editing in Resolve/Kdenlive (feed it RGBA).

    A/V sync: the microphone is captured with pw-record and fed to ffmpeg through its own pipe,
    starting at the instant the first video frame is written, so both streams start at t=0
    together. Video frames are timestamped by the wall clock as they arrive and ffmpeg emits
    constant frame rate (duplicating/dropping), so a dropped or late frame never shifts the sound.
    """

    RATE = 48000

    def __init__(self, path, width, height, fps, codec="hevc", alpha=False, audio: str | None = None,
                 audio_offset_ms: float = 0.0):
        self.alpha = alpha and codec == "prores4444"
        self.path = path
        self.audio_target = audio
        self.audio_offset_ms = audio_offset_ms
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        render = sorted(glob.glob("/dev/dri/renderD*"))
        vaapi = codec in ("hevc", "h264") and render
        if vaapi:
            cmd += ["-vaapi_device", render[0]]
        # raw inputs need no probing; ffmpeg's default 5 s analysis of the audio pipe would block the
        # first video frames for seconds
        raw = ["-probesize", "32", "-analyzeduration", "0"]
        cmd += ["-thread_queue_size", "64", *raw, "-use_wallclock_as_timestamps", "1",
                "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{width}x{height}", "-i", "pipe:0"]
        self._afd_r = self._afd_w = None
        if audio:
            self._afd_r, self._afd_w = os.pipe()
            cmd += ["-thread_queue_size", "1024", *raw, "-f", "s16le", "-ar", str(self.RATE), "-ac", "1",
                    "-i", f"pipe:{self._afd_r}"]
        cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio else [])
        cmd += ["-fps_mode", "cfr", "-r", str(fps)]
        if vaapi:
            cmd += ["-vf", "format=nv12,hwupload", "-c:v", f"{codec}_vaapi", "-qp", "18"]
        elif codec == "prores4444":
            cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt",
                    "yuva444p10le" if self.alpha else "yuv444p10le"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
        if audio:
            cmd += ["-c:a", "pcm_s16le"] if codec == "prores4444" else ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     pass_fds=(self._afd_r,) if audio else ())
        if audio:
            os.close(self._afd_r)
        self._started = threading.Event()
        self._closing = False
        # frames go through a writer thread: a slow disk or encoder never stalls the virtual camera
        import queue as _q
        self._vq = _q.Queue(maxsize=8)
        self.dropped = 0
        self._vt = threading.Thread(target=self._video_loop, daemon=True)
        self._vt.start()
        self._mic = None
        self._at = None
        if audio:
            if audio == "@default":
                tgt = []
            elif audio.endswith(".monitor"):        # record what a sink plays (e.g. a test sink)
                tgt = ["--target", audio[:-8], "-P", "{ stream.capture.sink = true }"]
            else:
                tgt = ["--target", audio]
            self._mic = subprocess.Popen(["pw-record", "--raw", "--rate", str(self.RATE), "--channels", "1",
                                          "--format", "s16", "--latency", "20ms", *tgt, "-"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._at = threading.Thread(target=self._audio_loop, daemon=True)
            self._at.start()

    def _audio_loop(self):
        chunk = self.RATE // 50 * 2          # 20 ms of s16 mono
        src = self._mic.stdout
        try:
            # mic audio from before the first video frame is discarded
            while not self._started.is_set() and not self._closing:
                if not src.read(chunk):
                    return
            # audio_offset_ms > 0 delays the sound (silence first), < 0 trims it
            off = int(self.audio_offset_ms * self.RATE / 1000) * 2
            if off > 0:
                os.write(self._afd_w, bytes(off))
            elif off < 0:
                src.read(-off)
            while not self._closing:
                b = src.read(chunk)
                if not b:
                    break
                os.write(self._afd_w, b)
        except (BrokenPipeError, OSError, ValueError):
            pass

    def _video_loop(self):
        while True:
            f = self._vq.get()
            if f is None:
                return
            try:
                self.proc.stdin.write(memoryview(f).cast("B"))
                if not self._started.is_set():
                    self._started.set()          # audio starts with the first frame ffmpeg receives
            except (BrokenPipeError, ValueError, OSError):
                return

    def send_bgra(self, bgra: np.ndarray):
        try:
            self._vq.put_nowait(np.array(bgra, copy=True, order="C"))  # caller reuses its buffer
        except Exception:  # noqa: BLE001 - queue full: encoder behind, drop (ffmpeg duplicates)
            self.dropped += 1

    def write(self, rgb: np.ndarray, alpha: np.ndarray | None = None):
        a = alpha if alpha is not None else np.full(rgb.shape[:2], 255, np.uint8)
        self.send_bgra(np.dstack([rgb[..., ::-1], a]))

    def close(self):
        self._vq.put(None)
        self._vt.join(timeout=30)
        self._closing = True
        if self._mic is not None:
            self._mic.terminate()
            try:
                self._mic.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._mic.kill()
        if self._at is not None:
            self._at.join(timeout=3)
        if self._afd_w is not None:
            try:
                os.close(self._afd_w)
            except OSError:
                pass
            self._afd_w = None
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.proc.wait(timeout=60)


def have_ffmpeg():
    return shutil.which("ffmpeg") is not None
