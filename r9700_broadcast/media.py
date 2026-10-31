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


def pw_target_args(target: str) -> list[str]:
    """pw-record arguments for a capture target: '@default' (default mic), '<sink>.monitor' (what a
    speaker/headset plays - desktop audio), or a source node name (a specific microphone)."""
    if target == "@default":
        return []
    # dont-fallback: a missing device must give silence, never quietly record the default mic instead
    if target.endswith(".monitor"):
        return ["--target", target[:-8], "-P", "{ stream.capture.sink = true node.dont-fallback = true }"]
    return ["--target", target, "-P", "{ node.dont-fallback = true }"]


def default_sink() -> str | None:
    try:
        name = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True,
                              timeout=3).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return name or None


def list_outputs() -> list[tuple[str, str]]:
    """[(sink node.name, description)] of speakers/headsets whose sound can be recorded."""
    out, name = [], None
    try:
        txt = subprocess.run(["pactl", "list", "sinks"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return out
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Description:") and name:
            out.append((name, line.split(":", 1)[1].strip()))
            name = None
    return out


def resolve_desktop(choice: str | None) -> str | None:
    """'none' | 'default' | a sink node name -> '<sink>.monitor' capture target (or None)."""
    if not choice or choice == "none" or shutil.which("pw-record") is None:
        return None
    sink = default_sink() if choice == "default" else choice
    return f"{sink}.monitor" if sink else None


class _AudioTrack:
    """One pw-record capture fed to ffmpeg through its own pipe."""

    def __init__(self, target: str, channels: int, title: str):
        self.target, self.channels, self.title = target, channels, title
        self.fd_r, self.fd_w = os.pipe()
        self.proc = None
        self.thread = None


class Recorder:
    """ffmpeg recorder with sound. 'h264'/'hevc' use the R9700 VCN (VAAPI); 'prores4444' keeps the
    alpha channel for editing in Resolve/Kdenlive (feed it RGBA).

    Sound: track 1 = microphone (mono), optional track 2 = desktop audio (stereo, what a speaker /
    headset plays), kept separate so they can be balanced in the editor.

    A/V sync: each audio source is captured with pw-record and fed to ffmpeg through its own pipe,
    starting at the instant the first video frame is written, so all streams start at t=0
    together. Video frames are timestamped by the wall clock as they arrive and ffmpeg emits
    constant frame rate (duplicating/dropping), so a dropped or late frame never shifts the sound.
    """

    RATE = 48000

    def __init__(self, path, width, height, fps, codec="hevc", alpha=False, audio: str | None = None,
                 audio_offset_ms: float = 0.0, crop=None, out_size: tuple[int, int] | None = None,
                 desktop: str | None = None):
        """audio: mic capture target (see pw_target_args). desktop: '<sink>.monitor' for a second,
        stereo track of desktop sound. crop: a vertical.VerticalFrame - records only that window of
        each frame (its position can move while recording; its size is fixed here). out_size: (w, h)
        to scale to, e.g. 1080x1920."""
        self.alpha = alpha and codec == "prores4444"
        self.path = path
        self.size, self.fps = (width, height), fps
        self.crop = crop
        self.crop_size = crop.size(width, height) if crop is not None else None
        if self.crop_size:
            width, height = self.crop_size
        scale = out_size if out_size and tuple(out_size) != (width, height) else None
        self.audio_target = audio
        self.desktop_target = desktop
        self.audio_offset_ms = audio_offset_ms
        self.tracks: list[_AudioTrack] = []
        if audio:
            self.tracks.append(_AudioTrack(audio, 1, "Microphone"))
        if desktop:
            self.tracks.append(_AudioTrack(desktop, 2, "Desktop audio"))
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
        for t in self.tracks:
            cmd += ["-thread_queue_size", "1024", *raw, "-f", "s16le", "-ar", str(self.RATE),
                    "-ac", str(t.channels), "-i", f"pipe:{t.fd_r}"]
        cmd += ["-map", "0:v"]
        for i, _ in enumerate(self.tracks):
            cmd += ["-map", f"{i + 1}:a"]
        cmd += ["-fps_mode", "cfr", "-r", str(fps)]
        if vaapi:
            sv = f",scale_vaapi=w={scale[0]}:h={scale[1]}" if scale else ""
            cmd += ["-vf", f"format=nv12,hwupload{sv}", "-c:v", f"{codec}_vaapi", "-qp", "18"]
        elif scale:
            cmd += ["-vf", f"scale={scale[0]}:{scale[1]}:flags=lanczos"]
        if vaapi:
            pass                                 # encoder chosen above
        elif codec == "prores4444":
            cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt",
                    "yuva444p10le" if self.alpha else "yuv444p10le"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
        for i, t in enumerate(self.tracks):
            if codec == "prores4444":
                cmd += [f"-c:a:{i}", "pcm_s16le"]
            else:
                cmd += [f"-c:a:{i}", "aac", f"-b:a:{i}", "192k" if t.channels == 1 else "256k"]
            # MP4 keeps a track's name in handler_name, MOV/MKV in title: set both
            cmd += [f"-metadata:s:a:{i}", f"title={t.title}", f"-metadata:s:a:{i}", f"handler_name={t.title}"]
        if self.tracks:
            cmd += ["-disposition:a:0", "default"]
        cmd.append(path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, pass_fds=tuple(t.fd_r for t in self.tracks))
        for t in self.tracks:
            os.close(t.fd_r)
        self._started = threading.Event()
        self._closing = False
        # frames go through a writer thread: a slow disk or encoder never stalls the virtual camera
        import queue as _q
        self._vq = _q.Queue(maxsize=8)
        self.dropped = 0
        self._vt = threading.Thread(target=self._video_loop, daemon=True)
        self._vt.start()
        for t in self.tracks:
            t.proc = subprocess.Popen(["pw-record", "--raw", "--rate", str(self.RATE), "--channels",
                                       str(t.channels), "--format", "s16", "--latency", "20ms",
                                       *pw_target_args(t.target), "-"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            t.thread = threading.Thread(target=self._audio_loop, args=(t,), daemon=True)
            t.thread.start()

    def _audio_loop(self, t: _AudioTrack):
        """Copy pw-record -> ffmpeg on the wall clock. If the device sends nothing (missing, unplugged,
        suspended) the gap is filled with silence, so the track keeps time with the video and ffmpeg's
        muxer never stalls waiting for it."""
        import select
        frame = 2 * t.channels
        chunk = self.RATE // 50 * frame          # 20 ms of s16
        fd = t.proc.stdout.fileno()
        bps = self.RATE * frame                  # bytes per second
        try:
            # audio from before the first video frame is discarded
            while not self._started.is_set() and not self._closing:
                if select.select([fd], [], [], 0.05)[0] and not os.read(fd, 65536):
                    fd = None
                    break
            t0 = time.perf_counter()
            # audio_offset_ms > 0 delays the sound (silence first), < 0 trims it
            off = int(self.audio_offset_ms * self.RATE / 1000) * frame
            written = 0
            if off > 0:
                os.write(t.fd_w, bytes(off))
                written = off
            trim = -off if off < 0 else 0
            carry = b""
            while not self._closing:
                b = b""
                if fd is not None and select.select([fd], [], [], 0.05)[0]:
                    b = os.read(fd, chunk * 4)
                    if not b:
                        fd = None                # pw-record ended
                elif fd is None:
                    time.sleep(0.02)
                if b:
                    if trim:
                        k = min(trim, len(b))
                        trim -= k
                        b = b[k:]
                    b = carry + b
                    n = len(b) - len(b) % frame  # whole sample frames only
                    carry = b[n:]
                    if n:
                        os.write(t.fd_w, b[:n])
                        written += n
                    continue
                # nothing arrived: pad to the wall clock once the gap is clearly a dropout
                due = int((time.perf_counter() - t0) * bps) + max(off, 0) - trim
                gap = due - written
                if gap > bps * 0.25 and not carry:
                    pad = gap - int(bps * 0.05)
                    pad -= pad % frame
                    os.write(t.fd_w, bytes(pad))
                    written += pad
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
        if self.crop is not None:
            bgra = self.crop.crop(bgra, self.crop_size)     # copies only the crop window below
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
        for t in self.tracks:
            t.proc.terminate()
            try:
                t.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                t.proc.kill()
        for t in self.tracks:
            t.thread.join(timeout=3)
            if t.fd_w is not None:
                try:
                    os.close(t.fd_w)
                except OSError:
                    pass
                t.fd_w = None
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.proc.wait(timeout=60)


def have_ffmpeg():
    return shutil.which("ffmpeg") is not None
