"""Pipelined live engine: camera -> GPU -> outputs, built for 30/60/100/120 fps presets.

    capture thread   MJPEG decode (OpenCV), keeps only the newest frame
    GPU thread       upload -> matte -> composite -> BGRA uint8 on GPU -> async D2H into pinned ring
    output thread    waits on the copy event, then feeds PipeWire camera / v4l2loopback / recorder /
                     preview. The GPU thread never waits for outputs, so frame N's outputs overlap
                     frame N+1's GPU work.

All control changes go through `command()` and run on the GPU thread between frames.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from .effects import Settings
from .media import FrameSource


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    width: int
    height: int
    fps: int
    engine: str
    note: str


PRESETS = [
    Preset("30", "30 FPS · Studio quality", 1920, 1080, 30, "birefnet-lite",
           "BiRefNet-lite on both GPUs - sharpest hair, ~40 fps headroom"),
    Preset("60", "60 FPS · Realtime", 1920, 1080, 60, "rvm",
           "RVM with HIP graphs - the default for calls and streams"),
    Preset("100", "100 FPS · 1080p high-speed", 1920, 1080, 100, "rvm",
           "OBSBOT Tiny SE 1080p100 MJPEG mode"),
    Preset("120", "120 FPS · 720p high-speed", 1280, 720, 120, "rvm",
           "OBSBOT Tiny SE 720p120 MJPEG mode"),
]
PRESET_BY_KEY = {p.key: p for p in PRESETS}


class Stats:
    def __init__(self):
        self.fps_in = 0.0
        self.fps_out = 0.0
        self.gpu_ms = 0.0
        self.latency_ms = 0.0
        self.dropped = 0
        self._n_out = 0
        self._t_out = time.perf_counter()

    def tick_out(self, now):
        self._n_out += 1
        if now - self._t_out >= 1.0:
            self.fps_out = self._n_out / (now - self._t_out)
            self._n_out, self._t_out = 0, now


class LivePipeline:
    QUEUE = 2          # frames in flight GPU->outputs: bounds latency when a preset overloads
    RING = QUEUE + 2   # slots in the queue + one being output + one being written

    def __init__(self, studio, on_preview=None, on_status=None, on_error=None, preview_fps=30):
        self.studio = studio
        self.source: FrameSource | None = None
        self.outputs: dict[str, object] = {}     # name -> sink with .send_bgra / .write_bgra
        self.on_preview = on_preview             # callable(bgra np.ndarray) - called <= preview_fps
        self.on_status = on_status
        self.on_error = on_error
        self.preview_period = 1.0 / preview_fps
        self.snap_cb = None                      # one-shot callable(bgra, has_alpha) for Snapshot
        self.stats = Stats()
        self._cmds: queue.Queue = queue.Queue()
        self._outq: queue.Queue = queue.Queue(maxsize=self.QUEUE)
        self._ring = None
        self._ring_i = 0
        self._warm = 0
        self.source_error = None
        self._running = True
        self._gpu_t = threading.Thread(target=self._gpu_loop, name="r9700-gpu", daemon=True)
        self._out_t = threading.Thread(target=self._out_loop, name="r9700-out", daemon=True)
        self._gpu_t.start()
        self._out_t.start()

    # ---- control (any thread) ------------------------------------------------------------------
    def command(self, fn, *args, **kw):
        """Run fn(pipeline, *args) on the GPU thread before the next frame."""
        self._cmds.put((fn, args, kw))

    def set_source(self, uri, width, height, fps):
        def do(p):
            if p.source:
                p.source.close()
                p.source = None
            p.source_error = None
            if uri:
                try:
                    p.source = FrameSource(uri, width, height, fps)
                except Exception as e:  # noqa: BLE001 - camera busy / unplugged / bad file
                    p.source_error = f"{e} (is another app using the camera?)"
                    if p.on_error:
                        p.on_error(p.source_error)
                    return
                if p.studio:
                    p.studio.engine.reset()
                    p.studio.compositor.framer.reset()
        self.command(do)

    def add_output(self, name, sink):
        self.command(lambda p: p._swap_output(name, sink))

    def remove_output(self, name):
        self.command(lambda p: p._swap_output(name, None))

    def _swap_output(self, name, sink):
        old = self.outputs.pop(name, None)
        if sink is not None:
            self.outputs[name] = sink
        if old is not None:
            # let the output thread finish frames already queued for it
            self._outq.put(("close", old))

    def stop(self):
        self._running = False
        self._gpu_t.join(timeout=5)
        self._outq.put(None)
        self._out_t.join(timeout=5)
        for o in list(self.outputs.values()):
            try:
                o.close()
            except Exception:  # noqa: BLE001
                pass
        if self.source:
            self.source.close()
        if self.studio is not None:
            self.studio.close()

    # ---- GPU thread ----------------------------------------------------------------------------
    def _ensure_ring(self, h, w):
        if self._ring is None or self._ring[0][0].shape != (h, w, 4):
            self._ring = [(torch.empty((h, w, 4), dtype=torch.uint8, pin_memory=True),
                           torch.cuda.Event()) for _ in range(self.RING)]

    def _gpu_loop(self):
        seq = -1
        last_status = 0.0
        frames_in = 0
        t_in = time.perf_counter()
        while self._running:
            try:
                while not self._cmds.empty():
                    fn, args, kw = self._cmds.get_nowait()
                    fn(self, *args, **kw)
                if self.source is None or self.studio is None:
                    time.sleep(0.05)
                    continue
                seq2, frame = self.source.read(seq, timeout=0.25)
                if frame is None:
                    continue
                if seq != -1 and seq2 > seq + 1 and self._warm > 30:
                    self.stats.dropped += seq2 - seq - 1
                self._warm += 1
                seq = seq2
                t0 = time.perf_counter()
                st = self.studio
                with torch.cuda.device(st.device):
                    src = st.upload(frame)
                    out, alpha, _fgr, _used = st.process_tensor(src)
                    a = st.compositor.out_alpha
                    bgra = self._to_bgra(out, a)
                    h, w = bgra.shape[:2]
                    self._ensure_ring(h, w)
                    host, ev = self._ring[self._ring_i]
                    self._ring_i = (self._ring_i + 1) % self.RING
                    host.copy_(bgra, non_blocking=True)
                    ev.record()
                self._outq.put(("frame", host, ev, t0, a is not None))
                frames_in += 1
                now = time.perf_counter()
                if now - t_in >= 1.0:
                    self.stats.fps_in = frames_in / (now - t_in)
                    frames_in, t_in = 0, now
                if self.on_status and now - last_status > 0.5:
                    last_status = now
                    self.on_status(self)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                if self.on_error:
                    self.on_error(f"{type(e).__name__}: {e}")
                time.sleep(0.5)

    @staticmethod
    def _to_bgra(out: torch.Tensor, alpha: torch.Tensor | None) -> torch.Tensor:
        rgb = (out[0].float() * 255 + 0.5).clamp_(0, 255).to(torch.uint8)       # 3,H,W
        if alpha is not None:
            a = (alpha[0].float() * 255 + 0.5).clamp_(0, 255).to(torch.uint8)  # 1,H,W
        else:
            a = torch.full_like(rgb[:1], 255)
        return torch.cat([rgb[2:3], rgb[1:2], rgb[0:1], a], 0).permute(1, 2, 0).contiguous()

    # ---- output thread -------------------------------------------------------------------------
    def _out_loop(self):
        last_preview = 0.0
        while True:
            item = self._outq.get()
            if item is None:
                return
            if item[0] == "close":
                try:
                    item[1].close()
                except Exception:  # noqa: BLE001
                    pass
                continue
            _, host, ev, t0, has_alpha = item
            ev.synchronize()
            bgra = host.numpy()
            now = time.perf_counter()
            self.stats.gpu_ms = 0.8 * self.stats.gpu_ms + 0.2 * (now - t0) * 1000 if self.stats.gpu_ms else (now - t0) * 1000
            for name, sink in list(self.outputs.items()):
                try:
                    sink.send_bgra(bgra)
                except Exception as e:  # noqa: BLE001
                    if self.on_error:
                        self.on_error(f"{name}: {e}")
            self.stats.latency_ms = (time.perf_counter() - t0) * 1000
            self.stats.tick_out(time.perf_counter())
            snap, self.snap_cb = self.snap_cb, None
            if snap is not None:
                try:
                    snap(bgra.copy(), has_alpha)     # full-resolution output frame for Snapshot
                except Exception as e:  # noqa: BLE001
                    if self.on_error:
                        self.on_error(f"snapshot: {e}")
            if self.on_preview and now - last_preview >= self.preview_period:
                last_preview = now
                self.on_preview(bgra.copy(), has_alpha)
