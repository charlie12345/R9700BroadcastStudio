"""Command line: process a video file offline, or run the camera headless into the virtual cam.

  r9700-broadcast file in.mp4 out.mov --engine birefnet --mode alpha       # ProRes 4444 + alpha
  r9700-broadcast file in.mp4 out.mp4 --mode green                          # virtual green screen
  r9700-broadcast live --preset 120 --mode transparent --output pipewire   # headless, alpha into OBS
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import cv2
import numpy as np

from .effects import MODES, Settings
from .matting import ENGINES


def _devices(arg):
    from .studio import visible_devices
    return [d.strip() for d in arg.split(",")] if arg else visible_devices()


def cmd_file(a):
    from .media import Recorder
    from .studio import Studio
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alpha_out = a.mode == "alpha"
    st = Studio(a.engine, _devices(a.gpus), Settings(mode="remove" if alpha_out else a.mode))
    if a.background:
        st.set_background_image(cv2.cvtColor(cv2.imread(a.background), cv2.COLOR_BGR2RGB))
    codec = "prores4444" if alpha_out or a.output.endswith(".mov") else a.codec
    rec = Recorder(a.output, w, h, fps, codec=codec, alpha=alpha_out)
    t0 = time.time()
    i = 0
    # pipelined multi-GPU engines lag one frame; feed the last frame again at the end to flush
    pending_flush = getattr(st.engine, "n", 1) - 1
    frame = None
    last_used = None
    while True:
        ok, f = cap.read()
        if not ok:
            if pending_flush <= 0 or frame is None:
                break
            pending_flush -= 1
            f = frame
        frame = f
        src = st.upload(f)
        out, alpha, fgr, used = st.process_tensor(src)
        if used is last_used:
            continue  # pipeline warm-up repeats a frame
        last_used = used
        if alpha_out:
            from .effects import refine_alpha
            al = refine_alpha(alpha, st.settings, h / 1080) if alpha is not None else None
            fg = fgr if fgr is not None else used
            rec.write(st.download(fg), None if al is None else st.download(al.expand(1, 3, h, w))[..., 0])
        else:
            rec.write(st.download(out))
        i += 1
        if i % 30 == 0:
            el = time.time() - t0
            print(f"\r{i}/{n} frames  {i / el:.1f} fps", end="", flush=True)
    rec.close()
    if hasattr(st.engine, "close"):
        st.engine.close()
    print(f"\ndone: {a.output}  ({i} frames, {i / (time.time() - t0):.1f} fps)")


def cmd_live(a):
    from .backgrounds import BackgroundProvider
    from .pipeline import PRESET_BY_KEY, LivePipeline
    from .studio import Studio
    pr = PRESET_BY_KEY[a.preset]
    engine = a.engine or pr.engine
    devs = _devices(a.gpus)
    st = Studio(engine, devs if engine.startswith("birefnet") else devs[:1],
                Settings(mode=a.mode, auto_frame=a.auto_frame, mirror=a.mirror, blur_strength=a.blur,
                         bg_blur=a.bg_blur))
    if a.background:
        st.compositor.background_src = BackgroundProvider(a.background)
    pl = LivePipeline(st)
    pl.set_source(a.camera, pr.width, pr.height, pr.fps)
    t_open = time.time()
    while pl.source is None:
        if pl.source_error or time.time() - t_open > 20:
            pl.stop()
            sys.exit(f"error: {pl.source_error or 'camera did not open'}")
        time.sleep(0.05)
    w, h, fps = pl.source.width, pl.source.height, int(round(pl.source.fps))
    if a.output in ("pipewire", "both", "all"):
        from .pwcam import PipeWireCamera
        pl.add_output("pipewire", PipeWireCamera(w, h, fps))
    if a.output in ("v4l2", "both", "all"):
        from .v4l2out import V4L2Camera
        pl.add_output("v4l2", V4L2Camera(w, h, fps, "YUYV"))
    if a.output in ("v4l2alpha", "all"):
        from .v4l2out import V4L2Camera
        pl.add_output("v4l2alpha", V4L2Camera(w, h, fps, "ABGR32"))
    print(f"{a.camera} {w}x{h}@{fps} -> {a.output}  engine={engine} mode={a.mode}  Ctrl+C to stop")
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("_stop", True))
    while not globals().get("_stop"):
        time.sleep(1)
        s = pl.stats
        print(f"\rin {s.fps_in:6.1f}  out {s.fps_out:6.1f} fps  {s.latency_ms:5.1f} ms  dropped {s.dropped}",
              end="", flush=True)
    print()
    pl.stop()
    if hasattr(st.engine, "close"):
        st.engine.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="r9700-broadcast", description="Virtual Greenscreen R9700 Broadcast Studio")
    sub = p.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--engine", default="rvm", choices=list(ENGINES))
    common.add_argument("--gpus", default=None, help="e.g. cuda:0,cuda:1 (default: all visible)")
    common.add_argument("--background", help="image for --mode replace")

    f = sub.add_parser("file", parents=[common], help="process a video file")
    f.add_argument("input")
    f.add_argument("output")
    f.add_argument("--mode", default="alpha", choices=list(MODES) + ["alpha"])
    f.add_argument("--codec", default="hevc", choices=["hevc", "h264", "x264", "prores4444"])
    f.set_defaults(fn=cmd_file)

    l = sub.add_parser("live", help="camera -> virtual camera, no GUI")
    l.add_argument("--engine", default=None, choices=list(ENGINES), help="default: the preset's engine")
    l.add_argument("--gpus", default=None)
    l.add_argument("--background", help="image or video for --mode replace")
    l.add_argument("--camera", default="/dev/video0")
    l.add_argument("--preset", default="60", choices=["30", "60", "100", "120"])
    l.add_argument("--mode", default="blur", choices=list(MODES))
    l.add_argument("--output", default="pipewire", choices=["pipewire", "v4l2", "v4l2alpha", "both", "all"],
                   help="v4l2 = /dev/video42 YUYV (chat apps); v4l2alpha = /dev/video43 ABGR32 (OBS, transparent)")
    l.add_argument("--blur", type=float, default=0.6, help="background blur amount 0..1")
    l.add_argument("--bg-blur", type=float, default=0.0, help="blur the replacement background 0..1")
    l.add_argument("--auto-frame", action="store_true")
    l.add_argument("--mirror", action="store_true")
    l.set_defaults(fn=cmd_live)

    g = sub.add_parser("gui", help="open the studio window (default)")
    g.set_defaults(fn=lambda a: __import__("r9700_broadcast.app", fromlist=["main"]).main())

    a = p.parse_args(argv)
    if not a.cmd:
        from .app import main as gui
        return gui()
    a.fn(a)


if __name__ == "__main__":
    main()
