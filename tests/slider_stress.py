"""Regression for the 'drag the feather slider -> app aborts' crash: sweep every slider-driven
setting quickly while the live pipeline runs at 60 fps. Exits non-zero / aborts if the GPU faults."""
import random, sys, time
from r9700_broadcast.effects import Settings
from r9700_broadcast.pipeline import LivePipeline
from r9700_broadcast.studio import Studio


def main():
    cam = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    s = Settings(mode="blur", replace_dummy=None) if False else Settings(mode="blur")
    st = Studio("rvm", ["cuda:0"], s)
    pl = LivePipeline(st, on_error=lambda e: print("ERROR", e, flush=True))
    pl.set_source(cam, 1920, 1080, 60)
    time.sleep(4)
    rnd = random.Random(1)
    t0 = time.time(); n = 0
    modes = ["blur", "replace", "green", "transparent", "remove", "matte"]
    import os
    only = os.environ.get("ONLY")
    base = Settings(mode="blur")
    while time.time() - t0 < secs:
        if only:
            v = {"feather": rnd.choice([x * 0.5 for x in range(21)]),
                 "choke": rnd.choice([x * 0.5 for x in range(-16, 17)]),
                 "blur_strength": rnd.random(), "bg_blur": rnd.random(), "light_wrap": rnd.random(),
                 "auto_frame": rnd.random() < 0.5, "mode": rnd.choice(modes),
                 "denoise": rnd.random() * 0.5, "auto_on": True}[only]
            if only == "auto_on":
                only_attr = "auto_frame"
            else:
                only_attr = only
            setattr(s, only_attr, v)
            n += 1
            time.sleep(0.02)
            continue
        s.feather = rnd.choice([x * 0.5 for x in range(21)])
        s.choke = rnd.choice([x * 0.5 for x in range(-16, 17)])
        s.blur_strength = rnd.random()
        s.bg_blur = rnd.random()
        s.light_wrap = rnd.random()
        s.relight = rnd.random() * 0.5
        s.vignette = rnd.random() * 0.5
        s.denoise = rnd.random() * 0.5
        s.auto_frame = rnd.random() < 0.3
        s.frame_zoom = 1 + rnd.random() * 2
        if n % 25 == 0:
            s.mode = rnd.choice(modes)
        n += 1
        time.sleep(0.02)   # ~50 slider events/s, like dragging
    print(f"survived {n} setting changes, out {pl.stats.fps_out:.1f} fps, dropped {pl.stats.dropped}", flush=True)
    pl.stop()


if __name__ == "__main__":
    main()
