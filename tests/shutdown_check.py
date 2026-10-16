"""Shutdown must be quick and clean: run the live pipeline with the PipeWire camera for a few
seconds, stop it step by step, and report how long each step takes (fails if any hangs)."""
import faulthandler
import sys
import time

faulthandler.dump_traceback_later(45, exit=True)   # a hang prints every thread's stack, then exits

from r9700_broadcast.effects import Settings
from r9700_broadcast.pipeline import LivePipeline
from r9700_broadcast.pwcam import PipeWireCamera
from r9700_broadcast.studio import Studio

T0 = time.time()


def log(m):
    print(f"{time.time() - T0:6.2f}s {m}", flush=True)


def main():
    cam = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    st = Studio("rvm", ["cuda:0"], Settings(mode="transparent"))
    pl = LivePipeline(st)
    pl.set_source(cam, 1920, 1080, 60)
    while pl.source is None:
        time.sleep(0.05)
    pl.add_output("pipewire", PipeWireCamera(pl.source.width, pl.source.height, int(round(pl.source.fps))))
    time.sleep(5)
    log(f"running at {pl.stats.fps_out:.1f} fps; stopping")
    t = time.time()
    pl.stop()
    log(f"pipeline.stop() took {time.time() - t:.2f}s")
    log("main done")


if __name__ == "__main__":
    main()
    log("interpreter exiting")
