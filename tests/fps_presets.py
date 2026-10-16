"""End-to-end FPS check per preset: OBSBOT -> pipeline -> PipeWire camera -> external consumer.

usage: python tests/fps_presets.py [camera] [modes...]
The consumer (system python3 + GStreamer) runs in a separate process so we measure what OBS /
a browser would actually receive.
"""
import subprocess, sys, time, os
from r9700_broadcast.effects import Settings
from r9700_broadcast.pipeline import LivePipeline, PRESETS
from r9700_broadcast.pwcam import PipeWireCamera
from r9700_broadcast.studio import Studio

CONSUMER = r'''
import os
import sys, time, gi
gi.require_version("Gst","1.0"); from gi.repository import Gst
Gst.init(None)
p=Gst.parse_launch("pipewiresrc target-object=r9700_broadcast_camera ! video/x-raw,format=BGRA ! fakesink name=s sync=false")
c=[0]; p.get_by_name("s").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, lambda *a:(c.__setitem__(0,c[0]+1),Gst.PadProbeReturn.OK)[1])
p.set_state(Gst.State.PLAYING); time.sleep(3); c0=c[0]; t=time.time(); time.sleep(6)
print((c[0]-c0)/(time.time()-t)); p.set_state(Gst.State.NULL)
'''
def main():
    cam = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    modes = sys.argv[2:] or ["blur"]
    only = os.environ.get("PRESETS", "").split(",") if os.environ.get("PRESETS") else None
    for pr in [p for p in PRESETS if not only or p.key in only]:
        for mode in modes:
            devs = ["cuda:0", "cuda:1"] if pr.engine.startswith("birefnet") else ["cuda:0"]
            st = Studio(pr.engine, devs, Settings(mode=mode, subject_lock=os.environ.get("LOCK", "1") == "1"))
            if os.environ.get("EXPO") == "1":
                from r9700_broadcast import uvc
                uvc.lock_exposure_for_fps(cam, pr.fps)
            pl = LivePipeline(st)
            pl.set_source(cam, pr.width, pr.height, pr.fps)
            pw = PipeWireCamera(pr.width, pr.height, pr.fps)
            pl.add_output("pipewire", pw)
            time.sleep(float(os.environ.get("WARM", "4")))   # warm-up (graph capture, MIOpen, subject lock)
            got = subprocess.run(["/usr/bin/python3", "-c", CONSUMER], capture_output=True, text=True).stdout.strip()
            print(f"{pr.label:30s} mode={mode:11s} camera {pl.stats.fps_in:6.1f} fps | pipeline out {pl.stats.fps_out:6.1f} fps"
                  f" | OBS/consumer receives {float(got or 0):6.1f} fps | glass-to-output {pl.stats.latency_ms:5.1f} ms"
                  f" | dropped {pl.stats.dropped} | lock {st.subject_status()}", flush=True)
            pl.stop()
            if hasattr(st.engine, "close"):
                st.engine.close()
            del st
            import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
