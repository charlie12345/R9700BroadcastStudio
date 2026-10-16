"""Transparent mode end to end: the alpha matte arrives in an external PipeWire consumer (what OBS
Studio's 'Video Capture Device (PipeWire)' source receives)."""
import os
import subprocess
import sys
import tempfile
import time
from r9700_broadcast.effects import Settings
from r9700_broadcast.pipeline import LivePipeline
from r9700_broadcast.pwcam import PipeWireCamera
from r9700_broadcast.studio import Studio

CONSUMER = r'''
import sys, gi
gi.require_version("Gst","1.0"); from gi.repository import Gst
Gst.init(None)
p=Gst.parse_launch("pipewiresrc target-object=r9700_broadcast_camera ! video/x-raw,format=BGRA ! appsink name=s max-buffers=1 drop=true sync=false")
s=p.get_by_name("s"); p.set_state(Gst.State.PLAYING)
for _ in range(60): smp=s.emit("try-pull-sample", Gst.SECOND)
caps=smp.get_caps().get_structure(0); w,h=caps.get_value("width"),caps.get_value("height")
b=smp.get_buffer(); ok,m=b.map(Gst.MapFlags.READ)
a=bytes(m.data)[3::4]; b.unmap(m); p.set_state(Gst.State.NULL)
open(sys.argv[1],"wb").write(bytes([w>>8,w&255,h>>8,h&255])+a)
n=len(a); z=a.count(0); o=a.count(255)
print(f"{w}x{h} alpha: transparent(0) {z/n*100:.1f}%  opaque(255) {o/n*100:.1f}%  soft edge {(n-z-o)/n*100:.2f}%")
'''


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "testdata/talking_head_720p.mp4"
    st = Studio("rvm", ["cuda:0"], Settings(mode="transparent"))
    pl = LivePipeline(st)
    pl.set_source(src, 1280, 720, 60)
    time.sleep(1.5)
    pw = PipeWireCamera(pl.source.width, pl.source.height, int(round(pl.source.fps)))
    pl.add_output("pipewire", pw)
    time.sleep(2)
    out = os.path.join(tempfile.gettempdir(), "r9700_alpha.raw")
    r = subprocess.run(["/usr/bin/python3", "-c", CONSUMER, out], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr[-800:])
    pl.stop()


if __name__ == "__main__":
    main()
