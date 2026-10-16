"""A/V sync check for the recorder: white frames + simultaneous beeps (played to the default sink and
recorded from its monitor), then measure the offset between flash and beep in the file.
usage: python tests/av_sync.py OUT.mp4 [codec]"""
import subprocess
import sys
import threading
import time

import numpy as np

from r9700_broadcast.media import Recorder

out = sys.argv[1]
codec = sys.argv[2] if len(sys.argv) > 2 else "h264"
# a private null sink: the beeps are inaudible and nothing else is recorded
sink = "r9700_avsync_test"
mod = subprocess.run(["pactl", "load-module", "module-null-sink", f"sink_name={sink}",
                      "sink_properties=device.description=r9700-avsync-test"], capture_output=True, text=True).stdout.strip()
time.sleep(0.5)
w, h, fps = 640, 360, 60
rec = Recorder(out, w, h, fps, codec, audio=f"{sink}.monitor")
player = subprocess.Popen(["pw-cat", "--playback", "--target", sink, "--raw", "--rate", "48000", "--channels", "1",
                           "--format", "s16", "--latency", "10ms", "-"], stdin=subprocess.PIPE)
beeps = []


def audio_clock():
    """Stream silence continuously; a 30 ms 1 kHz beep each time a flash is due."""
    n = 480     # 10 ms
    t = np.arange(1440) / 48000
    beep = (np.sin(2 * np.pi * 1000 * t) * 20000).astype(np.int16).tobytes()
    sil = bytes(n * 2)
    t0 = time.perf_counter()
    k = 0
    while k < 400:
        k += 1
        if beeps and beeps[0] <= time.perf_counter():
            beeps.pop(0)
            player.stdin.write(beep)
        else:
            player.stdin.write(sil)
        d = t0 + k * 0.01 - time.perf_counter()
        if d > 0:
            time.sleep(d)


th = threading.Thread(target=audio_clock, daemon=True)
th.start()
time.sleep(0.5)
frame = np.zeros((h, w, 4), np.uint8)
t0 = time.perf_counter()
flashes = []
for i in range(fps * 3):
    white = i % fps == 30
    frame[:] = 255 if white else 0
    if white:
        beeps.append(time.perf_counter())
        flashes.append(i)
    rec.send_bgra(frame)
    time.sleep(max(0, t0 + (i + 1) / fps - time.perf_counter()))
rec.close()
th.join()
player.stdin.close()
player.wait()
subprocess.run(["pactl", "unload-module", mod])

# analyse
v = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=pts_time",
                    "-f", "lavfi", f"movie={out},signalstats", "-show_entries", "frame_tags=lavfi.signalstats.YAVG",
                    "-of", "csv=p=0"], capture_output=True, text=True).stdout.split()
vt = [float(x.split(",")[0]) for x in v if x.count(",") == 1 and float(x.split(",")[1]) > 128]
pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
                     capture_output=True).stdout
a = np.abs(np.frombuffer(pcm, np.int16).astype(np.float32))
env = a[: len(a) // 48 * 48].reshape(-1, 48).max(1)        # 1 ms envelope
on = np.nonzero((env[1:] > 3000) & (env[:-1] <= 3000))[0] / 1000.0
print("flash times:", [round(x, 3) for x in vt])
print("beep  times:", [round(x, 3) for x in on])
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time,duration", "-of", "csv=p=0",
                      out], capture_output=True, text=True).stdout.split()
print("streams:", dur)
if vt and len(on):
    offs = [min(on, key=lambda b: abs(b - f)) - f for f in vt]
    print("audio - video offset (ms):", [round(o * 1000) for o in offs])
