"""Live-path test of the subject lock: Studio over a recorded clip at real-time pacing.

Reports the GPU-thread time per frame (must stay well under the frame budget), when the tracker
locks on, and writes alphas for the contact sheet.
usage: python tests/subject_lock_live.py VIDEO OUT.npz [fps]
"""
import sys
import time

import cv2
import numpy as np
import torch

from r9700_broadcast.effects import Settings
from r9700_broadcast.studio import Studio


def main():
    video, out = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    cap = cv2.VideoCapture(video)
    frames = [f for f in (cap.read()[1] for _ in range(600)) if f is not None][5:]
    import os
    lock = os.environ.get("LOCK", "1") == "1"
    st = Studio("rvm", ["cuda:0"], Settings(mode="matte", subject_lock=lock, keep_accessories=lock is False and os.environ.get("GUARD", "0") == "1"))
    alphas, times, status = [], [], []
    t_next = time.perf_counter()
    lock_t = None
    t0 = time.perf_counter()
    for loop in range(int(os.environ.get('LOOPS', '3'))):              # tracker needs a few seconds to load + lock on
        for i, f in enumerate(frames):
            t_next += 1 / fps
            t = time.perf_counter()
            src = st.upload(f)
            _, a, _, _ = st.process_tensor(src)
            a8 = (a[0, 0].float() * 255).byte()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            s = st.subject_status()
            if status and s != status[-1] or (not status and loop == 0 and i == 0):
                print(f"{time.perf_counter() - t0:6.2f}s loop {loop} frame {i}: {s}", flush=True)
            if loop != int(os.environ.get('LOOPS', '3')) - 1:
                status.append(s)
            if s == "locked" and lock_t is None:
                lock_t = time.perf_counter() - t0
                print(f"locked after {lock_t:.1f} s", flush=True)
            if loop == int(os.environ.get('LOOPS', '3')) - 1:
                alphas.append(a8.cpu().numpy())
                status.append(s)
            d = t_next - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                t_next = time.perf_counter()
    status = status[-len(frames):]
    ts = np.array(times[-len(frames):]) * 1000
    print(f"GPU thread: median {np.median(ts):.1f} ms  p99 {np.percentile(ts, 99):.1f} ms  max {ts.max():.1f} ms")
    print("status last pass:", {k: status.count(k) for k in set(status)})
    st.close()
    np.savez_compressed(out, rvm_lock=np.stack(alphas))


if __name__ == "__main__":
    main()
