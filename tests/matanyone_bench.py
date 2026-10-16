"""Speed + output dump of MatAnyone 2 on recorded footage (first-frame mask from RVM).
usage: python tests/matanyone_bench.py VIDEO SIZE [DEVICE] [OUTDIR]"""
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "MatAnyone2"))


def main():
    video, size = sys.argv[1], int(sys.argv[2])
    dev = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    out = sys.argv[4] if len(sys.argv) > 4 else None
    torch.cuda.set_device(dev)
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.get_default_model import get_matanyone2_model
    root = os.path.join(os.path.dirname(__file__), "..")
    net = get_matanyone2_model(os.path.join(root, "models/matanyone2/matanyone2.pth"), dev)
    proc = InferenceCore(net, cfg=net.cfg)
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < 600:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    h, w = frames[0].shape[:2]
    nh, nw = size, int(round(w * size / h / 16)) * 16
    from r9700_broadcast.matting import RVMEngine
    rvm = RVMEngine(dev)
    rvm.use_graph = False
    to_t = lambda f: torch.from_numpy(f[..., ::-1].copy()).to(dev).permute(2, 0, 1).float().div(255)
    with torch.inference_mode():
        for f in frames[:30]:     # let RVM's recurrent state settle; its alpha seeds the first-frame mask
            seed = rvm(to_t(f)[None].half()).alpha[0, 0].float()
        mask = (seed > 0.5).float() * 255
        import torch.nn.functional as F
        mask = F.interpolate(mask[None, None], size=(nh, nw), mode="nearest")[0, 0]
        times, alphas = [], []
        for i, f in enumerate(frames):
            img = F.interpolate(to_t(f)[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=AMP):
                if i == 0:
                    proc.step(img, mask, objects=[1])
                    p = proc.step(img, first_frame_pred=True)
                elif i <= 10:
                    p = proc.step(img, first_frame_pred=True)
                else:
                    p = proc.step(img)
                a = proc.output_prob_to_mask(p)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t)
            if out:
                alphas.append((a.clamp(0, 1) * 255).byte().cpu().numpy())
    ts = np.array(times[20:]) * 1000
    print(f"MatAnyone2 {nw}x{nh} amp={AMP}: median {np.median(ts):.1f} ms  p95 {np.percentile(ts, 95):.1f} ms "
          f"-> {1000 / np.median(ts):.1f} fps  mem {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GB")
    if out:
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, f"ma2_{nh}.npy"), np.stack(alphas))


AMP = os.environ.get("AMP", "1") == "1"


if __name__ == "__main__":
    main()
