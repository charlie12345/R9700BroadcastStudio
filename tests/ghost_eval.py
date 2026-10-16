"""Ghosting evaluation on recorded footage.

Runs each method over the clip, then reports, per method:
  ghost%   share of the subject's *interior* (head/headset/body, eroded union silhouette) that is
           semi-transparent (0.1 < alpha < 0.9) - the "see-through headset / temple" artefact
  hole%    interior pixels with alpha < 0.1 (dropped entirely)
  flicker  mean |alpha_t - alpha_t-1| inside the interior on frames where the subject is still
and writes a contact sheet of head crops composited over magenta (ghosting is obvious there).

usage: python tests/ghost_eval.py VIDEO OUTDIR [extra .npy alpha stacks named label=path ...]
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from r9700_broadcast.matting import RVMEngine, SemanticGuard

DEV = os.environ.get("DEV", "cuda:0")


def load(video, n=600):
    cap = cv2.VideoCapture(video)
    fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def to_t(f):
    return torch.from_numpy(f[..., ::-1].copy()).to(DEV).permute(2, 0, 1)[None].half().div(255)


@torch.inference_mode()
def run_methods(frames, methods):
    out = {}
    for name, make in methods.items():
        step = make()
        al = []
        for f in frames:
            a = step(to_t(f))
            al.append((a[0, 0].float().clamp(0, 1) * 255).byte().cpu().numpy())
        out[name] = np.stack(al)
        print(f"ran {name}", flush=True)
    return out


def rvm_plain():
    e = RVMEngine(DEV)
    e.use_graph = False
    return lambda s: e(s).alpha


def rvm_guard():
    e = RVMEngine(DEV)
    e.use_graph = False
    g = SemanticGuard(DEV)
    return lambda s: g(s, e(s).alpha)


def rvm_core(lag=0, short=384, show_core=False):
    def make():
        from r9700_broadcast.coretrack import CoreTracker, fuse_core, tracker_size
        e = RVMEngine(DEV)
        e.use_graph = False
        tr = CoreTracker(DEV, short, seeder=os.environ.get("SEEDER", "1") == "1")
        tr.seed_stride = 30
        hist = []

        def step(s):
            a = e(s).alpha
            mid = lambda t: F.interpolate(t.float(), size=(720, 1280), mode="bilinear", align_corners=False)
            hist.append(tr.step(mid(s), mid(a)))
            core = hist[max(0, len(hist) - 1 - lag)]
            del hist[:-4]
            if core is None:
                return a if not show_core else torch.zeros_like(a)
            if show_core:
                return F.interpolate(core, size=s.shape[-2:], mode='bilinear', align_corners=False)
            return fuse_core(a, core)
        return step
    return make


def metrics(alphas, frames, interior):
    a = alphas.astype(np.float32) / 255
    semi = ((a > 0.1) & (a < 0.9) & interior).sum() / interior.sum()
    hole = ((a <= 0.1) & interior).sum() / interior.sum()
    g = np.stack([cv2.cvtColor(cv2.resize(f, (a.shape[2], a.shape[1])), cv2.COLOR_BGR2GRAY) for f in frames])
    still = np.abs(np.diff(g.astype(np.float32), axis=0)).mean((1, 2)) < 1.5
    da = np.abs(np.diff(a, axis=0))
    flick = (da * interior[1:]).sum((1, 2)) / interior[1:].sum((1, 2)).clip(1)
    return semi * 100, hole * 100, float(flick[still].mean() * 255) if still.any() else float("nan")


def main():
    video, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    frames = load(video)[int(os.environ.get('SKIP', '0')):]
    H, W = frames[0].shape[:2]
    methods = {"rvm": rvm_plain, "rvm+guard": rvm_guard, "core(seeded)": rvm_core(show_core=True), "rvm+core": rvm_core(),
               "rvm+core lag2": rvm_core(2)}
    only = os.environ.get("METHODS")
    if only:
        methods = {k: v for k, v in methods.items() if k in only.split(",") or k == "rvm"}
    res = run_methods(frames, methods)
    for extra in sys.argv[3:]:
        label, path = extra.split("=", 1)
        st = np.load(path)
        res[label] = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) for x in st])[:len(frames)]
    # interior = pixels every method calls solid-ish *or* any method calls solid, then eroded well
    # inside: the subject's body/head, never the true hair/edge transition band
    union = np.zeros(res["rvm"].shape, bool)
    for v in res.values():
        union |= v > 128
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    interior = np.stack([cv2.erode(u.astype(np.uint8), k) > 0 for u in union])
    print(f"{'method':14s} ghost%  hole%  flicker")
    for name, a in res.items():
        s, h, fl = metrics(a, frames, interior)
        print(f"{name:14s} {s:6.2f} {h:6.2f} {fl:7.2f}")
    # contact sheet: the 8 frames with most disagreement, head crop over magenta
    dis = np.stack([np.abs(v.astype(np.int16) - res["rvm"]).mean((1, 2)) for v in res.values()]).max(0)
    dis[:int(os.environ.get('PICK_FROM', '0'))] = -1
    picks = sorted(np.argsort(dis)[-8:])
    ys, xs = np.nonzero(union.any(0))
    cy0, cx = max(0, ys.min() - 20), int(np.median(xs))
    crop = lambda im: im[cy0:cy0 + 500, max(0, cx - 400):cx + 400]
    rows = []
    for i in picks:
        tiles = [crop(frames[i])]
        for name, a in res.items():
            al = a[i][..., None].astype(np.float32) / 255
            comp = frames[i] * al + np.array([255, 0, 255], np.float32) * (1 - al)
            t = crop(comp.astype(np.uint8)).copy()
            cv2.putText(t, f"{name} f{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            tiles.append(t)
        rows.append(np.hstack([cv2.resize(t, (400, 250)) for t in tiles]))
    cv2.imwrite(os.path.join(outdir, "ghost_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    np.savez_compressed(os.path.join(outdir, "alphas.npz"), **{k.replace("+", "_"): v for k, v in res.items()})


if __name__ == "__main__":
    main()
