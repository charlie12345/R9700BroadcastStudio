"""Precise "who is the subject" mask, used to (re)seed the MatAnyone 2 core tracker.

"Grounded SAM 2" recipe (Ren et al. 2024, Grounded-SAM; Ravi et al. 2024, SAM 2):

  person       a semantic person map (DeepLabV3) gives the subject's box and interior points;
               SAM 2.1 turns them into a pixel-accurate person mask
  accessories  SAM 2 treats a headset as a separate object from the person wearing it, and a black
               ear cup on a black chair is invisible to colour/semantic cues. Grounding DINO (open-
               vocabulary detector) finds "headphones / headset / microphone" by name; each box,
               given to SAM 2, returns the whole accessory (both ear cups, band, boom mic)

The union is the subject the tracker follows. The chair, not being named, stays background.
All models are permissively licensed: SAM 2.1 Apache-2.0, Grounding DINO Apache-2.0, DeepLab BSD.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

from .matting import MODELS, ROOT

SAM2_DIR = os.path.join(ROOT, "third_party", "sam2")
SAM2_CKPTS = {"base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
              "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt")}
GDINO_DIR = os.path.join(MODELS, "grounding-dino-base")
ACCESSORIES = "headphones. headset. microphone."


def available(variant="large") -> bool:
    return os.path.isfile(os.path.join(MODELS, "sam2", SAM2_CKPTS[variant][1])) and \
        os.path.isdir(os.path.join(SAM2_DIR, "sam2"))


def grounding_available() -> bool:
    return os.path.isfile(os.path.join(GDINO_DIR, "config.json"))


class SubjectSeeder:
    def __init__(self, device, variant="large"):
        if SAM2_DIR not in sys.path:
            sys.path.insert(0, SAM2_DIR)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg, ck = SAM2_CKPTS[variant]
        self.device = torch.device(device)
        self.pred = SAM2ImagePredictor(build_sam2(cfg, os.path.join(MODELS, "sam2", ck), device=str(device)))
        self.gd = None
        if grounding_available():
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self.gd_proc = AutoProcessor.from_pretrained(GDINO_DIR)
            self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_DIR).to(self.device).eval()
        self.last_accessories = None
        self.last_body = None

    def _predict(self, pts=None, labels=None, box=None):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            ms, sc, _ = self.pred.predict(
                point_coords=None if pts is None else np.asarray(pts, np.float32),
                point_labels=None if labels is None else np.asarray(labels, np.int32),
                box=None if box is None else np.asarray(box, np.float32), multimask_output=True)
        return ms > 0, sc

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, text: str = ACCESSORIES, threshold: float = 0.3):
        """Open-vocabulary boxes [(label, score, [x0,y0,x1,y1]), ...]."""
        if self.gd is None:
            return []
        from PIL import Image
        img = Image.fromarray(rgb)
        inp = self.gd_proc(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.gd(**inp)
        r = self.gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[img.size[::-1]])[0]
        return [(l, float(s), [float(v) for v in b])
                for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["text_labels"])]

    def __call__(self, rgb: np.ndarray, person: np.ndarray, chair: np.ndarray | None = None) -> np.ndarray | None:
        """rgb HxWx3 uint8; person/chair HxW float probabilities -> HxW bool seed, or None if nobody."""
        H, W = person.shape
        p = person > 0.5
        n, lab, stats, _ = cv2.connectedComponentsWithStats(p.astype(np.uint8))
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.01 * H * W]
        if not keep:
            return None
        self.pred.set_image(rgb)
        body = np.zeros((H, W), bool)
        for i in keep:                                     # every person in shot
            blob = lab == i
            x, y, w, h = stats[i, :4]
            core = blob & (person > 0.9)
            ys, xs = np.nonzero(core if core.any() else blob)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            ms, sc = self._predict([[cx, cy], [cx, (ys.min() + cy) // 2]], [1, 1], box=[x, y, x + w, y + h])
            ious = [(m & blob).sum() / max(1, (m | blob).sum()) for m in ms]
            best = int(np.argmax([s if iou > 0.5 else -1 for s, iou in zip(sc, ious)]))
            body |= ms[best]
        acc = self._accessories(rgb, body, chair)
        self.last_accessories = acc
        self.last_body = body
        return body | acc

    @staticmethod
    def _best_boxes(dets, iou_thr=0.5):
        """Non-maximum suppression across labels: "headphones" and "headset" fire on the same object,
        and the looser duplicates tend to include the chair wing behind it."""
        def iou(a, b):
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua if ua > 0 else 0.0
        kept = []
        for _l, _s, box in sorted(dets, key=lambda d: -d[1]):
            if all(iou(box, k) < iou_thr for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _chair_points(body, chair, head_w, box):
        """Negative prompts: points DeepLab is sure are chair, clearly away from the head (so never on
        an ear cup), on each side of the head inside / near the accessory box."""
        if chair is None:
            return []
        H, W = body.shape
        dist = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
        x0, y0, x1, y1 = [int(v) for v in box]
        pad = int(0.5 * head_w)
        win = np.zeros_like(body)
        win[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = True
        sure = (chair > 0.6) & (dist > 0.35 * head_w) & win
        pts = []
        cx = (x0 + x1) / 2
        for side in (np.arange(W) < cx, np.arange(W) >= cx):
            c = sure & side[None, :]
            if c.sum() > 50:
                d = cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 5)
                py, px = np.unravel_index(np.argmax(d), d.shape)
                pts.append([px, py])
        return pts

    def _accessories(self, rgb, body, chair=None):
        acc = np.zeros_like(body)
        if not body.any():
            return acc
        ys, xs = np.nonzero(body)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = 0.15 * (bx1 - bx0)
        top = ys.min()
        hh = max(1, int(0.25 * (ys.max() - top)))
        widths = body[top:top + hh].sum(1)
        head_w = float(np.median(widths[widths > 0])) if (widths > 0).any() else float(bx1 - bx0)
        near = cv2.dilate(body.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        for x0, y0, x1, y1 in self._best_boxes(self.detect(rgb)):
            # only things worn by the subject: the box must overlap the subject's (padded) box
            if x1 < bx0 - pad or x0 > bx1 + pad or y1 < by0 - pad or y0 > by1:
                continue
            if (x1 - x0) * (y1 - y0) > 0.6 * (bx1 - bx0) * (by1 - by0):
                continue                                   # a box around the whole person
            neg = self._chair_points(body, chair, head_w, (x0, y0, x1, y1))
            ms, sc = self._predict(neg or None, [0] * len(neg) or None, box=[x0, y0, x1, y1])
            best, best_s = None, -1.0
            for m, s in zip(ms, sc):
                a = m.sum()
                if a == 0 or (m & body).sum() > 0.5 * a:
                    continue                               # the person, not the accessory
                if s > best_s:
                    best, best_s = m, s
            if best is not None and (best & near).any():   # touching the subject: worn, not on a desk
                acc |= best
        # worn accessories hug the head: ear cups/band/mic stay within ~1/4 head-width of the head and
        # hair. A loose detection box makes SAM add the chair wing behind the ear cup, which reaches
        # much further out.
        if acc.any():
            dist = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
            # sideways (ear cups, mic) up to 1/4 head-width; above the head a band hugs the hair, so
            # anything standing off the top is the headrest behind it
            reach = np.full(body.shape, 0.25 * head_w, np.float32)
            head_cols = body[top:top + hh].any(0)          # columns the head itself spans
            col_top = np.where(head_cols, body.argmax(0), 0)
            yy = np.arange(body.shape[0])[:, None]
            above = (yy < col_top[None, :]) & head_cols[None, :]
            reach[above] = 0.06 * head_w
            acc &= dist <= reach
        # drop specks (SAM sometimes adds a few pixels of the chair's texture)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(acc.astype(np.uint8))
        if n > 2:
            big = stats[1:, cv2.CC_STAT_AREA].max()
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 0.03 * big:
                    acc[lab == i] = False
        return acc
