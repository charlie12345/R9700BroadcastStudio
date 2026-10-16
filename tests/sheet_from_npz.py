"""Contact sheet (head crops over magenta) from ghost_eval's alphas.npz.
usage: python tests/sheet_from_npz.py VIDEO NPZ OUT.jpg SKIP FROM [methods,comma]"""
import sys

import cv2
import numpy as np

video, npz, out, skip, start = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
z = np.load(npz)
names = sys.argv[6].split(",") if len(sys.argv) > 6 else list(z.keys())
cap = cv2.VideoCapture(video)
frames = [f for f in (cap.read()[1] for _ in range(skip + len(z[names[0]]))) if f is not None][skip:]
dis = np.abs(z[names[0]].astype(np.int16) - z[names[-1]]).mean((1, 2))
dis[:start] = -1
picks = sorted(np.argsort(dis)[-8:])
rows = []
for i in picks:
    tiles = [frames[i][120:620, 560:1360]]
    for n in names:
        al = z[n][i][..., None].astype(np.float32) / 255
        comp = (frames[i] * al + np.array([255, 0, 255], np.float32) * (1 - al)).astype(np.uint8)[120:620, 560:1360].copy()
        cv2.putText(comp, f"{n} f{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(comp)
    rows.append(np.hstack([cv2.resize(t, (480, 300)) for t in tiles]))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
