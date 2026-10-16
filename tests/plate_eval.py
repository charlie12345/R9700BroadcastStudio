"""Clean-plate assist evaluation on real camera noise.

Plate = live frames of the empty room from the camera. Test frames = new live frames with a
matted person + a held 'product' box composited in. Reports alpha on the box (should be ~1 with
plate assist) and false-positive alpha on the empty background (should be ~0).
Usage: python tests/plate_eval.py /dev/video0 testdata/talking_head_720p.mp4 out.jpg
"""
import sys, time
import cv2, numpy as np, torch
from r9700_broadcast.studio import Studio
from r9700_broadcast.effects import Settings
from r9700_broadcast.media import FrameSource

cam_uri, clip, out_path = sys.argv[1:4]
src = FrameSource(cam_uri, 1920, 1080, 30)
seq = -1
def grab():
    global seq
    while True:
        seq, f = src.read(seq)
        if f is not None:
            return f.copy()

# person + alpha from the interview clip (BiRefNet, studio quality), scaled into the room
st_hi = Studio("birefnet-lite", ["cuda:0"], Settings(mode="matte"))
cap = cv2.VideoCapture(clip)
cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
_, person = cap.read()
t = st_hi.upload(person)
_, a, fgr, _ = st_hi.process_tensor(t)
p_rgb = st_hi.download(fgr)[..., ::-1]
p_a = a[0, 0].float().cpu().numpy()
H, W = 1080, 1920
p_rgb = cv2.resize(p_rgb, (1280, 720)); p_a = cv2.resize(p_a, (1280, 720))
ox, oy = 640, 360
box = (560, 640, 860, 900)   # x0,y0,x1,y1 of the held 'product' (left of the person's body)

def composite(bg):
    f = bg.astype(np.float32)
    # held box: saturated red/blue packaging with a label
    x0, y0, x1, y1 = box
    f[y0:y1, x0:x1] = (40, 40, 200)
    f[y0 + 60:y0 + 120, x0 + 30:x1 - 30] = (230, 230, 230)
    sub = f[oy:oy + 720, ox:ox + 1280]
    sub[:] = p_rgb * p_a[..., None] + sub * (1 - p_a[..., None])
    return f.clip(0, 255).astype(np.uint8)

st = Studio("rvm", ["cuda:0"], Settings(mode="matte"))
print("capturing plate (stay out of shot)…")
st.start_plate_capture()
while st.capturing_plate:
    st.process(grab())
plate_ready = st.plate.ready
gt = np.zeros((H, W), np.float32)
gt[oy:oy + 720, ox:ox + 1280] = p_a
gt[box[1]:box[3], box[0]:box[2]] = 1

def run(use_plate):
    st.plate.rescue = st.plate.suppress = use_plate
    st.engine.reset()
    for _ in range(40):
        frame = composite(grab())
        _, alpha, _, _ = st.process_tensor(st.upload(frame))
    a = alpha[0, 0].float().cpu().numpy()
    x0, y0, x1, y1 = box
    obj = a[y0:y1, x0:x1].mean()
    bgmask = gt < 0.01
    fp = a[bgmask].mean()
    mad = np.abs(a - gt).mean()
    return a, frame, obj, fp, mad

a0, frame, obj0, fp0, mad0 = run(False)
a1, _, obj1, fp1, mad1 = run(True)
print(f"plate ready: {plate_ready}")
print(f"AI only      : box alpha {obj0:.3f}  bg false-positive {fp0:.4f}  MAD vs GT {mad0:.4f}")
print(f"AI + plate   : box alpha {obj1:.3f}  bg false-positive {fp1:.4f}  MAD vs GT {mad1:.4f}")
viz = np.hstack([cv2.resize(frame, (640, 360)),
                 cv2.resize(cv2.cvtColor((a0 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (640, 360)),
                 cv2.resize(cv2.cvtColor((a1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (640, 360))])
cv2.imwrite(out_path, viz)

# Camouflage check: paint a patch of the person's shirt with exactly the plate pixels behind it.
# Suppression must not punch a hole there.
st.plate.rescue = st.plate.suppress = True
st.engine.reset()
for _ in range(30):
    bgf = grab()
    frame = composite(bgf)
    frame[820:900, 1180:1260] = bgf[820:900, 1180:1260]
    _, alpha, _, _ = st.process_tensor(st.upload(frame))
hole = alpha[0, 0, 820:900, 1180:1260].float().mean().item()
print(f"camouflage patch alpha (want ~1): {hole:.3f}")
src.close()
