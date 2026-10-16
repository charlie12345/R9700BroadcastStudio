"""Procedurally generate the built-in background set (no third-party images = no licensing)."""
import os, subprocess, sys
import cv2, numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "backgrounds")
os.makedirs(OUT, exist_ok=True)
W, H = 1920, 1080
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
u, v = xx / W, yy / H
rng = np.random.default_rng(7)


def save(name, rgb):
    cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(np.clip(rgb, 0, 1).__mul__(255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 93])


def radial(cx, cy, s):
    return np.exp(-(((u - cx) * 16 / 9) ** 2 + (v - cy) ** 2) / s)


def bokeh(base, n, colors, rmin, rmax, blur):
    img = base.copy()
    for _ in range(n):
        c = colors[rng.integers(len(colors))]
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(rmin, rmax)
        layer = np.zeros((H, W), np.float32)
        cv2.circle(layer, (int(x), int(y)), int(r), 1.0, -1, cv2.LINE_AA)
        img += layer[..., None] * np.array(c, np.float32) * rng.uniform(0.15, 0.45)
    return cv2.GaussianBlur(img, (0, 0), blur)


grain = lambda s: rng.normal(0, s, (H, W, 1)).astype(np.float32)

# 1 studio grey seamless paper
g = 0.18 + 0.32 * radial(0.5, 0.42, 0.35)
save("01 Studio Grey.jpg", np.repeat(g[..., None], 3, 2) + grain(0.006))
# 2 midnight blue
c = np.stack([0.02 + 0.10 * radial(0.3, 0.3, 0.4), 0.04 + 0.16 * radial(0.3, 0.3, 0.4), 0.10 + 0.40 * radial(0.35, 0.35, 0.5)], -1)
save("02 Midnight Blue.jpg", c + grain(0.006))
# 3 sunset gradient
t = v[..., None]
top, mid, bot = np.array([0.25, 0.10, 0.45]), np.array([0.95, 0.40, 0.35]), np.array([1.0, 0.75, 0.40])
img = np.where(t < 0.55, top + (mid - top) * (t / 0.55), mid + (bot - mid) * ((t - 0.55) / 0.45))
save("03 Sunset.jpg", img + grain(0.005))
# 4 warm bokeh (living room lights)
base = np.stack([0.10 + 0.08 * v, 0.07 + 0.05 * v, 0.05 + 0.03 * v], -1).astype(np.float32)
save("04 Warm Bokeh.jpg", bokeh(base, 70, [(1, .7, .35), (1, .55, .25), (1, .85, .6)], 30, 110, 9))
# 5 cool bokeh (city at night)
base = np.stack([0.02 + 0.03 * v, 0.03 + 0.05 * v, 0.07 + 0.08 * v], -1).astype(np.float32)
save("05 City Bokeh.jpg", bokeh(base, 90, [(.3, .6, 1), (.6, .4, 1), (1, .4, .6), (.4, 1, .9)], 20, 90, 8))
# 6 neon grid (retro)
img = np.zeros((H, W, 3), np.float32)
img[:] = np.stack([0.05 + 0.25 * (1 - v), 0.0 + 0.02 * v, 0.12 + 0.25 * (1 - v)], -1)
horizon = 0.55
for i in range(1, 40):
    y = horizon + (1 - horizon) * (i / 40) ** 2
    cv2.line(img, (0, int(y * H)), (W, int(y * H)), (1.0, 0.2, 0.8), 2, cv2.LINE_AA)
for i in range(-30, 31):
    cv2.line(img, (W // 2, int(horizon * H)), (int(W / 2 + i * W / 12), H), (0.3, 0.8, 1.0), 2, cv2.LINE_AA)
sun = radial(0.5, 0.4, 0.012)
img += sun[..., None] * np.array([1.0, 0.55, 0.3])
img[: int(horizon * H)] = np.maximum(img[: int(horizon * H)], 0)
save("06 Neon Grid.jpg", cv2.GaussianBlur(img, (0, 0), 1.2))

# 7/8 animated loops (10 s, seamless): ffmpeg lavfi so they are tiny
def ff(name, graph):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", graph, "-t", "10", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    os.path.join(OUT, name)], check=True)

ff("07 Aurora Loop.mp4",
   "gradients=s=1920x1080:r=30:c0=0x0b1a3a:c1=0x1f7a6b:c2=0x5b2a86:c3=0x0b1a3a:nb_colors=4:speed=0.0628:type=radial,"
   "gblur=sigma=40")
ff("08 Soft Gradient Loop.mp4",
   "gradients=s=1920x1080:r=30:c0=0x2a2d3a:c1=0x4a3f5c:c2=0x2f4a5a:nb_colors=3:speed=0.0628:type=linear,"
   "gblur=sigma=60")
print("\n".join(sorted(os.listdir(OUT))))
