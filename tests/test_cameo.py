"""Cameo bubble (you in a circle/box over any background) and camera-only mirroring."""
from dataclasses import replace

import pytest
import torch

from r9700_broadcast.effects import (Compositor, Settings, cameo_bounds, cameo_center,
                                     cameo_from_point)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 270, 480


def _scene():
    """Camera = red person (left half alpha=1 block), background = blue picture with a white
    stripe on its left edge so a mirrored background is detectable."""
    src = torch.zeros(1, 3, H, W, device=DEV)
    src[:, 0] = 1.0                                    # camera: all red
    a = torch.zeros(1, 1, H, W, device=DEV)
    a[..., 60:, 160:320] = 1.0                         # person in the middle
    bg = torch.zeros(1, 3, H, W, device=DEV)
    bg[:, 2] = 1.0
    bg[..., :, :20] = 1.0                              # white stripe at the left
    return src, a, bg


def _comp(bg):
    c = Compositor()
    c.set_background(bg)
    return c


def test_center_roundtrip_and_clamp():
    s = Settings(cameo_size=0.4)
    for x, y in ((0, 0), (1, 1), (0.25, 0.8)):
        s.cameo_x, s.cameo_y = x, y
        cx, cy, D = cameo_center(s, 1080, 1920)
        assert D / 2 <= cx <= 1920 - D / 2 and D / 2 <= cy <= 1080 - D / 2
        assert cameo_from_point(s, 1080, 1920, cx, cy) == pytest.approx((x, y))
    # dragging far outside the frame clamps to the edge
    assert cameo_from_point(s, 1080, 1920, -500, 5000) == (0.0, 1.0)
    D, (x0, x1), _ = cameo_bounds(s, 1080, 1920)
    assert x1 > x0


def test_default_is_lower_right_and_background_stays_visible():
    src, a, bg = _scene()
    s = Settings(mode="replace", layout="cameo", cameo_follow=False, light_wrap=0, cameo_size=0.4)
    out = _comp(bg)(src, a, src, s)
    cx, cy, D = cameo_center(s, H, W)
    assert cx > W / 2 and cy > H / 2                       # lower-right
    # the full-size person is gone: background shows where they stood (top-left of the person box)
    assert out[0, 2, 80, 200] > 0.9 and out[0, 0, 80, 200] < 0.1
    # inside the bubble there is camera content (red person or the blurred red room)
    assert out[0, 0, int(cy), int(cx)] > 0.5


@pytest.mark.parametrize("shape", ["circle", "rounded", "square", "cutout"])
def test_shapes_and_options_render(shape):
    src, a, bg = _scene()
    c = _comp(bg)
    for kw in ({}, {"cameo_pop": True}, {"cameo_ring": "rainbow"}, {"cameo_fill": "camera"},
               {"cameo_fill": "color"}, {"cameo_ring": "none", "cameo_shadow": False}):
        s = Settings(mode="replace", layout="cameo", cameo_shape=shape, **kw)
        out = c(src, a, src, s)
        assert out.shape == (1, 3, H, W) and torch.isfinite(out).all()


def test_transparent_cameo_alpha_only_in_bubble():
    src, a, bg = _scene()
    s = Settings(mode="transparent", layout="cameo", cameo_follow=False, cameo_x=0, cameo_y=0)
    c = _comp(bg)
    c(src, a, src, s)
    al = c.out_alpha
    cx, cy, D = cameo_center(s, H, W)
    assert al[0, 0, int(cy), int(cx)] > 0.99
    assert al[0, 0, H - 5, W - 5] < 0.01                   # far corner fully transparent


def test_mirror_flips_camera_not_background():
    src, a, bg = _scene()
    a = torch.zeros_like(a)
    a[..., :, :100] = 1.0                                   # person on the left edge
    s = Settings(mode="replace", light_wrap=0, mirror=True)
    out = _comp(bg)(src, a, src, s)
    assert out[0, 0, 100, W - 50] > 0.9                    # person now on the right
    assert out[0, 1, 100, 5] > 0.9                         # white stripe still on the left


def test_face_mirrored():
    from r9700_broadcast.props import Face
    f = Face((100, 50, 40, 40), (110, 60), (130, 60), (120, 70), (120, 80), 0.9)
    m = f.mirrored(480)
    assert m.box[0] == 480 - 140
    assert m.leye == (480 - 130, 60) and m.reye == (480 - 110, 60)


def test_rgba_stamp_grows_alpha():
    import numpy as np
    from r9700_broadcast.props import stamp
    layer = torch.zeros(1, 4, 40, 40, device=DEV)
    spr = np.zeros((10, 10, 4), np.uint8)
    spr[..., 1] = 255
    spr[..., 3] = 255
    stamp(layer, spr, 20, 20, DEV)
    assert layer[0, 3, 20, 20] == 1.0 and layer[0, 1, 20, 20] == 1.0 and layer[0, 3, 0, 0] == 0.0
