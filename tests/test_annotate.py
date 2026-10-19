"""Drawing on the stream: shape snapping, undo/redo/fade bookkeeping, rasterising and the GPU
layer order (under you / over you / live layer on top)."""
import math
import time

import numpy as np
import pytest
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _ann(qapp, w=480, h=270):
    from r9700_broadcast.annotate import Annotator
    a = Annotator()
    a.size = (w, h)
    return a


def _drag(a, tool, x0, y0, x1, y1, n=12):
    a.set_tool(tool)
    a.press(x0, y0)
    for i in range(1, n + 1):
        a.move(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, True)
    a.release(x1, y1)


def test_snap_shapes():
    from r9700_broadcast.annotate import snap_shape
    circ = [(0.5 + 0.1 * math.cos(t) / 1.78, 0.5 + 0.1 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 40)]
    assert snap_shape(circ)[0] == "ellipse"
    box = [(0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6), (0.2, 0.6), (0.2, 0.4),
           (0.2, 0.21)]
    box = [p for a, b in zip(box, box[1:]) for p in [(a[0] + (b[0] - a[0]) * k / 5, a[1] + (b[1] - a[1]) * k / 5)
                                                     for k in range(5)]]
    assert snap_shape(box)[0] == "rect"
    line = [(0.1 + 0.05 * i, 0.3 + 0.002 * (i % 2)) for i in range(10)]
    assert snap_shape(line)[0] == "line"
    squiggle = [(0.1 + 0.03 * i, 0.3 + 0.08 * math.sin(i)) for i in range(20)]
    assert snap_shape(squiggle) is None


def test_undo_redo_clear_and_steps(qapp):
    a = _ann(qapp)
    _drag(a, "arrow", 0.1, 0.1, 0.5, 0.5)
    a.set_tool("step")
    for x in (0.2, 0.3):
        a.press(x, 0.8)
        a.release(x, 0.8)
    assert [s.tool for s in a.strokes] == ["arrow", "step", "step"] and a.strokes[-1].num == 2
    a.undo()
    assert a.next_step == 2 and len(a.strokes) == 2
    a.redo()
    assert a.strokes[-1].num == 2
    a.clear()
    assert a.strokes == [] and a.next_step == 1
    a.set_tool("rect")
    a.press(0.3, 0.3)
    a.release(0.3, 0.3)       # a click with a shape tool draws nothing
    assert a.strokes == []


def test_eraser_removes_hit_stroke(qapp):
    a = _ann(qapp)
    _drag(a, "rect", 0.1, 0.1, 0.3, 0.3)
    _drag(a, "line", 0.6, 0.6, 0.9, 0.9)
    a.set_tool("eraser")
    a.press(0.75, 0.75)
    a.release(0.75, 0.75)
    assert [s.tool for s in a.strokes] == ["rect"]


def test_render_and_fade(qapp):
    a = _ann(qapp)
    _drag(a, "pen", 0.1, 0.5, 0.9, 0.5)
    a.tick()
    v, img = a.strokes_layer
    assert img.shape == (270, 480, 4) and img[135, 240, 3] > 200 and img[20, 20, 3] == 0
    a.fade = 0.05
    time.sleep(0.05 + 0.6 + 0.05)
    a.tick()
    assert a.strokes == [] and a.strokes_layer[1] is None


def test_live_layers(qapp):
    a = _ann(qapp)
    a.set_tool("spotlight")
    a.move(0.5, 0.5, False)
    a.tick()
    live = a.live_layer[1]
    assert live[135, 240, 3] == 0 and live[5, 5, 3] > 100     # hole at the pointer, dark elsewhere
    a.set_tool("magnifier")
    a.move(0.25, 0.25, False)
    assert a.magnifier == (0.25, 0.25)
    a.tick()
    assert a.live_layer[1] is None


def test_compositor_layer_order(qapp):
    from r9700_broadcast.effects import Compositor, Settings
    a = _ann(qapp)
    _drag(a, "rect", 0.0, 0.0, 1.0, 1.0)
    a.strokes[-1].width = 400            # a thick box covering the person's corner too
    a.tick()
    H, W = 270, 480
    src = torch.zeros(1, 3, H, W, device=DEV)
    src[:, 1] = 1.0                       # green person
    al = torch.zeros(1, 1, H, W, device=DEV)
    al[..., :, :] = 1.0                   # person everywhere
    c = Compositor()
    c.annotations = a
    out = c(src, al, src, Settings(mode="replace", light_wrap=0))
    assert out[0, 1, 135, 20] > 0.9        # under you: hidden by the person
    out = c(src, al, src, Settings(mode="replace", light_wrap=0, draw_over_me=True))
    assert out[0, 0, 135, 20] > 0.9 and out[0, 1, 135, 20] < 0.2   # over you: red box on top


def test_magnify_zooms():
    from r9700_broadcast.effects import magnify
    img = torch.zeros(1, 3, 270, 480, device=DEV)
    img[..., 130:140, 235:245] = 1.0      # 10 px white square at the centre
    out = magnify(img, 0.5, 0.5, zoom=2.0)
    assert out[0, 0, 135 + 8, 240].item() == pytest.approx(1.0, abs=0.05)   # now ~20 px wide
    assert img[0, 0, 135 + 8, 240].item() == 0.0
