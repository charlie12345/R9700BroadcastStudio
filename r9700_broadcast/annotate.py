"""Draw on the stream: pen, highlighter, arrows, shapes, text, numbered steps, laser pointer,
spotlight and magnifier, drawn in the preview and composited into every output.

Strokes live in normalised output coordinates (0..1), so they survive preset/resolution changes
and work over any background (a shared screen most of all). The GUI thread rasterises with
QPainter into premultiplied BGRA layers; the GPU thread uploads a layer only when its version
changes:

    strokes layer  pen / highlighter / shapes / text / steps  (under you, or over you)
    live layer     laser trail and spotlight                  (always on top)
    magnifier      a GPU loupe at the pointer (Compositor), not rasterised here
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPainterPathStroker,
                         QPen, QPolygonF, QRadialGradient)

# (key, button label, tooltip with shortcut)
TOOLS = [
    ("move", "✋", "Move the bubble / stop drawing (Esc)"),
    ("pen", "✏️", "Pen (P)"),
    ("highlighter", "🖍️", "Highlighter - see-through marker (H)"),
    ("arrow", "➜", "Arrow (A)"),
    ("line", "╱", "Straight line (I)"),
    ("rect", "▭", "Box (R) - hold Shift for a square"),
    ("ellipse", "◯", "Circle / oval (O) - hold Shift for a circle"),
    ("text", "T", "Text (T) - click, type, Enter"),
    ("step", "①", "Numbered steps (N) - click to drop 1, 2, 3…"),
    ("laser", "🔴", "Laser pointer (L) - glowing dot with a fading trail"),
    ("spotlight", "🔦", "Spotlight (S) - dims everything except around the pointer"),
    ("magnifier", "🔍", "Magnifier (M) - zooms in under the pointer"),
    ("eraser", "⌫", "Eraser (X) - click or drag over a drawing"),
]
SHORTCUTS = {"Esc": "move", "P": "pen", "H": "highlighter", "A": "arrow", "I": "line", "R": "rect",
             "O": "ellipse", "T": "text", "N": "step", "L": "laser", "S": "spotlight", "M": "magnifier",
             "X": "eraser"}
COLORS = [("#ff2d3d", "Red"), ("#ffd400", "Yellow"), ("#2ee85a", "Green"), ("#2d8cff", "Blue"),
          ("#ffffff", "White"), ("#111111", "Black")]
WIDTHS = {"Thin": 3.0, "Medium": 6.0, "Thick": 12.0}     # px at 1080p
FADES = {"Keep": 0.0, "Fade 3 s": 3.0, "Fade 6 s": 6.0, "Fade 12 s": 12.0}
FADE_OUT = 0.6         # seconds a fading drawing takes to disappear
LASER_TRAIL = 0.45     # seconds of laser trail
DRAW_TOOLS = {"pen", "highlighter", "arrow", "line", "rect", "ellipse"}


class Stroke:
    __slots__ = ("tool", "color", "width", "pts", "text", "num", "t_end", "constrain")

    def __init__(self, tool, color, width, pts, text="", num=0, constrain=False):
        self.tool, self.color, self.width, self.pts = tool, QColor(color), width, pts
        self.text, self.num, self.constrain = text, num, constrain
        self.t_end = None      # set when finished; fading starts from here


# ---------------------------------------------------------------------------------------------
# shape snapping: a rough hand-drawn circle/box/line becomes a clean one
# ---------------------------------------------------------------------------------------------

def snap_shape(pts, aspect=16 / 9):
    """Return ("ellipse"|"rect"|"line", [p0, p1]) for a stroke that is clearly one of those
    shapes, else None. pts are normalised; aspect = W/H so distances are measured in pixels."""
    if len(pts) < 6:
        return None
    p = np.asarray(pts, np.float64) * (aspect, 1.0)
    span = np.ptp(p, axis=0)
    size = float(max(span.max(), 1e-6))
    start, end = p[0], p[-1]
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    if size < 0.02:
        return None
    closed = np.linalg.norm(end - start) < 0.25 * size and length > 2.2 * size
    if not closed:
        d = end - start
        L = float(np.linalg.norm(d))
        if L < 0.6 * length or L < 1e-6:
            return None
        n = np.array([-d[1], d[0]]) / L
        if np.abs((p - start) @ n).max() < 0.06 * L:
            return "line", [tuple(pts[0]), tuple(pts[-1])]
        return None
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 1e-6), max((y1 - y0) / 2, 1e-6)
    ell = np.abs(np.hypot((p[:, 0] - cx) / rx, (p[:, 1] - cy) / ry) - 1).mean()
    dx = np.minimum(np.abs(p[:, 0] - x0), np.abs(p[:, 0] - x1)) / (2 * rx)
    dy = np.minimum(np.abs(p[:, 1] - y0), np.abs(p[:, 1] - y1)) / (2 * ry)
    rect = np.minimum(dx, dy).mean()
    box = [(x0 / aspect, y0), (x1 / aspect, y1)]
    if rect < 0.045 and rect < ell * 0.8:
        return "rect", box
    if ell < 0.16:
        return "ellipse", box
    return None


# ---------------------------------------------------------------------------------------------

class Annotator:
    """All drawing state. Mouse methods take normalised (x, y). Thread-safe: the GUI thread edits
    and renders, the GPU thread only reads `strokes_layer` / `live_layer` / `magnifier`."""

    def __init__(self):
        self.tool = "move"
        self.color = COLORS[0][0]
        self.width = WIDTHS["Medium"]
        self.fade = 0.0
        self.snap = True
        self.aspect = 16 / 9
        self.strokes: list[Stroke] = []
        self.redo_stack: list[Stroke] = []
        self.cur: Stroke | None = None
        self.next_step = 1
        self.pointer = None                 # (x, y) while the pointer is over the preview
        self.trail: deque = deque()          # laser (x, y, t)
        self._lock = threading.Lock()
        self._dirty = True
        self._live_on = False
        self.strokes_layer = (0, None)       # (version, premultiplied BGRA ndarray | None)
        self.live_layer = (0, None)
        self.size = (1920, 1080)            # output size, set by the compositor side

    # ---- state ---------------------------------------------------------------------------------
    @property
    def magnifier(self):
        return self.pointer if self.tool == "magnifier" else None

    def set_tool(self, tool):
        with self._lock:
            if self.cur is not None:
                self._finish()
            self.tool = tool
            self.trail.clear()
            self._dirty = True

    def undo(self):
        with self._lock:
            if self.strokes:
                s = self.strokes.pop()
                self.redo_stack.append(s)
                if s.tool == "step":
                    self.next_step = max(1, s.num)
                self._dirty = True

    def redo(self):
        with self._lock:
            if self.redo_stack:
                s = self.redo_stack.pop()
                s.t_end = time.monotonic()
                self.strokes.append(s)
                if s.tool == "step":
                    self.next_step = s.num + 1
                self._dirty = True

    def clear(self):
        with self._lock:
            self.redo_stack = list(reversed(self.strokes)) + self.redo_stack if self.strokes else self.redo_stack
            self.strokes = []
            self.cur = None
            self.next_step = 1
            self._dirty = True

    # ---- input ---------------------------------------------------------------------------------
    def press(self, x, y, constrain=False):
        with self._lock:
            self.pointer = (x, y)
            t = self.tool
            if t in DRAW_TOOLS:
                w = self.width * (3.5 if t == "highlighter" else 1.0)
                self.cur = Stroke(t, self.color, max(w, 14.0) if t == "highlighter" else w,
                                  [(x, y)], constrain=constrain)
                self.redo_stack.clear()
            elif t == "step":
                s = Stroke("step", self.color, self.width, [(x, y)], num=self.next_step)
                s.t_end = time.monotonic()
                self.strokes.append(s)
                self.next_step += 1
                self.redo_stack.clear()
            elif t == "eraser":
                self._erase(x, y)
            elif t == "laser":
                self.trail.append((x, y, time.monotonic()))
            self._dirty = True

    def move(self, x, y, pressed, constrain=False):
        with self._lock:
            self.pointer = (x, y)
            if self.cur is not None and pressed:
                self.cur.constrain = constrain
                if self.cur.tool in ("pen", "highlighter"):
                    lx, ly = self.cur.pts[-1]
                    if abs(x - lx) + abs(y - ly) > 0.0008:
                        self.cur.pts.append((x, y))
                else:
                    self.cur.pts[1:] = [(x, y)]
            elif self.tool == "eraser" and pressed:
                self._erase(x, y)
            elif self.tool == "laser":
                self.trail.append((x, y, time.monotonic()))
            self._dirty = True

    def release(self, x, y):
        with self._lock:
            if self.cur is not None:
                if self.cur.tool not in ("pen", "highlighter"):
                    self.cur.pts[1:] = [(x, y)]
                self._finish()
            self._dirty = True

    def leave(self):
        with self._lock:
            self.pointer = None
            self._dirty = True

    def add_text(self, x, y, text):
        if not text.strip():
            return
        with self._lock:
            s = Stroke("text", self.color, self.width, [(x, y)], text=text.strip())
            s.t_end = time.monotonic()
            self.strokes.append(s)
            self.redo_stack.clear()
            self._dirty = True

    def _finish(self):
        s, self.cur = self.cur, None
        if s.tool not in ("pen", "highlighter"):
            (x0, y0), (x1, y1) = s.pts[0], s.pts[-1]
            if abs(x1 - x0) * self.aspect + abs(y1 - y0) < 0.004:
                return                                 # a click with a shape tool: nothing
        if self.snap and s.tool in ("pen", "highlighter"):
            snapped = snap_shape(s.pts, self.aspect)
            if snapped is not None:
                kind, pts = snapped
                if s.tool == "highlighter":            # a highlighter only straightens
                    if kind == "line":
                        s.pts = pts
                else:
                    s.tool, s.pts = kind, pts
        s.t_end = time.monotonic()
        self.strokes.append(s)

    def _erase(self, x, y):
        r = 0.02
        keep = []
        for s in self.strokes:
            pts = s.pts
            if s.tool in ("rect", "ellipse", "line", "arrow") and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                pts = [(x0 + (x1 - x0) * k / 16, y0 + (y1 - y0) * j / 16)
                       for k in range(17) for j in (0, 16)] + \
                      [(x0 + (x1 - x0) * k / 16, y0 + (y1 - y0) * k / 16) for k in range(17)] + \
                      [(x0 + (x1 - x0) * j / 16, y0 + (y1 - y0) * k / 16) for k in range(17) for j in (0, 16)]
            hit = any(abs(px - x) * self.aspect < r and abs(py - y) < r for px, py in pts)
            if not hit:
                keep.append(s)
            else:
                self.redo_stack.append(s)
        self.strokes = keep

    # ---- rendering (GUI thread) ----------------------------------------------------------------
    def busy(self) -> bool:
        """True when a re-render is needed this tick."""
        return self._dirty or bool(self.trail) or (self.fade > 0 and bool(self.strokes)) or self._live_on

    def tick(self, now=None):
        """Re-rasterise what changed. Cheap no-op when nothing is drawn."""
        if not self.busy():
            return
        now = time.monotonic() if now is None else now
        w, h = self.size
        self.aspect = w / h
        with self._lock:
            self._dirty = False
            alive = []
            for s in self.strokes:
                if self.fade > 0 and s.t_end is not None and now - s.t_end > self.fade + FADE_OUT:
                    continue
                alive.append(s)
            self.strokes = alive
            draw = list(alive) + ([self.cur] if self.cur is not None else [])
            while self.trail and now - self.trail[0][2] > LASER_TRAIL:
                self.trail.popleft()
            trail = list(self.trail)
            pointer = self.pointer
            tool = self.tool
        v, _ = self.strokes_layer
        self.strokes_layer = (v + 1, self._render_strokes(draw, w, h, now) if draw else None)
        live = None
        if (tool == "laser" and (trail or pointer)) or (tool == "spotlight" and pointer):
            live = self._render_live(tool, trail, pointer, w, h, now)
        self._live_on = live is not None
        v, _ = self.live_layer
        self.live_layer = (v + 1, live)

    @staticmethod
    def _image(w, h):
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        return img

    @staticmethod
    def _array(img: QImage) -> np.ndarray:
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        return np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()].copy()

    def _render_strokes(self, strokes, w, h, now):
        img = self._image(w, h)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        k = h / 1080.0
        for s in strokes:
            op = 1.0
            if self.fade > 0 and s.t_end is not None:
                age = now - s.t_end - self.fade
                if age > 0:
                    op = max(0.0, 1 - age / FADE_OUT)
            p.setOpacity(op)
            self._draw(p, s, w, h, k)
        p.end()
        return self._array(img)

    def _draw(self, p: QPainter, s: Stroke, w, h, k):
        pts = [QPointF(x * w, y * h) for x, y in s.pts]
        col = QColor(s.color)
        width = s.width * k
        if s.tool == "highlighter":
            col.setAlphaF(0.38)
            pen = QPen(col, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap, Qt.PenJoinStyle.RoundJoin)
        else:
            pen = QPen(col, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        shadow = QPen(QColor(0, 0, 0, 90), width + 3 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if s.tool in ("pen", "highlighter"):
            path = QPainterPath(pts[0])
            if len(pts) == 1:
                path.lineTo(pts[0] + QPointF(0.01, 0))
            elif len(pts) == 2 or s.tool == "highlighter" and len(pts) == 2:
                path.lineTo(pts[1])
            else:      # smooth: quadratic curves through the midpoints
                for a, b in zip(pts[1:-1], pts[2:]):
                    path.quadTo(a, (a + b) / 2)
                path.lineTo(pts[-1])
            if s.tool == "pen":
                p.setPen(shadow)
                p.drawPath(path)
            p.setPen(pen)
            p.drawPath(path)
        elif s.tool in ("line", "arrow", "rect", "ellipse") and len(pts) >= 2:
            a, b = pts[0], pts[-1]
            if s.constrain and s.tool in ("rect", "ellipse"):
                d = max(abs(b.x() - a.x()), abs(b.y() - a.y()))
                b = QPointF(a.x() + math.copysign(d, b.x() - a.x()), a.y() + math.copysign(d, b.y() - a.y()))
            elif s.constrain:     # lines/arrows: snap to 45 degrees
                ang = round(math.atan2(b.y() - a.y(), b.x() - a.x()) / (math.pi / 4)) * (math.pi / 4)
                L = math.hypot(b.x() - a.x(), b.y() - a.y())
                b = QPointF(a.x() + L * math.cos(ang), a.y() + L * math.sin(ang))
            r = QRectF(a, b).normalized()
            for pn in (shadow, pen):
                p.setPen(pn)
                if s.tool == "rect":
                    p.drawRoundedRect(r, 4 * k, 4 * k)
                elif s.tool == "ellipse":
                    p.drawEllipse(r)
                else:
                    end = b
                    if s.tool == "arrow":
                        L = math.hypot(b.x() - a.x(), b.y() - a.y())
                        head = min(max(30 * k, width * 5), L * 0.6) if L > 0 else 0
                        if L > 0:
                            end = b - (b - a) * (head * 0.8 / L)
                    p.drawLine(a, end)
            if s.tool == "arrow":
                L = math.hypot(b.x() - a.x(), b.y() - a.y())
                if L > 1:
                    head = min(max(30 * k, width * 5), L * 0.6)
                    ux, uy = (b.x() - a.x()) / L, (b.y() - a.y()) / L
                    base = QPointF(b.x() - ux * head, b.y() - uy * head)
                    nx, ny = -uy * head * 0.55, ux * head * 0.55
                    tri = QPolygonF([b, base + QPointF(nx, ny), base - QPointF(nx, ny)])
                    p.setPen(QPen(QColor(0, 0, 0, 90), 2 * k))
                    p.setBrush(QBrush(s.color))
                    p.drawPolygon(tri)
                    p.setBrush(Qt.BrushStyle.NoBrush)
        elif s.tool == "text":
            font = QFont("Noto Sans")
            font.setBold(True)
            font.setPixelSize(max(12, int((22 + s.width * 3) * k)))
            path = QPainterPath()
            path.addText(pts[0], font, s.text)
            outline = QPen(QColor(0, 0, 0, 200) if s.color.lightness() > 90 else QColor(255, 255, 255, 220),
                           max(2.0, 4 * k), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                           Qt.PenJoinStyle.RoundJoin)
            p.strokePath(path, outline)
            p.fillPath(path, QBrush(s.color))
        elif s.tool == "step":
            rad = (18 + s.width * 1.5) * k
            c = pts[0]
            p.setPen(QPen(QColor(255, 255, 255), max(2.0, 3 * k)))
            p.setBrush(QBrush(s.color))
            p.drawEllipse(c, rad, rad)
            font = QFont("Noto Sans")
            font.setBold(True)
            font.setPixelSize(int(rad * 1.2))
            p.setFont(font)
            p.setPen(QColor(0, 0, 0) if s.color.lightness() > 150 else QColor(255, 255, 255))
            p.drawText(QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad), Qt.AlignmentFlag.AlignCenter,
                       str(s.num))
            p.setBrush(Qt.BrushStyle.NoBrush)

    def _render_live(self, tool, trail, pointer, w, h, now):
        img = self._image(w, h)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        k = h / 1080.0
        if tool == "spotlight" and pointer:
            p.fillRect(0, 0, w, h, QColor(0, 0, 0, 165))
            c = QPointF(pointer[0] * w, pointer[1] * h)
            rad = 170 * k
            g = QRadialGradient(c, rad)
            g.setColorAt(0.0, QColor(0, 0, 0, 255))
            g.setColorAt(0.82, QColor(0, 0, 0, 255))
            g.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(g))
            p.drawEllipse(c, rad, rad)
        elif tool == "laser":
            col = QColor(self.color)
            pts = [(QPointF(x * w, y * h), t) for x, y, t in trail]
            for (a, ta), (b, tb) in zip(pts, pts[1:]):
                life = max(0.0, 1 - (now - tb) / LASER_TRAIL)
                c2 = QColor(col)
                c2.setAlphaF(0.75 * life)
                p.setPen(QPen(c2, (4 + 8 * life) * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(a, b)
            if pointer:
                c = QPointF(pointer[0] * w, pointer[1] * h)
                rad = 22 * k
                g = QRadialGradient(c, rad)
                core = QColor(255, 255, 255)
                glow = QColor(col)
                glow.setAlpha(150)
                edge = QColor(col)
                edge.setAlpha(0)
                g.setColorAt(0.0, core)
                g.setColorAt(0.25, col)
                g.setColorAt(0.5, glow)
                g.setColorAt(1.0, edge)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(g))
                p.drawEllipse(c, rad, rad)
        p.end()
        return self._array(img)


# ---------------------------------------------------------------------------------------------
# GPU side
# ---------------------------------------------------------------------------------------------

class LayerUpload:
    """Uploads one of the Annotator's layers to the GPU when its version changes.
    Returns (premultiplied rgb (1,3,H,W), alpha (1,1,H,W)) float tensors, or None."""

    def __init__(self, get):
        self.get = get
        self._cache = (None, None)

    def __call__(self, h, w, device):
        import torch
        import torch.nn.functional as F
        ver, arr = self.get()
        if arr is None:
            return None
        if self._cache[0] == (ver, h, w):
            return self._cache[1]
        t = torch.from_numpy(arr).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255   # BGRA
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        val = (t[:, [2, 1, 0]], t[:, 3:4])
        self._cache = ((ver, h, w), val)
        return val
