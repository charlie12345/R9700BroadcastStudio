"""Snapshot + thumbnail maker.

Snapshot saves the full-resolution output frame (PNG; keeps transparency in the Transparent mode).
The thumbnail maker turns it into a 1280x720 YouTube/Twitch thumbnail: cover-crop to 16:9, optional
punch (contrast/saturation), and a bold outlined title, exported as JPEG (< 2 MB, YouTube's limit)
or PNG.
"""
from __future__ import annotations

import os
import time

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QColor, QFont, QFontMetricsF, QGuiApplication, QImage, QPainter, QPainterPath,
                         QPen, QPixmap)
from PyQt6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
                             QPlainTextEdit, QPushButton, QSlider, QVBoxLayout)

PICTURES = os.path.join(os.path.expanduser("~/Pictures"), "R9700 Broadcast")
THUMB_SIZE = (1280, 720)


def bgra_to_qimage(bgra) -> QImage:
    h, w = bgra.shape[:2]
    return QImage(bgra.data, w, h, 4 * w, QImage.Format.Format_ARGB32).copy()


def save_snapshot(bgra, has_alpha: bool) -> tuple[str, QImage]:
    os.makedirs(PICTURES, exist_ok=True)
    img = bgra_to_qimage(bgra)
    if not has_alpha:
        img = img.convertToFormat(QImage.Format.Format_RGB32)
    path = os.path.join(PICTURES, time.strftime("Snapshot %Y-%m-%d %H-%M-%S.png"))
    img.save(path)
    return path, img


def cover(img: QImage, size=THUMB_SIZE, focus_y: float = 0.4) -> QImage:
    """Scale to fill `size` and crop (keeps the upper-middle, where faces are)."""
    W, H = size
    s = max(W / img.width(), H / img.height())
    scaled = img.scaled(int(img.width() * s + 0.5), int(img.height() * s + 0.5),
                        Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
    x = (scaled.width() - W) // 2
    y = int((scaled.height() - H) * focus_y)
    return scaled.copy(x, y, W, H)


def punch(img: QImage, amount: float) -> QImage:
    """Thumbnail 'pop': contrast + saturation boost (amount 0..1)."""
    if amount <= 0:
        return img
    import numpy as np
    a = img.convertToFormat(QImage.Format.Format_RGB32)
    w, h = a.width(), a.height()
    arr = np.frombuffer(a.bits().asarray(h * a.bytesPerLine()), np.uint8).reshape(h, a.bytesPerLine() // 4, 4)[:, :w, :3]
    f = arr.astype(np.float32) / 255
    f = (f - 0.5) * (1 + 0.35 * amount) + 0.5                     # contrast
    g = f.mean(2, keepdims=True)
    f = g + (f - g) * (1 + 0.6 * amount)                          # saturation
    out = np.dstack([(f.clip(0, 1) * 255 + 0.5).astype(np.uint8), np.full((h, w, 1), 255, np.uint8)])
    return QImage(out.tobytes(), w, h, 4 * w, QImage.Format.Format_RGB32).copy()


def render_thumbnail(img: QImage, title: str = "", size_pct: float = 12.0, color=QColor("#ffffff"),
                     outline=QColor("#000000"), position: str = "bottom", pop: float = 0.5) -> QImage:
    out = punch(cover(img), pop)
    if not title.strip():
        return out
    W, H = out.width(), out.height()
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Impact")
    if not QFontMetricsF(font).horizontalAdvance("A"):
        font = QFont()
    font.setWeight(QFont.Weight.Black)
    font.setPixelSize(max(10, int(H * size_pct / 100)))
    fm = QFontMetricsF(font)
    lines = [ln for ln in title.upper().splitlines() if ln.strip()] or [""]
    # shrink to fit the width
    widest = max(fm.horizontalAdvance(ln) for ln in lines)
    if widest > W * 0.92:
        font.setPixelSize(max(10, int(font.pixelSize() * W * 0.92 / widest)))
        fm = QFontMetricsF(font)
    lh = fm.height() * 0.95
    block = lh * len(lines)
    margin = H * 0.06
    top = {"top": margin, "middle": (H - block) / 2}.get(position, H - margin - block)
    path = QPainterPath()
    for i, ln in enumerate(lines):
        x = (W - fm.horizontalAdvance(ln)) / 2
        path.addText(QPointF(x, top + i * lh + fm.ascent()), font, ln)
    if outline.alpha():
        pen = QPen(outline, max(2.0, font.pixelSize() * 0.12))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.strokePath(path, pen)
    p.fillPath(path, color)
    p.end()
    return out


class ThumbnailDialog(QDialog):
    def __init__(self, img: QImage, snapshot_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thumbnail maker")
        self.img = img
        self.color = QColor("#ffffff")
        self.outline = QColor("#000000")
        v = QVBoxLayout(self)
        self.view = QLabel()
        self.view.setMinimumSize(640, 360)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.view, 1)
        f = QFormLayout()
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Title (optional) - new line for a second line")
        self.text.setFixedHeight(60)
        self.text.textChanged.connect(self.update_view)
        f.addRow("Title", self.text)
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(5, 25)
        self.size.setValue(12)
        self.size.valueChanged.connect(self.update_view)
        f.addRow("Text size", self.size)
        self.pos = QComboBox()
        for k in ("bottom", "top", "middle"):
            self.pos.addItem(k.capitalize(), k)
        self.pos.currentIndexChanged.connect(self.update_view)
        f.addRow("Position", self.pos)
        row = QHBoxLayout()
        cb = QPushButton("Text colour")
        cb.clicked.connect(lambda: self._pick("color"))
        ob = QPushButton("Outline colour")
        ob.clicked.connect(lambda: self._pick("outline"))
        self.outline_on = QCheckBox("Outline")
        self.outline_on.setChecked(True)
        self.outline_on.toggled.connect(self.update_view)
        row.addWidget(cb)
        row.addWidget(ob)
        row.addWidget(self.outline_on)
        f.addRow("", row)
        self.pop = QSlider(Qt.Orientation.Horizontal)
        self.pop.setRange(0, 100)
        self.pop.setValue(40)
        self.pop.valueChanged.connect(self.update_view)
        f.addRow("Pop (contrast / colour)", self.pop)
        v.addLayout(f)
        self.info = QLabel(f"Snapshot saved: {snapshot_path}")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        row = QHBoxLayout()
        b = QPushButton("Save thumbnail (JPG 1280×720)")
        b.clicked.connect(lambda: self.save("jpg"))
        row.addWidget(b)
        b = QPushButton("Save PNG")
        b.clicked.connect(lambda: self.save("png"))
        row.addWidget(b)
        b = QPushButton("Copy")
        b.clicked.connect(lambda: QGuiApplication.clipboard().setImage(self.render()))
        row.addWidget(b)
        b = QPushButton("Close")
        b.clicked.connect(self.accept)
        row.addWidget(b)
        v.addLayout(row)
        self.resize(900, 720)
        self.update_view()

    def _pick(self, which):
        c = QColorDialog.getColor(getattr(self, which), self)
        if c.isValid():
            setattr(self, which, c)
            self.update_view()

    def render(self) -> QImage:
        return render_thumbnail(self.img, self.text.toPlainText(), self.size.value(), self.color,
                                self.outline if self.outline_on.isChecked() else QColor(0, 0, 0, 0),
                                self.pos.currentData(), self.pop.value() / 100)

    def update_view(self):
        t = self.render()
        self.view.setPixmap(QPixmap.fromImage(t).scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self.update_view()

    def save(self, ext):
        os.makedirs(PICTURES, exist_ok=True)
        path = os.path.join(PICTURES, time.strftime(f"Thumbnail %Y-%m-%d %H-%M-%S.{ext}"))
        img = self.render()
        if ext == "jpg":
            q = 92
            while True:                              # YouTube's thumbnail limit is 2 MB
                img.save(path, "JPG", q)
                if os.path.getsize(path) < 2_000_000 or q <= 60:
                    break
                q -= 8
        else:
            img.save(path, "PNG")
        self.info.setText(f"Saved: {path}")
