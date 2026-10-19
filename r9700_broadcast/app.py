"""R9700 Broadcast Studio - Qt front end."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRect, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import (QAction, QColor, QFont, QGuiApplication, QIcon, QImage, QKeySequence,
                         QPainter, QPen, QPixmap, QShortcut)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog,
                             QDialogButtonBox,
                             QFileDialog,
                             QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QListView, QListWidget, QListWidgetItem, QMainWindow, QMenu, QPushButton,
                             QScrollArea, QSizePolicy, QSlider, QTabWidget, QVBoxLayout, QWidget)

from . import annotate as AN
from . import backgrounds as B
from . import screencap
from . import uvc
from .effects import (CAMEO_FILLS, CAMEO_RINGS, CAMEO_SHAPES, Settings, cameo_center,
                      cameo_from_point)
from .matting import ENGINES
from .pipeline import PRESET_BY_KEY, PRESETS, LivePipeline

CONFIG = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                      "r9700-broadcast", "config.json")
RECORD_DIR = os.path.expanduser("~/Videos/R9700 Broadcast")
ACCENT = "#e0263c"

STYLE = f"""
QWidget {{ background: #121317; color: #e8e8ec; font-size: 13px; }}
QFrame#bar, QFrame#strip {{ background: #181a20; border-radius: 10px; }}
QGroupBox {{ border: 1px solid #2a2d36; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT}; font-weight: 600; }}
QPushButton {{ background: #22252d; border: 1px solid #32353f; border-radius: 6px; padding: 6px 10px; }}
QPushButton:hover {{ background: #2b2e38; }}
QPushButton:checked {{ background: {ACCENT}; border-color: #ff5566; color: white; }}
QPushButton#live {{ font-weight: 600; padding: 7px 14px; }}
QComboBox {{ background: #22252d; border: 1px solid #32353f; border-radius: 6px; padding: 4px 8px; min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: #32353f; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: #a01c2c; border-radius: 2px; }}
QTabWidget::pane {{ border: 1px solid #2a2d36; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: #181a20; padding: 7px 8px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: #262932; color: white; }}
QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{ border-radius: 8px; padding: 4px; color: #c9cad3; }}
QListWidget::item:selected {{ background: #3a1d24; border: 2px solid {ACCENT}; color: white; }}
QListWidget::item:hover {{ background: #22252d; }}
QLabel#hint {{ color: #8b8d9a; font-size: 11px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #555a66; border-radius: 4px; background: #1d1f26; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: #ff5566; }}
QLabel#status {{ color: #a3a5b3; font-family: monospace; }}
QScrollArea {{ border: none; }}
"""

SPECIAL_TILES = [
    ("off", "Camera", "No effect"),
    ("blur", "Blur", "Blur the real background"),
    ("screen", "Share screen", "Your whole screen as the background - you stay on top of it"),
    ("window", "Share window", "One window (a browser, an editor…) as the background"),
    ("green", "Green screen", "Chroma green for OBS / vMix keying"),
    ("blue", "Blue screen", "Chroma blue"),
    ("transparent", "Transparent", "Real alpha via the PipeWire camera (OBS Studio)"),
    ("remove", "Solid colour", "Flat colour background"),
    ("matte", "Matte", "Black/white alpha matte"),
]


def _icon(pm: QPixmap) -> QIcon:
    """Same pixmap for normal + selected, so Qt doesn't tint the selected thumbnail."""
    ic = QIcon()
    ic.addPixmap(pm, QIcon.Mode.Normal)
    ic.addPixmap(pm, QIcon.Mode.Selected)
    return ic


def _hint(text):
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


def _tile_pixmap(kind: str, w=192, h=108, color=(0.12, 0.12, 0.14)) -> QPixmap:
    pm = QPixmap(w, h)
    pm.fill(QColor("#2a2d36"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if kind == "green":
        pm.fill(QColor(0, 177, 64))
    elif kind == "blue":
        pm.fill(QColor(0, 71, 187))
    elif kind == "remove":
        pm.fill(QColor.fromRgbF(*color))
    elif kind == "transparent":
        for y in range(0, h, 12):
            for x in range(0, w, 12):
                p.fillRect(x, y, 12, 12, QColor("#cfd0d6") if (x + y) // 12 % 2 else QColor("#8d8f99"))
    elif kind == "matte":
        pm.fill(QColor("black"))
    elif kind in ("screen", "window"):
        pm.fill(QColor("#e9ebf0"))
        p.fillRect(0, 0, w, 14, QColor("#343843"))
        p.setPen(QPen(QColor("#9da2b0"), 4))
        for i, ln in enumerate((0.7, 0.55, 0.8, 0.45, 0.62)):
            p.drawLine(12, 26 + i * 15, 12 + int((w - 24) * ln), 26 + i * 15)
        p.setPen(QPen(QColor(ACCENT), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(18, 34, 70, 26)                    # an annotation circle
        if kind == "window":
            p.setPen(QPen(QColor("#2d8cff"), 4))
            p.drawRect(6, 4, w - 12, h - 8)
    elif kind == "blur":
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(12):
            p.setBrush(QColor.fromHsvF((0.05 + i * 0.09) % 1, 0.35, 0.6, 0.8))
            p.drawEllipse((i * 41) % w - 12, (i * 29) % h - 12, 54, 54)
        p.fillRect(0, 0, w, h, QColor(20, 20, 26, 110))
    if kind != "off":
        p.setBrush(QColor(255, 255, 255, 215))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(w // 2 - 16, 26, 32, 32)
        p.drawRoundedRect(w // 2 - 36, 62, 72, 50, 24, 24)
    else:
        p.setPen(QColor("#9a9cab"))
        p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "camera")
    p.end()
    return pm


class _ThumbJob(QRunnable):
    class Sig(QObject):
        done = pyqtSignal(str, QImage)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.sig = _ThumbJob.Sig()

    def run(self):
        t = B.thumbnail(self.path)
        if t is None:
            return
        h, w = t.shape[:2]
        self.sig.done.emit(self.path, QImage(t.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


# ---------------------------------------------------------------------------------------------
class PreviewView(QWidget):
    """Draws the output; transparent output is shown over a checkerboard."""

    def __init__(self):
        super().__init__()
        self.img: QImage | None = None
        self.alpha = False
        self.message = "Starting…"
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._checker = QPixmap(32, 32)
        self._checker.fill(QColor("#8d8f99"))
        cp = QPainter(self._checker)
        cp.fillRect(0, 0, 16, 16, QColor("#cfd0d6"))
        cp.fillRect(16, 16, 16, 16, QColor("#cfd0d6"))
        cp.end()

        self.settings: Settings | None = None   # set by Main: the cameo bubble is dragged here
        self.ann: AN.Annotator | None = None     # set by Main: drawing tools
        self._drawing = False
        self._text_edit = None
        self.on_cameo_moved = None
        self.on_cameo_resized = None
        self._r = QRect()
        self._drag = None                        # grab offset (image px) while dragging the bubble
        self._hover = False
        self.setMouseTracking(True)

    def set_frame(self, img: QImage, alpha: bool):
        self.img, self.alpha, self.message = img, alpha, ""
        self.update()

    # -- cameo bubble: drag to move, scroll to resize ---------------------------------------------
    def _to_img(self, pos):
        if self.img is None or self._r.width() <= 0:
            return None
        k = self.img.width() / self._r.width()
        return (pos.x() - self._r.x()) * k, (pos.y() - self._r.y()) * k

    def _cameo_hit(self, pt):
        s = self.settings
        if s is None or s.layout != "cameo" or self.img is None or pt is None:
            return False
        cx, cy, D = cameo_center(s, self.img.height(), self.img.width())
        r = D / 2 * 1.08
        if s.cameo_shape == "circle":
            return (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2 <= r * r
        return abs(pt[0] - cx) <= r and abs(pt[1] - cy) <= r

    def _norm(self, pos):
        pt = self._to_img(pos)
        if pt is None:
            return None
        return (min(max(pt[0] / self.img.width(), 0.0), 1.0), min(max(pt[1] / self.img.height(), 0.0), 1.0))

    def _draw_tool(self):
        return self.ann is not None and self.ann.tool != "move"

    def _cursor_for_tool(self):
        t = self.ann.tool if self.ann else "move"
        if t in ("laser", "spotlight", "magnifier"):
            return Qt.CursorShape.BlankCursor
        if t == "text":
            return Qt.CursorShape.IBeamCursor
        if t == "eraser":
            return Qt.CursorShape.PointingHandCursor
        return Qt.CursorShape.CrossCursor if t != "move" else Qt.CursorShape.ArrowCursor

    def refresh_cursor(self):
        self.setCursor(self._cursor_for_tool())

    def _start_text(self, pos, n):
        from PyQt6.QtWidgets import QLineEdit
        if self._text_edit is not None:
            self._text_edit.deleteLater()
        ed = QLineEdit(self)
        ed.setPlaceholderText("Type, then Enter")
        ed.setStyleSheet(f"background: rgba(0,0,0,170); color: {self.ann.color}; font-size: 18px; "
                         f"font-weight: bold; border: 2px dashed {ACCENT}; padding: 2px 6px;")
        ed.setMinimumWidth(240)
        ed.move(int(pos.x()), int(pos.y()) - 20)
        ed.show()
        ed.setFocus()

        def done():
            if self._text_edit is ed:
                self.ann.add_text(n[0], n[1] + 0.012, ed.text())
                self._text_edit = None
                ed.deleteLater()
        ed.returnPressed.connect(done)
        ed.editingFinished.connect(done)
        self._text_edit = ed

    def leaveEvent(self, e):
        if self.ann is not None:
            self.ann.leave()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        pt = self._to_img(e.position())
        if self._draw_tool() and e.button() == Qt.MouseButton.LeftButton:
            n = self._norm(e.position())
            if n is None:
                return
            if self.ann.tool == "text":
                self._start_text(e.position(), n)
                return
            self._drawing = True
            self.ann.press(*n, constrain=bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            return
        if self._draw_tool() and e.button() == Qt.MouseButton.RightButton:
            self.ann.undo()          # right-click = undo the last drawing
            return
        if e.button() == Qt.MouseButton.LeftButton and self._cameo_hit(pt):
            cx, cy, _ = cameo_center(self.settings, self.img.height(), self.img.width())
            self._drag = (pt[0] - cx, pt[1] - cy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, e):
        if self._draw_tool():
            n = self._norm(e.position())
            if n is not None:
                self.ann.move(*n, pressed=self._drawing,
                              constrain=bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            return
        pt = self._to_img(e.position())
        if self._drag is not None and pt is not None:
            s = self.settings
            s.cameo_x, s.cameo_y = cameo_from_point(s, self.img.height(), self.img.width(),
                                                    pt[0] - self._drag[0], pt[1] - self._drag[1])
            self.update()
            return
        hover = self._cameo_hit(pt)
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover else Qt.CursorShape.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drawing:
            self._drawing = False
            n = self._norm(e.position())
            if n is not None:
                self.ann.release(*n)
            return
        if self._drag is not None:
            self._drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._hover else Qt.CursorShape.ArrowCursor)
            if self.on_cameo_moved:
                self.on_cameo_moved()
            self.update()

    def wheelEvent(self, e):
        if self._cameo_hit(self._to_img(e.position())) and self.on_cameo_resized:
            steps = e.angleDelta().y() / 120
            self.on_cameo_resized(min(0.9, max(0.12, self.settings.cameo_size * 1.06 ** steps)))
            e.accept()
            return
        super().wheelEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0b0e"))
        if self.img is None:
            p.setPen(QColor("#8b8d9a"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        sz = self.img.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        r = QRect((self.width() - sz.width()) // 2, (self.height() - sz.height()) // 2, sz.width(), sz.height())
        self._r = r
        if self.alpha:
            p.drawTiledPixmap(r, self._checker)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(r, self.img)
        s = self.settings
        if s is not None and s.layout == "cameo" and not self._draw_tool() and (self._hover or self._drag is not None):
            # outline where the bubble is (drawn at the live drag position, not the last frame's)
            k = r.width() / self.img.width()
            cx, cy, D = cameo_center(s, self.img.height(), self.img.width())
            box = QRectF(r.x() + (cx - D / 2) * k, r.y() + (cy - D / 2) * k, D * k, D * k)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(ACCENT), 2, Qt.PenStyle.DashLine))
            if s.cameo_shape == "circle":
                p.drawEllipse(box)
            else:
                p.drawRect(box)


class RegionDialog(QDialog):
    """Drag a box over a still of the shared screen; region = (x0, y0, x1, y1) in 0..1."""

    def __init__(self, parent, bgrx: np.ndarray, region=None):
        super().__init__(parent)
        self.setWindowTitle("Crop the screen share to a region")
        h, w = bgrx.shape[:2]
        self.img = QImage(bgrx.data, w, h, 4 * w, QImage.Format.Format_RGB32).copy()
        self.region = tuple(region) if region else None
        self._a = None
        v = QVBoxLayout(self)
        v.addWidget(_hint("Drag a box over the part to show. Only that part goes behind you, as large as "
                          "it fits."))
        self.canvas = QWidget()
        self.canvas.setMinimumSize(960, 540)
        self.canvas.paintEvent = self._paint
        self.canvas.mousePressEvent = self._press
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = lambda e: None
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        v.addWidget(self.canvas, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.resize(1100, 700)

    def _rect(self):
        sz = self.img.size().scaled(self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect((self.canvas.width() - sz.width()) // 2, (self.canvas.height() - sz.height()) // 2,
                     sz.width(), sz.height())

    def _n(self, pos):
        r = self._rect()
        return (min(max((pos.x() - r.x()) / r.width(), 0.0), 1.0),
                min(max((pos.y() - r.y()) / r.height(), 0.0), 1.0))

    def _press(self, e):
        self._a = self._n(e.position())
        self.region = None

    def _move(self, e):
        if self._a is None:
            return
        b = self._n(e.position())
        x0, x1 = sorted((self._a[0], b[0]))
        y0, y1 = sorted((self._a[1], b[1]))
        self.region = (x0, y0, x1, y1) if (x1 - x0) > 0.02 and (y1 - y0) > 0.02 else None
        self.canvas.update()

    def _paint(self, _):
        p = QPainter(self.canvas)
        p.fillRect(self.canvas.rect(), QColor("#0b0b0e"))
        r = self._rect()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(r, self.img)
        if self.region:
            x0, y0, x1, y1 = self.region
            box = QRectF(r.x() + x0 * r.width(), r.y() + y0 * r.height(),
                         (x1 - x0) * r.width(), (y1 - y0) * r.height())
            p.fillRect(QRectF(r), QColor(0, 0, 0, 120))
            p.drawImage(box, self.img, QRectF(x0 * self.img.width(), y0 * self.img.height(),
                                             (x1 - x0) * self.img.width(), (y1 - y0) * self.img.height()))
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawRect(box)


class Bridge(QObject):
    preview = pyqtSignal(QImage, bool)
    status = pyqtSignal(str)
    lock = pyqtSignal(str)
    snapshot = pyqtSignal(object, bool)
    error = pyqtSignal(str)
    uncheck = pyqtSignal(str)
    screen = pyqtSignal(object, str, str)     # (ScreenShare | None, kind, error)
    screen_ended = pyqtSignal(str)


# ---------------------------------------------------------------------------------------------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        import torch
        self.setWindowTitle("R9700 Broadcast Studio")
        self.resize(1560, 980)
        self.setAcceptDrops(True)
        self.cfg = self._load_config()
        self.settings = Settings()
        for k, v in self.cfg.get("settings", {}).items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, tuple(v) if k == "bg_color" else v)
        if self.cfg.get("config_version", 1) < 2:
            # v2: headset/chair fixes are on by default; older configs saved them as off
            self.settings.keep_accessories = True
            self.settings.subject_lock = True
            self.cfg["config_version"] = 2
        self.library = B.Library()
        self.n_gpus = torch.cuda.device_count()   # a GPU may be hidden (e.g. passed through to a VM)
        self.gpus = [g for g in (self.cfg.get("gpus") or []) if int(g.split(":")[1]) < self.n_gpus] \
            or [f"cuda:{i}" for i in range(max(1, self.n_gpus))]
        self.preset = PRESET_BY_KEY.get(str(self.cfg.get("preset", "60")), PRESETS[1])
        self.engine = self.cfg.get("engine", self.preset.engine)
        if self.engine not in ENGINES:
            self.engine = self.preset.engine
        self.want = {"pipewire": False, "v4l2": False, "v4l2alpha": False}
        self.provider = None
        self.screen_busy = False
        self.screen_pending = None      # PortalSession while the desktop's picker is open
        self.ann = AN.Annotator()
        self.uvc_ctrls = {}
        self.uvc_dev = None

        self.bridge = Bridge()
        self.bridge.preview.connect(lambda img, a: self.view.set_frame(img, a))
        self.bridge.status.connect(lambda s: self.status.setText(s))
        self.bridge.lock.connect(self._on_lock_status)
        self.bridge.snapshot.connect(self._on_snapshot)
        self.bridge.error.connect(self._on_error)
        self.bridge.screen.connect(self._on_screen)
        self.bridge.screen_ended.connect(self._on_screen_ended)
        self.bridge.uncheck.connect(lambda n: {"pipewire": self.pw_btn, "v4l2": self.v4_btn,
                                               "v4l2alpha": self.v4a_btn}[n].setChecked(False))
        self.pipe = LivePipeline(None, on_preview=self._emit_preview, on_status=self._emit_status,
                                 on_error=self.bridge.error.emit)

        root = QWidget()
        outer = QHBoxLayout(root)
        left = QVBoxLayout()
        left.addWidget(self._build_topbar())
        left.addWidget(self._build_drawbar())
        self.view = PreviewView()
        self.view.settings = self.settings
        self.view.ann = self.ann
        self.view.on_cameo_moved = self.changed
        self.view.on_cameo_resized = lambda x: self.cameo_size_slider.set_value(x)
        left.addWidget(self.view, 1)
        self.status = QLabel("")
        self.status.setObjectName("status")
        left.addWidget(self.status)
        self.errlabel = QLabel("")
        self.errlabel.setStyleSheet("color:#ff6b6b")
        self.errlabel.setWordWrap(True)
        left.addWidget(self.errlabel)
        left.addWidget(self._build_strip())
        outer.addLayout(left, 1)
        tabs = QTabWidget()
        tabs.addTab(self._scroll(self._build_bg_tab()), "Background")
        tabs.addTab(self._scroll(self._build_layout_tab()), "Layout")
        tabs.addTab(self._scroll(self._build_subject_tab()), "Subject")
        tabs.addTab(self._scroll(self._build_fun_tab()), "Fun")
        tabs.addTab(self._scroll(self._build_camera_tab()), "Camera")
        tabs.addTab(self._scroll(self._build_audio_tab()), "Audio")
        tabs.addTab(self._scroll(self._build_output_tab()), "Output")
        tabs.setFixedWidth(430)
        outer.addWidget(tabs)
        self.setCentralWidget(root)
        self.setStyleSheet(STYLE)

        for seq, fn in (("Ctrl+Q", self.close), ("Ctrl+O", self.add_backgrounds),
                        ("Ctrl+Z", self.ann.undo), ("Ctrl+Shift+Z", self.ann.redo), ("Ctrl+Y", self.ann.redo),
                        ("Delete", self.ann.clear)):
            a = QAction(self)
            a.setShortcut(seq)
            a.triggered.connect(fn)
            self.addAction(a)
        for key, tool in AN.SHORTCUTS.items():
            sc_ = QShortcut(QKeySequence(key), self)
            sc_.activated.connect(lambda t=tool: self._set_tool(t))
        self.ann_timer = QTimer(self, interval=16, timeout=self.ann.tick)   # ~60 Hz, no-op when idle
        self.ann_timer.start()
        self.save_timer = QTimer(self, singleShot=True, timeout=self._save_config)

        self._apply_engine()
        self._apply_source()
        bg = self.cfg.get("background")
        if bg == "screen":
            if self.settings.mode == "replace":   # resume the share (restore token: often no picker)
                self._start_screen(self.cfg.get("screen_kind", "monitor"))
        elif bg and os.path.exists(bg):
            self._set_background(bg, switch_mode=False)
        self._select_mode_tile(self.settings.mode)

    # ---- config ---------------------------------------------------------------------------------
    def _load_config(self):
        try:
            with open(CONFIG) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return {}

    def _save_config(self):
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        self.cfg.update(settings={k: v for k, v in asdict(self.settings).items() if k != "extra"},
                        engine=self.engine, gpus=self.gpus, preset=self.preset.key)
        with open(CONFIG, "w") as f:
            json.dump(self.cfg, f, indent=2)

    def changed(self):
        self.save_timer.start(800)

    # ---- pipeline callbacks (other threads -> Qt signals) ----------------------------------------
    def _emit_preview(self, bgra: np.ndarray, has_alpha: bool):
        h, w = bgra.shape[:2]
        fmt = QImage.Format.Format_ARGB32 if has_alpha else QImage.Format.Format_RGB32
        self.bridge.preview.emit(QImage(bgra.data, w, h, 4 * w, fmt).copy(), has_alpha)

    def _emit_status(self, p: LivePipeline):
        st, src = p.stats, p.source
        outs = " · ".join({"pipewire": "PipeWire cam", "v4l2": "V4L2 cam", "v4l2alpha": "V4L2 alpha cam",
                           "record": "● REC"}.get(k, k)
                          for k in p.outputs)
        eng = p.studio.engine_kind if p.studio else "loading…"
        plate = ""
        if p.studio and p.studio.plate.ready:
            plate = " · plate ✓"
        elif p.studio and p.studio.capturing_plate:
            plate = " · capturing plate…"
        self.bridge.status.emit(
            f"in {st.fps_in:5.1f} · out {st.fps_out:5.1f} fps · {st.latency_ms:4.1f} ms · {eng}"
            f" · {src.width}x{src.height}@{src.fps:g}" + (f" · {outs}" if outs else "") + plate
            + (f" · dropped {st.dropped}" if st.dropped else ""))
        if p.studio is not None:
            self.bridge.lock.emit(p.studio.subject_status())

    def _cfg_set(self, key, value):
        self.cfg[key] = value
        if hasattr(self, "save_timer"):
            self.save_timer.start(500)

    def _snapshot(self):
        secs = self.snap_timer.currentData() if hasattr(self, "snap_timer") else 0
        if secs:
            self._snap_left = secs

            def tick():
                if self._snap_left <= 0:
                    self.snap_btn.setText("📷 Snapshot")
                    self.pipe.snap_cb = self.bridge.snapshot.emit
                    return
                self.snap_btn.setText(f"📷 {self._snap_left}…")
                self._snap_left -= 1
                QTimer.singleShot(1000, tick)
            tick()
        else:
            self.pipe.snap_cb = self.bridge.snapshot.emit

    def _on_snapshot(self, bgra, has_alpha):
        from .thumbnail import ThumbnailDialog, save_snapshot
        path, img = save_snapshot(bgra, has_alpha)
        self._on_error(f"Snapshot saved: {path}")
        dlg = ThumbnailDialog(img, path, self)
        dlg.show()
        self._thumb_dlg = dlg

    def _on_lock_status(self, s):
        if hasattr(self, "lock_label"):
            icon = {"locked": "● ", "off": "○ ", "unavailable": "○ "}.get(s, "◌ ")
            self.lock_label.setText(icon + {"locked": "Locked on to you",
                                            "locking on": "Locking on… (look at the camera)",
                                            "looking for subject": "Nobody in shot",
                                            "starting": "Starting tracker on the second GPU…",
                                            "off": "Off",
                                            "unavailable": "MatAnyone 2 not installed"}.get(s, s))

    def _on_error(self, s):
        self.errlabel.setText(s)
        QTimer.singleShot(10000, lambda: self.errlabel.setText("") if self.errlabel.text() == s else None)

    # ---- layout helpers ------------------------------------------------------------------------
    @staticmethod
    def _scroll(w):
        sa = QScrollArea()
        sa.setWidget(w)
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return sa

    def _slider(self, form, label, attr, lo, hi, step, fmt="{:.2f}", on_change=None, value=None):
        val = getattr(self.settings, attr) if attr else value
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(int(round(lo / step)), int(round(hi / step)))
        s.setValue(int(round(val / step)))
        lab = QLabel(fmt.format(val))
        lab.setMinimumWidth(44)
        row = QHBoxLayout()
        row.addWidget(s, 1)
        row.addWidget(lab)

        def on(v):
            x = v * step
            lab.setText(fmt.format(x))
            if attr:
                setattr(self.settings, attr, x)
            if on_change:
                on_change(x)
            self.changed()
        s.valueChanged.connect(on)
        form.addRow(label, row)
        s.set_value = lambda x: s.setValue(int(round(x / step)))
        return s

    def _check(self, layout, text, attr):
        c = QCheckBox(text)
        c.setChecked(bool(getattr(self.settings, attr)))
        c.toggled.connect(lambda v: (setattr(self.settings, attr, bool(v)), self.changed()))
        layout.addWidget(c)
        return c

    # ---- top bar -------------------------------------------------------------------------------
    def _build_topbar(self):
        bar = QFrame()
        bar.setObjectName("bar")
        h = QHBoxLayout(bar)
        self.preset_combo = QComboBox()
        for p in PRESETS:
            self.preset_combo.addItem(p.label, p.key)
            self.preset_combo.setItemData(self.preset_combo.count() - 1, p.note, Qt.ItemDataRole.ToolTipRole)
        self.preset_combo.setCurrentIndex(self.preset_combo.findData(self.preset.key))
        self.preset_combo.activated.connect(self._on_preset)
        h.addWidget(QLabel("Preset"))
        h.addWidget(self.preset_combo)
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(200)
        self._refresh_cameras()
        self.cam_combo.activated.connect(self._on_camera)
        h.addWidget(QLabel("Camera"))
        h.addWidget(self.cam_combo, 1)
        self.engine_combo = QComboBox()
        for k, lab in ENGINES.items():
            self.engine_combo.addItem(lab, k)
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(self.engine)))
        self.engine_combo.activated.connect(self._on_engine)
        h.addWidget(self.engine_combo)
        h.addSpacing(8)
        self.pw_btn = QPushButton("● PipeWire camera")
        self.pw_btn.setObjectName("live")
        self.pw_btn.setCheckable(True)
        self.pw_btn.setToolTip("Native PipeWire camera: no driver, any frame rate, alpha for OBS Studio.\n"
                               "OBS: Sources → Video Capture Device (PipeWire) → R9700 Broadcast Camera")
        self.pw_btn.toggled.connect(lambda on: self._set_output("pipewire", on))
        h.addWidget(self.pw_btn)
        self.v4_btn = QPushButton("● V4L2 camera")
        self.v4_btn.setObjectName("live")
        self.v4_btn.setCheckable(True)
        self.v4_btn.setToolTip("Classic /dev/video camera for Zoom, Discord, Teams (needs v4l2loopback)")
        self.v4_btn.toggled.connect(lambda on: self._set_output("v4l2", on))
        h.addWidget(self.v4_btn)
        self.rec_btn = QPushButton("● Rec")
        self.rec_btn.setObjectName("live")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._toggle_record)
        h.addWidget(self.rec_btn)
        self.snap_btn = QPushButton("📷 Snapshot")
        self.snap_btn.setToolTip("Save the current output frame at full resolution and open the thumbnail maker.\n"
                                 "Timer: Output tab → Recording.")
        self.snap_btn.clicked.connect(self._snapshot)
        h.addWidget(self.snap_btn)
        return bar

    def _refresh_cameras(self):
        from .media import list_cameras
        cur = self.cfg.get("source")
        self.cam_combo.clear()
        for dev, label in list_cameras():
            self.cam_combo.addItem(label, dev)
        for f in self.cfg.get("recent_files", [])[-5:]:
            if os.path.exists(f):
                self.cam_combo.addItem("🎞 " + os.path.basename(f), f)
        self.cam_combo.addItem("Open video file as camera…", "__open__")
        i = self.cam_combo.findData(cur)
        self.cam_combo.setCurrentIndex(i if i >= 0 else 0)

    def _on_camera(self, _):
        if self.cam_combo.currentData() == "__open__":
            p, _ = QFileDialog.getOpenFileName(self, "Use a video file as the camera",
                                               os.path.expanduser("~/Videos"), B.FILE_FILTER)
            if not p:
                self.cam_combo.setCurrentIndex(0)
                return
            self.cfg["recent_files"] = ([f for f in self.cfg.get("recent_files", []) if f != p] + [p])[-5:]
            self.cfg["source"] = p
            self._refresh_cameras()
        self._apply_source()

    def _on_preset(self, _):
        self.preset = PRESET_BY_KEY[self.preset_combo.currentData()]
        self.engine = self.preset.engine
        self.engine_combo.setCurrentIndex(self.engine_combo.findData(self.engine))
        self._apply_engine()
        self._apply_source()
        if self.exposure_lock.isChecked():
            self._lock_exposure(True)
        self.changed()

    def _on_engine(self, _):
        self.engine = self.engine_combo.currentData()
        self._apply_engine()
        self.changed()

    def _apply_engine(self):
        kind, gpus, settings = self.engine, list(self.gpus), self.settings
        if hasattr(self, "status"):
            self.status.setText(f"loading {ENGINES[kind]} …")

        def do(p):
            from .studio import Studio
            if p.studio is None:
                p.studio = Studio(kind, gpus, settings)
            else:
                old = p.studio.engine
                p.studio.set_engine(kind, gpus)
                if old is not p.studio.engine and hasattr(old, "close"):
                    old.close()
            if self.provider is not None:
                p.studio.compositor.background_src = self.provider
            p.studio.compositor.annotations = self.ann
        self.pipe.command(do)

    def _apply_source(self):
        uri = self.cam_combo.currentData()
        if uri == "__open__":
            uri = None
        self.cfg["source"] = uri
        pr = self.preset
        self.pipe.set_source(uri, pr.width, pr.height, pr.fps)
        self.pipe.command(self._sync_outputs)
        self._load_uvc(uri)
        self.changed()
        if not uri:
            self.view.img = None
            self.view.message = "No camera found - plug one in, or pick “Open video file as camera…”"
            self.view.update()

    # ---- outputs -------------------------------------------------------------------------------
    def _set_output(self, name, on):
        self.want[name] = on
        self.pipe.command(self._sync_outputs)

    def _sync_outputs(self, p: LivePipeline):
        """GPU thread: (re)create sinks so they match the current source size and frame rate."""
        if p.source is None:
            return
        w, h, fps = p.source.width, p.source.height, max(1, int(round(p.source.fps)))
        for name, on in self.want.items():
            cur = p.outputs.get(name)
            ok = cur is not None and getattr(cur, "size", None) == (w, h) and getattr(cur, "fps", None) == fps
            if on and not ok:
                try:
                    if name == "pipewire":
                        from .pwcam import PipeWireCamera
                        sink = PipeWireCamera(w, h, fps)
                    else:
                        from .v4l2out import V4L2Camera
                        sink = V4L2Camera(w, h, fps, "ABGR32" if name == "v4l2alpha" else "YUYV")
                    p._swap_output(name, sink)
                except Exception as e:  # noqa: BLE001
                    self.want[name] = False
                    self.bridge.error.emit(str(e))
                    self.bridge.uncheck.emit(name)
            elif not on and cur is not None:
                p._swap_output(name, None)

    def _toggle_record(self, on):
        if on:
            codec = self.codec_combo.currentData()
            audio_choice = self.audio_combo.currentData()
            audio_adj = float(self.audio_sync.value())

            def do(p):
                from .media import Recorder
                if p.source is None:
                    return
                os.makedirs(RECORD_DIR, exist_ok=True)
                ext = "mov" if codec == "prores4444" else "mp4"
                path = os.path.join(RECORD_DIR, time.strftime(f"%Y-%m-%d %H-%M-%S.{ext}"))
                from .media import resolve_audio
                fps = max(1, int(round(p.source.fps)))
                # the picture reaches the recorder later than the sound (camera + processing):
                # delay the audio by the measured pipeline latency + ~2 frames of camera latency
                offset = p.stats.latency_ms + 2000 / fps + audio_adj
                rec = Recorder(path, p.source.width, p.source.height, fps, codec,
                               alpha=codec == "prores4444", audio=resolve_audio(audio_choice),
                               audio_offset_ms=offset)
                p._swap_output("record", rec)
                self.bridge.error.emit("Recording" + (f" with sound ({rec.audio_target})" if rec.audio_target
                                                      else " without sound") + f" → {path}")
            self.pipe.command(do)
        else:
            self.pipe.remove_output("record")
            self._on_error(f"Saved to {RECORD_DIR}")
        self.rec_btn.setText("■ Stop" if on else "● Rec")

    # ---- background strip ----------------------------------------------------------------------
    def _build_strip(self):
        f = QFrame()
        f.setObjectName("strip")
        v = QVBoxLayout(f)
        head = QHBoxLayout()
        t = QLabel("Backgrounds")
        t.setStyleSheet(f"color:{ACCENT}; font-weight:600")
        head.addWidget(t)
        head.addWidget(_hint("Drop images, GIFs or videos anywhere on the window · right-click to remove"), 1)
        add = QPushButton("+ Add images / videos…")
        add.clicked.connect(self.add_backgrounds)
        head.addWidget(add)
        v.addLayout(head)
        self.strip = QListWidget()
        self.strip.setViewMode(QListView.ViewMode.IconMode)
        self.strip.setFlow(QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setIconSize(QSize(160, 90))
        self.strip.setGridSize(QSize(176, 124))
        self.strip.setFixedHeight(146)
        self.strip.setMovement(QListView.Movement.Static)
        self.strip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.strip.customContextMenuRequested.connect(self._strip_menu)
        self.strip.itemClicked.connect(self._on_tile)
        v.addWidget(self.strip)
        self._fill_strip()
        return f

    def _fill_strip(self):
        self.strip.clear()
        self._tile_by_path = {}
        for kind, label, tip in SPECIAL_TILES:
            it = QListWidgetItem(_icon(_tile_pixmap(kind, color=self.settings.bg_color)), label)
            it.setData(Qt.ItemDataRole.UserRole, ("mode", kind))
            it.setToolTip(tip)
            self.strip.addItem(it)
        for path, builtin in self.library.items():
            name = os.path.splitext(os.path.basename(path))[0]
            if builtin and name[:2].isdigit():
                name = name[3:]
            it = QListWidgetItem(_icon(_tile_pixmap("off")),
                                 ("▶ " if B.kind_of(path) == "video" else "") + name)
            it.setData(Qt.ItemDataRole.UserRole, ("bg", path, builtin))
            it.setToolTip(path)
            self.strip.addItem(it)
            self._tile_by_path[path] = it
            job = _ThumbJob(path)
            job.sig.done.connect(self._on_thumb)
            QThreadPool.globalInstance().start(job)

    def _on_thumb(self, path, img):
        it = self._tile_by_path.get(path)
        if it is not None:
            it.setIcon(_icon(QPixmap.fromImage(img)))

    def _on_tile(self, it):
        d = it.data(Qt.ItemDataRole.UserRole)
        if d in (("mode", "screen"), ("mode", "window")):
            kind = "window" if d[1] == "window" else "monitor"
            prov = self.provider
            if isinstance(prov, screencap.ScreenBackground) and self.cfg.get("screen_kind") == kind:
                self._set_mode("replace")        # already sharing that: take over the output again
            else:
                self._start_screen(kind)
        elif d[0] == "mode":
            self._set_mode(d[1])
            if d[1] == "transparent" and not (self.pw_btn.isChecked() or self.v4a_btn.isChecked()):
                self._on_error("Transparent output needs an alpha-capable output: “● PipeWire camera” "
                               "(OBS: Video Capture Device (PipeWire)) or “● V4L2 alpha camera for OBS” "
                               "(Output tab; OBS: Video Capture Device (V4L2), 32-bit ABGR).")
        else:
            self._set_background(d[1])

    def _strip_menu(self, pos):
        it = self.strip.itemAt(pos)
        if it is None:
            return
        d = it.data(Qt.ItemDataRole.UserRole)
        if d[0] != "bg":
            return
        m = QMenu(self)
        if not d[2]:
            m.addAction("Remove from library", lambda: (self.library.remove(d[1]), self._fill_strip(),
                                                         self._select_mode_tile(self.settings.mode)))
        m.addAction("Show in folder", lambda: os.system(f'xdg-open "{os.path.dirname(d[1])}" &'))
        m.exec(self.strip.mapToGlobal(pos))

    def _select_mode_tile(self, mode):
        for i in range(self.strip.count()):
            d = self.strip.item(i).data(Qt.ItemDataRole.UserRole)
            bg = self.cfg.get("background")
            if (mode != "replace" and d[0] == "mode" and d[1] == mode) or \
               (mode == "replace" and bg == "screen" and
                d == ("mode", "window" if self.cfg.get("screen_kind") == "window" else "screen")) or \
               (mode == "replace" and d[0] == "bg" and d[1] == bg):
                self.strip.setCurrentRow(i)
                return

    def _set_mode(self, mode):
        self.settings.mode = mode
        self._select_mode_tile(mode)
        self.changed()

    def _set_background(self, path, switch_mode=True):
        try:
            prov = B.BackgroundProvider(path, self.settings.bg_fit)
        except Exception as e:  # noqa: BLE001
            self._on_error(f"Can't use {os.path.basename(path)}: {e}")
            return
        self._use_provider(prov, path, switch_mode)

    def _use_provider(self, prov, key, switch_mode=True):
        old, self.provider = self.provider, prov

        def do(p):
            if p.studio is not None:
                p.studio.compositor.background_src = prov
            if old is not None:
                # also ends a screen share that is being replaced; off the GPU thread so the
                # stream never hitches while a capture process shuts down
                threading.Thread(target=old.close, name="r9700-bg-close", daemon=True).start()
        self.pipe.command(do)
        self.cfg["background"] = key
        if switch_mode:
            self.settings.mode = "replace"
        self._select_mode_tile(self.settings.mode)
        self._sync_screen_ui()
        self.changed()

    # ---- screen share ----------------------------------------------------------------------------
    def _start_screen(self, kind="monitor", pick=False):
        """Share a monitor/window as the background. pick=True always shows the desktop's picker;
        otherwise the last choice is restored silently where the desktop supports it."""
        if self.screen_busy and self.screen_pending is not None:
            self.screen_pending.cancel()        # a new pick replaces one still waiting
        backend = screencap.backend_name()
        if backend is None:
            self._on_error(screencap.missing_reason())
            return
        geometry = None
        if backend != "portal":
            scr = QGuiApplication.screens()
            i = min(max(0, self.screen_combo.currentIndex()), len(scr) - 1)
            g, k = scr[i].geometry(), scr[i].devicePixelRatio()
            geometry = (int(g.x() * k), int(g.y() * k), int(g.width() * k), int(g.height() * k))
        token = None if pick else (self.cfg.get("screen_tokens") or {}).get(kind)
        self.screen_busy = True
        what = "window" if kind == "window" else "screen"
        self.screen_status.setText(f"Pick the {what} to share in your desktop's dialog…" if backend == "portal"
                                   else "Starting screen capture…")
        gen = self._screen_gen = getattr(self, "_screen_gen", 0) + 1

        def on_session(ps):
            self.screen_pending = ps

        def run():
            try:
                share = screencap.start_share(kind, token, geometry, on_end=self.bridge.screen_ended.emit,
                                              on_session=on_session)
                if gen != self._screen_gen:          # superseded by a newer pick
                    share.close()
                    return
                self.bridge.screen.emit(share, kind, "")
            except screencap.PortalCancelled:
                if gen == self._screen_gen:
                    self.bridge.screen.emit(None, kind, "")
            except Exception as e:  # noqa: BLE001
                if gen == self._screen_gen:
                    self.bridge.screen.emit(None, kind, str(e))
        threading.Thread(target=run, name="r9700-screen-pick", daemon=True).start()

    def _on_screen_ended(self, msg):
        self._on_error(msg.replace("screen share ended", "Screen share ended (window closed or sharing "
                                   "stopped from the desktop)") + " - pick it again from the strip.")

    def _on_screen(self, share, kind, err):
        self.screen_busy = False
        self.screen_pending = None
        if share is None:
            if err:
                self._on_error(f"Screen share: {err}")
            self._sync_screen_ui()
            return
        if share.restore_token:
            self.cfg.setdefault("screen_tokens", {})[kind] = share.restore_token
        self.cfg["screen_kind"] = kind
        # takes over the output from any mode (green screen, transparent, blur, …)
        self._use_provider(screencap.ScreenBackground(share, self.cfg.get("screen_region")), "screen")

    def _stop_screen(self):
        if not isinstance(self.provider, screencap.ScreenBackground):
            return
        fallback = next((p for p, _ in self.library.items()), None)
        if fallback:
            self._set_background(fallback, switch_mode=self.settings.mode == "replace")
        else:
            self._use_provider(None, None, switch_mode=False)
            self._set_mode("blur")

    def _sync_screen_ui(self):
        if not hasattr(self, "screen_status"):
            return
        prov = self.provider
        on = isinstance(prov, screencap.ScreenBackground)
        self.screen_stop.setEnabled(on)
        self.region_btn.setEnabled(on)
        self.region_reset.setEnabled(on and prov.region is not None)
        if on:
            self.screen_status.setText(f"Sharing: {prov.share.label}"
                                       + (" · cropped to a region" if prov.region else ""))
        elif not self.screen_busy:
            self.screen_status.setText("Not sharing")

    def add_backgrounds(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add backgrounds", os.path.expanduser("~/Pictures"),
                                                B.FILE_FILTER)
        self._import(paths)

    def _import(self, paths):
        if not paths:
            return
        added = self.library.add(paths)
        known = set(self.library.user)
        bad = [os.path.basename(p) for p in paths if os.path.abspath(p) not in known]
        if bad:
            self._on_error("Not a supported image/video: " + ", ".join(bad))
        if added:
            self._fill_strip()
            self._set_background(added[-1])
        elif paths and not bad:  # already in the library: just use it
            self._set_background(os.path.abspath(paths[-1]))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._import([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    # ---- tabs ----------------------------------------------------------------------------------
    def _build_bg_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._build_screen_group())
        g = QGroupBox("Background blur")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        f = QFormLayout()
        self.blur_slider = self._slider(f, "Amount", "blur_strength", 0, 1, 0.01)
        for lab, val in (("Light", 0.25), ("Medium", 0.55), ("Strong", 0.9)):
            b = QPushButton(lab)
            b.clicked.connect(lambda _=None, x=val: (self.blur_slider.set_value(x), self._set_mode("blur")))
            row.addWidget(b)
        gl.addLayout(row)
        gl.addLayout(f)
        gl.addWidget(_hint("Blurs your real room. The subject is excluded from the blur, so there's no halo."))
        v.addWidget(g)

        g = QGroupBox("Image / video background")
        f = QFormLayout(g)
        self._slider(f, "Blur it", "bg_blur", 0, 1, 0.01)
        fit = QComboBox()
        for k, lab in B.FITS.items():
            fit.addItem(lab, k)
        fit.setCurrentIndex(max(0, fit.findData(self.settings.bg_fit)))

        def on_fit(_):
            self.settings.bg_fit = fit.currentData()
            if self.provider is not None:
                self.provider.fit = self.settings.bg_fit
                self.provider._cache = None
            self.changed()
        fit.activated.connect(on_fit)
        f.addRow("Fit", fit)
        self._slider(f, "Dim", "bg_dim", 0, 0.9, 0.01)
        f.addRow(_hint("“Blur it” softens a picture or video background for a depth-of-field look. "
                       "Videos and GIFs loop at their own frame rate."))
        v.addWidget(g)

        g = QGroupBox("Edges && colour")
        f = QFormLayout(g)
        self._slider(f, "Light wrap", "light_wrap", 0, 1, 0.01)
        col = QPushButton("Solid colour…")
        col.clicked.connect(self._pick_color)
        f.addRow(col)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _build_screen_group(self):
        g = QGroupBox("Screen share background")
        gl = QVBoxLayout(g)
        backend = screencap.backend_name()
        row = QHBoxLayout()
        b = QPushButton("Share a screen…")
        b.clicked.connect(lambda: self._start_screen("monitor", pick=True))
        row.addWidget(b)
        if screencap.can_share_windows():
            b = QPushButton("Share a window…")
            b.clicked.connect(lambda: self._start_screen("window", pick=True))
            row.addWidget(b)
        self.screen_stop = QPushButton("Stop")
        self.screen_stop.clicked.connect(self._stop_screen)
        row.addWidget(self.screen_stop)
        gl.addLayout(row)
        row = QHBoxLayout()
        self.region_btn = QPushButton("Crop to a region…")
        self.region_btn.setToolTip("Show only part of what you share: drag a box over a still of it")
        self.region_btn.clicked.connect(self._choose_region)
        row.addWidget(self.region_btn)
        self.region_reset = QPushButton("Show all")
        self.region_reset.clicked.connect(lambda: self._set_region(None))
        row.addWidget(self.region_reset)
        gl.addLayout(row)
        self.screen_combo = QComboBox()
        if backend in ("x11", "gdi"):      # no desktop picker there: choose the monitor here
            for i, scr in enumerate(QGuiApplication.screens()):
                gm = scr.geometry()
                self.screen_combo.addItem(f"{i + 1}: {scr.name()} {gm.width()}x{gm.height()}")
            gl.addWidget(self.screen_combo)
        self.screen_status = QLabel("Not sharing")
        gl.addWidget(self.screen_status)
        if backend is None:
            gl.addWidget(_hint(screencap.missing_reason()))
        else:
            gl.addWidget(_hint("Shows your screen or one window behind you and takes over the output from any "
                               "mode. Your desktop asks what to share (KDE also offers “Share region”). "
                               "Draw on it with the tools above the preview. For a smaller you, use the "
                               "bubble in the Layout tab."))
        self._sync_screen_ui()
        return g

    def _choose_region(self):
        prov = self.provider
        if not isinstance(prov, screencap.ScreenBackground):
            return
        frame = prov.snapshot()
        if frame is None:
            self._on_error("No picture from the screen share yet.")
            return
        dlg = RegionDialog(self, frame, prov.region)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.region is not None:
            self._set_region(dlg.region)

    def _set_region(self, region):
        self.cfg["screen_region"] = list(region) if region else None
        prov = self.provider
        if isinstance(prov, screencap.ScreenBackground):
            prov.region = tuple(region) if region else None
        self._sync_screen_ui()
        self.changed()

    # ---- draw toolbar --------------------------------------------------------------------------
    def _build_drawbar(self):
        bar = QFrame()
        bar.setObjectName("bar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(4)
        lab = QLabel("Draw")
        lab.setStyleSheet(f"color:{ACCENT}; font-weight:600")
        h.addWidget(lab)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_btns = {}
        for key, label, tip in AN.TOOLS:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setFixedWidth(36)
            b.setChecked(key == "move")
            b.clicked.connect(lambda _=None, k=key: self._set_tool(k))
            self.tool_group.addButton(b)
            self.tool_btns[key] = b
            h.addWidget(b)
        h.addSpacing(8)
        self.color_btns = {}
        for col, name in AN.COLORS:
            b = QPushButton("")
            b.setCheckable(True)
            b.setToolTip(name)
            b.setFixedSize(22, 22)
            b.setStyleSheet(f"QPushButton {{ background: {col}; border: 2px solid #32353f; border-radius: 11px; }}"
                            f"QPushButton:checked {{ border: 3px solid white; }}")
            b.setChecked(col == self.ann.color)
            b.clicked.connect(lambda _=None, c=col: self._set_color(c))
            self.color_btns[col] = b
            h.addWidget(b)
        wcombo = QComboBox()
        for k, v in AN.WIDTHS.items():
            wcombo.addItem(k, v)
        wcombo.setCurrentIndex(1)
        wcombo.setToolTip("Line width")
        wcombo.activated.connect(lambda _: setattr(self.ann, "width", wcombo.currentData()))
        h.addWidget(wcombo)
        fcombo = QComboBox()
        for k, v in AN.FADES.items():
            fcombo.addItem(k, v)
        fcombo.setToolTip("Keep drawings, or let them fade away by themselves")
        fcombo.activated.connect(lambda _: setattr(self.ann, "fade", fcombo.currentData()))
        h.addWidget(fcombo)
        snap = QCheckBox("Snap")
        snap.setChecked(True)
        snap.setToolTip("Turn rough circles, boxes and lines into clean ones")
        snap.toggled.connect(lambda v: setattr(self.ann, "snap", bool(v)))
        h.addWidget(snap)
        top = QCheckBox("Over me")
        top.setChecked(self.settings.draw_over_me)
        top.setToolTip("Draw on top of you too (default: drawings sit on the background, under you)")
        top.toggled.connect(lambda v: (setattr(self.settings, "draw_over_me", bool(v)), self.changed()))
        h.addWidget(top)
        for label, tip, fn in (("↶", "Undo (Ctrl+Z, or right-click while drawing)", self.ann.undo),
                               ("↷", "Redo (Ctrl+Shift+Z)", self.ann.redo),
                               ("🗑", "Clear all drawings (Delete)", self.ann.clear)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setFixedWidth(34)
            b.clicked.connect(fn)
            h.addWidget(b)
        h.addStretch(1)
        return bar

    def _set_tool(self, tool):
        if self.view._text_edit is not None and tool != "text":
            self.view._text_edit.deleteLater()
            self.view._text_edit = None
        self.ann.set_tool(tool)
        self.tool_btns[tool].setChecked(True)
        self.view.refresh_cursor()

    def _set_color(self, col):
        self.ann.color = col
        for c, b in self.color_btns.items():
            b.setChecked(c == col)

    def _build_layout_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("You on screen")
        f = QFormLayout(g)
        lay = QComboBox()
        lay.addItem("Full size - you over the background", "full")
        lay.addItem("Bubble - you in a circle / box", "cameo")
        lay.setCurrentIndex(max(0, lay.findData(self.settings.layout)))
        self.layout_combo = lay

        def on_layout(_):
            self.settings.layout = lay.currentData()
            if self.settings.layout == "cameo":    # starts in the lower-right corner
                self.settings.cameo_x = self.settings.cameo_y = 1.0
            self.changed()
        lay.activated.connect(on_layout)
        f.addRow("Layout", lay)

        def combo(label, attr, items):
            c = QComboBox()
            for k, lab in items.items():
                c.addItem(lab, k)
            c.setCurrentIndex(max(0, c.findData(getattr(self.settings, attr))))
            c.activated.connect(lambda _: (setattr(self.settings, attr, c.currentData()), self.changed()))
            f.addRow(label, c)
            return c
        combo("Shape", "cameo_shape", CAMEO_SHAPES)
        self.cameo_size_slider = self._slider(f, "Size", "cameo_size", 0.12, 0.9, 0.01)
        self._slider(f, "Zoom", "cameo_zoom", 0.6, 2.5, 0.05)
        combo("Behind you", "cameo_fill", CAMEO_FILLS)
        combo("Ring", "cameo_ring", {k: k.capitalize() if k != "none" else "No ring" for k in CAMEO_RINGS})
        self._slider(f, "Ring width", "cameo_border", 0, 24, 1, fmt="{:.0f} px")
        box = QVBoxLayout()
        self._check(box, "Follow my head", "cameo_follow")
        self._check(box, "Pop out of the frame (head breaks over the top)", "cameo_pop")
        self._check(box, "Drop shadow", "cameo_shadow")
        f.addRow(box)
        row = QHBoxLayout()
        for lab, x, y in (("↖", 0, 0), ("↗", 1, 0), ("Centre", 0.5, 0.5), ("↙", 0, 1), ("↘", 1, 1)):
            b = QPushButton(lab)
            b.setToolTip("Move the bubble here")
            b.clicked.connect(lambda _=None, x=x, y=y: self._place_cameo(x, y))
            row.addWidget(b)
        f.addRow("Place", row)
        f.addRow(_hint("Drag the bubble in the preview to move it, scroll over it to resize. It works "
                       "over every background: screen share, pictures, videos, blur, green screen and "
                       "transparent (OBS)."))
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _place_cameo(self, x, y):
        if self.settings.layout != "cameo":
            self.settings.layout = "cameo"
            self.layout_combo.setCurrentIndex(self.layout_combo.findData("cameo"))
        self.settings.cameo_x, self.settings.cameo_y = x, y
        self.changed()

    def _pick_color(self):
        c = QColorDialog.getColor(QColor.fromRgbF(*self.settings.bg_color), self, "Background colour")
        if c.isValid():
            self.settings.bg_color = (c.redF(), c.greenF(), c.blueF())
            self._fill_strip()
            self._set_mode("remove")

    def _build_subject_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Matte")
        gl = QVBoxLayout(g)
        f = QFormLayout()
        matte_sliders = {
            "choke": self._slider(f, "Choke / grow", "choke", -8, 8, 0.5, "{:+.1f}"),
            "feather": self._slider(f, "Feather", "feather", 0, 10, 0.5, "{:.1f}"),
            "clip_black": self._slider(f, "Clip black", "clip_black", 0, 0.5, 0.01),
            "clip_white": self._slider(f, "Clip white", "clip_white", 0.5, 1, 0.01),
            "temporal_smooth": self._slider(f, "Temporal smooth", "temporal_smooth", 0, 0.9, 0.05,
                     on_change=lambda x: self.pipe.command(
                         lambda p: p.studio and setattr(p.studio.stabilizer, "strength", x)))}
        gl.addLayout(f)
        self._check(gl, "Edge colour decontamination", "decontaminate")
        reset = QPushButton("Reset matte to defaults")
        reset.setToolTip("Choke 0, feather 0, clip black 0.02, clip white 0.98, temporal 0.5. Extreme choke/clip values make "
                         "edges hard and pull the chair in; the subject lock handles headsets without them.")

        def do_reset():
            d = Settings()
            for k, sl in matte_sliders.items():
                sl.set_value(getattr(d, k))
        reset.clicked.connect(do_reset)
        gl.addWidget(reset)
        v.addWidget(g)

        g = QGroupBox("Subject lock (MatAnyone 2)")
        gl = QVBoxLayout(g)
        self._check(gl, "Lock on to me: headset, mic && glasses solid, chair removed", "subject_lock")
        row = QHBoxLayout()
        self.lock_label = QLabel("")
        self.lock_label.setObjectName("status")
        row.addWidget(self.lock_label, 1)
        relock = QPushButton("Re-lock on me")
        relock.setToolTip("Look at the camera, then click: the tracker re-detects you and your headset.")
        relock.clicked.connect(lambda: self.pipe.command(lambda p: p.studio and p.studio.relock_subject()))
        row.addWidget(relock)
        gl.addLayout(row)
        gl.addWidget(_hint(
            "Grounded SAM 2 finds you and anything you wear (“headphones”, “microphone”) once; MatAnyone 2 "
            "(CVPR 2026) then tracks exactly that on the second GPU, and RVM supplies the fine hair edges inside "
            "its outline. Nothing semi-transparent, no chair. Re-locks by itself if you leave and come back."))
        v.addWidget(g)

        g = QGroupBox("Headphones && chair")
        gl = QVBoxLayout(g)
        self._check(gl, "Keep headphones, mic && glasses solid (semantic guard)", "keep_accessories")
        gl.addWidget(_hint("Fallback used while the subject lock is starting or turned off. Uses a person/chair "
                           "segmenter on the second GPU when you have one."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Chair"))
        self.chair_combo = QComboBox()
        for k, lab in (("auto", "Auto (as the AI sees it)"), ("remove", "Remove chair"), ("keep", "Keep chair")):
            self.chair_combo.addItem(lab, k)
        self.chair_combo.setCurrentIndex(max(0, self.chair_combo.findData(self.settings.chair)))

        def on_chair(_):
            self.settings.chair = self.chair_combo.currentData()
            if self.settings.chair == "keep":
                self._on_error("Keep chair uses the clean plate: Subject tab → Capture clean plate, with you out "
                               "of shot. The chair is then cut out once and kept in every frame.")
            self.changed()
        self.chair_combo.activated.connect(on_chair)
        row.addWidget(self.chair_combo, 1)
        gl.addLayout(row)
        gl.addWidget(_hint("Remove works straight away and is pixel-exact after a clean plate. "
                           "Keep needs a clean plate."))
        v.addWidget(g)

        g = QGroupBox("Clean plate assist")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint("Step out of shot and capture the empty room. Things you hold up are kept, and "
                           "furniture the AI mistakes for you (like your chair) is removed. Keep the camera still."))
        row = QHBoxLayout()
        self.plate_btn = QPushButton("Capture clean plate (5 s)")
        self.plate_btn.clicked.connect(self._capture_plate)
        row.addWidget(self.plate_btn)
        clr = QPushButton("Clear")
        clr.clicked.connect(lambda: (self.pipe.command(lambda p: p.studio and p.studio.plate.clear()),
                                     self.plate_btn.setText("Capture clean plate (5 s)")))
        row.addWidget(clr)
        gl.addLayout(row)
        f = QFormLayout()
        self._slider(f, "Sensitivity", "plate_sensitivity", 0.03, 0.3, 0.01,
                     on_change=lambda x: self.pipe.command(
                         lambda p: p.studio and setattr(p.studio.plate, "threshold", x)))
        self._slider(f, "Furniture removal", "plate_furniture", 0, 1, 0.05,
                     on_change=lambda x: self.pipe.command(
                         lambda p: p.studio and setattr(p.studio.plate, "furniture", x)))
        gl.addLayout(f)
        gl.addWidget(_hint("Raise “Furniture removal” if the top of your chair still shows around you. "
                           "Too high can nibble clothing that matches the wall behind it."))
        v.addWidget(g)

        g = QGroupBox("Look")
        f = QFormLayout(g)
        self._slider(f, "Virtual key light", "relight", 0, 1, 0.01)
        self._slider(f, "Vignette", "vignette", 0, 1, 0.01)
        self._slider(f, "Video noise removal", "denoise", 0, 1, 0.01)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _capture_plate(self):
        self._countdown = 5

        def tick():
            if self._countdown > 0:
                self.plate_btn.setText(f"Leave the frame… {self._countdown}")
                self._countdown -= 1
                QTimer.singleShot(1000, tick)
            else:
                self.plate_btn.setText("Captured ✓ - re-capture")
                self.pipe.command(lambda p: p.studio and p.studio.start_plate_capture())
        tick()

    def _build_fun_tab(self):
        from .props import LABELS, PROPS
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Fun props (face-tracked)")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint("Comical overlays that ride your face. Pupils wobble, the propeller spins, "
                           "the bulb pulses and the bubble bobs. Props hide when the face tracker loses "
                           "you (sharp profile turns) instead of guessing."))
        for name in PROPS:
            c = QCheckBox(LABELS[name])
            c.setChecked(name in self.settings.fun_props)
            c.toggled.connect(lambda v, n=name: self._on_prop(n, v))
            gl.addWidget(c)
        row = QHBoxLayout()
        row.addWidget(QLabel("Speech bubble text"))
        self.bubble_edit = QLineEdit(self.settings.bubble_text)
        self.bubble_edit.setMaxLength(40)
        self.bubble_edit.editingFinished.connect(self._on_bubble_text)
        row.addWidget(self.bubble_edit, 1)
        gl.addLayout(row)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _on_prop(self, name, on):
        props = [p for p in self.settings.fun_props if p != name]
        if on:
            props.append(name)
        self.settings.fun_props = props
        self.changed()

    def _on_bubble_text(self):
        self.settings.bubble_text = self.bubble_edit.text().strip() or "HELLO!"
        self.changed()

    def _build_camera_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Smart framing (software)")
        gl = QVBoxLayout(g)
        self._check(gl, "Auto frame: crop and follow me", "auto_frame")
        f = QFormLayout()
        self._slider(f, "Zoom", "frame_zoom", 1, 3, 0.05)
        self._slider(f, "Smoothness", "frame_speed", 0.02, 0.3, 0.01)
        gl.addLayout(f)
        self._check(gl, "Mirror", "mirror")
        v.addWidget(g)
        self.uvc_box = QGroupBox("Camera hardware")
        self.uvc_layout = QVBoxLayout(self.uvc_box)
        v.addWidget(self.uvc_box)
        g = QGroupBox("High frame rate")
        gl = QVBoxLayout(g)
        self.exposure_lock = QCheckBox("Lock exposure to the frame rate")
        self.exposure_lock.toggled.connect(self._lock_exposure)
        gl.addWidget(self.exposure_lock)
        gl.addWidget(_hint("In a dim room, auto exposure can lengthen the shutter and quietly drop a 100/120 fps "
                           "camera to 30. This caps the exposure at one frame (raise Gain or add light)."))
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _load_uvc(self, dev):
        while self.uvc_layout.count():
            it = self.uvc_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
            elif it.layout():
                QWidget().setLayout(it.layout())
        self.uvc_dev = dev
        self.uvc_ctrls = uvc.list_controls(dev) if dev and dev.startswith("/dev/video") else {}
        self._uvc_sliders = {}
        if not self.uvc_ctrls:
            self.uvc_layout.addWidget(_hint("No hardware controls for this source."))
            return
        f = QFormLayout()
        for name, label in (("zoom_absolute", "Zoom"), ("pan_absolute", "Pan (gimbal)"),
                            ("tilt_absolute", "Tilt (gimbal)"), ("brightness", "Brightness"), ("gain", "Gain"),
                            ("sharpness", "Sharpness"), ("white_balance_temperature", "White balance K")):
            c = self.uvc_ctrls.get(name)
            if c is None or c.type != "int":
                continue
            s = QSlider(Qt.Orientation.Horizontal)
            step = max(1, c.step)
            s.setRange(c.min // step, c.max // step)
            s.setValue(c.value // step)
            s.valueChanged.connect(lambda val, n=name, st=step: uvc.set_control(self.uvc_dev, n, val * st))
            f.addRow(label, s)
            self._uvc_sliders[name] = s
        self.uvc_layout.addLayout(f)
        row = QHBoxLayout()
        if "pan_absolute" in self.uvc_ctrls:
            b = QPushButton("Centre gimbal")
            b.clicked.connect(lambda: [self._uvc_sliders[n].setValue(0) for n in ("pan_absolute", "tilt_absolute")
                                       if n in self._uvc_sliders])
            row.addWidget(b)
        for name, label in (("focus_automatic_continuous", "Autofocus"), ("white_balance_automatic", "Auto WB")):
            c = self.uvc_ctrls.get(name)
            if c is not None:
                cb = QCheckBox(label)
                cb.setChecked(bool(c.value))
                cb.toggled.connect(lambda on, n=name: uvc.set_control(self.uvc_dev, n, int(on)))
                row.addWidget(cb)
        self.uvc_layout.addLayout(row)
        self.uvc_layout.addWidget(_hint("OBSBOT Tiny: pan/tilt drive the real gimbal and zoom is the camera's own "
                                        "zoom. These are standard UVC controls, so OBSBOT Center isn't needed."))

    def _lock_exposure(self, on):
        if not self.uvc_dev or not self.uvc_ctrls:
            return
        if on:
            uvc.lock_exposure_for_fps(self.uvc_dev, self.preset.fps, self.uvc_ctrls)
        else:
            uvc.auto_exposure(self.uvc_dev, self.uvc_ctrls)

    def _build_audio_tab(self):
        from .audio import NoiseRemovalMic, list_microphones
        self.mic = NoiseRemovalMic()
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Microphone: AI noise removal")
        gl = QVBoxLayout(g)
        self.mic_combo = QComboBox()
        self.mic_combo.addItem("System default input", None)
        for name, desc in list_microphones():
            self.mic_combo.addItem(desc, name)
        i = self.mic_combo.findData(self.cfg.get("mic"))
        if i >= 0:
            self.mic_combo.setCurrentIndex(i)
        gl.addWidget(self.mic_combo)
        f = QFormLayout()
        self._slider(f, "Max reduction (dB)", None, 6, 100, 1, "{:.0f}",
                     value=self.cfg.get("mic_atten", 100), on_change=self._on_atten)
        gl.addLayout(f)
        self.mic_btn = QPushButton("Start noise-removal mic")
        self.mic_btn.setCheckable(True)
        self.mic_btn.setEnabled(NoiseRemovalMic.available())
        self.mic_btn.toggled.connect(self._toggle_mic)
        gl.addWidget(self.mic_btn)
        gl.addWidget(_hint("DeepFilterNet3 uses a few % of one CPU core. Choose “R9700 Broadcast Mic” as the "
                           "microphone in Discord / Zoom / OBS."))
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _on_atten(self, x):
        self.cfg["mic_atten"] = int(x)
        if getattr(self, "mic_btn", None) is not None and self.mic_btn.isChecked():
            QTimer.singleShot(400, lambda: self._toggle_mic(True))

    def _toggle_mic(self, on):
        if on:
            self.mic.start(self.mic_combo.currentData(), self.cfg.get("mic_atten", 100))
            self.cfg["mic"] = self.mic_combo.currentData()
            self.changed()
        else:
            self.mic.stop()
        self.mic_btn.setText("Stop noise-removal mic" if on else "Start noise-removal mic")

    def _build_output_tab(self):
        import torch
        from .media import find_loopback
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Virtual cameras")
        gl = QVBoxLayout(g)
        gl.addWidget(_hint(
            "<b>PipeWire camera</b> (recommended): no driver or sudo needed, runs at the preset's full frame rate "
            "(60/100/120 fps), and carries <b>alpha</b>.<br>OBS Studio: Sources → <i>Video Capture Device "
            "(PipeWire)</i> → “R9700 Broadcast Camera”. With the <i>Transparent</i> tile, OBS receives you with "
            "no background and needs no chroma key. Firefox, and Chrome with PipeWire camera enabled, can use it too."))
        gl.addWidget(_hint(
            "<b>V4L2 camera</b> (top bar): /dev/video42, YUYV, for apps that only speak V4L2 (Zoom, Discord, "
            "Teams). " + ("Installed." if find_loopback() else
                          "Not installed yet: run <tt>sudo scripts/setup-virtualcam.sh</tt> once.")))
        self.v4a_btn = QPushButton("● V4L2 alpha camera for OBS")
        self.v4a_btn.setObjectName("live")
        self.v4a_btn.setCheckable(True)
        self.v4a_btn.toggled.connect(lambda on: self._set_output("v4l2alpha", on))
        gl.addWidget(self.v4a_btn)
        gl.addWidget(_hint(
            "<b>V4L2 alpha camera</b>: /dev/video43, ABGR32 (B,G,R,<b>A</b>). In OBS use the classic "
            "<i>Video Capture Device (V4L2)</i> → “R9700 Broadcast Alpha (OBS)”, Video Format "
            "<b>32-bit ABGR</b>. Same transparency as the PipeWire camera, through OBS's long-standing V4L2 source. "
            + ("Installed." if find_loopback("R9700 Broadcast Alpha") else "Needs the same one-time setup script.")))
        v.addWidget(g)
        g = QGroupBox("Recording")
        gl = QVBoxLayout(g)
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("HEVC (R9700 VCN hardware)", "hevc")
        self.codec_combo.addItem("H.264 (R9700 VCN hardware)", "h264")
        self.codec_combo.addItem("ProRes 4444 + alpha (Resolve / Kdenlive)", "prores4444")
        gl.addWidget(self.codec_combo)
        f = QFormLayout()
        self.audio_combo = QComboBox()
        for k, lab in (("auto", "Auto (AI mic if on)"), ("studio", "AI noise-removed mic"),
                       ("default", "Default microphone"), ("none", "No sound")):
            self.audio_combo.addItem(lab, k)
        self.audio_combo.setCurrentIndex(max(0, self.audio_combo.findData(self.cfg.get("rec_audio", "auto"))))
        self.audio_combo.activated.connect(lambda _: self._cfg_set("rec_audio", self.audio_combo.currentData()))
        self.audio_combo.setToolTip("Auto: the R9700 Broadcast Mic (AI noise removal) when it is on, otherwise "
                                    "your default microphone.")
        f.addRow("Sound", self.audio_combo)
        self.audio_sync = QSlider(Qt.Orientation.Horizontal)
        self.audio_sync.setRange(-200, 200)
        self.audio_sync.setValue(int(self.cfg.get("rec_audio_adj", 0)))
        sync_lab = QLabel()
        def _sync(vv):
            sync_lab.setText(f"{vv:+d} ms")
            self._cfg_set("rec_audio_adj", vv)
        self.audio_sync.valueChanged.connect(_sync)
        _sync(self.audio_sync.value())
        row = QHBoxLayout()
        row.addWidget(self.audio_sync, 1)
        row.addWidget(sync_lab)
        f.addRow("Lip-sync trim", row)
        self.snap_timer = QComboBox()
        for secs, lab in ((0, "Instant"), (3, "3 s countdown"), (5, "5 s countdown")):
            self.snap_timer.addItem(lab, secs)
        self.snap_timer.setCurrentIndex(max(0, self.snap_timer.findData(self.cfg.get("snap_timer", 0))))
        self.snap_timer.activated.connect(lambda _: self._cfg_set("snap_timer", self.snap_timer.currentData()))
        f.addRow("Snapshot timer", self.snap_timer)
        gl.addLayout(f)
        gl.addWidget(_hint(f"Videos: {RECORD_DIR}. Snapshots and thumbnails: ~/Pictures/R9700 Broadcast. "
                           "Sound is synced automatically (pipeline latency is compensated); use the trim only if "
                           "lips look early (−) or late (+). Use ProRes with the Transparent tile to keep alpha."))
        v.addWidget(g)
        g = QGroupBox("GPUs")
        gl = QVBoxLayout(g)
        self.gpu_checks = []
        for i in range(self.n_gpus):
            c = QCheckBox(f"GPU{i}: {torch.cuda.get_device_name(i)}")
            c.setChecked(f"cuda:{i}" in self.gpus)
            c.toggled.connect(self._on_gpus)
            gl.addWidget(c)
            self.gpu_checks.append(c)
        gl.addWidget(_hint("Realtime engines use the first ticked GPU. Studio engines split frames across "
                           "all ticked GPUs."))
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _on_gpus(self, _):
        self.gpus = [f"cuda:{i}" for i, c in enumerate(self.gpu_checks) if c.isChecked()] or ["cuda:0"]
        self._apply_engine()
        self.changed()

    # ---- shutdown ------------------------------------------------------------------------------
    def closeEvent(self, e):
        if self.screen_pending is not None:
            self.screen_pending.cancel()
        self._save_config()
        self.mic.stop()
        self.pipe.stop()
        if self.provider is not None:
            self.provider.close()
        st = self.pipe.studio
        if st is not None and hasattr(st.engine, "close"):
            st.engine.close()
        super().closeEvent(e)


def main():
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    app = QApplication(sys.argv)
    app.setApplicationName("R9700 Broadcast Studio")
    app.setDesktopFileName("r9700-broadcast")
    app.setFont(QFont("Noto Sans", 10))
    w = Main()
    w.show()
    snap = os.environ.get("R9700_BROADCAST_SNAPSHOT")  # "path.png:seconds[:action,...]" - tests/docs
    if snap:
        path, secs, *acts = snap.split(":")
        for i, act in enumerate(",".join(acts).split(",") if acts else []):
            QTimer.singleShot(int(float(secs) * 400) + 300 * i, lambda a=act: _test_action(w, a))
        QTimer.singleShot(int(float(secs) * 1000), lambda: (w.grab().save(path), w.close()))
    sys.exit(app.exec())


def _test_action(w: Main, action: str):
    """Scripted UI actions for headless screenshot tests."""
    if action.startswith("tile="):
        name = action[5:].lower()
        for i in range(w.strip.count()):
            it = w.strip.item(i)
            if name in it.text().lower():
                w.strip.setCurrentItem(it)
                w._on_tile(it)
                return
    elif action == "pipewire":
        w.pw_btn.setChecked(True)
    elif action == "v4l2alpha":
        w.v4a_btn.setChecked(True)
    elif action == "v4l2":
        w.v4_btn.setChecked(True)
    elif action.startswith("import="):
        w._import(action[7:].split(";"))
    elif action.startswith("tab="):
        tabs = w.findChild(QTabWidget)
        for i in range(tabs.count()):
            if tabs.tabText(i).lower() == action[4:]:
                tabs.setCurrentIndex(i)
    elif action.startswith("tool="):
        w._set_tool(action[5:])
    elif action.startswith("draw="):           # draw=tool;x0;y0;x1;y1 (normalised), a straight drag
        tool, *xy = action[5:].split(";")
        x0, y0, x1, y1 = (float(t) for t in xy)
        w._set_tool(tool)
        w.ann.press(x0, y0)
        for i in range(1, 11):
            w.ann.move(x0 + (x1 - x0) * i / 10, y0 + (y1 - y0) * i / 10, pressed=True)
        w.ann.release(x1, y1)
    elif action.startswith("layout="):
        w.layout_combo.setCurrentIndex(w.layout_combo.findData(action[7:]))
        w.layout_combo.activated.emit(w.layout_combo.currentIndex())
    elif action.startswith("cameo="):          # cameo=x,y
        x, y = (float(t) for t in action[6:].split(";"))
        w._place_cameo(x, y)
    elif action.startswith("bgblur="):
        w.settings.bg_blur = float(action[7:])
