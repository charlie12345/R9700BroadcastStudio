"""Screen share as a background: capture a monitor or a window on any desktop.

Backends (picked automatically, see `backend_name()`):

* portal  - Wayland on KDE Plasma, GNOME, Sway/Hyprland/COSMIC (anything with xdg-desktop-portal):
            the desktop's own picker dialog -> a PipeWire stream, read by a `gst-launch-1.0
            pipewiresrc` subprocess. The portal's restore token is kept so the picker only shows once.
* x11     - X11 sessions (GNOME/Xorg, XFCE, Cinnamon, MATE, ...): ffmpeg x11grab of one monitor.
* gdi     - Windows: ffmpeg gdigrab of one monitor (untested - the rest of the app is Linux-only).

Every backend writes fixed-size BGRx frames (letterboxed, aspect kept) to a pipe; `ScreenBackground`
plugs into Compositor.background_src exactly like an image/video background.
"""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

CAPTURE_W, CAPTURE_H = 1920, 1080   # capture size; the GPU resizes to the output size

MONITOR, WINDOW = 1, 2               # portal source types


def backend_name() -> str | None:
    if sys.platform == "win32":
        return "gdi" if shutil.which("ffmpeg") else None
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        return "portal" if portal_available() else None
    if os.environ.get("DISPLAY"):
        return "x11" if shutil.which("ffmpeg") else None
    return None


def can_share_windows() -> bool:
    return backend_name() == "portal" and bool(_portal_prop("AvailableSourceTypes", 0) & WINDOW)


def missing_reason() -> str:
    if sys.platform == "win32":
        return "Screen share needs ffmpeg on PATH."
    if os.environ.get("WAYLAND_DISPLAY"):
        if not shutil.which("gst-launch-1.0"):
            return "Screen share needs GStreamer: sudo apt install gstreamer1.0-tools gstreamer1.0-pipewire"
        return ("Screen share needs xdg-desktop-portal with a ScreenCast backend "
                "(xdg-desktop-portal-kde / -gnome / -wlr / -hyprland) and the Python package jeepney.")
    return "Screen share needs ffmpeg (x11grab) and an X11 or Wayland session."


# ---------------------------------------------------------------------------------------------
# xdg-desktop-portal ScreenCast (Wayland)
# ---------------------------------------------------------------------------------------------

_BUS, _OBJ = "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop"
_IFACE = "org.freedesktop.portal.ScreenCast"


def _portal_prop(name, default=None):
    try:
        from jeepney import DBusAddress, Properties
        from jeepney.io.blocking import open_dbus_connection
        with open_dbus_connection(bus="SESSION") as conn:
            msg = Properties(DBusAddress(_OBJ, _BUS, _IFACE)).get(name)
            return conn.send_and_get_reply(msg, timeout=3).body[0][1]
    except Exception:  # noqa: BLE001
        return default


def portal_available() -> bool:
    if not shutil.which("gst-launch-1.0"):
        return False
    try:
        import jeepney  # noqa: F401
    except ImportError:
        return False
    return bool(_portal_prop("AvailableSourceTypes", 0) & MONITOR)


class PortalCancelled(RuntimeError):
    pass


class PortalSession:
    """One ScreenCast session. Blocks in `start()` until the user picks (or cancels) a source.
    The D-Bus connection stays open for the session's lifetime: closing it ends the share."""

    def __init__(self):
        from jeepney.io.blocking import open_dbus_connection
        self.conn = open_dbus_connection(bus="SESSION", enable_fds=True)
        self.session = None
        self.node = None
        self.fd = None
        self.size = None
        self.restore_token = None

    def _request(self, method, signature, *args, timeout=600):
        """Call a portal method that answers through a Request object's Response signal."""
        from jeepney import DBusAddress, MatchRule, message_bus, new_method_call
        from jeepney.io.blocking import Proxy
        token = "r9700_" + secrets.token_hex(6)
        sender = self.conn.unique_name[1:].replace(".", "_")
        path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
        rule = MatchRule(type="signal", interface="org.freedesktop.portal.Request", member="Response",
                         path=path)
        Proxy(message_bus, self.conn).AddMatch(rule)
        opts = dict(args[-1])
        opts["handle_token"] = ("s", token)
        args = (*args[:-1], opts)
        with self.conn.filter(rule) as q:
            self.conn.send_and_get_reply(
                new_method_call(DBusAddress(_OBJ, _BUS, _IFACE), method, signature, args), timeout=10)
            sig = self.conn.recv_until_filtered(q, timeout=timeout)
        code, results = sig.body
        if code == 1:
            raise PortalCancelled("screen share was cancelled")
        if code != 0:
            raise RuntimeError(f"portal {method} failed (code {code})")
        return results

    def start(self, source_types=MONITOR, restore_token: str | None = None):
        from jeepney import DBusAddress, new_method_call
        res = self._request("CreateSession", "a{sv}",
                            {"session_handle_token": ("s", "r9700_" + secrets.token_hex(6))})
        self.session = res["session_handle"][1]
        avail = _portal_prop("AvailableSourceTypes", MONITOR) or MONITOR
        cursors = _portal_prop("AvailableCursorModes", 1) or 1
        opts = {"types": ("u", (source_types & avail) or MONITOR),
                "multiple": ("b", False),
                "persist_mode": ("u", 2)}          # remember the choice until revoked
        if cursors & 2:
            opts["cursor_mode"] = ("u", 2)        # cursor baked into the frames: viewers see it
        if restore_token:
            opts["restore_token"] = ("s", restore_token)
        self._request("SelectSources", "oa{sv}", self.session, opts)
        res = self._request("Start", "osa{sv}", self.session, "", {})
        streams = res.get("streams", ("a(ua{sv})", []))[1]
        if not streams:
            raise RuntimeError("the portal returned no stream")
        self.node, props = streams[0]
        if "size" in props:
            self.size = tuple(props["size"][1])
        if "restore_token" in res:
            self.restore_token = res["restore_token"][1]
        reply = self.conn.send_and_get_reply(new_method_call(
            DBusAddress(_OBJ, _BUS, _IFACE), "OpenPipeWireRemote", "oa{sv}", (self.session, {})), timeout=10)
        self.fd = reply.body[0].to_raw_fd()
        return self

    def close(self):
        from jeepney import DBusAddress, new_method_call
        if self.session:
            try:
                self.conn.send(new_method_call(DBusAddress(self.session, _BUS, "org.freedesktop.portal.Session"),
                                               "Close"))
            except Exception:  # noqa: BLE001
                pass
            self.session = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


def gst_command(node, fd=None, w=CAPTURE_W, h=CAPTURE_H) -> list[str]:
    src = ["pipewiresrc", f"path={node}", "always-copy=true", "keepalive-time=500"]
    if fd is not None:
        src.insert(1, f"fd={fd}")
    return ["gst-launch-1.0", "-q", *src, "!", "videoconvert", "n-threads=4", "!",
            "videoscale", "add-borders=true", "n-threads=4", "!",
            f"video/x-raw,format=BGRx,width={w},height={h},pixel-aspect-ratio=1/1", "!",
            "fdsink", "fd=1", "sync=false"]


# ---------------------------------------------------------------------------------------------
# ffmpeg grabbers (X11 / Windows)
# ---------------------------------------------------------------------------------------------

def ffmpeg_command(backend, geometry, fps, w=CAPTURE_W, h=CAPTURE_H) -> list[str]:
    """geometry = (x, y, width, height) of the monitor in desktop pixels."""
    x, y, gw, gh = geometry
    if backend == "x11":
        inp = ["-f", "x11grab", "-draw_mouse", "1", "-framerate", str(fps), "-video_size", f"{gw}x{gh}",
               "-i", f"{os.environ.get('DISPLAY', ':0')}+{x},{y}"]
    else:  # gdi
        inp = ["-f", "gdigrab", "-draw_mouse", "1", "-framerate", str(fps), "-offset_x", str(x),
               "-offset_y", str(y), "-video_size", f"{gw}x{gh}", "-i", "desktop"]
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
          "format=bgr0")
    return ["ffmpeg", "-nostdin", "-v", "error", "-probesize", "32", "-analyzeduration", "0", *inp,
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr0", "-"]


# ---------------------------------------------------------------------------------------------
# Reader + GPU provider
# ---------------------------------------------------------------------------------------------

class ScreenCapture:
    """Runs a capture subprocess and keeps the newest BGRx frame. `error` is set when the stream
    ends (shared window closed, share revoked from the desktop's tray indicator, ...)."""

    def __init__(self, cmd: list[str], pass_fds=(), w=CAPTURE_W, h=CAPTURE_H, on_end=None):
        self.cmd, self.size = cmd, (w, h)
        self._pass_fds = tuple(pass_fds)
        self.on_end = on_end
        self.frame: np.ndarray | None = None
        self.seq = 0
        self.error: str | None = None
        self._stop = False
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      pass_fds=self._pass_fds, bufsize=w * h * 8)
        self._t = threading.Thread(target=self._run, name="r9700-screen", daemon=True)
        self._t.start()

    def _run(self):
        w, h = self.size
        n = w * h * 4
        while not self._stop:
            buf = self._proc.stdout.read(n)
            if len(buf) < n:
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
            self.seq += 1
        if not self._stop:
            err = b""
            try:
                self._proc.wait(timeout=2)
                err = self._proc.stderr.read() or b""
            except Exception:  # noqa: BLE001
                pass
            last = err.decode(errors="replace").strip().splitlines()[-1:] or ["stream ended"]
            self.error = "screen share ended: " + last[0]
            if self.on_end:
                self.on_end(self.error)

    def wait_first_frame(self, timeout=5.0) -> bool:
        t = time.monotonic() + timeout
        while self.frame is None and self.error is None and time.monotonic() < t:
            time.sleep(0.02)
        return self.frame is not None

    def close(self):
        self._stop = True
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._t.join(timeout=2)


class ScreenShare:
    """A running screen share (portal session or ffmpeg grabber) + its frame reader."""

    def __init__(self, capture: ScreenCapture, portal: PortalSession | None = None, label: str = "Screen"):
        self.capture, self.portal, self.label = capture, portal, label

    @property
    def restore_token(self):
        return self.portal.restore_token if self.portal else None

    def close(self):
        self.capture.close()
        if self.portal:
            self.portal.close()


def start_share(kind: str = "monitor", restore_token: str | None = None, geometry=None, fps: int = 30,
                on_end=None) -> ScreenShare:
    """Blocking (the portal waits for the user's pick): call from a worker thread.
    kind: "monitor" | "window". geometry (x, y, w, h) is only used by the x11/gdi backends."""
    backend = backend_name()
    if backend is None:
        raise RuntimeError(missing_reason())
    if backend == "portal":
        ps = PortalSession()
        try:
            ps.start(WINDOW if kind == "window" else MONITOR, restore_token)
            cap = ScreenCapture(gst_command(ps.node, ps.fd), pass_fds=(ps.fd,), on_end=on_end)
        except Exception:
            ps.close()
            raise
        label = "Window" if kind == "window" else "Screen"
        if ps.size:
            label += f" {ps.size[0]}x{ps.size[1]}"
        share = ScreenShare(cap, ps, label)
    else:
        if geometry is None:
            raise RuntimeError("pick a monitor")
        share = ScreenShare(ScreenCapture(ffmpeg_command(backend, geometry, fps), on_end=on_end),
                            label=f"Screen {geometry[2]}x{geometry[3]}")
    if not share.capture.wait_first_frame(8.0):
        err = share.capture.error or "no frames from the screen capture"
        share.close()
        raise RuntimeError(err)
    return share


class ScreenBackground:
    """GPU-side provider for Compositor.background_src. Screens are always fitted (letterboxed),
    never cropped, so nothing that is being shown gets cut off."""

    kind = "screen"

    def __init__(self, share: ScreenShare):
        self.share = share
        self.path = None
        self.fit = "contain"
        self._cache = None   # (seq, out_size, tensor)

    def __call__(self, h, w, device):
        import torch
        import torch.nn.functional as F
        cap = self.share.capture
        seq, frame = cap.seq, cap.frame
        if frame is None:
            return self._cache[2] if self._cache else None
        if self._cache is not None and self._cache[0] == seq and self._cache[1] == (h, w):
            return self._cache[2]
        t = torch.from_numpy(frame).pin_memory().to(device, non_blocking=True)
        t = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).half() / 255     # BGRx -> RGB
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        self._cache = (seq, (h, w), t)
        return t

    def close(self):
        self.share.close()
