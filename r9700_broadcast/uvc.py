"""UVC camera controls through v4l2-ctl (zoom / pan / tilt gimbal, exposure, focus, white balance).

The OBSBOT Tiny SE exposes its gimbal (pan_absolute / tilt_absolute), digital zoom and exposure as
standard UVC controls, so this works without OBSBOT's own software.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")


@dataclass
class Control:
    name: str
    type: str          # int | bool | menu
    min: int = 0
    max: int = 1
    step: int = 1
    default: int = 0
    value: int = 0
    inactive: bool = False
    menu: dict | None = None


def available() -> bool:
    return shutil.which("v4l2-ctl") is not None


def list_controls(dev: str) -> dict[str, Control]:
    if not available() or not dev.startswith("/dev/video"):
        return {}
    out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls-menus"], capture_output=True, text=True,
                         timeout=5).stdout
    ctrls: dict[str, Control] = {}
    last = None
    for line in out.splitlines():
        m = _LINE.match(line)
        if m:
            name, typ, rest = m.groups()
            kv = dict(re.findall(r"(\w+)=(-?\d+)", rest))
            c = Control(name, typ, int(kv.get("min", 0)), int(kv.get("max", 1)), int(kv.get("step", 1)),
                        int(kv.get("default", 0)), int(kv.get("value", 0)), "inactive" in rest,
                        {} if typ == "menu" else None)
            ctrls[name] = c
            last = c
        elif last is not None and last.menu is not None:
            mm = re.match(r"^\s*(\d+):\s*(.+)$", line)
            if mm:
                last.menu[int(mm.group(1))] = mm.group(2).strip()
    return ctrls


def set_control(dev: str, name: str, value: int) -> bool:
    r = subprocess.run(["v4l2-ctl", "-d", dev, "-c", f"{name}={int(value)}"], capture_output=True, timeout=5)
    return r.returncode == 0


def lock_exposure_for_fps(dev: str, fps: int, ctrls: dict[str, Control] | None = None) -> bool:
    """Manual exposure no longer than one frame interval (auto-exposure in dim rooms can stretch
    the shutter and silently drop a 100/120 fps camera to 30)."""
    ctrls = ctrls or list_controls(dev)
    if "auto_exposure" not in ctrls or "exposure_time_absolute" not in ctrls:
        return False
    manual = next((k for k, v in (ctrls["auto_exposure"].menu or {}).items() if "Manual" in v), 1)
    set_control(dev, "auto_exposure", manual)
    e = ctrls["exposure_time_absolute"]
    return set_control(dev, "exposure_time_absolute", max(e.min, min(e.max, int(10000 / fps))))  # 100 µs units


def auto_exposure(dev: str, ctrls: dict[str, Control] | None = None) -> bool:
    ctrls = ctrls or list_controls(dev)
    if "auto_exposure" not in ctrls:
        return False
    auto = next((k for k, v in (ctrls["auto_exposure"].menu or {}).items() if "Auto" in v), 0)
    return set_control(dev, "auto_exposure", auto)
