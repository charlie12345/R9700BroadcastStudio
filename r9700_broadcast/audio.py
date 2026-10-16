"""AI noise-removal microphone: DeepFilterNet3 (LADSPA) inside a private PipeWire filter-chain.

Starting it creates a virtual source "R9700 Broadcast Mic" that any app (Discord, Zoom, OBS) can
pick. Nothing is written to ~/.config/pipewire; stopping the process removes the device.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(HERE, "third_party", "deepfilter", "libdeep_filter_ladspa.so")

CONF = """
context.properties = {{ log.level = 0 }}
context.spa-libs = {{
    audio.convert.* = audioconvert/libspa-audioconvert
    support.*       = support/libspa-support
}}
context.modules = [
    {{ name = libpipewire-module-rt flags = [ ifexists nofail ] }}
    {{ name = libpipewire-module-protocol-native }}
    {{ name = libpipewire-module-client-node }}
    {{ name = libpipewire-module-adapter }}
    {{ name = libpipewire-module-filter-chain
        args = {{
            node.description = "R9700 Broadcast Mic (AI noise removal)"
            media.name       = "R9700 Broadcast Mic"
            filter.graph = {{
                nodes = [
                    {{
                        type   = ladspa
                        name   = deepfilter
                        plugin = "{plugin}"
                        label  = deep_filter_mono
                        control = {{
                            "Attenuation Limit (dB)" = {atten}
                            "Min processing threshold (dB)" = -15
                            "Max ERB processing threshold (dB)" = 35
                            "Max DF processing threshold (dB)" = 35
                            "Post Filter Beta" = 0
                        }}
                    }}
                ]
            }}
            audio.rate = 48000
            audio.position = [ MONO ]
            capture.props = {{
                node.name   = "r9700_broadcast_mic_in"
                node.passive = true
                {target}
            }}
            playback.props = {{
                node.name   = "r9700_broadcast_mic"
                media.class = Audio/Source
                audio.position = [ MONO ]
            }}
        }}
    }}
]
"""


def list_microphones() -> list[tuple[str, str]]:
    """[(node.name, description)] of hardware capture sources via pactl."""
    out = []
    try:
        txt = subprocess.run(["pactl", "list", "sources"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return out
    name = desc = None
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Description:"):
            desc = line.split(":", 1)[1].strip()
            if name and not name.endswith(".monitor") and "r9700_broadcast" not in name:
                out.append((name, desc))
    return out


class NoiseRemovalMic:
    def __init__(self):
        self.proc = None
        self.conf = None

    @staticmethod
    def available() -> bool:
        return os.path.exists(PLUGIN) and shutil.which("pipewire") is not None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, source_node: str | None = None, attenuation_db: float = 100):
        self.stop()
        target = f'target.object = "{source_node}"' if source_node else ""
        fd, self.conf = tempfile.mkstemp(prefix="r9700-mic-", suffix=".conf")
        with os.fdopen(fd, "w") as f:
            f.write(CONF.format(plugin=PLUGIN, atten=attenuation_db, target=target))
        env = dict(os.environ, RUST_LOG="error")
        self.proc = subprocess.Popen(["pipewire", "-c", self.conf], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     start_new_session=True)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self.conf and os.path.exists(self.conf):
            os.unlink(self.conf)
        self.conf = None
