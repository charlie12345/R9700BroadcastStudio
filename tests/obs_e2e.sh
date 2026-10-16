#!/bin/bash
# End-to-end test inside the real OBS Studio flatpak, invisibly (headless KWin):
#   r9700-broadcast (Transparent) -> PipeWire camera -> OBS "Video Capture Device (PipeWire)"
#   composited over a magenta colour source -> OBS recording in .obs-test/
# Uses a throwaway profile + scene collection and restores your OBS user.ini afterwards.
# usage: tests/obs_e2e.sh [preset 60|120] [source pipewire|v4l2]
#   pipewire: OBS "Video Capture Device (PipeWire)" <- our PipeWire camera (BGRA)
#   v4l2:     OBS "Video Capture Device (V4L2)"     <- /dev/video43 ABGR32 (needs setup-virtualcam.sh)
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PRESET=${1:-60}
SRC=${2:-pipewire}
if [ "$PRESET" = 120 ]; then W=1280 H=720 FPS=120; else W=1920 H=1080 FPS=60; fi
O=~/.var/app/com.obsproject.Studio/config/obs-studio
T="$HERE/.obs-test"; mkdir -p "$T"; rm -f "$T"/*.mkv
cp "$O/user.ini" "$T/user.ini.bak"
restore() {
  cp "$T/user.ini.bak" "$O/user.ini"
  rm -rf "$O/basic/profiles/R9700Test" "$O/basic/scenes/R9700Test.json" "$O/basic/scenes/R9700Test.json.bak"
}
trap restore EXIT
mkdir -p "$O/basic/profiles/R9700Test"
cat > "$O/basic/profiles/R9700Test/basic.ini" <<INI
[General]
Name=R9700Test
[Video]
BaseCX=$W
BaseCY=$H
OutputCX=$W
OutputCY=$H
FPSType=2
FPSNum=$FPS
FPSDen=1
[Output]
Mode=Simple
[SimpleOutput]
FilePath=$T
RecFormat2=mkv
RecQuality=HQ
RecEncoder=x264
INI
python3 - "$O/basic/scenes/R9700Test.json" $W $H $FPS $SRC <<'PY'
import json, sys
p, w, h, fps, src = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
if src == "v4l2":
    kind = "v4l2_input"
    cam = {"device_id": "/dev/video43", "input": 0, "pixelformat": 0x34325241,   # V4L2 ABGR32
           "resolution": (w << 32) | h, "framerate": (1 << 32) | fps, "buffering": False}
else:
    kind = "pipewire-camera-source"
    cam = {"device_id": "r9700_broadcast_camera",
           "format": json.dumps({"video_format": 12, "encoded": False}),        # 12 = BGRA (alpha)
           "resolution": json.dumps({"width": w, "height": h}),
           "framerate": json.dumps({"numerator": fps, "denominator": 1})}
u = ["11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222", "33333333-3333-4333-8333-333333333333"]
item = lambda n, i, k: {"name": n, "source_uuid": u[i], "visible": True, "id": k, "pos": {"x": 0, "y": 0},
                        "scale": {"x": 1, "y": 1}, "bounds_type": 0, "align": 5}
json.dump({"name": "R9700Test", "current_scene": "S", "current_program_scene": "S", "scene_order": [{"name": "S"}],
  "sources": [
    {"id": "color_source_v3", "versioned_id": "color_source_v3", "name": "Magenta", "uuid": u[0],
     "settings": {"color": 4294902015, "width": w, "height": h}},
    {"id": kind, "versioned_id": kind, "name": "R9700", "uuid": u[1], "settings": cam},
    {"id": "scene", "versioned_id": "scene", "name": "S", "uuid": u[2],
     "settings": {"id_counter": 2, "items": [item("Magenta", 0, 1), item("R9700", 1, 2)]}}]}, open(p, "w"))
PY
PYTHONWARNINGS=ignore timeout -s INT 45 "$HERE/bin/r9700-broadcast" live --preset "$PRESET" --mode transparent \
  --output $([ "$SRC" = v4l2 ] && echo v4l2alpha || echo pipewire) >"$T/producer.txt" 2>&1 &
sleep 6
SOCK=wayland-r9700-obstest
kwin_wayland --virtual --socket=$SOCK --width $W --height $H >/dev/null 2>&1 &
KW=$!
sleep 3
WAYLAND_DISPLAY=$SOCK flatpak run --env=WAYLAND_DISPLAY=$SOCK com.obsproject.Studio --multi --profile R9700Test --collection R9700Test \
  --startrecording --disable-updater --disable-shutdown-check >"$T/obs_stdout.txt" 2>&1 &
sleep 22
pkill -INT -f "obs --multi --profile R9700Test"; sleep 4; pkill -f "obs --multi --profile R9700Test"
kill $KW 2>/dev/null; wait 2>/dev/null
LOG=$(ls -t "$O"/logs/*.txt | head -1)
grep -E "\[pipewire\] +(Format|Size|Framerate)|state: \"error\"|v4l2-input: .*(Pixelformat|Resolution|Framerate|Failed|Unable)" "$LOG"
F=$(ls -t "$T"/*.mkv | head -1)
n=$(ffmpeg -v info -ss 5 -t 10 -i "$F" -vf "mpdecimate=hi=64:lo=32:frac=0.1,showinfo" -f null - 2>&1 | grep -c "Parsed_showinfo.*n:")
echo "OBS recording: $F  unique frames/s: $(echo "scale=1; $n/10" | bc)"
ffmpeg -v error -ss 10 -i "$F" -frames:v 1 -y "$T/frame.png" && echo "frame (you over magenta = alpha OK): $T/frame.png"
