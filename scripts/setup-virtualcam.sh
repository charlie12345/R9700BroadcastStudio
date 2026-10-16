#!/bin/bash
# One-time setup for the R9700 Broadcast V4L2 virtual cameras (v4l2loopback). Needs sudo.
#   /dev/video42  "R9700 Broadcast Camera"       YUYV   - Zoom, Discord, Teams, browsers
#   /dev/video43  "R9700 Broadcast Alpha (OBS)"  ABGR32 - OBS "Video Capture Device (V4L2)" with transparency
# DKMS builds the module for every installed kernel and signs it with the DKMS/Ubuntu MOK key
# (/var/lib/shim-signed/mok). With Secure Boot on, that key must be enrolled once (mokutil prompt).
set -euo pipefail
if [ "$(id -u)" != 0 ]; then exec sudo "$0" "$@"; fi
apt-get install -y v4l2loopback-dkms v4l2loopback-utils
K=$(uname -r)
if ! modinfo -k "$K" v4l2loopback >/dev/null 2>&1; then
  echo ">> building v4l2loopback for $K"
  dkms autoinstall -k "$K"
fi
cat > /etc/modprobe.d/r9700-broadcast-v4l2loopback.conf <<'CONF'
options v4l2loopback devices=2 video_nr=42,43 card_label="R9700 Broadcast Camera,R9700 Broadcast Alpha (OBS)" exclusive_caps=1,1 max_buffers=2
CONF
echo v4l2loopback > /etc/modules-load.d/r9700-broadcast-v4l2loopback.conf
modprobe -r v4l2loopback 2>/dev/null || true
modprobe v4l2loopback
sleep 0.5
for n in 42 43; do
  printf '/dev/video%s  [%s]\n' "$n" "$(cat /sys/class/video4linux/video$n/name 2>/dev/null || echo MISSING)"
done
[ -e /dev/video42 ] && [ -e /dev/video43 ] && echo "OK: virtual cameras ready" || { echo "FAILED: devices missing (see dmesg | tail)"; exit 1; }
