#!/bin/bash
# Adds R9700 Broadcast Studio to the Plasma app menu and puts the launcher on PATH.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
make -C "$HERE/native" -s || echo "note: build bin/r9700-pw-camera with: make -C native (needs libpipewire-0.3-dev)"
mkdir -p ~/.local/share/applications ~/.local/bin ~/.local/share/icons/hicolor/scalable/apps
ln -sf "$HERE/bin/r9700-broadcast" ~/.local/bin/r9700-broadcast
cp "$HERE/assets/r9700-broadcast.svg" ~/.local/share/icons/hicolor/scalable/apps/
cat > ~/.local/share/applications/r9700-broadcast.desktop <<DESK
[Desktop Entry]
Type=Application
Name=R9700 Broadcast Studio
Comment=Virtual greenscreen, background blur and noise removal on Radeon AI PRO R9700
Exec=$HERE/bin/r9700-broadcast
Icon=r9700-broadcast
Categories=AudioVideo;Video;
Keywords=webcam;green screen;background;blur;virtual camera;noise;
StartupWMClass=r9700-broadcast
DESK
update-desktop-database ~/.local/share/applications 2>/dev/null || true
echo "Installed. Launch from the app menu or run: r9700-broadcast"
