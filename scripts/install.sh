#!/bin/bash
# One-command installer for Virtual Greenscreen R9700 Broadcast Studio (Linux + AMD GPU with ROCm).
#
#   scripts/install.sh                 required models only (realtime engine)
#   scripts/install.sh --all           + Studio engines + subject lock (recommended, ~3.4 GB download)
#
# Options:
#   --studio / --lock / --all          optional model sets (see README "Optional downloads")
#   --accept-matanyone-license         accept MatAnyone 2's NON-COMMERCIAL license without asking
#   --decline-matanyone-license        skip MatAnyone 2 without asking
#   --rocm VER                         PyTorch ROCm wheel index (default 7.2 -> .../whl/rocm7.2)
#   --no-system-packages               don't run apt (install the packages yourself)
#   --no-desktop                       don't add the app-menu entry / ~/.local/bin launcher
#   --virtualcam                       also set up the V4L2 virtual camera for Zoom/Teams (sudo)
#
# Safe to re-run: every step skips what is already done.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

ROCM=7.2 SYS=1 DESKTOP=1 VCAM=0 FETCH=()
while [ $# -gt 0 ]; do
  case $1 in
    --studio|--lock|--all) FETCH+=("$1") ;;
    --accept-matanyone-license) export R9700_ACCEPT_MATANYONE_LICENSE=1 ;;
    --decline-matanyone-license) export R9700_ACCEPT_MATANYONE_LICENSE=0 ;;
    --rocm) ROCM=$2; shift ;;
    --no-system-packages) SYS=0 ;;
    --no-desktop) DESKTOP=0 ;;
    --virtualcam) VCAM=1 ;;
    -h|--help) sed -n 2,17p "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 1 ;;
  esac
  shift
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }

step "Checking the system"
[ "$(uname -s)" = Linux ] || die "Linux only (Windows/macOS are not supported)."
[ -e /dev/kfd ] || die "/dev/kfd missing: the AMD GPU kernel driver (amdgpu) is not loaded. Install an AMD GPU with a current kernel."
if ! { [ -r /dev/kfd ] && [ -w /dev/kfd ]; }; then
  die "No access to /dev/kfd. Run: sudo usermod -aG render,video \$USER   then log out and back in."
fi
lspci 2>/dev/null | grep -iE 'vga|display' | grep -i amd || echo "(lspci found no AMD display device; continuing)"

if [ $SYS = 1 ] && command -v apt-get >/dev/null; then
  PKGS=(python3-venv python3-dev ffmpeg pipewire-bin build-essential pkg-config libpipewire-0.3-dev git curl)
  MISSING=()
  for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
  if [ ${#MISSING[@]} -gt 0 ]; then
    step "Installing system packages: ${MISSING[*]} (sudo)"
    sudo apt-get update && sudo apt-get install -y "${MISSING[@]}"
  fi
elif [ $SYS = 1 ]; then
  echo "Not a Debian/Ubuntu system: install python3 (3.10+, with venv), ffmpeg, pipewire tools (pw-record),"
  echo "a C compiler, pkg-config, the PipeWire development headers, git and curl yourself."
fi
for c in ffmpeg pw-record git curl cc pkg-config; do command -v $c >/dev/null || die "'$c' not found (see above)."; done

PY=""
for c in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v $c >/dev/null && $c -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY=$c; break; fi
done
[ -n "$PY" ] || die "Python 3.10 or newer is required."

step "Python environment (.venv, using $PY)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
if .venv/bin/python -c 'import torch, sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null; then
  echo "PyTorch for ROCm already installed: $(.venv/bin/python -c 'import torch; print(torch.__version__)')"
else
  step "Installing PyTorch for ROCm $ROCM (large download, a few minutes)"
  .venv/bin/python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/rocm$ROCM"
fi
step "Installing Python packages"
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -c 'import torch, sys; sys.exit(0 if torch.version.hip else 1)' \
  || die "requirements.txt replaced PyTorch with a non-ROCm build. Re-run with --rocm matching your system."

step "Checking the GPU from PyTorch"
.venv/bin/python - <<'EOF' || die "PyTorch cannot use the GPU. Check 'rocminfo', your group membership (render, video), and the --rocm version."
import torch
n = torch.cuda.device_count()
assert n > 0, "no GPU visible"
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.gcnArchName}), {p.total_memory / 2**30:.0f} GB")
x = torch.randn(256, 256, device="cuda")
torch.cuda.synchronize()
print("  OK:", float((x @ x).abs().mean()) > 0)
EOF

step "Downloading models"
PATH="$ROOT/.venv/bin:$PATH" scripts/fetch-models.sh "${FETCH[@]}"

step "Building the PipeWire camera helper"
make -C native

if [ $DESKTOP = 1 ]; then
  step "Adding the app-menu entry and the r9700-broadcast command"
  scripts/install-desktop.sh
fi
if [ $VCAM = 1 ]; then
  step "Setting up the V4L2 virtual camera (sudo)"
  sudo scripts/setup-virtualcam.sh
fi

step "Self-test"
PYTHONPATH="$ROOT" .venv/bin/python -m r9700_broadcast --help >/dev/null && echo "  app imports OK"

printf '\n\033[1;32mInstalled.\033[0m Start it with:  %s\n' \
  "$([ $DESKTOP = 1 ] && echo 'r9700-broadcast   (or from the app menu)' || echo "$ROOT/bin/r9700-broadcast")"
[ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]] && [ $DESKTOP = 1 ] && \
  echo "Note: ~/.local/bin is not on your PATH yet; log out and back in, or run $ROOT/bin/r9700-broadcast"
exit 0
