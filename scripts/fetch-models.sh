#!/bin/bash
# Downloads third-party code and model weights (none of them are stored in this repository).
#
#   scripts/fetch-models.sh            required: RVM + DeepLabV3 + DeepFilterNet
#   scripts/fetch-models.sh --studio   + BiRefNet (Studio / Studio Max engines)
#   scripts/fetch-models.sh --lock     + SAM 2.1, Grounding DINO, MatAnyone 2 (subject lock)
#   scripts/fetch-models.sh --all      everything
#
# Each component keeps its own license; see THIRD_PARTY_NOTICES.md. MatAnyone 2 is NON-COMMERCIAL
# (NTU S-Lab License 1.0) and is only downloaded with --lock / --all after you accept it
# (interactive prompt, or R9700_ACCEPT_MATANYONE_LICENSE=1).
set -euo pipefail
cd "$(dirname "$0")/.."
STUDIO=0 LOCK=0
for a in "$@"; do
  case $a in
    --studio) STUDIO=1 ;;
    --lock) LOCK=1 ;;
    --all) STUDIO=1 LOCK=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done
hf_get() {  # repo dir [include patterns...]
  local repo=$1 dir=$2; shift 2
  local inc=(); for i in "$@"; do inc+=(--include "$i"); done
  if command -v hf >/dev/null; then hf download "$repo" "${inc[@]}" --local-dir "$dir"
  else huggingface-cli download "$repo" "${inc[@]}" --local-dir "$dir"; fi
}
mkdir -p models/torchvision models/sam2 models/matanyone2 third_party/deepfilter

# --- required -------------------------------------------------------------------------------------
# Robust Video Matting (GPL-3.0): realtime engine
[ -d third_party/RobustVideoMatting ] || git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting.git third_party/RobustVideoMatting
for m in mobilenetv3 resnet50; do
  [ -f models/rvm_$m.pth ] || curl -fL -o models/rvm_$m.pth https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_$m.pth
done
# DeepLabV3 (torchvision, BSD-3-Clause): person / chair semantics
for f in deeplabv3_mobilenet_v3_large-fc3c493d.pth; do
  [ -f models/torchvision/$f ] || curl -fL -o models/torchvision/$f https://download.pytorch.org/models/$f
done
# DeepFilterNet LADSPA plugin (MIT / Apache-2.0): noise-removal microphone
[ -f third_party/deepfilter/libdeep_filter_ladspa.so ] || curl -fL -o third_party/deepfilter/libdeep_filter_ladspa.so \
  https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/libdeep_filter_ladspa-0.5.6-x86_64-unknown-linux-gnu.so

# BlazeFace short-range (Google AI Edge, Apache-2.0): face box + eye/nose keypoints for fun props
mkdir -p models/mediapipe
[ -f models/mediapipe/blaze_face_short_range.tflite ] || curl -fL -o models/mediapipe/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# --- optional: Studio engines -------------------------------------------------------------------------
if [ $STUDIO = 1 ]; then
  hf_get ZhengPeng7/BiRefNet_lite-matting models/birefnet_lite_matting   # MIT
  hf_get ZhengPeng7/BiRefNet-matting models/birefnet_matting             # MIT
fi

# --- optional: subject lock ----------------------------------------------------------------------------
if [ $LOCK = 1 ]; then
  # SAM 2.1 (Apache-2.0)
  [ -d third_party/sam2 ] || git clone https://github.com/facebookresearch/sam2.git third_party/sam2
  git -C third_party/sam2 checkout -q 2b90b9f5ceec907a1c18123530e92e794ad901a4
  for c in large base_plus; do
    [ -f models/sam2/sam2.1_hiera_$c.pt ] || curl -fL -o models/sam2/sam2.1_hiera_$c.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_$c.pt
  done
  # Grounding DINO base (Apache-2.0)
  hf_get IDEA-Research/grounding-dino-base models/grounding-dino-base "*.json" "*.txt" "model.safetensors"
  # MatAnyone 2 (NTU S-Lab License 1.0: non-commercial use only)
  echo
  echo "MatAnyone 2 is licensed under the NTU S-Lab License 1.0 (NON-COMMERCIAL use only):"
  echo "  https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt"
  ok=${R9700_ACCEPT_MATANYONE_LICENSE:-}
  case $ok in 1|y|yes) ok=y ;; 0|n|no) ok=n ;; *)
    if [ -t 0 ]; then read -r -p "Download it and accept that license? [y/N] " ok || ok=n
    else echo "(non-interactive: set R9700_ACCEPT_MATANYONE_LICENSE=1 to accept)"; ok=n; fi ;;
  esac
  if [[ $ok =~ ^[Yy] ]]; then
    [ -d third_party/MatAnyone2 ] || git clone https://github.com/pq-yang/MatAnyone2.git third_party/MatAnyone2
    git -C third_party/MatAnyone2 checkout -q 0079197acd6d16a741f71558809c06c586c579e0
    [ -f models/matanyone2/matanyone2.pth ] || curl -fL -o models/matanyone2/matanyone2.pth \
      https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth
  else
    echo "Skipped MatAnyone 2: the app uses RVM + the semantic guard instead."
  fi
fi
echo "Done."
