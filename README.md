# Virtual Greenscreen R9700 Broadcast Studio

A virtual greenscreen studio for **AMD Radeon AI PRO R9700** (gfx1201, ROCm), built around
**AI green screen without a green screen**. Your webcam goes in, and a clean keyed subject comes out
over blur, a replacement image or video, solid colour, or a virtual chroma-green or blue screen
for OBS, or **real transparency straight into OBS Studio**. Outputs are a native PipeWire camera
(no driver, up to 120 fps, carries alpha) and a classic V4L2 virtual camera, plus an AI
noise-removal microphone.

```
bin/r9700-broadcast                                       # GUI
bin/r9700-broadcast live --preset 120 --mode transparent  # headless, alpha into OBS at 120 fps
bin/r9700-broadcast live --preset 60 --mode replace --background "assets/backgrounds/07 Aurora Loop.mp4"
bin/r9700-broadcast file in.mp4 out.mov --engine birefnet # offline: ProRes 4444 + alpha for Resolve
```

## Requirements

* **Linux** with PipeWire (tested on Ubuntu 26.04, kernel 7.3, Plasma 6). Windows/macOS are not supported.
  AMD GPUs with ROCm only.
* **AMD GPU with ROCm**: built and tested on Radeon AI PRO R9700 (gfx1201) with ROCm 10 / PyTorch 2.11
  and with the stock PyTorch 2.14 ROCm 7.2 wheel.
  Other ROCm-supported cards (RX 9070 / 7900 series) should work; one GPU is enough, a second one
  is used automatically for the subject lock and Studio engines.
* **Disk**: ~22 GB (PyTorch for ROCm ~16 GB installed, models ~3.4 GB).
* **VRAM**: ~1.2 GB for Realtime, ~4.2 GB with the subject lock, up to ~5.2 GB for Studio Max.
* Python 3.10+, `ffmpeg`, `pw-record`, a C compiler and `libpipewire-0.3-dev`, `git`, `curl`, and
  GStreamer with its PipeWire plugin (`gstreamer1.0-tools gstreamer1.0-pipewire`, for screen share).
  `scripts/install.sh` installs all of these on Debian/Ubuntu.
* Your user in the `render` and `video` groups (`sudo usermod -aG render,video $USER`, then log out/in).
* A UVC webcam. Tuned for the OBSBOT Tiny SE (1080p100 / 720p120), any UVC camera works.

## Install

### Quick install (one command)

```bash
git clone https://github.com/charlie12345/R9700BroadcastStudio.git
cd R9700BroadcastStudio
scripts/install.sh --all
```

That checks your system, installs missing packages (Debian/Ubuntu, asks for sudo), creates a
Python environment with PyTorch for ROCm, downloads the models (asks before the non-commercial
MatAnyone 2), builds the camera helper, adds **R9700 Broadcast Studio** to your app menu and runs a
self-test. About 10 GB of downloads (PyTorch for ROCm is 6.6 GB of it), ~22 GB of disk, and
5 minutes on a fast connection. It is safe to re-run.

Useful options: `--rocm 7.1` (other PyTorch ROCm wheel), `--virtualcam` (Zoom/Teams camera, sudo),
`--accept-matanyone-license` / `--decline-matanyone-license` (no prompt), `--no-system-packages`,
`--help`.

### Install with an AI assistant (Claude Code, Codex, Gemini CLI, Cursor, ChatGPT…)

Paste this into your AI coding assistant or command-line agent:

> Install Virtual Greenscreen R9700 Broadcast Studio from https://github.com/charlie12345/R9700BroadcastStudio on
> this computer. Clone it, then read and follow AGENTS.md in the repository exactly.

[`AGENTS.md`](AGENTS.md) gives the assistant the system checks, the one license question it must
ask you, the exact non-interactive install command, how to verify, and a troubleshooting table.
Chat-only assistants that cannot run commands can read the same file and walk you through it.

### Manual install

```bash
git clone https://github.com/charlie12345/R9700BroadcastStudio.git
cd R9700BroadcastStudio

# 1. system packages (Debian/Ubuntu names)
sudo apt install python3-venv python3-dev ffmpeg pipewire-bin build-essential pkg-config libpipewire-0.3-dev git curl

# 2. Python environment with PyTorch for ROCm (wheel index list: https://pytorch.org/get-started/locally/)
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
.venv/bin/pip install -r requirements.txt

# 3. models (see "Optional downloads" below)
PATH="$PWD/.venv/bin:$PATH" scripts/fetch-models.sh --all

# 4. PipeWire camera helper + app-menu entry + `r9700-broadcast` on PATH
make -C native
scripts/install-desktop.sh

# 5. optional: classic V4L2 virtual camera for Zoom / Teams / Discord (needs root, DKMS)
sudo scripts/setup-virtualcam.sh
```

If you already have a working ROCm PyTorch in another environment, you can reuse it instead of
step 2: create `.venv`, add a `.pth` file pointing at that environment's `site-packages`, and
install `requirements.txt` with `--no-deps` so pip does not replace your ROCm torch.

Tested end to end on a fresh clone: Ubuntu 26.04, Python 3.14, 2× Radeon AI PRO R9700, stock
PyTorch 2.14 ROCm 7.2 wheel. Install 4 min 18 s, all 46 tests pass, same speed as a custom ROCm build.

## Optional downloads

Nothing large is stored in this repository; `scripts/fetch-models.sh` downloads it.

| Set | Flag | What | Size | Without it |
|---|---|---|---|---|
| Required | (none) | RVM MobileNetV3 + ResNet50, DeepLabV3, DeepFilterNet plugin | ~0.2 GB | – |
| Studio engines | `--studio` | BiRefNet-lite matting, BiRefNet matting | ~0.9 GB | Studio / Studio Max presets hidden |
| Subject lock | `--lock` | SAM 2.1 large + base+, Grounding DINO base, MatAnyone 2 | ~2.3 GB | falls back to RVM + semantic guard (more headset/chair ghosting) |

MatAnyone 2 is **non-commercial** (NTU S-Lab License 1.0); the script asks before downloading it.

## Run

```bash
r9700-broadcast                    # GUI (or bin/r9700-broadcast, or the app menu entry)
r9700-broadcast live --preset 60 --mode blur
r9700-broadcast file in.mp4 out.mov --engine birefnet
r9700-broadcast --help
```

1. Close the webcam in other apps (OBS, browser tabs, Zoom). Only one app can use it at a time.
2. Pick an FPS preset and your camera in the top bar, then a background tile.
3. Look at the camera and wait ~10 s for **Subject → "Locked on to you"** (subject lock).
4. Turn on an output: **● PipeWire camera** (OBS, browsers), **● V4L2 camera** (Zoom/Teams), or
   **● Rec** (video + sound). **📷 Snapshot** saves a picture and opens the thumbnail maker.
5. If the edges look wrong after experimenting, **Subject → Reset matte to defaults**.

Tests: `PYTHONPATH=. .venv/bin/python -m pytest -q tests` (46 tests; some need a GPU, PipeWire or
ffmpeg and are skipped otherwise).

## The front end

* **Top bar**: FPS preset, camera (or "Open video file as camera…"), quality engine, and the
  ● PipeWire camera / ● V4L2 camera / ● Rec buttons.
* **Draw bar** (above the preview): pen, highlighter, arrow, line, box, circle, text, numbered
  steps, laser pointer, spotlight, magnifier and eraser, six colours, three widths, auto-fade, shape
  snapping, "over me", undo/redo/clear. See "Drawing on the stream" below.
* **Background strip**: one-click tiles for *Camera · Blur · Share screen · Share window · Green screen · Blue screen ·
  Transparent · Solid colour · Matte*, then your library and 8 built-in backgrounds (6 images,
  2 looping videos). Use **+ Add images / videos…**, *Ctrl+O*, or **drag files anywhere onto the
  window**. Right-click removes an item from the library (your files are never deleted).
  Thumbnails are generated in the background and cached.
* **Tabs**:
  * **Background**: blur amount with Light/Medium/Strong presets; **"Blur it"** (depth of field
    for a picture or video background); Fill/Fit/Stretch; dim; light wrap; solid colour.
  * **Layout**: *Full size* (you over the background) or *Bubble* (you in a circle, rounded
    square, square or frameless cut-out). See "Screen share and the bubble" below.
  * **Subject**: matte choke/feather/clip, temporal smoothing, clean plate (sensitivity +
    furniture removal), virtual key light, vignette, video noise removal.
  * **Camera**: software auto-frame and mirror, plus **OBSBOT hardware controls** (gimbal pan/tilt,
    zoom, gain, white balance, autofocus, Centre gimbal) and *Lock exposure to the frame rate*.
  * **Audio**: noise-removal mic.
  * **Output**: setup notes, recording codec, and GPU selection.

### Screen share and the bubble

* **Screen share as the background**: the *Share screen* / *Share window* tiles (or *Background →
  Share a screen… / Share a window…*) put your monitor or one window behind you, with the mouse
  pointer in it, and take over the output from any mode (green screen, transparent, blur…).
  Screens are letterboxed, never cropped. **Crop to a region…** shows only part of the share (drag a
  box over a still of it); it works on every desktop, and KDE's own *Share region* works too.
  How it captures:

  | Desktop | Capture |
  |---|---|
  | Wayland: KDE Plasma, GNOME, Sway/Hyprland/COSMIC (any xdg-desktop-portal with ScreenCast) | the desktop's own picker → PipeWire stream (GStreamer `pipewiresrc`), read straight from the PipeWire graph. The portal's `OpenPipeWireRemote` is only used inside Flatpak: with xdg-desktop-portal 1.21 + KDE it deadlocks the whole portal for window/region streams. Tick "Allow … without asking next time" in KDE's dialog to skip it next time |
  | X11 sessions (GNOME on Xorg, XFCE, Cinnamon, MATE, …) | ffmpeg `x11grab`, pick the monitor in the Background tab |
  | Windows | ffmpeg `gdigrab` backend is written but untested; the rest of the app is Linux-only today |

* **Bubble (Layout tab)**: you in a circle / rounded square / square / frameless cut-out over *any*
  background: screen share, pictures, videos, blur, green/blue screen and transparent (OBS gets
  alpha only inside the bubble). It starts in the lower-right corner. **Drag it in the preview** to
  move it, **scroll over it** to resize, or use the corner buttons. Options: zoom, follow my head,
  what fills the bubble behind you (blurred room / real room / solid colour), ring colour (white,
  red, black, animated rainbow, none) and width, drop shadow, and **pop out**, where your head and
  headset break out over the top of the frame. Fun props ride along inside the bubble.
* **Drawing on the stream** (draw bar above the preview; drawings go into every output and
  recording, on any background):

  | Tool | Key | |
  |---|---|---|
  | ✋ Move | Esc | stop drawing; drag the bubble |
  | ✏️ Pen · 🖍️ Highlighter | P · H | snap turns a rough circle/box/line into a clean one; the highlighter is see-through and straightens |
  | ➜ Arrow · ╱ Line · ▭ Box · ◯ Circle | A · I · R · O | Shift = 45° / square / circle |
  | T Text · ① Steps | T · N | click, type, Enter · click to drop 1, 2, 3… |
  | 🔴 Laser · 🔦 Spotlight · 🔍 Magnifier | L · S · M | glowing dot with fading trail · dim all but the pointer · 2.2× loupe |
  | ⌫ Eraser | X | click or drag over a drawing |

  Ctrl+Z / Ctrl+Shift+Z undo/redo (right-click also undoes), Delete clears. *Fade* makes drawings
  disappear by themselves after 3/6/12 s. Drawings sit on the background under you; tick *Over me*
  to draw on top of you too.
* **Mirror** now flips only the camera, so a screen share or a picture with text in it never
  comes out backwards.

### Background formats

| | Formats | How |
|---|---|---|
| Images | png jpg webp bmp tiff avif heic jxl tga exr … | OpenCV, ffmpeg fallback |
| Video / animated | mp4 mov mkv webm avi m4v mpg wmv flv ts ogv mxf **gif** apng … | streamed by ffmpeg, looped at native rate, scaled on the fly (no RAM preload) |

All 18 sample formats are covered by `tests/test_backgrounds.py`.

## FPS presets (OBSBOT Tiny SE, measured end to end)

Measured with `tests/fps_presets.py`: OBSBOT → pipeline → PipeWire camera → a separate
GStreamer process (what OBS receives).

| Preset | Camera mode | Engine | Received fps | Glass→output |
|---|---|---|---|---|
| 30 FPS · Studio quality | 1080p30 | BiRefNet-lite, 2 GPUs | 30.0 | ~10–38 ms |
| 60 FPS · Realtime | 1080p60 | RVM + HIP graphs | 60.0 | 7–11 ms |
| 100 FPS · 1080p high-speed | 1080p100 MJPEG | RVM + HIP graphs | 99.8–100.0 | 7–9 ms |
| 120 FPS · 720p high-speed | 720p120 MJPEG | RVM + HIP graphs | 120.0 | 5–6 ms |

0 dropped frames in every blur/replace/transparent run, and 0 drops over 30 s steady state with
the GUI and preview running. Without the camera limit, the GPU path runs at 141 fps (1080p) and
186 fps (720p).

What made it fast:
* **HIP graph replay for RVM**: CPU launch time 3.5 → 1.1 ms/frame, bit-identical output.
* **Compositor rewrite**: blur 7.4 → 1.8 ms (one fused normalised-convolution pass on a pyramid
  level); replace 5.9 → 1.35 ms (light wrap at ¼ res, cached constants, no CPU syncs).
* **Three-thread pipeline** (`pipeline.py`): capture / GPU / output, with BGRA built on the GPU
  and an async D2H copy into a pinned ring.
* **V4L2 capture queue of 4 buffers**: with a single buffer, GIL jitter cost about 10% of frames
  at 100/120 fps.

About "drivers" for higher frame rates: the OBSBOT Tiny SE needs no special driver on Linux. Its
1080p100 and 720p120 modes are standard UVC MJPEG and are used directly. OBSBOT Center (vendor
app) is Windows/macOS only; its gimbal, zoom and exposure are standard UVC controls and appear in
the Camera tab. The only thing to watch at 100/120 fps is auto exposure in a dim room; use *Lock
exposure to the frame rate*.

## Getting it into OBS Studio (high fps, with alpha)

### Which virtual camera to use

| Option | OBS source | Alpha | Max fps | Setup | Works in Zoom/Discord | Status |
|---|---|---|---|---|---|---|
| **PipeWire camera** (ours) | Video Capture Device (PipeWire) | ✓ BGRA | 120 verified | none | ✗ (Firefox/Chrome only) | **verified in OBS 32.2.2** |
| v4l2loopback chat cam `/dev/video42` | Video Capture Device (V4L2) | ✗ (YUYV) | any | `sudo scripts/setup-virtualcam.sh` (DKMS) | ✓ | for Zoom / Discord / Teams |
| **v4l2loopback alpha cam** `/dev/video43` | Video Capture Device (V4L2), format *32-bit ABGR* | ✓ ABGR32* | any | same script | ✗ | OBS fallback through its mature V4L2 source |
| NDI (DistroAV plugin) | NDI Source | ✓ | high | proprietary NDI runtime + plugin | ✗ | not used: lossy, extra CPU and latency, only useful across PCs |
| Custom OBS plugin (shared memory) | own source | ✓ | any | must match the Flatpak's libobs ABI | ✗ | not worth it given the PipeWire results |

\* OBS maps V4L2 `ABGR32` → `VIDEO_FORMAT_BGRA`, and v4l2loopback 0.15.3 accepts `ABGR32`.
`r9700_broadcast/v4l2out.py` writes it with raw ioctls (pyvirtualcam can't emit ABGR32). Its struct
layouts are checked against `<linux/videodev2.h>` by `tests/test_v4l2out.py`. ABGR32 is B,G,R,A in
memory, the same bytes as the pipeline's BGRA frames, so no conversion is needed.
Test it in OBS with `tests/obs_e2e.sh 60 v4l2`.

**About PipeWire trouble in OBS with the raw webcam**: OBS's *PipeWire camera source* opening
the OBSBOT directly and negotiating **YUY2 640×360 @ 30**. OBS's PipeWire camera source skips
compressed formats, and the OBSBOT's only uncompressed mode is 640×360, so 1080p MJPG was
impossible that way. The V4L2 source (MJPG 1080p100) was the right call *for the raw webcam*.
That problem doesn't apply here: this app reads the OBSBOT over V4L2 MJPG, and OBS only sees our
**uncompressed BGRA** stream at full resolution and rate.

### Steps

1. In the studio, click **● PipeWire camera**, then pick **Transparent** (or any other look).
2. Close the OBSBOT in OBS first: remove or hide your existing *Video Capture Device (V4L2)*
   source. Only one app can open the webcam, and the studio needs it.
3. In OBS: **Sources → + → Video Capture Device (PipeWire)** → **R9700 Broadcast Camera**. In its
   properties keep *Video format* = **BGRA** (first in the list; this carries the alpha), and pick
   the resolution/frame rate of your preset.
4. Set the OBS canvas FPS (Settings → Video) to 60, 100 or 120 to match.

### Verified inside the real OBS (`tests/obs_e2e.sh 60|120`)

`tests/obs_e2e.sh` runs the OBS Flatpak invisibly in a headless KWin with a throwaway profile
(your `user.ini` is backed up and restored). It puts our camera over a magenta colour source and
records:

| Preset | OBS negotiated | Unique frames/s in OBS's recording | Alpha |
|---|---|---|---|
| 60 | BGRA 1920×1080 @ 60/1 | 60.0 | ✓ you over magenta |
| 120 | BGRA 1280×720 @ 120/1 | 120.0 | ✓ |

Gotcha found by this test: OBS's camera source connects with the *single format selected in its
properties*. A source with no format chosen fails with "no more input formats". The properties
dialog selects the first format (BGRA) for you; scripted scene files must set it.

## Subject lock: no ghosting around headsets and chairs (default)

Every matting net (RVM, BiRefNet, MODNet...) decides "person or background" from colour and edges.
A **black/red headset in front of a black/red gaming chair** therefore turns semi-transparent
("ghosting"), the side of the head ghosts while you turn, and the chair's top and wings are kept.
Bigger matting models do not fix it; the information to separate the two is *semantic*: which
object is you.

The subject lock is built from current research:

| Stage | Model | Paper / license |
|---|---|---|
| Who is the subject | SAM 2.1 (person, pixel-accurate) + Grounding DINO ("headphones / headset / microphone") → SAM 2.1 per accessory, DeepLabV3 person map for prompts | Ravi et al. 2024 (Apache-2.0); Liu et al. ECCV 2024 (Apache-2.0); "Grounded SAM", Ren et al. 2024 |
| Track that subject | **MatAnyone 2** (memory-propagation video matting) at 384p on the second GPU | Yang et al., *CVPR 2026* (NTU S-Lab License 1.0, non-commercial) |
| Fine hair edges | RVM at full resolution, every frame | Lin et al., WACV 2022 (GPL-3.0) |
| Fusion | `alpha = clamp(RVM, erode(core), dilate(core))`, band ≈ 10 px | MatAnyone's "stable core, detailed boundary" idea |

* Inside the tracked subject the edge matte **cannot go see-through** (headset, temples, mic);
  outside it, it **cannot keep the chair**; within ~10 px of the outline RVM's hair detail stays.
* Seeding is automatic and robust: four candidate seeds ~0.5 s apart; the one with both ear cups,
  the best agreement with RVM and no extra area wins (loose detection boxes otherwise pull in the
  chair wing). A worn accessory may reach at most ¼ head-width sideways and hug the head on top.
  It re-seeds itself if you leave and come back; **Subject → Re-lock on me** forces it.
* The tracker lags the camera by 1-2 frames. Its matte is shifted by your motion and the band
  widens only where the picture is moving, so hands and turning heads are never clipped while
  static furniture stays out.
* Main-GPU cost: **~0.2 ms/frame** (measured 7.7 → 7.9 ms at 100 fps pacing). Startup: locks on
  about 10 s after the camera starts (model load + seed candidates); until then the semantic guard
  below is used.
* Measured on the recorded desk clip (`tests/ghost_eval.py`, `tests/subject_lock_live.py`): chair
  top/wings gone, both ear cups and the boom mic solid, no semi-transparent temple.

The older **semantic guard** (DeepLabV3 fill + motion check) remains as the fallback while the
lock starts, when it is switched off, or when MatAnyone 2 isn't installed.
**Chair: Auto / Remove / Keep**: with the lock the chair is removed anyway; *Keep* needs a clean
plate and keeps a crisp chair cut-out (BiRefNet on the empty-room plate) in every frame.

## Recording with sound, snapshots and thumbnails

* **● Rec** records video *and* sound. Output → Recording → **Microphone** picks track 1: Auto (the AI
  noise-removed "R9700 Broadcast Mic" when it is on, otherwise the system default), the AI mic, the
  system default, or any specific microphone (⟳ rescans after plugging one in). **Desktop audio**
  (off by default) adds track 2: what a speaker/headset plays (game, browser, call), stereo, kept
  separate from the mic so you can balance them in Resolve / Kdenlive. Live **level meters** show
  both while the Output or Audio tab is open (the Audio tab also shows the noise-removal mic's raw
  input vs. its cleaned output), and a small pair sits next to ● Rec while recording. A device
  that disappears mid-take is filled with silence; the file still closes cleanly. The mic is captured with
  `pw-record` into its own ffmpeg pipe, starting at the first video frame; video is stamped by the
  wall clock and written at constant frame rate, so dropped frames never shift the sound. Audio
  is delayed by the measured pipeline latency + 2 frames (camera); a ±200 ms lip-sync trim is
  there if needed. Measured flash/beep offset in the file: 16–36 ms, most of which is the test
  beep's own playback latency (`tests/av_sync.py`, on a private null sink). AAC 192k in MP4,
  PCM in ProRes MOV. Frames go through a writer thread: a slow disk never stalls the cameras.
* **Vertical 9:16 (Shorts / Reels / TikTok)**: **Vertical 9:16** in the top bar shows/hides a template beside the
  preview: a live 9:16 view with the apps' caption/button **safe zones** shaded, and a dashed frame on
  the main preview. Drag either one to move the frame (also while recording); scroll or use **Zoom**
  for a tighter crop. **Record** picks *Long form only*, *Vertical only* or *Long form + Vertical*;
  ● Rec then writes `Longform_<name>.mp4` and/or `Vertical_<name>.mp4` (same sound, same length).
  *Name* blank = date and time; an existing name gets ` (2)` instead of being overwritten. Vertical
  is 1080×1920 (scaled on the VCN with `scale_vaapi`) or the native crop (608×1080 from 1080p).
* **📷 Snapshot** saves the full-resolution output frame as PNG (transparent in the Transparent
  mode) to `~/Pictures/R9700 Broadcast/` and opens the **thumbnail maker**: 1280×720 cover crop,
  "pop" (contrast/colour), bold outlined title (auto-shrinks to fit), JPG under YouTube's 2 MB
  limit, PNG, or copy to clipboard. Optional 3 s / 5 s countdown.

## VRAM (measured with rocm-smi above idle, live camera)

| Configuration | Main GPU | Second GPU | Total |
|---|---|---|---|
| Realtime 60, guard off | 1.2 GB | – | **1.2 GB** |
| Realtime 60, guard on (default) | 1.25 GB | 0.7 GB | **1.9 GB** |
| Realtime 120 @720p, guard on | 1.0 GB | 0.75 GB | **1.7 GB** |
| Studio (BiRefNet-lite ×2) | 2.2 GB | 1.7 GB | **3.7 GB** |
| Studio Max (BiRefNet ×2) | 2.9 GB | 2.3 GB | **5.2 GB** |
| **Realtime 60 + subject lock (default)** | 0.9 GB | 3.3 GB | **4.2 GB** |

On a single GPU everything lands on that card (about the same totals), so 8 GB cards can run
Realtime, and 16–24 GB cards (RX 7900 XTX, RX 9070 XT) can run every mode.

## Stability notes

* `expandable_segments` is stripped from `PYTORCH_CUDA_ALLOC_CONF` at import. With HIP graphs on
  ROCm, the allocator unmapped memory a captured graph still used (when MIOpen retried a
  workspace allocation for a new blur size). That gave `HSA_STATUS_ERROR_MEMORY_FAULT`, and the
  app aborted when you dragged the Feather slider. Reproduced and fixed: `tests/slider_stress.py`
  survives about 3,000 setting changes per minute.
* Blurs use cumulative-sum box filters ("boxes for Gauss", exact variance). They need no
  per-radius MIOpen kernels, so dragging sliders never compiles anything.
* hipBLASLt on gfx1201 page-faults in a fused fp16 GEMM+bias kernel (`Cijk_*_HHS_BH_Bias_*`), hit
  by MatAnyone 2's transformer. The package routes all GEMMs through plain rocBLAS
  (`TORCH_BLAS_PREFER_HIPBLASLT=0`, `DISABLE_ADDMM_CUDA_LT=1`, `ROCBLAS_USE_HIPBLASLT=0`, plus
  `preferred_blas_library("cublas")` in case torch was imported first).
* No blocking host↔device copies per frame (auto-frame reads its box asynchronously, and
  theta is built from GPU basis tensors). All effect paths are pre-warmed per resolution.

## Where Corridor Crew comes in

Corridor Crew's **CorridorKey** (Niko Pueringer, 2026) is a neural *keyer*. It takes a green-screen
plate plus a coarse "alpha hint" from GVM, VideoMaMa, BiRefNet or a rough chroma key. From those
it predicts a **linear alpha and the true un-premultiplied foreground colour** for every pixel, so
motion blur, defocus and hair keep their real colour instead of picking up a green fringe. It still
needs a green screen, and its license forbids repackaging it into a product.

We take CorridorKey's *ideas* and apply them to footage with **no** screen:

1. **Coarse hint → refine.** The AI segments the person (RVM's recurrent net, or BiRefNet at
   576×1024). A guided filter then lifts that to full resolution using the real image edges
   (`matting.guided_filter`).
2. **Straight foreground colour.** RVM predicts the foreground colour directly. For BiRefNet we
   recover it with blur-fusion foreground estimation (Forte & Pitié 2021,
   `matting.blur_fusion_foreground`). Either way, the old background's colour is removed from hair
   and edges before compositing, which is CorridorKey's main selling point.
3. **Clean plate: the VFX trick that replaces the green screen.** Leave the shot for 5 seconds.
   The app records a median plate plus a per-pixel noise map. Every frame, a difference key against
   that plate is fused with the AI matte:
   * **rescue**: things you hold (mug, guitar, a product) that a person-segmenter ignores;
   * **suppress**: false positives such as your empty gaming chair or a poster with a face.
   It compensates for auto-exposure (a per-channel gain fitted on background pixels only) and
   ignores flicker (noise map). It only removes *large* regions that match the plate, so a shirt
   the same colour as the wall behind it never gets holes punched in it.

Measured results (`tests/plate_eval.py`, live OBSBOT frames with a composited person and a held box):

| | held box alpha | bg false positives | mean abs error vs truth |
|---|---|---|---|
| AI only (RVM) | 0.000 | 0.0000 | 0.0391 |
| AI + clean plate | **0.994** | 0.0011 | **0.0042** |

On the empty room, RVM and BiRefNet both mark the gaming chair as about 12% foreground. With the
plate, that drops to under 0.1%. A patch of shirt painted with the exact plate pixels behind it
keeps alpha 0.992.

## Engines and measured speed (R9700, ROCm 10, torch 2.11)

| Engine | Where | 1080p | Notes |
|---|---|---|---|
| Realtime: RVM ResNet50 | 1 GPU | **100 fps @1080p / 120 fps @720p live** (camera-limited); 4.4 ms GPU matte | temporal, default, HIP graphs |
| Realtime: RVM MobileNetV3 | 1 GPU | same | lowest power |
| Studio: BiRefNet-lite matting | 2 GPUs | ~38–42 fps (24 ms/frame) | sharper hair |
| Studio Max: BiRefNet matting | 2 GPUs | ~23 fps (43 ms/frame) | best edges; also for offline renders |

Per-frame engines run a motion-adaptive temporal stabiliser so edges don't shimmer.

## Using it in OBS

* Easiest: add a **Video Capture Device → R9700 Broadcast Camera** source with mode **Remove** or
  **Replace**.
* Compositing in OBS: set mode **Green screen**, then add OBS's *Chroma Key* filter (key colour
  green). The edges are already decontaminated, so use low similarity/smoothness.
* For editing: record **ProRes 4444 + alpha**, or use `file … --mode alpha`, and drop it straight
  into Resolve/Kdenlive.

## ROCm notes

* `TORCH_BLAS_PREFER_HIPBLASLT=0` is set by the package. hipBLASLt intermittently raised
  `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` (a `Cijk_*_HHS_BH_Bias` kernel) on gfx1201 when two
  processes ran BiRefNet concurrently. With rocBLAS: 4/4 stress runs clean, same speed.
* The torchvision in this ROCm build has no C++ ops, so `deform_conv2d` (BiRefNet's decoder) uses
  a pure-PyTorch implementation, `r9700_broadcast/ops.py`, unit-tested against `conv2d`.
* Threads do not scale across GPUs (GIL + HIP runtime contention: 2 GPUs were *slower* than 1).
  `MultiGPU` uses one spawned process per extra card and ships only the 576×1024 network input
  and output. Full-resolution finishing stays on the main GPU.
* If a GPU is hidden (e.g. passed through to a VM), the app filters its saved GPU list by the
  visible device count.

## Layout

```
r9700_broadcast/
  matting.py   engines (RVM, BiRefNet, MultiGPU), guided filter, blur-fusion, PlateAssist, stabiliser
  effects.py   compositor: modes, matte refine, light wrap, relight, vignette, auto-frame, denoise
  studio.py    per-frame graph shared by GUI / CLI
  pipeline.py  3-thread live engine + FPS presets
  pwcam.py     PipeWire camera (shared-memory triple buffer -> native helper)
  backgrounds.py  library, thumbnails, any-format image/video backgrounds
  uvc.py       camera hardware controls (OBSBOT gimbal/zoom/exposure)
  media.py     camera/file source (latest-frame, low latency), v4l2loopback sink, ffmpeg recorder
  audio.py     DeepFilterNet PipeWire virtual mic
  app.py       PyQt6 GUI          cli.py  file / live / gui commands
  ops.py       deform_conv2d fallback
native/        pw_camera.c (PipeWire Video/Source producer)
assets/backgrounds/  built-ins, generated by scripts/make_builtin_backgrounds.py
tests/         pytest (46) + plate_eval.py, fps_presets.py, alpha_pipewire.py (live camera / PipeWire)
```

## License and credits

The code in this repository is released under the **MIT License** (`LICENSE`).

Models and third-party code are **not** included; `scripts/fetch-models.sh` downloads them and
each keeps its own license. Full list, credits and citations: **`THIRD_PARTY_NOTICES.md`**.
In short:

* Robust Video Matting (realtime engine): **GPL-3.0**. A bundle that ships RVM with this app must
  be GPL-3.0 as a whole.
* MatAnyone 2 (subject lock): **NTU S-Lab License 1.0, non-commercial use only**. Optional.
* BiRefNet: MIT. SAM 2.1, Grounding DINO: Apache-2.0. DeepLabV3 (torchvision): BSD-3-Clause.
  DeepFilterNet: MIT / Apache-2.0.
* No CorridorKey code or weights are used.

Not affiliated with AMD, OBSBOT, OBS Project or Corridor Digital. "Radeon" and "R9700" are
trademarks of AMD.
