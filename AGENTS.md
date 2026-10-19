# Instructions for AI assistants installing or working on Virtual Greenscreen R9700 Broadcast Studio

You are helping a user install or run this app on their own Linux PC. Follow these steps in order.
Report each step's result to the user. Do not skip the license question in step 3.

## 1. Check the machine first (read-only)

```bash
uname -s                                  # must be Linux
ls -l /dev/kfd                            # AMD GPU compute driver must exist
groups                                    # user must be in "render" and "video"
lspci | grep -iE 'vga|display'            # expect an AMD Radeon GPU
command -v apt-get                        # Debian/Ubuntu? (installer uses apt; otherwise see README)
```

Stop and tell the user if:
* not Linux, or there is no AMD GPU / no `/dev/kfd`: this app needs an AMD GPU with ROCm
  (other GPU brands are not supported);
* the user is not in `render` and `video`: they must run `sudo usermod -aG render,video $USER`
  and log out/in, then you continue.

## 2. Get the code

```bash
git clone https://github.com/charlie12345/R9700BroadcastStudio.git
cd R9700BroadcastStudio
```

## 3. Ask the user ONE question before installing

MatAnyone 2 (the "subject lock" that stops headsets/chairs from ghosting) is licensed
**for non-commercial use only** (NTU S-Lab License 1.0). Ask the user:

> "The best-quality mode uses MatAnyone 2, which is free for non-commercial use only. Do you
> accept that license? (If not, the app still works with slightly more ghosting around headsets.)"

Do not decide this for the user.

## 4. Run the installer (non-interactive)

If they accepted:

```bash
scripts/install.sh --all --accept-matanyone-license
```

If they declined:

```bash
scripts/install.sh --all --decline-matanyone-license
```

* It uses `sudo apt-get` for missing system packages. If sudo needs a password and you cannot
  provide one, tell the user to run that one command themselves (`! scripts/install.sh ...` in
  tools that support it), or install the packages it prints and re-run with `--no-system-packages`.
* Downloads ~10 GB (PyTorch for ROCm ~6.6 GB + models ~3.4 GB) and needs ~22 GB of free disk.
  About 5 minutes on a fast connection, much longer on a slow one: use a long command timeout
  (30 min or more), or run it in the background and poll.
* It is safe to re-run; finished steps are skipped.
* Default PyTorch wheel: ROCm 7.2. If the GPU check fails, retry with `--rocm 7.1` or `--rocm 6.4`.
* Optional, needs sudo, only if the user wants the camera in Zoom/Teams/Discord: add `--virtualcam`.

Success ends with `Installed. Start it with: r9700-broadcast`.

## 5. Verify (optional but recommended)

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests      # expect: 71 passed
```

## 6. Tell the user how to start it

* Start: `r9700-broadcast` or the app-menu entry "R9700 Broadcast Studio"
  (or `./bin/r9700-broadcast` from the repo folder).
* Close the webcam in every other app first (only one app can use it).
* In the app: pick FPS preset + camera, pick a background tile, wait ~10 s for
  "Subject: Locked on to you", then press ● PipeWire camera (OBS / browsers),
  ● V4L2 camera (Zoom / Teams, needs `--virtualcam`), or ● Rec.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/dev/kfd` missing | amdgpu driver not loaded; needs a recent kernel and an AMD GPU |
| `Permission denied` on `/dev/kfd` | `sudo usermod -aG render,video $USER`, then log out/in |
| GPU check fails / `hipErrorNoBinaryForGpu` | re-run `scripts/install.sh --rocm 7.1` (or another version from https://pytorch.org/get-started/locally/) |
| `make -C native` fails | `sudo apt install libpipewire-0.3-dev pkg-config build-essential` |
| Camera busy | close OBS / browser / Zoom tabs using the webcam |
| No V4L2 camera in Zoom | `sudo scripts/setup-virtualcam.sh`; with Secure Boot, enroll the DKMS key when prompted and reboot |

## Rules for code changes in this repo

* Run `PYTHONPATH=. .venv/bin/python -m pytest -q tests` after changes.
* Never commit model weights, `models/`, `third_party/` downloads, `.venv/` or `testdata/`.
* Keep `THIRD_PARTY_NOTICES.md` up to date when adding a model or library.
