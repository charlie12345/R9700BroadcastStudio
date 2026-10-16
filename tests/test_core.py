import pytest
import torch
import torch.nn.functional as F

from r9700_broadcast.effects import AutoFramer, Compositor, Settings, refine_alpha, CHROMA_GREEN
from r9700_broadcast.matting import PlateAssist, box_filter, guided_filter, blur_fusion_foreground

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_box_filter_matches_avg_pool_interior():
    x = torch.rand(1, 1, 40, 50, device=DEV)
    r = 3
    ref = F.avg_pool2d(x, 2 * r + 1, 1)
    assert torch.allclose(box_filter(x, r)[..., r:-r, r:-r], ref, atol=1e-5)


def test_box_filter_constant_at_edges():
    x = torch.full((1, 1, 20, 20), 0.7, device=DEV)
    assert torch.allclose(box_filter(x, 5), x, atol=1e-5)


def test_guided_filter_upsamples_and_preserves_flat_alpha():
    g = torch.rand(1, 3, 120, 160, device=DEV)
    s = torch.ones(1, 1, 30, 40, device=DEV)
    out = guided_filter(g, s, 2, 1e-4)
    assert out.shape == (1, 1, 120, 160)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-3)


def test_blur_fusion_removes_background_bleed():
    # red subject on the left, blue background on the right, semi-transparent seam observed purple
    H, W = 64, 64
    img = torch.zeros(1, 3, H, W, device=DEV)
    a = torch.zeros(1, 1, H, W, device=DEV)
    a[..., :32] = 1
    a[..., 32:36] = 0.5
    img[:, 0] = a[:, 0]              # red where subject
    img[:, 2] = 1 - a[:, 0]          # blue where background
    fg = blur_fusion_foreground(img, a, scale=1.0)
    seam = fg[0, :, :, 33]
    assert seam[0].mean() > 0.8 and seam[2].mean() < 0.3  # recovered straight red, not purple


def test_compositor_modes():
    c = Compositor()
    src = torch.rand(1, 3, 108, 192, device=DEV)
    a1 = torch.ones(1, 1, 108, 192, device=DEV)
    a0 = torch.zeros_like(a1)
    s = Settings(mode="green", light_wrap=0, clip_black=0, clip_white=1)
    assert torch.allclose(c(src, a1, None, s), src, atol=1e-5)
    g = c(src, a0, None, s)
    assert torch.allclose(g[0, :, 0, 0], torch.tensor(CHROMA_GREEN, device=DEV), atol=1e-5)
    s.mode = "matte"
    assert torch.allclose(c(src, a0, None, s), torch.zeros_like(src))
    s.mode = "blur"
    out = c(src, a0, None, s)
    assert out.std() < src.std()  # blurred
    s.mode = "off"
    assert torch.allclose(c(src, None, None, s), src)


def test_refine_alpha_choke_and_grow():
    a = torch.zeros(1, 1, 50, 50, device=DEV)
    a[..., 20:30, 20:30] = 1
    s = Settings(choke=2, clip_black=0, clip_white=1)
    assert refine_alpha(a, s, 1.0).sum() < a.sum()
    s.choke = -2
    assert refine_alpha(a, s, 1.0).sum() > a.sum()


def test_autoframer_crop_keeps_size_and_tracks_subject():
    img = torch.rand(1, 3, 180, 320, device=DEV)
    a = torch.zeros(1, 1, 180, 320, device=DEV)
    a[..., 40:180, 220:300] = 1      # subject on the right
    fr = AutoFramer()
    for _ in range(100):
        box = fr.update(a, zoom=2.0, speed=0.2)
        if DEV == "cuda":
            torch.cuda.synchronize()  # the box is read back asynchronously, one frame later
    assert box[0] > 0.6               # moved right
    assert fr.crop(img, box).shape == img.shape


def test_plate_assist_rescue_and_suppress():
    torch.manual_seed(0)
    H, W = 216, 384
    plate = torch.rand(1, 3, H, W, device=DEV) * 0.5
    pa = PlateAssist()
    for _ in range(30):
        assert pa.add_sample(plate + 0.005 * torch.randn_like(plate)) in (False, True)
    assert pa.ready
    img = plate.clone()
    img[..., 80:200, 150:250] = 0.9           # person
    img[..., 120:160, 100:140] = 0.05         # held object beside them
    ai = torch.zeros(1, 1, H, W, device=DEV)
    ai[..., 80:200, 150:250] = 1              # AI finds the person...
    ai[..., 10:60, 300:370] = 1               # ...plus a false positive that matches the plate
    out = pa(img, ai)
    assert out[..., 125:155, 105:135].mean() > 0.9   # object rescued
    assert out[..., 15:55, 305:365].mean() < 0.1     # false positive removed
    assert out[..., 100:180, 170:230].mean() > 0.99  # person intact


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_multigpu_returns_frames_in_order():
    from r9700_broadcast.matting import build_engine
    eng = build_engine("birefnet-lite", ["cuda:0", "cuda:1"])
    try:
        frames = [torch.rand(1, 3, 144, 256, device="cuda:0", dtype=torch.half) for _ in range(8)]
        got = []
        for f in frames:
            m = eng(f)
            got.append(m.src)
        # warm-up repeats frame 0 once, then strictly ordered with one frame of latency
        assert got[0] is frames[0] and got[1] is frames[0]
        assert all(got[i] is frames[i - 1] for i in range(2, 8))
    finally:
        eng.close()


def test_denoiser_reduces_static_noise_but_follows_motion():
    from r9700_broadcast.effects import TemporalDenoiser
    torch.manual_seed(0)
    base = torch.full((1, 3, 64, 64), 0.5, device=DEV)
    d = TemporalDenoiser()
    for _ in range(20):
        out = d(base + 0.01 * torch.randn_like(base), 1.0)
    assert (out - base).std() < 0.006
    moved = base.clone()
    moved[..., :32] = 0.9                       # big change -> must pass straight through
    out = d(moved, 1.0)
    assert (out[..., :32] - 0.9).abs().mean() < 0.02


def test_plate_furniture_removes_chair_adjacent_to_person():
    torch.manual_seed(0)
    H, W = 540, 960                              # safe guard ~28 px here, aggressive ~4 px
    plate = torch.rand(1, 3, H, W, device=DEV) * 0.5
    pa = PlateAssist()
    for _ in range(30):
        pa.add_sample(plate + 0.005 * torch.randn_like(plate))
    img = plate.clone()
    img[..., 250:540, 380:580] = 0.9            # person (changed vs plate)
    ai = torch.zeros(1, 1, H, W, device=DEV)
    ai[..., 250:540, 380:580] = 1
    ai[..., 120:250, 390:570] = 1               # chair headrest directly above the head, matches plate
    band = (..., slice(228, 242), slice(420, 540))   # 8-22 px above the head
    safe = pa(img, ai)[band].mean()
    pa.furniture = 1.0
    out = pa(img, ai)
    assert out[band].mean() < 0.2 < safe
    assert out[..., 300:520, 400:560].mean() > 0.99   # person still intact


def test_plate_chair_keep_and_remove():
    """keep: the chair cut from the plate becomes foreground. remove: chair pixels that still match the
    empty plate go transparent, while things in front of the chair (headset) stay."""
    H, W = 216, 384
    torch.manual_seed(0)
    plate = torch.rand(1, 3, H, W, device=DEV) * 0.3
    plate[..., 20:200, 120:270] = 0.05                       # dark chair
    pa = PlateAssist()
    for _ in range(30):
        pa.add_sample(plate + 0.003 * torch.randn_like(plate))
    pa.chair_mask = torch.zeros(1, 1, H, W, device=DEV, dtype=torch.half)
    pa.chair_mask[..., 20:200, 120:270] = 1                   # as BiRefNet would cut it from the plate
    img = plate.clone()
    img[..., 60:216, 150:240] = 0.8                           # person sitting in the chair
    img[..., 70:110, 140:150] = 0.12                          # dark headset cup, slightly off the chair colour
    alpha = torch.zeros(1, 1, H, W, device=DEV)
    alpha[..., 60:216, 150:240] = 1
    alpha[..., 20:60, 130:260] = 1                            # matte wrongly kept the headrest
    alpha[..., 70:110, 140:150] = 1
    kept = pa.apply_chair(img, alpha, "keep")
    assert kept[..., 30:190, 125:145].mean() > 0.99           # chair sides are foreground
    removed = pa.apply_chair(img, alpha, "remove")
    assert removed[..., 25:55, 135:255].mean() < 0.1          # headrest gone
    assert removed[..., 80:200, 160:230].mean() > 0.99        # person intact
    assert removed[..., 75:105, 142:148].mean() > 0.9         # headset in front of the chair stays
