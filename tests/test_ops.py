import torch, torch.nn.functional as F
from r9700_broadcast.ops import deform_conv2d


def test_zero_offset_matches_conv2d():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 17, 23)
    w = torch.randn(6, 8, 3, 3)
    b = torch.randn(6)
    off = torch.zeros(2, 18, 17, 23)
    m = torch.ones(2, 9, 17, 23)
    ref = F.conv2d(x, w, b, padding=1)
    assert torch.allclose(deform_conv2d(x, off, w, b, padding=1, mask=m), ref, atol=1e-4)


def test_integer_offset_is_shift():
    torch.manual_seed(1)
    x = torch.randn(1, 4, 12, 12)
    w = torch.randn(3, 4, 1, 1)
    off = torch.zeros(1, 2, 12, 12)
    off[:, 1] = 1.0  # dx = +1
    out = deform_conv2d(x, off, w)
    ref = F.conv2d(x, w)
    assert torch.allclose(out[..., :-1], ref[..., 1:], atol=1e-4)
    assert torch.allclose(out[..., -1], torch.zeros_like(out[..., -1]), atol=1e-6)


def test_stride_and_mask():
    torch.manual_seed(2)
    x = torch.randn(1, 4, 16, 16)
    w = torch.randn(5, 4, 3, 3)
    off = torch.zeros(1, 18, 8, 8)
    m = torch.full((1, 9, 8, 8), 0.5)
    ref = F.conv2d(x, w, stride=2, padding=1) * 0.5
    assert torch.allclose(deform_conv2d(x, off, w, stride=2, padding=1, mask=m), ref, atol=1e-4)
