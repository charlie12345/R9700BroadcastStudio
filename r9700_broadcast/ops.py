"""Pure-PyTorch fallbacks for torchvision ops missing from the ROCm torchvision build."""
import torch
import torch.nn.functional as F


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    """Same semantics as torchvision.ops.deform_conv2d (DCNv1/v2), built on grid_sample."""
    B, C, H, W = input.shape
    Co, Cg, kh, kw = weight.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    Ho, Wo = offset.shape[-2:]
    K = kh * kw
    G = offset.shape[1] // (2 * K)
    dev, dt = input.device, input.dtype

    off = offset.view(B, G, K, 2, Ho, Wo)
    ky, kx = torch.meshgrid(torch.arange(kh, device=dev) * dh, torch.arange(kw, device=dev) * dw, indexing="ij")
    ky = ky.reshape(1, 1, K, 1, 1).to(dt)
    kx = kx.reshape(1, 1, K, 1, 1).to(dt)
    ys = (torch.arange(Ho, device=dev) * sh - ph).to(dt).view(1, 1, 1, Ho, 1)
    xs = (torch.arange(Wo, device=dev) * sw - pw).to(dt).view(1, 1, 1, 1, Wo)
    py = ys + ky + off[:, :, :, 0]
    px = xs + kx + off[:, :, :, 1]
    gx = px * (2.0 / max(W - 1, 1)) - 1.0
    gy = py * (2.0 / max(H - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).view(B * G, K * Ho, Wo, 2)

    x = input.reshape(B * G, C // G, H, W)
    sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.view(B, G, C // G, K, Ho, Wo)
    if mask is not None:
        sampled = sampled * mask.view(B, G, 1, K, Ho, Wo)
    cols = sampled.reshape(B, C, K, Ho * Wo)

    groups = C // Cg
    cols = cols.view(B, groups, Cg * K, Ho * Wo)
    w = weight.view(groups, Co // groups, Cg * K)
    out = torch.einsum("gok,bgkn->bgon", w, cols).reshape(B, Co, Ho, Wo)
    if bias is not None:
        out = out + bias.view(1, Co, 1, 1)
    return out


def torchvision_has_ops():
    try:
        torch.ops.torchvision.deform_conv2d
        import torchvision.extension as ext
        ext._assert_has_ops()
        return True
    except Exception:
        return False
