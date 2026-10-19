"""Fun props: face-tracked comical overlays (unit tests). Drawing itself is verified
visually on real frames; here we pin the math: tracker smoothing/coasting, anchor
geometry, and sprite compositing."""
import numpy as np
import pytest
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _face(cx=960, cy=540, w=330, h=330, score=0.9):
    from r9700_broadcast.props import Face
    fw = w
    return Face(box=(cx - w / 2, cy - h / 2, w, h),
                leye=(cx - 0.075 * fw, cy - 0.105 * fw), reye=(cx + 0.075 * fw, cy - 0.105 * fw),
                nose=(cx, cy - 0.02 * fw), mouth=(cx, cy + 0.05 * fw), score=score)


def test_tracker_smooths_jitter_and_coasts():
    from r9700_broadcast.props import FaceTracker
    tr = FaceTracker.__new__(FaceTracker)  # no model: exercise update() only
    tr._ema, tr._miss, tr.every, tr._i = 0.6, 0, 1, 0
    tr._last, tr._coast, tr.min_score = None, 10, 0.5
    f1 = _face(cx=960)
    assert tr.update([f1]).box[0] == pytest.approx(960 - 165)
    f2 = _face(cx=990)  # 30px jump: EMA must lag (0.6*990 + 0.4*960 = 978 center)
    out = tr.update([f2])
    assert (out.box[0] + out.box[2] / 2) == pytest.approx(978)
    for _ in range(10):  # detector drops: coast keeps last face
        assert tr.update([]) is not None
    assert tr.update([]) is None  # ... then gives up


def test_tracker_takes_best_detection():
    from r9700_broadcast.props import FaceTracker
    tr = FaceTracker.__new__(FaceTracker)
    tr._ema, tr._miss, tr.every, tr._i = 1.0, 0, 1, 0
    tr._last, tr._coast, tr.min_score = None, 10, 0.5
    out = tr.update([_face(cx=100, score=0.6), _face(cx=900, score=0.95)])
    assert (out.box[0] + out.box[2] / 2) == pytest.approx(900)


def test_anchors_put_hats_above_and_eyes_on_eyes():
    from r9700_broadcast.props import anchors
    a = anchors(_face(cx=960, cy=540, w=330, h=330))
    assert a["head_top"][1] < 540 - 165          # above the face box (on the hair)
    assert abs(a["head_top"][0] - 960) < 5
    assert a["leye"] == pytest.approx((960 - 0.075 * 330, 540 - 0.105 * 330))
    assert a["face_w"] == pytest.approx(330)


def test_stamp_composites_alpha_correctly():
    from r9700_broadcast.props import stamp
    out = torch.zeros(1, 3, 100, 100, device=DEV)
    sprite = np.zeros((20, 20, 4), np.uint8)
    sprite[:, :, 0] = 255                        # red
    sprite[:, :, 3] = 128                        # 50% alpha
    stamp(out, sprite, 50, 50, DEV if DEV == "cuda" else "cpu")
    px = out[0, :, 50, 50].float().cpu().numpy()   # frame is float [0,1]; no rescale
    assert px[0] == pytest.approx(0.5, abs=0.02) and px[1] < 0.02 and px[2] < 0.02


def test_draw_all_props_covers_expected_spots():
    from r9700_broadcast.props import draw_props
    out = torch.zeros(1, 3, 1080, 1920, device=DEV)
    face = _face(cx=960, cy=540, w=330, h=330)
    for name in ("googly_eyes", "sunglasses", "propeller_hat", "santa_hat", "elf_hat",
                 "rabbit_ears", "light_bulb", "speech_bubble"):
        img = draw_props(out.clone(), face, [name], t=1.0, bubble_text="HI!")
        assert (img > 0).any(), name             # the prop actually drew something
    both = draw_props(out.clone(), face, ["googly_eyes"], t=1.0)
    le = both[0, :, int(face.leye[1]), int(face.leye[0])].float().cpu().numpy()
    assert le.mean() > 0.5                         # eye area is bright white, not background


def test_settings_roundtrip_fun_props():
    from r9700_broadcast.effects import Settings
    from dataclasses import asdict
    import json
    s = Settings(fun_props=["santa_hat", "speech_bubble"], bubble_text="HO HO")
    d = json.loads(json.dumps({k: v for k, v in asdict(s).items() if k != "extra"}))
    s2 = Settings()
    for k, v in d.items():
        if hasattr(s2, k):
            setattr(s2, k, v)
    assert s2.fun_props == ["santa_hat", "speech_bubble"] and s2.bubble_text == "HO HO"
