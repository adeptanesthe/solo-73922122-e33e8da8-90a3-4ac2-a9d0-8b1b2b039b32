"""点选分割单元测试：颜色/邻接分割、连通域约束与各类不可靠情形。"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.segmentation import MAX_CROP, SegmentationError, encode_rle, segment_crop


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_scene(size=120, blobs=((30, 30, 16, (220, 40, 40)),),
               bg=(40, 120, 40)) -> bytes:
    """纯色草地背景 + 若干实心彩色圆（动物），返回 PNG 字节。"""
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for cx, cy, r, color in blobs:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return png_bytes(img)


def decode_rle(rle, n):
    out = np.zeros(n, dtype=np.uint8)
    pos = 0
    for i, length in enumerate(rle):
        if i % 2 == 1:
            out[pos:pos + length] = 1
        pos += length
    assert pos == n
    return out


# ---------------------------------------------------------------- 成功路径

def test_segment_success_and_area_matches_rle():
    data = make_scene(blobs=((30, 30, 16, (220, 40, 40)),))
    rect = [10.0, 10.0, 40.0, 40.0]  # 整个圆都在框内
    res = segment_crop(data, rect, foreground=[(26, 30), (34, 30)],
                       background=[(14, 14)])
    assert res["ok"] is True
    assert res["crop"] == {"x": 10, "y": 10, "width": 40, "height": 40}
    mask = decode_rle(res["mask_rle"], 40 * 40).reshape(40, 40).astype(bool)
    assert mask.sum() == res["area_px"]
    # 前景点在掩膜内、背景点在掩膜外
    assert mask[30 - 10, 26 - 10] and mask[30 - 10, 34 - 10]
    assert not mask[14 - 10, 14 - 10]
    # 掩膜是单块 8 邻接连通区域
    from scipy import ndimage
    _, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    assert n == 1
    # 圆面积的合理范围（πr² ≈ 804，Pillow 填充含边界像素，略大）
    assert 700 <= res["area_px"] <= 900


def test_segment_picks_only_component_of_foreground_seed():
    # 框内有两个同色物体，前景点只在其中一个上 → 只掩膜该块
    data = make_scene(blobs=(
        (25, 25, 8, (220, 40, 40)),
        (90, 90, 8, (220, 40, 40)),
    ))
    res = segment_crop(data, [0, 0, 120, 120],
                       foreground=[(23, 25), (27, 25)], background=[(60, 60)])
    mask = decode_rle(res["mask_rle"], 120 * 120).reshape(120, 120).astype(bool)
    assert mask[25, 25] and not mask[90, 90]
    from scipy import ndimage
    _, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    assert n == 1


def test_background_point_never_inside_mask():
    data = make_scene(blobs=((30, 30, 16, (220, 40, 40)),))
    # 背景点放在圆外的草地上
    res = segment_crop(data, [10, 10, 40, 40],
                       foreground=[(26, 30), (34, 30)], background=[(44, 44)])
    mask = decode_rle(res["mask_rle"], 1600).reshape(40, 40).astype(bool)
    assert not mask[44 - 10, 44 - 10]


# ---------------------------------------------------------------- 不可靠情形

def test_solid_color_region_rejected():
    data = png_bytes(Image.new("RGB", (60, 60), (100, 100, 100)))
    with pytest.raises(SegmentationError, match="纯色"):
        segment_crop(data, [5, 5, 30, 30],
                     foreground=[(10, 10), (20, 20)], background=[(30, 30)])


def test_disconnected_foreground_seeds_rejected():
    # 两个同色圆，前景点分别落在两个圆上 → 无法连成一块
    data = make_scene(blobs=(
        (25, 25, 8, (220, 40, 40)),
        (90, 90, 8, (220, 40, 40)),
    ))
    with pytest.raises(SegmentationError, match="连通"):
        segment_crop(data, [0, 0, 120, 120],
                     foreground=[(25, 25), (90, 90)], background=[(60, 60)])


def test_crop_larger_than_256_rejected():
    data = make_scene(size=300)
    with pytest.raises(SegmentationError, match="256"):
        segment_crop(data, [0, 0, 257, 100],
                     foreground=[(10, 10), (20, 10)], background=[(50, 50)])
    # 另一边超限同样拒绝
    with pytest.raises(SegmentationError, match="256"):
        segment_crop(data, [0, 0, 100, 257],
                     foreground=[(10, 10), (20, 10)], background=[(50, 50)])
    # 恰好在上限内可以进入颜色判定（本例为纯色，会以纯色原因失败而非尺寸原因）
    with pytest.raises(SegmentationError) as ei:
        segment_crop(data, [0, 0, MAX_CROP, MAX_CROP],
                     foreground=[(10, 10), (20, 10)], background=[(50, 50)])
    assert "256" not in str(ei.value)


def test_decode_failure_rejected():
    with pytest.raises(SegmentationError, match="解码"):
        segment_crop(b"not-an-image-at-all", [0, 0, 10, 10],
                     foreground=[(2, 2), (4, 4)], background=[(8, 8)])


def test_points_outside_box_rejected():
    data = make_scene()
    with pytest.raises(SegmentationError, match="框外"):
        segment_crop(data, [10, 10, 20, 20],
                     foreground=[(15, 15), (40, 15)], background=[(12, 12)])


def test_too_few_seeds_rejected():
    data = make_scene()
    with pytest.raises(SegmentationError, match="至少"):
        segment_crop(data, [10, 10, 30, 30],
                     foreground=[(20, 20)], background=[(12, 12)])
    with pytest.raises(SegmentationError, match="至少"):
        segment_crop(data, [10, 10, 30, 30],
                     foreground=[(20, 20), (21, 21)], background=[])


def test_rle_roundtrip():
    mask = np.zeros((6, 5), dtype=bool)
    mask[1:4, 1:4] = True
    rle = encode_rle(mask)
    back = decode_rle(rle, 30).reshape(6, 5).astype(bool)
    assert np.array_equal(back, mask)
