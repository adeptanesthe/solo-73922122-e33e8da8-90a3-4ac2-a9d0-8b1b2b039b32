"""点选分割：在检测框内依据像素颜色与邻接关系生成单块动物掩膜。

方法（仅使用 Pillow / NumPy / SciPy）
------------------------------------
1. Pillow 解码整帧并按检测框 1:1 裁剪（不缩放，面积即原图像素面积），
   裁剪超过 256×256 或解码失败直接报错；
2. 以前景/背景种子点（含 3×3 邻域）分别估计 RGB 对角高斯颜色模型；
3. 对裁剪内每个像素计算前景对背景的对数似然比，取似然比 > 0 者为
   前景候选；
4. 用 SciPy ``ndimage.label``（8 邻接）对前景候选做连通域分析，
   取包含全部前景点的那一个连通域作为唯一掩膜（单块动物）；
5. 背景点必须被判为背景、不得进入掩膜；前景种子必须落在同一连通域。

不能可靠分割时抛出 :class:`SegmentationError`（中文提示），不返回面积：
- 框内裁剪超过 256×256；
- 图片解码失败；
- 标点数量不足或落在框外；
- 纯色区域：前景/背景种子平均颜色一致，颜色无法区分；
- 颜色模型无法同时解释前景与背景种子；
- 前景点分属不同连通域，无法连成一块。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

# 框内裁剪的最大边长（像素）：超过则拒绝分割
MAX_CROP = 256
# 前景/背景种子平均颜色的最小可分距离（RGB 欧氏距离，0~255 量纲）
MEAN_DELTA_MIN = 1.0
# 颜色通道方差下限，避免纯色种子导致除零、似然比爆炸
VAR_FLOOR = 1.0
# 单类种子点数量上限，防止滥用
MAX_POINTS_PER_CLASS = 200


class SegmentationError(Exception):
    """可预期的分割失败，消息可直接展示给用户。"""


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    """解码图片为 H×W×3 的 uint8 RGB 数组；失败抛 SegmentationError。"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # 立即触发解码，截断/损坏文件在此报错
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except SegmentationError:
        raise
    except Exception as exc:  # Pillow 的各类解码异常统一转成业务错误
        raise SegmentationError("图片解码失败，无法读取该帧像素。") from exc


def _crop_box(rect, img_w: int, img_h: int):
    """把 [x, y, w, h] 规范为裁剪到画面内的整数像素框 (x0, y0, x1, y1)。"""
    x, y, w, h = (float(v) for v in rect)
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + h)))
    return x0, y0, x1, y1


def _normalize_points(points, x0: int, y0: int, cw: int, ch: int):
    """校验点并换算为裁剪内整数坐标（去重、保序）。"""
    if not points:
        return []
    seen = set()
    out = []
    for p in points:
        px, py = float(p[0]), float(p[1])
        if not (np.isfinite(px) and np.isfinite(py)):
            raise SegmentationError("标点含有非法坐标。")
        cx = int(round(px)) - x0
        cy = int(round(py)) - y0
        if not (0 <= cx < cw and 0 <= cy < ch):
            raise SegmentationError("存在检测框外的标点，所有点必须位于所选检测框内。")
        if (cx, cy) not in seen:
            seen.add((cx, cy))
            out.append((cx, cy))
    return out


def _seed_samples(rgb: np.ndarray, points):
    """收集种子点及其 3×3 邻域的像素样本（裁剪越界自动忽略）。"""
    h, w, _ = rgb.shape
    coords = set()
    for cx, cy in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx, yy = cx + dx, cy + dy
                if 0 <= xx < w and 0 <= yy < h:
                    coords.add((yy, xx))
    if not coords:
        raise SegmentationError("标点含有非法坐标。")
    rows = np.array([r for r, _ in coords], dtype=np.intp)
    cols = np.array([c for _, c in coords], dtype=np.intp)
    return rgb[rows, cols].astype(np.float64)


def _gaussian_stats(samples: np.ndarray):
    """对角高斯：每通道均值与加了下限的方差。"""
    mean = samples.mean(axis=0)
    var = np.maximum(samples.var(axis=0), VAR_FLOOR)
    return mean, var


def _log_likelihood(rgb: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """每个像素在该对角高斯下的对数似然（H×W）。"""
    x = rgb.astype(np.float64)
    diff = x - mean
    return -0.5 * np.sum(diff * diff / var + np.log(2.0 * np.pi * var), axis=2)


def encode_rle(mask: np.ndarray) -> list[int]:
    """把二值掩膜编码为行程长度数组。

    行程按 off/on 交替、且始终以 off 行程开头（首项可为 0），
    末尾的 off 行程也保留；解码时按下标奇偶判断是否为前景。
    """
    flat = mask.astype(np.uint8).reshape(-1)
    runs: list[int] = []
    prev = 0
    count = 0
    for v in flat:
        iv = int(v)
        if iv == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = iv
    runs.append(count)
    return runs


def segment_crop(image_bytes: bytes, rect, foreground, background) -> dict:
    """对一帧的指定检测框执行点选分割。

    参数
    ----
    image_bytes: 整帧图片的原始字节（会话内保存的上传内容）。
    rect: 检测框 ``[x, y, w, h]``，原图像素坐标。
    foreground, background: ``[(x, y), ...]`` 前景/背景种子点（原图坐标）。

    返回
    ----
    成功时返回::

        {
          "ok": True,
          "crop": {"x", "y", "width", "height"},
          "mask_rle": [int, ...],   # 裁剪大小的二值掩膜 RLE
          "area_px": int,           # 掩膜像素数 = 原图像素面积
        }

    无法可靠分割时抛出 :class:`SegmentationError`，不给出面积。
    """
    if len(foreground) < 2 or len(background) < 1:
        raise SegmentationError("至少需要 2 个前景点和 1 个背景点。")
    if len(foreground) > MAX_POINTS_PER_CLASS or len(background) > MAX_POINTS_PER_CLASS:
        raise SegmentationError(f"单类标点不能超过 {MAX_POINTS_PER_CLASS} 个。")

    rgb = _decode_rgb(image_bytes)
    img_h, img_w = rgb.shape[:2]
    x0, y0, x1, y1 = _crop_box(rect, img_w, img_h)
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        raise SegmentationError("检测框与画面没有交集，无法裁剪。")
    if cw > MAX_CROP or ch > MAX_CROP:
        raise SegmentationError(
            f"框内裁剪为 {cw}×{ch} 像素，超过 {MAX_CROP}×{MAX_CROP} 上限，"
            "无法进行点选分割。"
        )

    fg_pts = _normalize_points(foreground, x0, y0, cw, ch)
    bg_pts = _normalize_points(background, x0, y0, cw, ch)
    if len(fg_pts) < 2 or len(bg_pts) < 1:
        raise SegmentationError("至少需要 2 个前景点和 1 个背景点（重复点只计一次）。")

    crop = rgb[y0:y1, x0:x1, :]

    fg_samples = _seed_samples(crop, fg_pts)
    bg_samples = _seed_samples(crop, bg_pts)
    fg_mean, fg_var = _gaussian_stats(fg_samples)
    bg_mean, bg_var = _gaussian_stats(bg_samples)

    # 纯色区域：两类种子的平均颜色几乎相同 → 颜色无法区分前后景
    if float(np.linalg.norm(fg_mean - bg_mean)) < MEAN_DELTA_MIN:
        raise SegmentationError(
            "前景点与背景点位于颜色一致的纯色区域，无法依据像素颜色区分前后景，"
            "不能可靠分割，不提供面积。"
        )

    ll_fg = _log_likelihood(crop, fg_mean, fg_var)
    ll_bg = _log_likelihood(crop, bg_mean, bg_var)
    ratio = ll_fg - ll_bg
    fg_like = ratio > 0.0

    # 种子点自身必须与颜色模型的判定一致，否则说明颜色不可分
    for cx, cy in fg_pts:
        if not fg_like[cy, cx]:
            raise SegmentationError(
                "颜色模型无法把前景点判为前景，前后景颜色无法可靠区分，"
                "不能可靠分割，不提供面积。"
            )
    for cx, cy in bg_pts:
        if fg_like[cy, cx]:
            raise SegmentationError(
                "颜色模型无法把背景点判为背景，前后景颜色无法可靠区分，"
                "不能可靠分割，不提供面积。"
            )

    # 8 邻接连通域：前景必须同属一个连通域，背景点不得入选
    structure = np.ones((3, 3), dtype=bool)
    labels, n_labels = ndimage.label(fg_like, structure=structure)
    seed_labels = {labels[cy, cx] for cx, cy in fg_pts}
    if 0 in seed_labels or len(seed_labels) != 1:
        raise SegmentationError(
            "标出的前景无法连成同一连通区域，不能可靠分割，不提供面积。"
        )
    fg_label = seed_labels.pop()
    for cx, cy in bg_pts:
        if labels[cy, cx] == fg_label:
            # 理论上已被上面的似然比检查拦截，这里做防御性保证
            raise SegmentationError(
                "背景点落入了前景连通区域，不能可靠分割，不提供面积。"
            )

    mask = labels == fg_label
    area = int(mask.sum())
    if area <= 0:
        raise SegmentationError("生成的掩膜为空，不能可靠分割，不提供面积。")

    return {
        "ok": True,
        "crop": {"x": x0, "y": y0, "width": cw, "height": ch},
        "mask_rle": encode_rle(mask),
        "area_px": area,
    }
