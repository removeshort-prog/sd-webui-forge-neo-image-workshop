"""Anime image censoring for the Forge Neo image workshop.

The censor workflow deliberately handles still images only. Detection is
optional and lazy so regular compression, watermark and upscaling features do
not require the anime model package.
"""

from __future__ import annotations

import csv
import re
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .engine import Cancelled, check_pixels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ANIME_DEFAULT_TARGETS = ("penis", "pussy")
ANIME_EXTRA_TARGETS = {"nipple_f": "乳头（二次元）"}


@dataclass
class CensorOptions:
    extra_targets: tuple[str, ...] = ()
    confidence: float = 0.25
    shape: str = "fit"
    mode: str = "mosaic"
    dilate_px: int = 15
    strength: int = 100
    max_megapixels: float = 64

    def validate(self):
        if not 0.01 <= float(self.confidence) <= 0.99:
            raise ValueError("置信度阈值必须在 0.01～0.99 之间")
        if self.shape not in ("fit", "ellipse", "rect"):
            raise ValueError("遮罩形状无效")
        if self.mode not in ("mosaic", "blur"):
            raise ValueError("打码方式无效")
        if not 0 <= int(self.dilate_px) <= 300:
            raise ValueError("扩边缘必须在 0～300 像素之间")
        if not 4 <= int(self.strength) <= 500:
            raise ValueError("马赛克粒度必须在 4～500 之间")
        if not 1 <= float(self.max_megapixels) <= 256:
            raise ValueError("像素上限必须在 1～256 百万像素之间")


def _dependency_error():
    return ValueError(
        "二次元检测需要安装 dghs-imgutils；请运行 Forge Neo 虚拟环境中的 requirements-censor.txt"
    )


class AnimeDetector:
    def __init__(self):
        try:
            from imgutils.detect import detect_censors
        except ImportError as exc:
            raise _dependency_error() from exc
        self.detect_censors = detect_censors

    def detect(self, image_bgr, targets, confidence):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.detect_censors(Image.fromarray(rgb), conf_threshold=confidence)
        return [
            (int(x0), int(y0), int(x1), int(y1))
            for (x0, y0, x1, y1), label, score in results
            if label in targets and float(score) >= confidence
        ]


def build_detector():
    return AnimeDetector()


def targets_for(options: CensorOptions):
    options.validate()
    allowed = set(ANIME_DEFAULT_TARGETS) | set(ANIME_EXTRA_TARGETS)
    return list(ANIME_DEFAULT_TARGETS) + [
        value for value in options.extra_targets if value in allowed
    ]


def _ellipse_mask(shape_hw, box):
    height, width = shape_hw
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(
        mask,
        ((x0 + x1) // 2, (y0 + y1) // 2),
        (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)),
        0, 0, 360, 255, -1,
    )
    return mask


def _rectangle_mask(shape_hw, box):
    height, width = shape_hw
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), np.uint8)
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255
    return mask


def _grabcut_mask(image, box):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    box_width, box_height = x1 - x0, y1 - y0
    if box_width < 12 or box_height < 12:
        return None
    pad = int(max(box_width, box_height) * 0.35) + 5
    crop_x0, crop_y0 = max(0, x0 - pad), max(0, y0 - pad)
    crop_x1, crop_y1 = min(width, x1 + pad), min(height, y1 + pad)
    crop = image[crop_y0:crop_y1, crop_x0:crop_x1]
    scale = min(1.0, 480.0 / max(crop.shape[:2]))
    if scale != 1:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
    rect = (
        round((x0 - crop_x0) * scale), round((y0 - crop_y0) * scale),
        max(1, round(box_width * scale)), max(1, round(box_height * scale)),
    )
    labels = np.zeros(crop.shape[:2], np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, labels, rect, background, foreground, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    foreground_mask = np.where(
        (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    if foreground_mask.sum() / 255 < 0.20 * rect[2] * rect[3]:
        return None
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
    )
    if scale != 1:
        foreground_mask = cv2.resize(
            foreground_mask, (crop_x1 - crop_x0, crop_y1 - crop_y0), interpolation=cv2.INTER_NEAREST
        )
    mask = np.zeros((height, width), np.uint8)
    mask[crop_y0:crop_y1, crop_x0:crop_x1] = foreground_mask
    guard = np.zeros((height, width), np.uint8)
    guard[max(0, y0 - 6):min(height, y1 + 6), max(0, x0 - 6):min(width, x1 + 6)] = 255
    return cv2.bitwise_and(mask, guard)


def build_mask(image, boxes, shape="fit"):
    total = np.zeros(image.shape[:2], np.uint8)
    for box in boxes:
        if shape == "rect":
            mask = _rectangle_mask(image.shape[:2], box)
        elif shape == "ellipse":
            mask = _ellipse_mask(image.shape[:2], box)
        else:
            mask = _grabcut_mask(image, box)
            if mask is None:
                mask = _ellipse_mask(image.shape[:2], box)
        total = cv2.bitwise_or(total, mask)
    return total


def dilate_mask(mask, pixels):
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask, kernel)


def render_censor(image, mask, options):
    mask = dilate_mask(mask, int(options.dilate_px))
    if mask.max() == 0:
        return image.copy(), 0
    height, width = image.shape[:2]
    if options.mode == "blur":
        kernel = max(5, (max(height, width) // max(1, int(options.strength))) * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        censored = cv2.GaussianBlur(image, (kernel, kernel), 0)
        detail = kernel
    else:
        block = max(4, int(max(height, width) / max(1, int(options.strength))))
        small = cv2.resize(image, (max(1, width // block), max(1, height // block)), interpolation=cv2.INTER_LINEAR)
        censored = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        detail = block
    output = image.copy()
    output[mask > 0] = censored[mask > 0]
    return output, detail


def _rgba_to_bgr(image):
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_rgba(image, alpha):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(rgb).convert("RGBA")
    output.putalpha(alpha)
    return output


def process_still(source, destination, detector, options, targets, check_cancel):
    with Image.open(source) as opened:
        check_pixels(opened.size, options.max_megapixels)
        if getattr(opened, "n_frames", 1) > 1:
            raise ValueError("自动打码只支持静态单帧图片")
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        original_alpha = image.getchannel("A")
        bgr = _rgba_to_bgr(image)
        boxes = detector.detect(bgr, targets, options.confidence)
        if boxes:
            processed, _ = render_censor(bgr, build_mask(bgr, boxes, options.shape), options)
            image = _bgr_to_rgba(processed, original_alpha)
        check_cancel()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fmt = destination.suffix.lower()
    if fmt in (".jpg", ".jpeg"):
        rgb = Image.new("RGB", image.size, "white")
        rgb.paste(image.convert("RGB"), mask=image.getchannel("A"))
        rgb.save(destination, "JPEG", quality=95, optimize=True)
    else:
        image.save(
            destination,
            format={".webp": "WEBP", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP"}.get(fmt, "PNG"),
            optimize=fmt == ".png",
        )
    return {"boxes": len(boxes)}


def media_sources(mode, uploads, directory, recursive, sort_mode, allow_directories=True, excluded_root=None):
    if mode == "本机文件夹":
        if not allow_directories:
            raise ValueError("Forge 已使用 --hide-ui-dir-config 禁用本机目录访问")
        root = Path(str(directory or "").strip()).expanduser()
        if not root.is_dir():
            raise ValueError("请选择有效的本机图片文件夹")
        paths = root.rglob("*") if recursive else root.glob("*")
        excluded = Path(excluded_root).resolve() if excluded_root else None
        files = [
            path for path in paths
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and not (excluded and (path.resolve() == excluded or excluded in path.resolve().parents))
        ]
    else:
        files = []
        for item in uploads or []:
            if isinstance(item, dict):
                files.append(Path(item.get("path") or item.get("name") or ""))
            elif hasattr(item, "name"):
                files.append(Path(item.name))
            else:
                files.append(Path(item))
    files = [path for path in files if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError("没有找到静态图片")
    key = lambda path: tuple((1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", str(path)))
    if sort_mode == "修改时间从旧到新":
        return sorted(files, key=lambda path: (path.stat().st_mtime, key(path)))
    if sort_mode == "修改时间从新到旧":
        return sorted(files, key=lambda path: (-path.stat().st_mtime, key(path)))
    if sort_mode == "文件名自然排序":
        return sorted(files, key=lambda path: (key(path.name), key(path)))
    return sorted(files, key=key)


@dataclass
class CensorBatchResult:
    directory: Path
    total: int
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    outputs: list[Path] = field(default_factory=list)
    previews: list = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    archive: Path | None = None
    report: Path | None = None
    elapsed: float = 0


def run_censor_batch(sources, destination, options, check_cancel=lambda: None,
                     progress=lambda *_: None, make_zip=True, detector=None):
    options.validate()
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = Path(destination).resolve() / stamp
    directory.mkdir(parents=True, exist_ok=False)
    result = CensorBatchResult(directory, len(sources))
    started = time.monotonic()
    detector = detector or build_detector()
    targets = targets_for(options)
    for index, source in enumerate(sources, 1):
        try:
            check_cancel()
            progress((index - 1) / len(sources), f"{index}/{len(sources)} · {source.name}")
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source.stem).strip(" .")[:100] or "image"
            output = directory / f"{index:05d}_{stem}{source.suffix.lower()}"
            stats = process_still(source, output, detector, options, targets, check_cancel)
            result.outputs.append(output)
            result.completed += 1
            if len(result.previews) < 20:
                with Image.open(output) as image:
                    preview = image.convert("RGBA")
                    preview.thumbnail((1024, 1024))
                    preview.load()
                result.previews.append((preview, output.name))
            result.rows.append({
                "输入文件": source.name, "输出文件": output.name, "状态": "成功",
                "检测框": stats["boxes"], "错误": "",
            })
        except Cancelled:
            result.cancelled = True
            break
        except Exception as exc:
            result.failed += 1
            result.rows.append({"输入文件": source.name, "状态": "失败", "错误": f"{type(exc).__name__}: {exc}"})
        progress(index / len(sources), f"已完成 {index}/{len(sources)}")
    result.report = directory / "自动打码报告.csv"
    fields = ["输入文件", "输出文件", "状态", "检测框", "错误"]
    with result.report.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.rows)
    if make_zip and result.outputs:
        result.archive = directory / "自动打码结果.zip"
        with zipfile.ZipFile(result.archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in [*result.outputs, result.report]:
                archive.write(path, arcname=path.name)
    result.elapsed = time.monotonic() - started
    return result
