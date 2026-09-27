"""Optional automatic censoring engine for images, GIFs and videos.

The detector packages are intentionally imported lazily. The regular image
workshop remains usable when NudeNet or dghs-imgutils is not installed.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

from .engine import Cancelled, check_pixels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".ts"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | GIF_EXTENSIONS | VIDEO_EXTENSIONS

REAL_DEFAULT_TARGETS = (
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
)
REAL_EXTRA_TARGETS = {
    "FEMALE_GENITALIA_COVERED": "女性性器官（遮挡）",
    "ANUS_COVERED": "肛门（遮挡）",
    "FEMALE_BREAST_EXPOSED": "女性胸部（裸露）",
    "BUTTOCKS_EXPOSED": "臀部（裸露）",
}
ANIME_DEFAULT_TARGETS = ("penis", "pussy")
ANIME_EXTRA_TARGETS = {"nipple_f": "乳头（二次元）"}
TARGET_LABELS = {
    **REAL_EXTRA_TARGETS, **ANIME_EXTRA_TARGETS,
    "__none__": "只检测核心部位（默认）",
}


@dataclass
class CensorOptions:
    engine: str = "anime"
    extra_targets: tuple[str, ...] = ()
    confidence: float = 0.25
    shape: str = "fit"
    mode: str = "mosaic"
    dilate_px: int = 15
    strength: int = 100
    detect_every: int = 2
    hold: int = 8
    max_megapixels: float = 64

    def validate(self):
        if self.engine not in ("anime", "real"):
            raise ValueError("自动打码检测引擎无效")
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
        if not 1 <= int(self.detect_every) <= 60:
            raise ValueError("检测间隔必须在 1～60 帧之间")
        if not 0 <= int(self.hold) <= 300:
            raise ValueError("漏检保持必须在 0～300 帧之间")
        if not 1 <= float(self.max_megapixels) <= 256:
            raise ValueError("像素上限必须在 1～256 百万像素之间")


def _dependency_error(engine):
    if engine == "anime":
        return ValueError("二次元检测需要安装 dghs-imgutils；请运行 Forge Neo 虚拟环境中的 requirements-censor.txt")
    return ValueError("真人检测需要安装 NudeNet；请运行 Forge Neo 虚拟环境中的 requirements-censor.txt")


class RealDetector:
    def __init__(self):
        try:
            from nudenet import NudeDetector
        except ImportError as exc:
            raise _dependency_error("real") from exc
        self.detector = NudeDetector()

    def detect(self, image_bgr, targets, confidence):
        try:
            results = self.detector.detect(image_bgr)
        except Exception:
            # Older NudeNet releases require a path and fail on non-ASCII paths.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                cv2.imencode(".png", image_bgr)[1].tofile(str(temporary))
                results = self.detector.detect(str(temporary))
            finally:
                temporary.unlink(missing_ok=True)
        boxes = []
        for item in results or []:
            if item.get("class") in targets and float(item.get("score", 0)) >= confidence:
                x, y, width, height = item["box"]
                boxes.append((int(x), int(y), int(x + width), int(y + height)))
        return boxes


class AnimeDetector:
    def __init__(self):
        try:
            from imgutils.detect import detect_censors
        except ImportError as exc:
            raise _dependency_error("anime") from exc
        self.detect_censors = detect_censors

    def detect(self, image_bgr, targets, confidence):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.detect_censors(Image.fromarray(rgb), conf_threshold=confidence)
        return [
            (int(x0), int(y0), int(x1), int(y1))
            for (x0, y0, x1, y1), label, score in results
            if label in targets and float(score) >= confidence
        ]


def build_detector(name):
    return AnimeDetector() if name == "anime" else RealDetector()


def targets_for(options: CensorOptions):
    options.validate()
    if options.engine == "anime":
        allowed = set(ANIME_DEFAULT_TARGETS) | set(ANIME_EXTRA_TARGETS)
        return list(ANIME_DEFAULT_TARGETS) + [value for value in options.extra_targets if value in allowed]
    allowed = set(REAL_DEFAULT_TARGETS) | set(REAL_EXTRA_TARGETS)
    return list(REAL_DEFAULT_TARGETS) + [value for value in options.extra_targets if value in allowed]


def _ellipse_mask(shape_hw, box):
    height, width = shape_hw
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, ((x0 + x1) // 2, (y0 + y1) // 2),
                (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)), 0, 0, 360, 255, -1)
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
    rect = (round((x0 - crop_x0) * scale), round((y0 - crop_y0) * scale),
            max(1, round(box_width * scale)), max(1, round(box_height * scale)))
    labels = np.zeros(crop.shape[:2], np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, labels, rect, background, foreground, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    foreground_mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if foreground_mask.sum() / 255 < 0.20 * rect[2] * rect[3]:
        return None
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if scale != 1:
        foreground_mask = cv2.resize(foreground_mask, (crop_x1 - crop_x0, crop_y1 - crop_y0), interpolation=cv2.INTER_NEAREST)
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


class FrameCensor:
    def __init__(self, detector, options, targets, check_cancel=lambda: None):
        self.detector = detector
        self.options = options
        self.targets = targets
        self.check_cancel = check_cancel
        self.index = 0
        self.last_boxes = []
        self.misses = 0
        self.hit_frames = 0
        self.total_boxes = 0

    def __call__(self, image):
        self.check_cancel()
        if self.index % max(1, int(self.options.detect_every)) == 0:
            boxes = self.detector.detect(image, self.targets, self.options.confidence)
            if boxes:
                self.last_boxes, self.misses = boxes, 0
            else:
                self.misses += 1
                boxes = self.last_boxes if self.misses <= self.options.hold else []
                if not boxes:
                    self.last_boxes = []
        else:
            boxes = self.last_boxes
        self.index += 1
        if not boxes:
            return image, 0
        self.total_boxes += len(boxes)
        output, _ = render_censor(image, build_mask(image, boxes, self.options.shape), self.options)
        self.hit_frames += 1
        return output, len(boxes)


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
        image.save(destination, format={".webp": "WEBP", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP"}.get(fmt, "PNG"), optimize=fmt == ".png")
    return {"frames": 1, "hit_frames": int(bool(boxes)), "boxes": len(boxes)}


def process_gif(source, destination, frame_censor, check_cancel):
    with Image.open(source) as opened:
        loop = opened.info.get("loop", 0)
        frames, durations = [], []
        for frame in ImageSequence.Iterator(opened):
            check_cancel()
            rgba = frame.convert("RGBA")
            output, _ = frame_censor(_rgba_to_bgr(rgba))
            frames.append(_bgr_to_rgba(output, rgba.getchannel("A")))
            durations.append(frame.info.get("duration", 80))
    if not frames:
        raise ValueError("GIF 没有可处理的帧")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=durations, loop=loop, disposal=2, optimize=False)
    return {"frames": len(frames), "hit_frames": frame_censor.hit_frames, "boxes": frame_censor.total_boxes}


def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def process_video(source, destination, frame_censor, check_cancel, progress=lambda *_: None):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("无法打开视频；请检查 OpenCV 支持的编码格式")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("视频没有有效的宽高")
    if width % 2 or height % 2:
        raise ValueError("视频宽高必须为偶数，避免 H.264 编码改变画面尺寸")
    temp_dir = Path(tempfile.mkdtemp(prefix="forge_censor_"))
    raw = temp_dir / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("视频编码器初始化失败")
    frames = 0
    try:
        while True:
            check_cancel()
            ok, frame = capture.read()
            if not ok:
                break
            processed, _ = frame_censor(frame)
            writer.write(processed)
            frames += 1
            if frames % max(1, total // 50 or 1) == 0:
                progress(frames / max(1, total), f"视频帧 {frames}/{total or '?'}")
    except BaseException:
        capture.release()
        writer.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        capture.release()
        writer.release()
    final = destination.with_suffix(".mp4")
    ffmpeg = get_ffmpeg()
    merged = False
    if ffmpeg:
        ffmpeg_output = temp_dir / "final.mp4"
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-i", str(source),
                   "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "18",
                   "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(ffmpeg_output)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=7200)
            if completed.returncode == 0 and ffmpeg_output.exists() and ffmpeg_output.stat().st_size:
                shutil.move(str(ffmpeg_output), final)
                merged = True
        except (OSError, subprocess.SubprocessError):
            pass
    if not merged:
        shutil.move(str(raw), final)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return {"frames": frames, "hit_frames": frame_censor.hit_frames, "boxes": frame_censor.total_boxes, "path": final}


def media_sources(mode, uploads, directory, recursive, sort_mode, allow_directories=True, excluded_root=None):
    if mode == "本机文件夹":
        if not allow_directories:
            raise ValueError("Forge 已使用 --hide-ui-dir-config 禁用本机目录访问")
        root = Path(str(directory or "").strip()).expanduser()
        if not root.is_dir():
            raise ValueError("请选择有效的本机媒体文件夹")
        paths = root.rglob("*") if recursive else root.glob("*")
        excluded = Path(excluded_root).resolve() if excluded_root else None
        files = [path for path in paths if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
                 and not (excluded and (path.resolve() == excluded or excluded in path.resolve().parents))]
    else:
        files = []
        for item in uploads or []:
            if isinstance(item, dict):
                files.append(Path(item.get("path") or item.get("name") or ""))
            elif hasattr(item, "name"):
                files.append(Path(item.name))
            else:
                files.append(Path(item))
    files = [path for path in files if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS]
    if not files:
        raise ValueError("没有找到图片、GIF 或视频")
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
    detector = detector or build_detector(options.engine)
    targets = targets_for(options)
    for index, source in enumerate(sources, 1):
        try:
            check_cancel()
            progress((index - 1) / len(sources), f"{index}/{len(sources)} · {source.name}")
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source.stem).strip(" .")[:100] or "media"
            extension = ".mp4" if source.suffix.lower() in VIDEO_EXTENSIONS else source.suffix.lower()
            output = directory / f"{index:05d}_{stem}{extension}"
            frame_censor = FrameCensor(detector, options, targets, check_cancel)
            if source.suffix.lower() in IMAGE_EXTENSIONS:
                stats = process_still(source, output, detector, options, targets, check_cancel)
            elif source.suffix.lower() in GIF_EXTENSIONS:
                stats = process_gif(source, output, frame_censor, check_cancel)
            else:
                stats = process_video(source, output, frame_censor, check_cancel, progress)
                output = stats.pop("path")
            result.outputs.append(output)
            result.completed += 1
            if len(result.previews) < 20:
                if output.suffix.lower() in VIDEO_EXTENSIONS:
                    preview = Image.new("RGB", (320, 180), "#222")
                else:
                    with Image.open(output) as image:
                        preview = image.convert("RGBA")
                        preview.thumbnail((1024, 1024))
                        preview.load()
                result.previews.append((preview, output.name))
            result.rows.append({"输入文件": source.name, "输出文件": output.name, "媒体类型": source.suffix.upper().lstrip("."),
                               "状态": "成功", "帧数": stats.get("frames", 1), "打码帧": stats.get("hit_frames", 0),
                               "检测框累计": stats.get("boxes", 0), "错误": ""})
        except Cancelled:
            result.cancelled = True
            break
        except Exception as exc:
            result.failed += 1
            result.rows.append({"输入文件": source.name, "状态": "失败", "错误": f"{type(exc).__name__}: {exc}"})
        progress(index / len(sources), f"已完成 {index}/{len(sources)}")
    result.report = directory / "自动打码报告.csv"
    fields = ["输入文件", "输出文件", "媒体类型", "状态", "帧数", "打码帧", "检测框累计", "错误"]
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
