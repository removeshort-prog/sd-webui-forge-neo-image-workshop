from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps, PngImagePlugin, features

RESAMPLE = Image.Resampling.LANCZOS
EXTENSIONS = {"WEBP": ".webp", "PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif"}
INPUT_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".avif"}
POSITIONS = ("左上", "上中", "右上", "左中", "居中", "右中", "左下", "下中", "右下")


class Cancelled(Exception):
    pass


@dataclass
class Options:
    scale: float = 1.0
    model: str = "Lanczos"
    alpha_mode: str = "模型遮罩（匹配 ComfyUI）"
    formats: tuple = ("WEBP",)
    quality: int = 90
    webp_lossless: bool = False
    png_compression: int = 9
    background: str = "#ffffff"
    strip_metadata: bool = True
    preserve_alpha: bool = True
    image_enabled: bool = False
    image_mode: str = "按比例匹配模板并铺满"
    image_width: float = 30
    image_opacity: float = 100
    image_position: str = "居中"
    margin: float = 3
    text_enabled: bool = False
    text: str = ""
    text_size: float = 4.5
    text_color: str = "#ffffff"
    text_opacity: float = 28
    text_position: str = "右下"
    text_tile: bool = False
    text_angle: float = -28
    text_spacing: float = 5
    font_path: str = ""
    max_megapixels: float = 64

    def validate(self):
        ranges = {
            "scale": (0.05, 8), "quality": (1, 100), "png_compression": (0, 9),
            "image_width": (1, 100), "image_opacity": (0, 100), "margin": (0, 30),
            "text_size": (0.5, 25), "text_opacity": (0, 100), "text_angle": (-180, 180),
            "text_spacing": (1, 50), "max_megapixels": (1, 256),
        }
        for name, (low, high) in ranges.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必须在 {low}～{high} 之间")
        if not self.formats or any(fmt not in EXTENSIONS for fmt in self.formats):
            raise ValueError("请至少选择一种有效输出格式")
        if "WEBP" in self.formats and not features.check("webp"):
            raise ValueError("当前 Pillow 不支持 WebP，请检查 Forge 的 Pillow 安装")
        if self.alpha_mode not in ("模型遮罩（匹配 ComfyUI）", "Lanczos（快速）"):
            raise ValueError("未知透明通道处理方式")
        if self.image_mode not in ("按比例匹配模板并铺满", "等比 Logo"):
            raise ValueError("未知图片水印模式")
        for position in (self.image_position, self.text_position):
            if position not in POSITIONS:
                raise ValueError("未知水印位置")
        ImageColor.getrgb(self.background)
        ImageColor.getrgb(self.text_color)
        if self.text_enabled and not self.text.strip():
            raise ValueError("已启用文字水印，请填写文字")
        if len(self.text) > 1000:
            raise ValueError("水印文字请控制在 1000 个字符以内")


@dataclass
class LoadedImage:
    image: Image.Image
    metadata: dict


def check_pixels(size, max_megapixels):
    if size[0] * size[1] > max_megapixels * 1_000_000:
        raise ValueError(f"图像 {size[0]}×{size[1]} 超过 {max_megapixels:g} 百万像素限制")


def load_image(path, max_megapixels=64):
    with Image.open(path) as opened:
        check_pixels(opened.size, max_megapixels)
        if getattr(opened, "n_frames", 1) > 1:
            raise ValueError("暂不支持动画或多页图片，请先导出单帧")
        oriented = ImageOps.exif_transpose(opened)
        info = dict(oriented.info)
        exif = oriented.getexif()
        # Orientation was applied to pixels. Do not carry stale geometry or thumbnails.
        for tag in (274, 256, 257, 40962, 40963, 513, 514):
            if tag in exif:
                del exif[tag]
        if exif:
            info["exif"] = exif.tobytes()
        else:
            info.pop("exif", None)
        return LoadedImage(oriented.convert("RGBA"), info)


def upscale_rgba(image, options, model_upscale=None, check_cancel=lambda: None):
    target = tuple(max(1, round(side * options.scale)) for side in image.size)
    check_pixels(target, options.max_megapixels)
    check_cancel()
    if options.scale <= 1 or options.model in ("None", "Lanczos", "Nearest"):
        resample = Image.Resampling.NEAREST if options.model == "Nearest" else RESAMPLE
        return image.resize(target, resample) if target != image.size else image.copy()
    if model_upscale is None:
        raise ValueError("当前没有可用的 Forge 超分模型接口")
    # ComfyUI LoadImage's MASK is 1-alpha; JoinImageWithAlpha inverts it back.
    rgb = model_upscale(image.convert("RGB"), options.model, options.scale, options.max_megapixels, check_cancel)
    rgb = rgb.convert("RGB").resize(target, RESAMPLE)
    alpha = image.getchannel("A")
    check_cancel()
    if alpha.getextrema()[0] == alpha.getextrema()[1]:
        alpha = Image.new("L", target, alpha.getextrema()[0])
    elif options.alpha_mode == "模型遮罩（匹配 ComfyUI）":
        mask = ImageOps.invert(alpha).convert("RGB")
        mask = model_upscale(mask, options.model, options.scale, options.max_megapixels, check_cancel)
        alpha = ImageOps.invert(mask.convert("RGB").getchannel("R").resize(target, RESAMPLE))
    else:
        alpha = alpha.resize(target, RESAMPLE)
    check_cancel()
    rgb.putalpha(alpha)
    return rgb


def load_font(path, size, text):
    if path:
        try:
            return ImageFont.truetype(str(path), size=size)
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法加载指定字体：{Path(path).name}") from exc
    candidates = [
        str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"),
        str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/simhei.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    if all(ord(char) < 128 for char in text):
        candidates.extend(["DejaVuSans.ttf", "Arial.ttf"])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    raise ValueError("没有找到适用字体，请上传支持水印文字的 TTF/OTF/TTC 字体")


def placement(canvas, mark, position, margin):
    row, col = divmod(POSITIONS.index(position), 3)
    pad = round(min(canvas) * margin / 100)
    return (
        (pad, (canvas[0] - mark[0]) // 2, canvas[0] - mark[0] - pad)[col],
        (pad, (canvas[1] - mark[1]) // 2, canvas[1] - mark[1] - pad)[row],
    )


def change_opacity(image, percent):
    mark = image.copy()
    mark.putalpha(mark.getchannel("A").point([round(i * percent / 100) for i in range(256)]))
    return mark


def composite(base, overlay, preserve_alpha):
    if preserve_alpha:
        # Paint onto the subject's color while retaining its exact coverage.
        result = Image.composite(overlay.convert("RGB"), base.convert("RGB"), overlay.getchannel("A"))
        result.putalpha(base.getchannel("A"))
        return result
    return Image.alpha_composite(base, overlay)


def make_text_mark(size, options):
    font_size = max(8, round(min(size) * options.text_size / 100))
    font = load_font(options.font_path, font_size, options.text)
    stroke = max(1, round(font_size / 24))
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bounds = draw.multiline_textbbox((0, 0), options.text, font=font, stroke_width=stroke)
    padding = stroke + 2
    mark_size = (max(1, bounds[2] - bounds[0] + 2 * padding), max(1, bounds[3] - bounds[1] + 2 * padding))
    check_pixels(mark_size, options.max_megapixels)
    mark = Image.new("RGBA", mark_size)
    alpha = round(options.text_opacity * 255 / 100)
    color = ImageColor.getrgb(options.text_color)[:3]
    ImageDraw.Draw(mark).multiline_text(
        (padding - bounds[0], padding - bounds[1]), options.text, font=font,
        fill=(*color, alpha), stroke_width=stroke, stroke_fill=(0, 0, 0, alpha), align="center",
    )
    # A long signature must fit the canvas; Pillow's alpha-aware resize avoids dark fringes.
    pad = round(min(size) * options.margin / 100)
    max_width = max(1, size[0] - 2 * pad)
    max_height = max(1, size[1] - 2 * pad)
    mark.thumbnail((max_width, max_height), RESAMPLE)
    if options.text_angle:
        mark = mark.rotate(options.text_angle, resample=Image.Resampling.BICUBIC, expand=True)
        mark.thumbnail((max_width, max_height), RESAMPLE)
    return mark


def apply_watermarks(image, options, templates=()):
    result = image
    used = ""
    if options.image_enabled:
        if not templates:
            raise ValueError("已启用图片水印，请上传至少一张模板")
        used, source = min(templates, key=lambda pair: abs(pair[1].width / pair[1].height - image.width / image.height))
        if options.image_mode == "按比例匹配模板并铺满":
            mark = source.resize(image.size, RESAMPLE)
            origin = (0, 0)
        else:
            pad = round(min(image.size) * options.margin / 100)
            width = min(max(1, round(image.width * options.image_width / 100)), max(1, image.width - 2 * pad))
            # A small Logo is allowed to grow to the chosen width.
            factor = min(width / source.width, max(1, image.height - 2 * pad) / source.height)
            mark = source.resize((max(1, round(source.width * factor)), max(1, round(source.height * factor))), RESAMPLE)
            origin = placement(image.size, mark.size, options.image_position, options.margin)
        overlay = Image.new("RGBA", image.size)
        overlay.alpha_composite(change_opacity(mark, options.image_opacity), dest=origin)
        result = composite(result, overlay, options.preserve_alpha)
    if options.text_enabled:
        mark = make_text_mark(image.size, options)
        overlay = Image.new("RGBA", image.size)
        if options.text_tile:
            gap = max(1, round(min(image.size) * options.text_spacing / 100))
            step_x, step_y = mark.width + gap, mark.height + gap
            for row, y in enumerate(range(-mark.height, image.height, step_y)):
                offset = step_x // 2 if row % 2 else 0
                for x in range(-mark.width - offset, image.width, step_x):
                    overlay.alpha_composite(mark, dest=(x, y))
        else:
            overlay.alpha_composite(mark, dest=placement(image.size, mark.size, options.text_position, options.margin))
        result = composite(result, overlay, options.preserve_alpha)
    return result, used


def process_image(loaded, options, templates=(), model_upscale=None, check_cancel=lambda: None):
    options.validate()
    image = upscale_rgba(loaded.image, options, model_upscale, check_cancel)
    check_cancel()
    return apply_watermarks(image, options, templates)


def save_image(image, path, fmt, options, metadata=None):
    # Start from pixels so clearing metadata also removes Pillow's implicit EXIF/ICC.
    clean = Image.new("RGBA", image.size)
    clean.paste(image)
    kwargs = {}
    metadata = metadata or {}
    if not options.strip_metadata:
        if metadata.get("icc_profile"):
            kwargs["icc_profile"] = metadata["icc_profile"]
        if metadata.get("exif"):
            kwargs["exif"] = metadata["exif"]
        if fmt == "PNG":
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                if isinstance(key, str) and isinstance(value, str):
                    pnginfo.add_itxt(key, value)
            kwargs["pnginfo"] = pnginfo
    if fmt == "WEBP":
        kwargs.update(quality=int(options.quality), lossless=options.webp_lossless, method=6, exact=True)
    elif fmt == "PNG":
        kwargs["compress_level"] = int(options.png_compression)
    elif fmt == "JPEG":
        background = Image.new("RGB", clean.size, ImageColor.getrgb(options.background)[:3])
        background.paste(clean.convert("RGB"), mask=clean.getchannel("A"))
        clean = background
        kwargs.update(quality=int(options.quality), optimize=True, progressive=True, subsampling=0)
    elif fmt == "TIFF":
        kwargs["compression"] = "tiff_deflate"
    else:
        raise ValueError(f"不支持的格式：{fmt}")
    # Exclusive creation protects existing files even when filenames collide.
    path = Path(path)
    with path.open("xb") as handle:
        try:
            clean.save(handle, format=fmt, **kwargs)
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise
