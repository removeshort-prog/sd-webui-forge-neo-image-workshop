from __future__ import annotations

import csv
import re
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .engine import Cancelled, EXTENSIONS, INPUT_EXTENSIONS, load_image, process_image, save_image


def file_path(value):
    if isinstance(value, (str, Path)):
        return Path(value)
    if isinstance(value, dict):
        return Path(value.get("path") or value.get("name") or "")
    if hasattr(value, "name"):
        return Path(value.name)
    raise ValueError("无法识别上传的文件")


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", str(path)))


def collect_sources(mode, uploads, directory, recursive, sort_mode, allow_directories=True, excluded_root=None):
    if mode == "本机文件夹":
        if not allow_directories:
            raise ValueError("Forge 已使用 --hide-ui-dir-config 禁用本机目录访问")
        root = Path(str(directory).strip()).expanduser()
        if not str(directory).strip() or not root.is_dir():
            raise ValueError("请选择有效的本机输入文件夹")
        paths = root.rglob("*") if recursive else root.glob("*")
        excluded = Path(excluded_root).resolve() if excluded_root else None
        files = [path for path in paths if path.is_file() and path.suffix.lower() in INPUT_EXTENSIONS
                 and not (excluded and (path.resolve() == excluded or excluded in path.resolve().parents))]
    else:
        files = [file_path(item) for item in (uploads or [])]
    if not files:
        raise ValueError("没有找到图片，请上传图片或选择输入文件夹")
    if sort_mode == "修改时间从旧到新":
        return sorted(files, key=lambda path: (path.stat().st_mtime, natural_key(path)))
    if sort_mode == "修改时间从新到旧":
        return sorted(files, key=lambda path: (-path.stat().st_mtime, natural_key(path)))
    if sort_mode == "文件名自然排序":
        return sorted(files, key=lambda path: (natural_key(path.name), natural_key(path)))
    return sorted(files, key=natural_key)


@dataclass
class BatchResult:
    directory: Path
    total: int
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    outputs: list = field(default_factory=list)
    previews: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    archive: Path | None = None
    report: Path | None = None
    elapsed: float = 0


def run_batch(sources, destination, options, templates=(), model_upscale=None,
              check_cancel=lambda: None, progress=lambda *args: None, make_zip=True):
    options.validate()
    if options.image_enabled and not templates:
        raise ValueError("已启用图片水印，请上传模板")
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = Path(destination).resolve() / stamp
    directory.mkdir(parents=True, exist_ok=False)
    result = BatchResult(directory, len(sources))
    started = time.monotonic()
    for index, path in enumerate(sources, start=1):
        try:
            check_cancel()
            progress((index - 1) / len(sources), f"{index}/{len(sources)} · {path.name}")
            loaded = load_image(path, options.max_megapixels)
            image, watermark = process_image(loaded, options, templates, model_upscale, check_cancel)
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", path.stem).strip(" .")[:100] or "image"
            prefix = f"{index:05d}_{stem}"
            source_bytes = path.stat().st_size
            first_output = None
            for fmt in dict.fromkeys(options.formats):
                check_cancel()
                output = directory / (prefix + EXTENSIONS[fmt])
                save_image(image, output, fmt, options, loaded.metadata)
                result.outputs.append(output)
                if first_output is None:
                    first_output = output
                byte_count = output.stat().st_size
                result.rows.append({
                    "输入文件": path.name, "输出文件": output.name, "状态": "成功", "格式": fmt,
                    "尺寸": f"{image.width}×{image.height}", "原始字节": source_bytes,
                    "输出字节": byte_count, "体积变化": f"{(byte_count / max(1, source_bytes) - 1) * 100:+.1f}%",
                    "图片水印": watermark, "错误": "",
                })
            result.completed += 1
            if len(result.previews) < 20 and first_output:
                preview = load_image(first_output, options.max_megapixels).image
                preview.thumbnail((1024, 1024))
                result.previews.append((preview, first_output.name))
        except Cancelled:
            result.cancelled = True
            break
        except Exception as exc:
            result.failed += 1
            result.rows.append({"输入文件": path.name, "状态": "失败", "错误": f"{type(exc).__name__}: {exc}"})
        progress(index / len(sources), f"已完成 {index}/{len(sources)}")
    result.report = directory / "处理报告.csv"
    fields = ["输入文件", "输出文件", "状态", "格式", "尺寸", "原始字节", "输出字节", "体积变化", "图片水印", "错误"]
    with result.report.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result.rows:
            # Keep filenames/text from being interpreted as spreadsheet formulas.
            writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value for key, value in row.items()})
    if make_zip and result.outputs:
        progress(1, "正在打包下载文件")
        result.archive = directory / "处理结果.zip"
        # Encoded images are already compressed; ZIP_STORED avoids a slow second compression.
        with zipfile.ZipFile(result.archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in [*result.outputs, result.report]:
                archive.write(path, arcname=path.name)
    result.elapsed = time.monotonic() - started
    return result
