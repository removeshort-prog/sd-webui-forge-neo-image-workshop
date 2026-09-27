from __future__ import annotations

import html
import threading
import uuid
from dataclasses import fields
from pathlib import Path

import gradio as gr
from modules import call_queue, shared

from . import forge_adapter
from .batch import collect_sources, file_path, run_batch
from .censor import (
    ANIME_EXTRA_TARGETS, IMAGE_EXTENSIONS, CensorOptions, media_sources, run_censor_batch,
)
from .engine import Cancelled, Options, POSITIONS, load_image

_jobs = {}
_jobs_lock = threading.Lock()


def directories_allowed():
    return not getattr(shared.cmd_opts, "hide_ui_dir_config", False)


def default_output():
    from modules.paths_internal import script_path

    return Path(script_path) / "outputs" / "image-workshop"


def open_output_folder(last_output, configured_output):
    if not directories_allowed():
        raise gr.Error("Forge 已禁用本机目录访问")
    from modules import util

    try:
        latest = str(last_output or "").strip()
        configured = str(configured_output or "").strip()
        directory = Path(latest or configured or default_output()).expanduser().resolve()
        if latest and not directory.is_dir():
            raise ValueError("最近一次输出文件夹已被移动或删除")
        directory.mkdir(parents=True, exist_ok=True)
        util.open_folder(str(directory))
    except (OSError, ValueError) as exc:
        raise gr.Error(f"无法打开输出文件夹：{exc}") from exc


def cancel_job(session):
    with _jobs_lock:
        event = _jobs.get(session)
        if event:
            event.set()
    return "<p>已请求停止，将在当前模型步骤结束后保存已完成的结果。</p>" if event else "<p>当前没有本面板的处理任务。</p>"


def summary_html(result):
    esc = html.escape
    rows = []
    for row in result.rows[:100]:
        cells = [row.get(key, "") for key in ("输入文件", "格式", "尺寸", "体积变化", "状态", "错误")]
        rows.append("<tr>" + "".join(f"<td>{esc(str(cell))}</td>" for cell in cells) + "</tr>")
    status = "已停止" if result.cancelled else "处理结束"
    return (
        f"<div class='fiw-report'><p><b>{status}</b> · 完整处理 {result.completed}/{result.total} 张 · "
        f"失败 {result.failed} 张 · 导出 {len(result.outputs)} 个文件 · {result.elapsed:.1f} 秒</p>"
        f"<p>保存位置：<code>{esc(str(result.directory))}</code></p>"
        "<table><thead><tr><th>图片</th><th>格式</th><th>尺寸</th><th>体积变化</th><th>状态</th><th>错误</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p>体积变化相对于输入文件；超分或增加水印后可能变大。完整记录见 CSV。预览最多 20 张，单文件下载最多 100 个，ZIP 包含所有输出。</p></div>"
    )


def censor_summary_html(result):
    esc = html.escape
    rows = []
    for row in result.rows[:100]:
        cells = [row.get(key, "") for key in ("输入文件", "状态", "检测框", "错误")]
        rows.append("<tr>" + "".join(f"<td>{esc(str(cell))}</td>" for cell in cells) + "</tr>")
    status = "已停止" if result.cancelled else "处理结束"
    return (
        f"<div class='fiw-report'><p><b>{status}</b> · 完成 {result.completed}/{result.total} 张图片 · "
        f"失败 {result.failed} 张 · {result.elapsed:.1f} 秒</p>"
        f"<p>保存位置：<code>{esc(str(result.directory))}</code></p>"
        "<table><thead><tr><th>输入</th><th>状态</th><th>检测框</th><th>错误</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p>自动检测不是人工审核的替代品，请抽查导出的图片。</p></div>"
    )


def execute(values, session, first_only, progress):
    event = threading.Event()
    with _jobs_lock:
        if session in _jobs:
            return [], [], "<p>本会话已有任务，请等待完成或先停止。</p>", gr.update()
        _jobs[session] = event
    try:
        options = Options(**{field.name: values[field.name] for field in fields(Options) if field.name in values})
        font = values.get("font_upload")
        options.font_path = str(file_path(font)) if font else ""
        options.validate()
        allow_dirs = directories_allowed()
        destination = default_output()
        output_directory = str(values.get("output_dir") or "").strip()
        if output_directory:
            if not allow_dirs:
                raise ValueError("Forge 已禁用自定义目录")
            destination = Path(output_directory).expanduser()
        input_directory = str(values.get("input_dir") or "").strip()
        if values["input_mode"] == "本机文件夹" and input_directory:
            source_dir = Path(input_directory).expanduser().resolve()
            output_dir = destination.resolve()
            if source_dir == output_dir or output_dir in source_dir.parents:
                raise ValueError("输出目录不能等于输入目录或是它的上级目录，请选择单独的结果文件夹")
        sources = collect_sources(
            values["input_mode"], values.get("uploads"), input_directory, values["recursive"],
            values["sort_mode"], allow_dirs, destination,
        )
        if first_only:
            sources = sources[:1]
        templates = []
        if options.image_enabled:
            for item in values.get("watermarks") or []:
                path = file_path(item)
                templates.append((path.name, load_image(path, options.max_megapixels).image))
            if not templates:
                raise ValueError("请先上传图片水印模板")

        def check_cancel():
            if event.is_set() or shared.state.interrupted:
                raise Cancelled()

        progress(0, desc="等待 Forge 处理队列")
        # Share Forge's GPU lock with txt2img/img2img/extras; restore state even on failure.
        with call_queue.queue_lock:
            shared.state.begin(job="图片工坊")
            try:
                result = run_batch(
                    sources, destination, options, templates, forge_adapter.upscale, check_cancel,
                    lambda fraction, description: progress(fraction, desc=description), values["make_zip"],
                )
            finally:
                shared.state.end()
                shared.state.interrupted = False
                shared.state.skipped = False
                shared.state.stopping_generation = False
        downloads = ([str(result.archive)] if result.archive else []) + [str(result.report)]
        downloads.extend(str(path) for path in result.outputs[:100])
        return result.previews, downloads, summary_html(result), str(result.directory)
    except Exception as exc:
        return [], [], f"<div class='error'><b>无法处理：</b>{html.escape(str(exc))}</div>", gr.update()
    finally:
        with _jobs_lock:
            _jobs.pop(session, None)


def execute_censor(values, session, first_only, progress):
    event = threading.Event()
    with _jobs_lock:
        if session in _jobs:
            return [], [], "<p>本会话已有任务，请等待完成或先停止。</p>", gr.update()
        _jobs[session] = event
    try:
        options = CensorOptions(
            extra_targets=tuple(values.get("censor_targets") or []),
            confidence=float(values["censor_confidence"]), shape=values["censor_shape"],
            mode=values["censor_mode"], dilate_px=int(values["censor_dilate"]),
            strength=int(values["censor_strength"]), max_megapixels=float(values["censor_max_megapixels"]),
        )
        options.validate()
        allow_dirs = directories_allowed()
        configured_output = str(values.get("censor_output_dir") or "").strip()
        destination = Path(configured_output).expanduser() if configured_output else default_output() / "censored"
        input_directory = str(values.get("censor_input_dir") or "").strip()
        if values["censor_input_mode"] == "本机文件夹" and input_directory:
            source_dir = Path(input_directory).expanduser().resolve()
            output_dir = destination.resolve()
            if source_dir == output_dir or output_dir in source_dir.parents:
                raise ValueError("自动打码输出目录不能等于输入目录或是它的上级目录")
        sources = media_sources(
            values["censor_input_mode"], values.get("censor_uploads"), input_directory,
            bool(values["censor_recursive"]), values["censor_sort"], allow_dirs, destination,
        )
        if first_only:
            sources = sources[:1]

        def check_cancel():
            if event.is_set() or shared.state.interrupted:
                raise Cancelled()

        progress(0, desc="等待 Forge 自动打码任务")
        with call_queue.queue_lock:
            shared.state.begin(job="自动打码")
            try:
                result = run_censor_batch(
                    sources, destination, options, check_cancel,
                    lambda fraction, description: progress(fraction, desc=description),
                    bool(values["censor_make_zip"]),
                )
            finally:
                shared.state.end()
                shared.state.interrupted = False
                shared.state.skipped = False
                shared.state.stopping_generation = False
        downloads = ([str(result.archive)] if result.archive else []) + [str(result.report)]
        downloads.extend(str(path) for path in result.outputs[:100])
        return result.previews, downloads, censor_summary_html(result), str(result.directory)
    except Exception as exc:
        return [], [], f"<div class='error'><b>无法自动打码：</b>{html.escape(str(exc))}</div>", gr.update()
    finally:
        with _jobs_lock:
            _jobs.pop(session, None)


def create_ui():
    form = {}
    censor_form = {}

    def add(name, component):
        form[name] = component
        return component

    file_type = "filepath"
    allow_dirs = directories_allowed()
    with gr.Blocks(analytics_enabled=False) as panel:
        gr.HTML("<div class='fiw-header'><h2>图片工坊</h2><p>透明图超分 · 自定义水印 · 批量压缩导出</p></div>")
        session = gr.State(value=lambda: uuid.uuid4().hex)
        last_output = gr.State(value="")
        censor_last_output = gr.State(value="")
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Accordion("1 · 输入图片", open=True):
                    mode = add("input_mode", gr.Radio(
                        ["上传图片", "本机文件夹"] if allow_dirs else ["上传图片"], value="上传图片", label="输入方式"))
                    add("uploads", gr.File(label="拖入一张或多张原始图片（保留透明通道）", file_count="multiple", type=file_type,
                                          file_types=[".png", ".webp", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".avif"]))
                    with gr.Group(visible=False) as folder_group:
                        add("input_dir", gr.Textbox(label="本机输入文件夹", placeholder="例如 D:\\待处理图片"))
                        add("recursive", gr.Checkbox(label="包含子文件夹", value=True))
                    add("sort_mode", gr.Dropdown(["路径自然排序", "文件名自然排序", "修改时间从旧到新", "修改时间从新到旧"],
                                               value="路径自然排序", label="处理顺序"))
                with gr.Accordion("2 · 缩放与透明图超分", open=True):
                    add("scale", gr.Slider(0.05, 8, value=1, step=0.05, label="缩放倍数（0.65 = 缩小；2 = 放大两倍）"))
                    with gr.Row():
                        model = add("model", gr.Dropdown(forge_adapter.model_names(), value="Lanczos", label="超分模型"))
                        refresh = gr.Button("刷新列表", size="sm")
                    add("alpha_mode", gr.Radio(["模型遮罩（匹配 ComfyUI）", "Lanczos（快速）"],
                                               value="模型遮罩（匹配 ComfyUI）", label="透明通道处理"))
                    gr.Markdown("倍率大于 1 时使用选定模型。Lanczos / Nearest 是普通缩放。模型遮罩会分别处理彩色图和反向透明遮罩，耗时约为单次超分的两倍。")
                with gr.Accordion("3 · 自定义水印", open=True):
                    with gr.Tabs():
                        with gr.Tab("图片 / 模板"):
                            add("image_enabled", gr.Checkbox(label="启用图片水印", value=False))
                            add("watermarks", gr.File(label="水印图片（可上传多张，自动匹配最接近底图比例的一张）", file_count="multiple",
                                                       type=file_type, file_types=[".png", ".webp", ".jpg", ".jpeg", ".tif", ".bmp"]))
                            add("image_mode", gr.Radio(["按比例匹配模板并铺满", "等比 Logo"], value="按比例匹配模板并铺满", label="图片水印模式"))
                            add("image_width", gr.Slider(1, 100, value=30, step=1, label="Logo 宽度（图片宽度 %；铺满模式忽略）"))
                            add("image_opacity", gr.Slider(0, 100, value=100, step=1, label="图片水印不透明度 %"))
                            add("image_position", gr.Dropdown(list(POSITIONS), value="居中", label="Logo 位置"))
                        with gr.Tab("文字 / 防盗署名"):
                            add("text_enabled", gr.Checkbox(label="启用文字水印", value=False))
                            add("text", gr.Textbox(label="自定义文字（支持多行）", value="Made by Me", lines=2))
                            add("font_upload", gr.File(label="自定义字体（可选；Windows 默认使用微软雅黑）", type=file_type,
                                                       file_types=[".ttf", ".otf", ".ttc"]))
                            with gr.Row():
                                add("text_color", gr.ColorPicker(label="文字颜色", value="#ffffff"))
                                add("text_position", gr.Dropdown(list(POSITIONS), value="右下", label="单处文字位置"))
                            add("text_size", gr.Slider(0.5, 25, value=4.5, step=0.5, label="字号（图片短边 %）"))
                            add("text_opacity", gr.Slider(0, 100, value=28, step=1, label="文字不透明度 %"))
                            add("text_tile", gr.Checkbox(label="全图重复平铺文字", value=False))
                            add("text_angle", gr.Slider(-180, 180, value=-28, step=1, label="文字旋转角度"))
                            add("text_spacing", gr.Slider(1, 50, value=5, step=1, label="平铺间距（图片短边 %）"))
                    add("margin", gr.Slider(0, 30, value=3, step=0.5, label="边距（图片短边 %）"))
                    add("preserve_alpha", gr.Checkbox(label="保持透明轮廓（水印仅绘制在原图主体上）", value=True))
                with gr.Accordion("4 · 压缩与导出", open=True):
                    add("formats", gr.CheckboxGroup(["WEBP", "PNG", "JPEG", "TIFF"], value=["WEBP"], label="输出格式（可多选）"))
                    add("quality", gr.Slider(1, 100, value=90, step=1, label="WebP / JPEG 质量（越低通常越小）"))
                    add("webp_lossless", gr.Checkbox(label="WebP 无损（开启后质量值影响编码速度，不降低画质）", value=False))
                    add("png_compression", gr.Slider(0, 9, value=9, step=1, label="PNG 压缩级别（始终无损）"))
                    add("background", gr.ColorPicker(label="JPEG 透明区域填充色", value="#ffffff"))
                    add("strip_metadata", gr.Checkbox(label="清除 metadata / EXIF / 提示词", value=True))
                    add("make_zip", gr.Checkbox(label="生成 ZIP 批量下载包", value=True))
                    add("output_dir", gr.Textbox(label="输出文件夹（留空使用 Forge/outputs/image-workshop）", visible=allow_dirs))
                    add("max_megapixels", gr.Slider(1, 256, value=64, step=1, label="单张及模型中间图像上限（百万像素）"))
                    gr.Markdown("WebP / PNG / TIFF 支持透明；JPEG 会填充底色。仅处理静态单帧图片。输出写入独立任务文件夹。")
            with gr.Column(scale=6):
                with gr.Row():
                    start = gr.Button("开始批量处理", variant="primary")
                    preview = gr.Button("试处理首张")
                    stop = gr.Button("停止")
                gallery = gr.Gallery(label="实际导出效果（最多 20 张缩略图）", columns=2, height=520, elem_id="fiw-gallery")
                with gr.Row():
                    open_folder = gr.Button("📂 打开输出文件夹", size="sm", scale=0, min_width=180,
                                            visible=allow_dirs, elem_id="fiw-open-folder")
                downloads = gr.File(label="下载 ZIP / 报告 / 单张图片", file_count="multiple", interactive=False)
                status = gr.HTML("<p>上传图片并选择输出格式，即可开始。试处理首张也会按完整设置导出文件。</p>")

                with gr.Accordion("5 · 二次元图片自动打码", open=False):
                    gr.Markdown(
                        "使用 dghs-imgutils 检测二次元图片中的目标区域，再用马赛克或高斯模糊覆盖。"
                        "只处理静态图片；检测结果只用于打码，请务必人工抽查。"
                    )
                    def add_censor(name, component):
                        censor_form[name] = component
                        return component

                    censor_mode = add_censor("censor_input_mode", gr.Radio(
                        ["上传图片", "本机文件夹"] if allow_dirs else ["上传图片"], value="上传图片", label="输入方式"))
                    add_censor("censor_uploads", gr.File(
                        label="上传图片（可多选）", file_count="multiple", type=file_type,
                        file_types=sorted(IMAGE_EXTENSIONS)))
                    with gr.Group(visible=False) as censor_folder_group:
                        add_censor("censor_input_dir", gr.Textbox(label="本机图片文件夹", placeholder="例如 D:\\待处理图片"))
                        add_censor("censor_recursive", gr.Checkbox(label="包含子文件夹", value=True))
                    add_censor("censor_sort", gr.Dropdown(
                        ["路径自然排序", "文件名自然排序", "修改时间从旧到新", "修改时间从新到旧"],
                        value="路径自然排序", label="处理顺序"))
                    add_censor("censor_confidence", gr.Slider(
                        0.01, 0.99, value=0.25, step=0.01, label="置信度阈值"))
                    add_censor("censor_targets", gr.CheckboxGroup(
                        [(label, code) for code, label in ANIME_EXTRA_TARGETS.items()],
                        value=[], label="额外检测部位（核心部位始终开启）"))
                    with gr.Row():
                        add_censor("censor_shape", gr.Radio(
                            [("贴合轮廓（GrabCut，失败回退椭圆）", "fit"), ("椭圆", "ellipse"), ("矩形", "rect")],
                            value="fit", label="遮罩形状"))
                        add_censor("censor_mode", gr.Radio(
                            [("马赛克", "mosaic"), ("高斯模糊", "blur")], value="mosaic", label="打码方式"))
                    with gr.Row():
                        add_censor("censor_dilate", gr.Slider(0, 100, value=15, step=1, label="扩边缘（像素）"))
                        add_censor("censor_strength", gr.Slider(4, 300, value=100, step=1, label="马赛克粒度 / 模糊强度"))
                    add_censor("censor_output_dir", gr.Textbox(
                        label="自动打码输出文件夹（留空使用 Forge/outputs/image-workshop/censored）",
                        visible=allow_dirs))
                    with gr.Row():
                        add_censor("censor_make_zip", gr.Checkbox(label="生成 ZIP 和 CSV 报告", value=True))
                        add_censor("censor_max_megapixels", gr.Slider(
                            1, 256, value=64, step=1, label="单帧像素上限（百万像素）"))
                    gr.Markdown(
                        "首次使用请先安装 `requirements-censor.txt` 中的二次元检测依赖。"
                    )
                    with gr.Row():
                        censor_start = gr.Button("开始自动打码", variant="primary")
                        censor_preview = gr.Button("试处理首张图片")
                        censor_stop = gr.Button("停止打码")
                        censor_open_folder = gr.Button("📂 打开打码输出文件夹", size="sm", scale=0, min_width=190,
                                                        visible=allow_dirs)
                    censor_gallery = gr.Gallery(label="自动打码结果预览（最多 20 张）", columns=2, height=360,
                                                elem_id="fiw-censor-gallery")
                    censor_downloads = gr.File(label="下载打码 ZIP / 报告 / 图片", file_count="multiple", interactive=False)
                    censor_status = gr.HTML("<p>尚未运行自动打码。</p>")
        inputs = set(form.values()) | {session}

        def run(data, progress=gr.Progress()):
            return execute({key: data[component] for key, component in form.items()}, data[session], False, progress)

        def run_preview(data, progress=gr.Progress()):
            return execute({key: data[component] for key, component in form.items()}, data[session], True, progress)

        def run_censor(data, progress=gr.Progress()):
            return execute_censor({key: data[component] for key, component in censor_form.items()}, data[session], False, progress)

        def run_censor_preview(data, progress=gr.Progress()):
            return execute_censor({key: data[component] for key, component in censor_form.items()}, data[session], True, progress)

        def refresh_models(current):
            names = forge_adapter.model_names()
            return gr.update(choices=names, value=current if current in names else "Lanczos")

        mode.change(lambda value: gr.update(visible=value == "本机文件夹" and allow_dirs), inputs=mode, outputs=folder_group, queue=False)
        censor_mode.change(lambda value: gr.update(visible=value == "本机文件夹" and allow_dirs),
                          inputs=censor_mode, outputs=censor_folder_group, queue=False)
        refresh.click(refresh_models, inputs=model, outputs=model, queue=False)
        start.click(run, inputs=inputs, outputs=[gallery, downloads, status, last_output])
        preview.click(run_preview, inputs=inputs, outputs=[gallery, downloads, status, last_output])
        stop.click(cancel_job, inputs=session, outputs=status, queue=False)
        open_folder.click(open_output_folder, inputs=[last_output, form["output_dir"]], outputs=[],
                          queue=False, show_progress=False)
        censor_inputs = set(censor_form.values()) | {session}
        censor_start.click(run_censor, inputs=censor_inputs,
                           outputs=[censor_gallery, censor_downloads, censor_status, censor_last_output])
        censor_preview.click(run_censor_preview, inputs=censor_inputs,
                             outputs=[censor_gallery, censor_downloads, censor_status, censor_last_output])
        censor_stop.click(cancel_job, inputs=session, outputs=censor_status, queue=False)
        censor_open_folder.click(open_output_folder, inputs=[censor_last_output, censor_form["censor_output_dir"]],
                                 outputs=[], queue=False, show_progress=False)
    return [(panel, "图片工坊", "forge_image_workshop")]
