"""Small, lazy adapter: the image engine is usable and testable without Forge."""

from .engine import RESAMPLE, check_pixels


def model_names():
    from modules import shared

    names = [item.name for item in getattr(shared, "sd_upscalers", []) if item.name not in ("None", "Lanczos", "Nearest")]
    return ["Lanczos", "Nearest", *dict.fromkeys(names)]


def upscale(image, model_name, factor, max_megapixels, check_cancel):
    from modules import shared

    selected = next((item for item in shared.sd_upscalers if item.name == model_name), None)
    if selected is None:
        raise ValueError(f"找不到超分模型 {model_name}，请刷新模型列表或重启 Forge")
    target = tuple(max(1, round(side * factor)) for side in image.size)
    result = image
    # do_upscale avoids Forge Neo's rounding-to-8 and lets us detect silent model failures.
    for _ in range(4):
        check_cancel()
        native_scale = max(1, float(getattr(selected, "scale", 4) or 4))
        check_pixels((round(result.width * native_scale), round(result.height * native_scale)), max_megapixels)
        old_size = result.size
        selected.scaler.scale = factor
        result = selected.scaler.do_upscale(result, selected.data_path)
        check_cancel()
        if result is None or result.width <= old_size[0] or result.height <= old_size[1]:
            raise RuntimeError(f"模型 {model_name} 没有放大图像；请检查 Forge 控制台的模型加载错误")
        check_pixels(result.size, max_megapixels)
        if result.width >= target[0] and result.height >= target[1]:
            return result.resize(target, RESAMPLE)
    raise RuntimeError("超分模型连续运行 4 次仍未达到目标尺寸")
