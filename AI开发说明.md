# Forge Neo 图片工坊：AI / 开发者交接说明

本扩展只负责图片压缩、多格式导出、自定义水印和透明图超分。二次元自动打码已经迁移到独立的 `[自动打码]` 扩展，不要把两个扩展重新合并。

## 数据流

```text
读取图片 → EXIF 方向校正 → 普通缩放或透明图超分
        → 图片水印 → 文字水印 → WebP/PNG/JPEG/TIFF 编码
```

## 目录与职责

```text
scripts/image_workshop.py             Forge 扩展入口，只注册 on_ui_tabs
forge_image_workshop/ui.py            Gradio 面板、参数转换、Forge 状态和任务锁
forge_image_workshop/engine.py        RGBA 缩放、透明超分、水印、格式编码
forge_image_workshop/forge_adapter.py Forge sd_upscalers 到插件内核的适配
forge_image_workshop/batch.py         图片批处理、CSV/ZIP、预览
tests/test_engine.py                  图像和批处理回归
```

扩展入口不加载任何自动打码模型或额外检测依赖。Forge 用户只安装图片工坊时，不会触发二次元检测包和模型下载。

## 透明图超分

彩色图与反向 Alpha 遮罩分别经过 Forge 超分模型，再合并回 RGBA。ComfyUI 的 MASK 表示透明程度，因此输入遮罩是 `1-Alpha`，合并时再次反转。全不透明或恒定透明度图片会跳过无意义的遮罩模型调用；Lanczos 只对 Alpha 做普通插值。

## Forge 规则

- 任务通过 `call_queue.queue_lock` 与 Forge 其他 GPU 任务串行。
- `shared.state.begin()` 与 `shared.state.end()` 必须成对出现，失败和取消也要恢复 Forge 状态。
- `--hide-ui-dir-config` 开启时隐藏目录输入、输出目录和打开文件夹按钮；后端仍拒绝目录操作。
- 输出写入唯一任务目录，不能覆盖原文件，也不能让输出目录成为输入目录的上级。
- 模型静默返回原图时必须报错，不能把普通插值当成超分成功。

## 修改建议

1. 先为纯函数增加回归测试，再接 Gradio。
2. 新增图片格式时同步修改输入扩展名、输出扩展名、报告和测试。
3. 保留 Alpha、metadata 清理、文件名清理、CSV 公式注入防护、取消清理和输出目录安全检查。
4. 水印颜色合成不能重复乘 Alpha；保存成功前要确认文件真实存在并可重新读取。
5. 任何 UI 参数变化都同步更新 README、CHANGELOG 和本交接文档。

## 验证

```powershell
python -B -m unittest discover -s tests -v
```

纯算法测试、Gradio 页面构建和真实 Forge 模型推理分别记录，不能用其中一项代替另外两项。
