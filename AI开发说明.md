# Forge Neo 图片工坊：AI / 开发者交接说明

这份文档写给以后接手本项目的 AI、维护者和扩展开发者。先理解数据流，再改 UI；不要把自动打码、透明超分和普通水印混成一个不可测试的大函数。

## 项目目标

插件在 Forge Neo 里提供一个独立的“图片工坊”页签，当前有两条处理路径：

1. **常规图像路径**：读取图片 → EXIF 方向校正 → 缩放或透明图超分 → 图片水印 → 文字水印 → WebP/PNG/JPEG/TIFF 编码。
2. **二次元打码路径**：读取静态图片 → dghs-imgutils 输出检测框 → 生成几何/轮廓遮罩 → 扩边缘 → 马赛克或高斯模糊 → 保存图片和 CSV/ZIP 报告。

两条路径共享输入目录安全检查、任务目录、停止事件和 Forge 任务锁，但处理内核独立。后续 AI 修改时，优先保持这种边界。

## 目录与职责

```text
scripts/image_workshop.py             Forge 扩展入口，只注册 on_ui_tabs
forge_image_workshop/ui.py            Gradio 面板、参数转换、Forge 状态和任务锁
forge_image_workshop/engine.py        RGBA 缩放、超分、水印、格式编码
forge_image_workshop/forge_adapter.py Forge sd_upscalers 到插件内核的适配
forge_image_workshop/batch.py         常规图片批处理、CSV/ZIP、预览
forge_image_workshop/censor.py        二次元图片检测、遮罩、图片处理
tests/test_engine.py                  常规路径回归
tests/test_censor.py                  自动打码纯算法回归
requirements-censor.txt               自动打码可选依赖
AI开发说明.md                         本文档
```

扩展入口不能在导入时加载 dghs-imgutils。Forge 用户可能只想压缩图片；可选依赖导入失败不能让整个 Forge 启动失败。

## 自动打码原理

检测器接收 BGR `numpy.ndarray`，返回半开区间框 `(x0, y0, x1, y1)`。`AnimeDetector` 使用 dghs-imgutils 的 `detect_censors`；默认检测 `penis`、`pussy`，可选 `nipple_f`。这些标签是外部模型的契约。

检测器只负责“哪里可能需要处理”，不负责渲染。渲染步骤固定为：

```text
检测框
  └─ shape=rect       直接矩形填充
  └─ shape=ellipse    椭圆填充
  └─ shape=fit        GrabCut 贴合轮廓，失败回退椭圆
       ↓
合并多个 mask → 椭圆核膨胀 dilate_px → 全图生成马赛克/模糊图 → 只在 mask 区域替换
```

`strength` 对马赛克表示格子粒度分母，对模糊表示高斯核强度近似值。不要直接把检测框裁掉或填纯色，否则会产生明显矩形边界，也不符合原工具的处理意图。

自动打码只接受静态单帧图片；动画和视频会被输入层过滤或明确拒绝。

## 为什么没有把检测依赖写进主 requirements

`dghs-imgutils` 会引入模型下载和额外版本约束，只对自动打码有用，写入主依赖会让普通图片工坊安装变慢。因此：

- 主插件导入不能依赖这个包。
- `requirements-censor.txt` 是显式可选依赖。
- Windows 用户可双击 `安装自动打码依赖.bat`。
- Forge Neo 环境应使用 `venv/Scripts/python.exe -m pip`，不要用系统 Python 混装。
- 首次检测模型可能从网络下载；安装成功不等于模型下载成功。

若未来 Forge 提供正式的扩展依赖安装回调，可以把可选安装接入回调，但仍要保留 lazy import 和清晰的缺包错误。

## Forge Neo 集成规则

- `scripts/image_workshop.py` 只负责 `script_callbacks.on_ui_tabs(create_ui)`。
- GPU/模型任务通过 `call_queue.queue_lock` 与 Forge 其他任务串行，避免超分和生成同时抢显存。
- 处理前调用 `shared.state.begin(job=...)`，`finally` 中必须调用 `shared.state.end()` 并恢复 `interrupted/skipped/stopping_generation`。
- 不修改 Forge 源码，不 monkey patch Gradio。
- `--hide-ui-dir-config` 开启时隐藏本机目录、输出目录和打开文件夹按钮；后端也必须再次拒绝目录操作。
- 输出写入带时间和随机后缀的独立任务目录，禁止输出目录等于输入目录或成为输入目录上级。
- Gradio 回调返回值必须与输出组件数量严格一致；新增一个 `State` 输出时，所有成功、失败、忙碌分支都要返回四元组。

## 修改建议

1. 先给纯函数加测试，再接 Gradio。遮罩合成、Alpha 保留、检测间隔和取消逻辑都可以不用启动 Forge 测试。
2. 新增图片格式时，同时修改 `IMAGE_EXTENSIONS`、输入 UI、输出扩展名、报告和测试；不要只改文件选择器。
3. 不要把 detector 对象写进全局缓存，除非明确处理并发、模型显存释放和不同会话的参数隔离。
4. 不要把检测框坐标直接用于不同尺寸的帧。检测间隔复用框时，至少在文档中说明移动目标的拖尾风险。
5. 任何“保存成功”的状态都应在文件真实存在并可重新读取后再记录；临时文件失败要清理。
6. 对 CSV 的用户文件名和错误文字做公式注入防护；对 HTML 报告使用 `html.escape`。
7. 不要自动覆盖已有输出。当前任务目录唯一化，文件名使用序号加清理后的 stem。
8. 更新 UI 文案时同步更新 README、CHANGELOG 和本交接文档，尤其是依赖、模型下载和未验证范围。

## 验证顺序

```powershell
# 纯图像和打码算法回归
python -B -m unittest discover -s tests -v

```

交接时要区分纯算法测试、Gradio 组件/回调构建和真实模型推理。不能只因为 Python 能导入就声称 Forge 页面和检测模型都可用；在目标 Forge Neo 环境中手动打开“图片工坊”验证 UI 和模型即可。

## 当前限制与后续方向

- 自动打码检测依赖外部模型，结果必须人工抽查；本项目不承诺检测召回率或法律合规性。
- 当前 UI 没有把检测框可视化编辑；可考虑增加“预览首张 + 应用参数”，但不要阻塞批处理接口。
- 未来可增加并发安全的 detector 缓存和检测框预览。
