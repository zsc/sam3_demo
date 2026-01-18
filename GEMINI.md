# gemini.md — SAM3(MPS) 视频分割 Demo（Python + HTML，端口 8000）

> 你是一个资深全栈/算法工程师。请生成一个**可运行**的最小可用 Demo 项目：浏览器访问 `http://localhost:8000`，左右并排展示 **RGB 视频** 与 **SAM3 分割结果**（把多张图拼成一张大图以保证帧同步显示），并分别显示 **RGB FPS** 与 **Seg FPS**。支持在 UI 上用 **文字 / 点选 / 画框** 三种方式交互式提示（prompt），调整后**实时生效**。全部文档（README 等）用中文。

文档在 doc.txt

---

## 1) 必须满足的硬性约束

1. **推理后端：PyTorch + MPS**（Apple Silicon）
   - `device = torch.device("mps")` 优先；若不可用回退 `cpu`。
   - dtype：默认尝试 `torch.float16`；若 MPS 报错/不支持则回退 `torch.float32`。
   - 不允许依赖 triton。

2. **SAM3 必须使用 🤗 Transformers 的实现**
   - 不用官方 sam3 仓库（因 triton 依赖导致 MPS 不可用）。
   - 必须支持 **流式（streaming）逐帧推理**。

3. **服务端端口固定 8000**
   - `python -m app` 或 `python app.py` 启动后，监听 `0.0.0.0:8000`。

4. **UI 交互实时生效**
   - 文本 prompt：输入框实时更新。
   - 点选 prompt：在画面上点点（正/负点）实时更新。
   - 框选 prompt：拖拽画框（正/负框）实时更新。

5. **画面显示要求**
   - 浏览器端 side-by-side：左侧 RGB，右侧分割（overlay 或 mask）。
   - 为了帧同步，后端**将多张图拼接**成一张：
     - 例如：`[RGB | overlay | mask]` 或 `[RGB | overlay]` 横向拼接。
   - 页面上分别显示两类 FPS：
     - `RGB FPS`（视频读取/输入帧率）
     - `Seg FPS`（推理/分割输出帧率）

6. **项目输出内容**
   - 生成：`README.md`（中文）、`.gitignore`、`requirements.txt`、可运行代码。
   - 代码写完后：运行静态检查（至少 ruff；可选 mypy/pyright，但不要卡住）。
   - 最后给出 `git init && git add && git commit` 的命令（不能真的推远端）。

---

## 2) 目标用户体验（UI 规格）

### 2.1 页面布局

- 顶部工具栏：
  - 视频源选择：
    - 本地视频路径输入（默认提供一个示例路径占位，或允许从后端 `static/` 读一个样例 mp4）。
    - 或者下拉选择：`demo.mp4` / `webcam`（如果做 webcam，允许作为可选项，不强制）。
  - Prompt 模式切换（单选）：`文本` / `点` / `框`
  - 正负标签切换（单选）：`正` / `负`
  - 清空按钮：清空当前 session 的所有 prompt
  - 阈值滑条：`score_threshold` 与 `mask_threshold`（默认 0.5）

- 主画布：
  - 显示后端发送的**拼接大图**（例如 2~3 列拼接）。
  - 鼠标交互：
    - 点模式：点击一次发送一个点（x,y）与 label（正/负）。
    - 框模式：按下-拖拽-松开生成 bbox（x1,y1,x2,y2）与 label。

- 右侧/底部状态：
  - `RGB FPS:`
  - `Seg FPS:`
  - 当前 prompt 概览（文本、点数量、框数量）

### 2.2 帧同步显示逻辑

- 前端不做任何“推理帧缓存/对齐”。
- 后端每次发送的是**同一时刻**的一张拼接图（RGB 与 Seg 同帧），前端直接绘制即可。

---

## 3) 后端架构（SPEC）

### 3.1 技术选型

- Web 框架：FastAPI
- 实时通道：WebSocket（同一 ws 既收 prompt，也推送图像帧）
- 视频读取：OpenCV（cv2.VideoCapture）或 imageio
- 图像拼接与编码：PIL + numpy；输出 JPEG/PNG（建议 JPEG 减小带宽）

### 3.2 目录结构（建议）

```
.
├── app.py
├── sam3_engine.py
├── web/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── static/
│   └── demo.mp4              # 可选示例视频
├── README.md
├── requirements.txt
├── .gitignore
└── pyproject.toml             # ruff 配置
```

### 3.3 运行方式

- `pip install -r requirements.txt`
- `python app.py`
- 浏览器打开 `http://localhost:8000`

---

## 4) 推理与 Prompt 设计（核心逻辑）

你需要支持两条推理路径：

### 路径 A：文本提示（PCS / Video）

- 用 `Sam3VideoModel` + `Sam3VideoProcessor`。
- 以“流式逐帧”方式处理：帧到达即推理并立刻返回该帧结果。
- 文本 prompt 变化：更新 inference_session，后续帧立即生效。

### 路径 B：交互提示（点/框，PVS / TrackerVideo）

- 用 `Sam3TrackerVideoModel` + `Sam3TrackerVideoProcessor`。
- 点/框属于“某个对象”的提示；最小实现可以只维护一个 obj_id=1。
- 点/框变化：通过 processor 将输入写入 session，后续帧 propagate。

> 说明：如果同时要支持 文本+点/框 的“组合提示”，可以先不做；最小版允许两种模式互斥（文本模式走 A，点/框模式走 B）。

---

## 5) WebSocket 协议（必须实现）

### 5.1 客户端 -> 服务端（JSON）

统一消息格式：

```json
{ "type": "...", "payload": { ... } }
```

必须支持：

1) `start`
```json
{ "type": "start", "payload": { "source": "static/demo.mp4" } }
```

2) `set_text`
```json
{ "type": "set_text", "payload": { "text": "person" } }
```

3) `add_point`
```json
{ "type": "add_point", "payload": { "x": 210, "y": 350, "label": 1 } }
```

4) `add_box`
```json
{ "type": "add_box", "payload": { "x1": 75, "y1": 275, "x2": 1725, "y2": 850, "label": 1 } }
```

5) `clear_prompts`
```json
{ "type": "clear_prompts", "payload": {} }
```

6) `set_thresholds`
```json
{ "type": "set_thresholds", "payload": { "score_threshold": 0.5, "mask_threshold": 0.5 } }
```

### 5.2 服务端 -> 客户端（JSON header + 二进制）

为了效率：建议用 **二进制帧**发送 JPEG bytes；同时每隔 N 帧或每帧发送一个 JSON 状态。

- 二进制：`<jpeg_bytes>`（拼接图）
- JSON：
```json
{
  "type": "stats",
  "payload": {
    "rgb_fps": 29.7,
    "seg_fps": 12.4,
    "mode": "text|point|box",
    "prompt_summary": {"text": "person", "points": 3, "boxes": 1}
  }
}
```

> 若你想简化：也可以每帧发送一个 JSON：`{ image_b64, stats }`，但性能会差一些。

---

## 6) 画面合成规则（必须）

### 6.1 输入

- `rgb_frame`: HxWx3 uint8
- `masks`: 可能是多个实例掩码（NxHxW）。

### 6.2 输出

- `overlay_frame`: RGB 上叠加 mask（半透明）。
- `mask_vis`: 单通道或伪彩（最简单：把所有实例 mask 做 OR 得到一个二值 mask）。

### 6.3 拼接

- 横向拼接：
  - 最小：`[rgb_frame | overlay_frame]`
  - 推荐：`[rgb_frame | overlay_frame | mask_vis]`

编码为 JPEG 后发送。

---

## 7) 性能与 FPS 计算

- `RGB FPS`：以视频读取成功的帧时间间隔计算（滑动窗口平均，例如最近 30 帧）。
- `Seg FPS`：以一次推理完成的耗时计算（同样滑动窗口）。
- UI 每秒刷新一次 stats（避免频繁 DOM 更新）。

---

## 8) 依赖（requirements.txt 建议）

- fastapi
- uvicorn[standard]
- opencv-python
- numpy
- pillow
- transformers
- accelerate
- torch  （注意：macOS 安装 torch/mps 通常走官方 pip；这里 requirements 里可以不锁死版本，README 里写安装建议）
- ruff

可选：
- mypy
- types-Pillow
- types-requests

---

## 9) 静态检查要求

至少做到：

- `ruff check .`
- `ruff format .` 或 black（任选其一，建议 ruff-format）

如果你加了 mypy：
- `mypy .`（别强制到无法通过的严格级别）

---

## 10) README.md（中文）必须包含

1. 项目简介（做什么、截图/示意）
2. 环境要求（macOS + MPS，Python 版本建议 3.10+）
3. 安装步骤（含 torch(mps) 安装提示）
4. 运行方式（启动后端、浏览器访问）
5. UI 使用说明（文本/点/框，正负标签，清空）
6. 常见问题：
   - MPS dtype 不支持怎么办
   - FPS 很低怎么办（降低分辨率、降低 max_frame_num_to_track 等）

---

## 11) .gitignore（必须）

至少忽略：
- venv/
- __pycache__/
- .DS_Store
- *.pyc
- .ruff_cache/
- .mypy_cache/
- outputs/（如果你生成调试图）

---

## 12) 核心 API 使用示例（务必在代码里落地）

> 下面给出“必须用到”的 Transformers SAM3 核心调用方式。你生成的代码需要与这些示例一致（可封装，但不要换成别的实现）。

### 12.1 文本视频（Sam3VideoModel + Streaming）

要求：
- 初始化 session
- add_text_prompt
- 逐帧 streaming：对每一帧，先用 processor 做预处理，然后把 `frame=inputs.pixel_values[0]` 传给 model
- postprocess_outputs 并拿到 masks/boxes/scores

伪代码（必须体现同等步骤）：

```python
from transformers import Sam3VideoModel, Sam3VideoProcessor
import torch

model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

session = processor.init_video_session(
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=dtype,
)

session = processor.add_text_prompt(inference_session=session, text="person")

# streaming loop
inputs = processor(images=frame_rgb, device=device, return_tensors="pt")
model_outputs = model(
    inference_session=session,
    frame=inputs.pixel_values[0],
    reverse=False,
)
processed = processor.postprocess_outputs(
    session,
    model_outputs,
    original_sizes=inputs.original_sizes,
)
```

### 12.2 点/框视频（Sam3TrackerVideoModel）

要求：
- init_video_session
- 在第 0 帧（或当前帧）通过 `add_inputs_to_inference_session` 写入点/框提示
- 使用 `model.propagate_in_video_iterator(session)` 或 streaming 逐帧（最小版任选其一）

伪代码（必须体现同等步骤）：

```python
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype)
processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

session = processor.init_video_session(inference_device=device, dtype=dtype)

# add one positive click to obj_id=1 at frame 0
processor.add_inputs_to_inference_session(
    inference_session=session,
    frame_idx=0,
    obj_ids=1,
    input_points=[[[[210, 350]]]],
    input_labels=[[[1]]],
    original_size=[H, W],
)

out = model(inference_session=session, frame_idx=0)
mask = processor.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=False)[0]

# propagate (or streaming)
for o in model.propagate_in_video_iterator(session):
    ...
```

---

## 13) 交付清单（最终输出必须包含）

- [ ] `app.py`（FastAPI + WS + 静态文件服务）
- [ ] `sam3_engine.py`（封装 SAM3 推理、session、prompt 更新、mask 合成）
- [ ] `web/index.html`、`web/app.js`、`web/style.css`（纯前端，不用框架）
- [ ] `requirements.txt`
- [ ] `README.md`（中文）
- [ ] `.gitignore`
- [ ] `pyproject.toml`（ruff 配置）

并在 README 里给出：

```bash
ruff check .
ruff format .
# 可选：mypy .

git init
git add .
git commit -m "feat: sam3 mps streaming demo"
```

---

## 14) 质量门槛

- Demo 必须能在**无 CUDA**的 macOS(MPS) 上跑。
- UI 操作后，下一帧起就能看到分割变化。
- 代码清晰、可读、注释适度，错误处理要有（视频路径无效、ws 断开、模型加载失败等）。

---
模型在下面找（尽量只用 model.safetensors）
% ls -lh ~/.cache/modelscope/hub/models/facebook/sam3/
total 13481744
-rw-r--r--  1 georgezhou  staff    25K  1 18 16:02 config.json
-rw-r--r--  1 georgezhou  staff    73B  1 18 16:02 configuration.json
-rw-r--r--  1 georgezhou  staff   7.2K  1 18 16:02 LICENSE
-rw-r--r--  1 georgezhou  staff   512K  1 18 16:02 merges.txt
-rw-r--r--  1 georgezhou  staff   3.2G  1 18 16:05 model.safetensors
-rw-r--r--  1 georgezhou  staff   1.7K  1 18 16:02 processor_config.json
-rw-r--r--  1 georgezhou  staff    25K  1 18 16:02 README.md
-rw-r--r--  1 georgezhou  staff   3.2G  1 18 16:05 sam3.pt
-rw-r--r--  1 georgezhou  staff   588B  1 18 16:02 special_tokens_map.json
-rw-r--r--  1 georgezhou  staff   799B  1 18 16:02 tokenizer_config.json
-rw-r--r--  1 georgezhou  staff   3.5M  1 18 16:02 tokenizer.json
-rw-r--r--  1 georgezhou  staff   842K  1 18 16:02 vocab.json

