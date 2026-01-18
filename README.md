# SAM3 (MPS) 视频分割 Demo

这是一个基于 SAM3 (Segment Anything Model 3) 的实时视频分割 Demo，专为 Apple Silicon (MPS) 优化。支持通过文本、点选和框选三种交互方式进行对象分割与跟踪。

## 功能特性

- **双推理模式**：支持 PCS (文本提示) 与 PVS (视觉提示)。
- **高性能推理**：利用 Apple Silicon 的 MPS 加速，支持流式逐帧推理。
- **实时交互**：在 UI 上实时调整文本、点或框，结果立即生效。
- **帧同步展示**：左右并排展示 RGB 原图、分割覆盖图 (Overlay) 与 二值掩码 (Mask)。
- **性能监控**：实时显示 RGB 读取帧率与分割推理帧率 (Seg FPS)。

## 环境要求

- **操作系统**: macOS (建议最新版本以获得最佳 MPS 支持)
- **硬件**: Apple Silicon (M1/M2/M3 等)
- **Python**: 3.10+
- **关键依赖**: 
  - `torch` (需支持 mps)
  - `transformers`
  - `fastapi` & `uvicorn`
  - `opencv-python`

## 安装步骤

1. **克隆项目**
   ```bash
   git clone <repo_url>
   cd sam3_demo
   ```

2. **安装依赖**
   建议使用虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   *注意：如果你的环境中尚未安装支持 MPS 的 PyTorch，请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 进行安装。*

3. **下载模型**
   项目启动时会自动从 Hugging Face 下载 `facebook/sam3`。请确保网络通畅。

## 运行方式

1. **启动后端**
   ```bash
   python app.py
   ```
   后端将运行在 `http://0.0.0.0:8000`。

2. **访问 UI**
   打开浏览器访问 `http://localhost:8000`。

3. **使用说明**
   - **视频源**：输入 `webcam` 使用摄像头，或输入本地文件路径（如 `static/demo.mp4`）。
   - **文本模式**：在输入框中输入对象名称（如 "person"），按下回车或等待实时生效。
   - **交互模式**：切换到“交互提示”，在左侧 RGB 画面上：
     - **点击**：添加正向/负向点提示。
     - **拖拽**：添加正向/负向框提示。
   - **清空**：点击“清空提示”重置当前所有 Prompt。

## 常见问题

- **MPS 相关报错**：部分 SAM3 算子在 `float16` 下可能在某些系统版本上报错，系统会自动回退到 `float32`。
- **FPS 较低**：SAM3 模型较大，FPS 取决于芯片性能。可以尝试减小输入视频分辨率或降低 `max_frame_num_to_track`。

## 开发与检查

```bash
# 格式化与检查
ruff format .
ruff check .

# Git 提交示例
git init
git add .
git commit -m "feat: sam3 mps streaming demo"
```
