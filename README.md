# SAM3 (MPS) 视频分割 Demo

这是一个基于 SAM3 (Segment Anything Model 3) 的实时视频分割 Demo，专为 Apple Silicon (MPS) 优化。支持通过文本、点选和框选三种交互方式进行对象分割与跟踪。

## 核心亮点

- **MPS 原生支持**：特意移除了 `triton` 依赖，确保模型能在 macOS (MPS) 上流畅运行。
- **FP16 高效推理**：强制使用 `float16` 半精度推理，大幅节省显存并提升在 Apple Silicon 上的运行速度。
- **本地权重加载**：优先支持从 ModelScope 下载的本地权重，无需依赖 Hugging Face 连接。
- **Transformers 最新版**：基于 `transformers` 的 git main 分支开发，使用最新的 SAM3 API。

## 功能特性

- **双推理模式**：支持 PCS (文本提示) 与 PVS (交互式视觉提示)。
- **实时交互**：在 UI 上实时调整文本、点或框，结果下一帧立即生效。
- **帧同步展示**：后端自动将多图拼接，前端展示 [RGB 原图 | 分割覆盖图 | 二值掩码]。
- **性能监控**：实时显示 RGB 读取帧率 (RGB FPS) 与分割推理帧率 (Seg FPS)。

## 环境要求

- **操作系统**: macOS (建议最新版本以获得最佳 MPS 支持)
- **硬件**: Apple Silicon (M1/M2/M3/M4 等)
- **Python**: 3.10+

### 关键依赖安装

由于本项目依赖 `transformers` 的最新特性且需避免 `triton`，建议如下安装：

```bash
# 1. 基础依赖
pip install fastapi uvicorn[standard] opencv-python numpy pillow accelerate torch

# 2. 安装 Transformers (Git Main)
pip install git+https://github.com/huggingface/transformers.git
```

## 模型准备

本项目使用 **ModelScope** 提供的权重文件（无需特殊 License 审批，便于国内访问）。
代码会自动尝试加载 `~/.cache/modelscope/hub/models/facebook/sam3/`。

如果该路径不存在，脚本会回退尝试从 Hugging Face 在线加载（可能较慢）。

## 运行方式

1. **克隆项目**
   ```bash
   git clone <repo_url>
   cd sam3_demo
   ```

2. **启动后端**
   ```bash
   python app.py
   ```
   后端将运行在 `http://0.0.0.0:8000`。
   *首次启动会加载模型，请耐心等待。*

3. **访问 UI**
   打开浏览器访问 `http://localhost:8000`。

4. **使用说明**
   - **视频源**：输入 `webcam` 使用摄像头，或输入本地文件路径。
   - **文本模式**：在输入框中输入对象名称（如 "person"），实时更新分割。
   - **交互模式**：切换到“交互提示”，在左侧 RGB 区域操作：
     - **点击**：添加正向/负向点提示。
     - **拖拽**：添加正向/负向框提示。
   - **清空**：点击“清空提示”重置当前所有状态。

## 常见问题

- **MPS 报错 (mps.add)**：本项目已通过强制 `float16` 严格转换修复了此问题。
- **FPS 性能**：
  - SAM3 推理开销较大，建议在 M2 Pro/Max 或更高芯片上运行以获得更流畅的体验。
  - 减小浏览器窗口或降低视频源分辨率可有效提升 Seg FPS。

## 开发与检查

```bash
# 格式化与检查
ruff format .
ruff check .
```
