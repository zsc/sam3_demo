# SAM3 (MPS) 视频分割 Demo

这是一个基于 SAM3 (Segment Anything Model 3) 的实时视频分割 Demo，专为 Apple Silicon (MPS) 优化。支持通过文本、点选和框选三种交互方式进行对象分割与跟踪。

## 功能特性

- **双推理模式**：支持 PCS (文本提示) 与 PVS (交互式视觉提示)。
- **MPS 深度优化**：
  - 强制使用 `float16` 半精度推理，大幅提升 Apple Silicon 上的运行速度。
  - 严格执行输入 Tensor 的类型转换，避免 MPS 在混合精度运算时的崩溃。
- **实时交互**：在 UI 上实时调整文本、点或框，结果下一帧立即生效。
- **帧同步展示**：后端自动将多图拼接，前端展示 [RGB 原图 | 分割覆盖图 | 二值掩码]。
- **性能监控**：实时显示 RGB 读取帧率 (RGB FPS) 与分割推理帧率 (Seg FPS)。

## 环境要求

- **操作系统**: macOS (建议最新版本以获得最佳 MPS 支持)
- **硬件**: Apple Silicon (M1/M2/M3/M4 等)
- **Python**: 3.10+
- **关键依赖**: 
  - `torch` (需支持 mps)
  - `transformers` (需支持 SAM3)
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

3. **模型准备**
   项目默认优先加载 ModelScope 缓存目录下的模型：
   `~/.cache/modelscope/hub/models/facebook/sam3/`
   
   请确保该目录下包含 `config.json`, `model.safetensors`, `processor_config.json` 等文件。如果路径不存在，系统将尝试从 Hugging Face 远程加载。

## 运行方式

1. **启动后端**
   ```bash
   python app.py
   ```
   后端将运行在 `http://0.0.0.0:8000`。

2. **访问 UI**
   打开浏览器访问 `http://localhost:8000`。

3. **使用说明**
   - **视频源**：输入 `webcam` 使用摄像头，或输入本地文件路径。
   - **文本模式**：在输入框中输入对象名称（如 "person"），实时更新分割。
   - **交互模式**：切换到“交互提示”，在左侧 RGB 区域操作：
     - **点击**：添加正向/负向点提示。
     - **拖拽**：添加正向/负向框提示。
   - **清空**：点击“清空提示”重置当前所有状态。

## 常见问题

- **MPS 报错 (mps.add)**：本项目已通过强制 `float16` 严格转换修复了此问题。
- **FPS 性能**：
  - SAM3 推理开销较大，建议在 M2 或更高芯片上运行以获得更流畅的体验。
  - 减小浏览器窗口或降低视频源分辨率可有效提升 Seg FPS。

## 开发与检查

```bash
# 格式化与检查
ruff format .
ruff check .

# Git 提交
git add .
git commit -m "feat: updated sam3 mps demo with float16 support"
```