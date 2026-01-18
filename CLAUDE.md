python + html demo（端口 8000）, side-by-side 显示 RGB 视频和 sam3 seg（几张图粘在一起，实现帧同步显示）, 和分别的帧率。sam3 可通过 UI 上输入文字在 UI 上可调并实时生效。用 pytorch 调MPS.
文档都用中文。生成README.md, .gitignore。代码写完了跑下静态检查。没问题了交下 git。


用下面 transformers 的实现，不用官方 sam3（那个依赖 triton 无法用 mps）

>>> from transformers import Sam3Processor, Sam3Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> model = Sam3Model.from_pretrained("facebook/sam3").to(device)
>>> processor = Sam3Processor.from_pretrained("facebook/sam3")

>>> # Load image
>>> image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> # Segment using text prompt
>>> inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Post-process results
>>> results = processor.post_process_instance_segmentation(
...     outputs,
...     threshold=0.5,
...     mask_threshold=0.5,
...     target_sizes=inputs.get("original_sizes").tolist()
... )[0]

>>> print(f"Found {len(results['masks'])} objects")
>>> # Results contain:
>>> # - masks: Binary masks resized to original image size
>>> # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
>>> # - scores: Confidence scores

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
