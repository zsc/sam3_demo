import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
from transformers import (
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)


class SAM3Engine:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # MPS mixed precision often causes "mps.add op requires same element type" errors
        # Force float32 for stability
        self.dtype = torch.float32
        
        print(f"Using device: {self.device}, dtype: {self.dtype}")

        # Models and processors
        self.pcs_model = None
        self.pcs_processor = None
        self.pvs_model = None
        self.pvs_processor = None

        # Current mode and session
        self.mode = "text"  # "text" or "interactive"
        self.session = None

        # Prompt state
        self.text_prompt = ""
        self.points = []  # List of [x, y, label]
        self.boxes = []  # List of [x1, y1, x2, y2, label]

        # Thresholds
        self.score_threshold = 0.5
        self.mask_threshold = 0.5

        # Stats
        self.seg_times = []

        self._load_models()

    def _load_models(self):
        print("Loading SAM3 models...")
        # Use local path provided by user
        model_id = os.path.expanduser("~/.cache/modelscope/hub/models/facebook/sam3/")

        if not os.path.exists(model_id):
            print(f"Warning: Local path {model_id} not found. Fallback to 'facebook/sam3'")
            model_id = "facebook/sam3"

        print(f"Loading from: {model_id}")

        try:
            self.pcs_model = Sam3VideoModel.from_pretrained(model_id).to(
                self.device, dtype=self.dtype
            )
            self.pcs_processor = Sam3VideoProcessor.from_pretrained(model_id)

            self.pvs_model = Sam3TrackerVideoModel.from_pretrained(model_id).to(
                self.device, dtype=self.dtype
            )
            self.pvs_processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            # If it fails even with float32, we can't do much
            pass

    def init_session(self, mode="text"):
        self.mode = mode
        if mode == "text":
            self.session = self.pcs_processor.init_video_session(
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=self.dtype,
            )
            if self.text_prompt:
                self.session = self.pcs_processor.add_text_prompt(
                    inference_session=self.session, text=self.text_prompt
                )
        else:
            self.session = self.pvs_processor.init_video_session(
                inference_device=self.device, dtype=self.dtype
            )

    def set_text_prompt(self, text: str):
        print(f"Setting text prompt: {text}")
        self.text_prompt = text
        if self.mode == "text" and self.session:
            try:
                self.session = self.pcs_processor.add_text_prompt(
                    inference_session=self.session, text=text
                )
                print("Text prompt added to session.")
            except Exception as e:
                print(f"Error adding text prompt: {e}")
        elif self.mode != "text":
            self.init_session(mode="text")

    def add_point(self, x: int, y: int, label: int):
        self.points.append([x, y, label])
        if self.mode != "interactive":
            self.init_session(mode="interactive")

    def add_box(self, x1: int, y1: int, x2: int, y2: int, label: int):
        self.boxes.append([x1, y1, x2, y2, label])
        if self.mode != "interactive":
            self.init_session(mode="interactive")

    def clear_prompts(self):
        self.text_prompt = ""
        self.points = []
        self.boxes = []
        if self.session:
            self.init_session(mode=self.mode)

    def process_frame(self, frame_rgb: np.ndarray, frame_idx: int) -> np.ndarray:
        start_time = time.time()
        H, W = frame_rgb.shape[:2]

        masks = None

        if self.mode == "text":
            if not self.text_prompt:
                # No prompt, just return original frame or empty mask
                return self._compose_result(frame_rgb, None)

            inputs = self.pcs_processor(images=frame_rgb, device=self.device, return_tensors="pt")
            with torch.no_grad():
                model_outputs = self.pcs_model(
                    inference_session=self.session,
                    frame=inputs.pixel_values[0].to(self.device, dtype=self.dtype),
                    reverse=False,
                )
            processed = self.pcs_processor.postprocess_outputs(
                self.session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )

            # Debug output
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx} (Text): Found {len(processed['masks'])} masks")
                if len(processed["masks"]) > 0:
                    print(f"  Scores: {processed.get('scores', 'N/A')}")
                    m_min = processed["masks"].min().item()
                    m_max = processed["masks"].max().item()
                    print(f"  Mask range: {m_min:.3f} to {m_max:.3f}")

            if len(processed["masks"]) > 0:
                # processed["masks"] is list of masks [N, H, W]
                masks = processed["masks"]

        else:  # interactive mode
            inputs = self.pvs_processor(images=frame_rgb, device=self.device, return_tensors="pt")

            # For interactive mode, we need to add inputs to session if any
            if self.points or self.boxes:
                # Only add if it's the first frame or we want to update
                # In this simple demo, we add prompts to the current frame to see immediate effect
                input_points = []
                input_labels = []

                if self.points:
                    pts = [[p[:2] for p in self.points]]
                    lbls = [[p[2] for p in self.points]]
                    input_points.append(pts)
                    input_labels.append(lbls)

                pts = []
                lbls = []
                for p in self.points:
                    pts.append([p[0], p[1]])
                    lbls.append(p[2])

                boxes_in = []
                boxes_lbls = []
                for b in self.boxes:
                    boxes_in.append([b[0], b[1], b[2], b[3]])
                    boxes_lbls.append(b[4])

                if pts or boxes_in:
                    self.pvs_processor.add_inputs_to_inference_session(
                        inference_session=self.session,
                        frame_idx=frame_idx,
                        obj_ids=1,
                        input_points=[[pts]] if pts else None,
                        input_labels=[[lbls]] if lbls else None,
                        input_boxes=[[boxes_in]] if boxes_in else None,
                        original_size=[H, W],
                    )

            with torch.no_grad():
                model_outputs = self.pvs_model(
                    inference_session=self.session,
                    frame=inputs.pixel_values[0].to(self.device, dtype=self.dtype),
                )

            masks = self.pvs_processor.post_process_masks(
                [model_outputs.pred_masks], original_sizes=[[H, W]], binarize=False
            )[0]

        end_time = time.time()
        self.seg_times.append(end_time - start_time)
        if len(self.seg_times) > 30:
            self.seg_times.pop(0)

        return self._compose_result(frame_rgb, masks)

    def _compose_result(self, rgb: np.ndarray, masks: Optional[torch.Tensor]) -> np.ndarray:
        # rgb: HxWx3 uint8
        # masks: [N, H, W] or [N, 1, H, W]
        H, W = rgb.shape[:2]
        overlay = rgb.copy()
        mask_vis = np.zeros((H, W), dtype=np.uint8)

        if masks is not None and len(masks) > 0:
            if masks.ndim == 4:
                masks = masks.squeeze(1)  # [N, H, W]

            # Move to CPU for visualization
            masks_np = masks.cpu().float().numpy()

            # Create a combined mask for simple visualization
            combined_mask = np.zeros((H, W), dtype=float)
            for i in range(len(masks_np)):
                m = masks_np[i]
                combined_mask = np.maximum(combined_mask, m)

            # Apply threshold
            binary_mask = (combined_mask > self.mask_threshold).astype(np.uint8)
            mask_vis = (binary_mask * 255).astype(np.uint8)

            # Overlay: Red tint for mask
            overlay_mask = binary_mask[:, :, np.newaxis]
            color = np.array([255, 0, 0], dtype=np.uint8)

            mask_rgb = (overlay_mask * color).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)

        # Side-by-side: [RGB | Overlay | Mask]
        mask_vis_3ch = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

        combined = np.hstack([rgb, overlay, mask_vis_3ch])
        return combined

    def get_seg_fps(self) -> float:
        if not self.seg_times:
            return 0.0
        return 1.0 / (sum(self.seg_times) / len(self.seg_times))
