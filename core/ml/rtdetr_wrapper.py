from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image


DEFAULT_RTDETR_NAMES = {
    0: "bubble",
    1: "text_bubble",
    2: "text_free",
}


class RTDetrDetectionBoxes:
    """Small compatibility shim for the Ultralytics Boxes API used by the app."""

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


@dataclass
class RTDetrDetectionResult:
    """Small compatibility shim for the Ultralytics Results API used by the app."""

    boxes: RTDetrDetectionBoxes
    names: dict[int, str]
    orig_shape: tuple[int, int]
    masks: None = None


class TransformersRTDETRWrapper:
    """Expose a Transformers RT-DETR object detector through a YOLO-like call API."""

    def __init__(
        self,
        model_dir_or_repo_id: str | Path,
        device: torch.device | str = "cpu",
        local_files_only: Optional[bool] = None,
    ):
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        self.model_source = str(model_dir_or_repo_id)
        self.device = torch.device(device)
        source_path = Path(model_dir_or_repo_id)
        if local_files_only is None:
            local_files_only = source_path.exists()

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_source,
            local_files_only=local_files_only,
        )
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            self.model_source,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.names = self._resolve_names()

    def _resolve_names(self) -> dict[int, str]:
        names = DEFAULT_RTDETR_NAMES.copy()
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if isinstance(id2label, dict):
            for key, value in id2label.items():
                try:
                    class_id = int(key)
                except (TypeError, ValueError):
                    continue
                if value and not str(value).upper().startswith("LABEL_"):
                    names[class_id] = str(value)
        return names

    @staticmethod
    def _to_rgb_pil(source: Any) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, np.ndarray):
            array = source
            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")
            if array.ndim == 3 and array.shape[2] >= 3:
                # OpenCV callers pass BGR arrays; RT-DETR expects RGB images.
                rgb = array[:, :, :3][:, :, ::-1]
                return Image.fromarray(rgb).convert("RGB")
        raise TypeError(f"Unsupported RT-DETR input type: {type(source)!r}")

    def _make_result(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        orig_shape: tuple[int, int],
        classes: Optional[Sequence[int]] = None,
        output_device: Optional[torch.device | str] = None,
    ) -> RTDetrDetectionResult:
        if classes is not None:
            allowed = torch.tensor(list(classes), device=labels.device, dtype=labels.dtype)
            keep = (labels[:, None] == allowed[None, :]).any(dim=1)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        target_device = torch.device(output_device) if output_device is not None else self.device
        boxes = boxes.to(device=target_device, dtype=torch.float32)
        scores = scores.to(device=target_device, dtype=torch.float32)
        labels = labels.to(device=target_device, dtype=torch.float32)

        return RTDetrDetectionResult(
            boxes=RTDetrDetectionBoxes(boxes, scores, labels),
            names=self.names,
            orig_shape=orig_shape,
            masks=None,
        )

    def __call__(
        self,
        source: Any,
        conf: float = 0.25,
        device: Optional[torch.device | str] = None,
        classes: Optional[Sequence[int]] = None,
        **_: Any,
    ) -> list[RTDetrDetectionResult]:
        image = self._to_rgb_pil(source)
        width, height = image.size

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        target_sizes = torch.tensor([[height, width]], device=self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            detections = self.processor.post_process_object_detection(
                outputs,
                threshold=float(conf),
                target_sizes=target_sizes,
            )[0]

        output_device = torch.device(device) if device is not None else self.device
        result = self._make_result(
            boxes=detections.get("boxes", torch.empty((0, 4), device=self.device)),
            scores=detections.get("scores", torch.empty((0,), device=self.device)),
            labels=detections.get("labels", torch.empty((0,), device=self.device)),
            orig_shape=(height, width),
            classes=classes,
            output_device=output_device,
        )
        return [result]
