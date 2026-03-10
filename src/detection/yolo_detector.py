"""YOLOv8/YOLOv9 detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: np.ndarray
    class_id: int
    confidence: float
    class_name: str


class YOLODetector:
    def __init__(
        self,
        weights: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 960,
        fp16: bool = True,
    ) -> None:
        self.model = YOLO(weights)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.fp16 = fp16 and device.startswith("cuda")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        r0 = results[0]
        names: Dict[int, str] = r0.names
        boxes = r0.boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()

        for box, class_id, score in zip(xyxy, cls, conf):
            detections.append(
                Detection(
                    xyxy=box,
                    class_id=int(class_id),
                    confidence=float(score),
                    class_name=names.get(int(class_id), str(class_id)),
                )
            )
        return detections

    def export_tensorrt(self, dynamic: bool = True, workspace: int = 4) -> None:
        """Export the loaded model to TensorRT engine."""
        self.model.export(format="engine", dynamic=dynamic, workspace=workspace, half=self.fp16)
