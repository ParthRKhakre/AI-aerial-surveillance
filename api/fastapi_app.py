"""Optional FastAPI backend for frame-level detection inference."""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.detection.yolo_detector import YOLODetector


class FrameRequest(BaseModel):
    image_b64: str


class DetectionResponse(BaseModel):
    bbox_xyxy: List[float]
    class_id: int
    class_name: str
    confidence: float


@lru_cache(maxsize=1)
def get_detector() -> YOLODetector:
    weights = os.getenv("YOLO_WEIGHTS", "models/yolov8n.pt")
    return YOLODetector(weights=weights)


app = FastAPI(title="Aerial Surveillance Detection API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=List[DetectionResponse])
def detect(payload: FrameRequest) -> List[DetectionResponse]:
    try:
        image_bytes = base64.b64decode(payload.image_b64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Image decode failed")

    detections = get_detector().infer(frame)
    return [
        DetectionResponse(
            bbox_xyxy=det.xyxy.tolist(),
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
        )
        for det in detections
    ]
