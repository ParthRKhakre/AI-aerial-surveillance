"""Visualization utilities for dual-window aerial surveillance UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from src.tracking.byte_tracker_wrapper import TrackResult


@dataclass
class RenderToggles:
    show_boxes: bool = True
    show_labels: bool = True
    show_ids: bool = True
    show_trails: bool = True
    show_fps: bool = True


class SurveillanceRenderer:
    def __init__(self) -> None:
        self.colors: Dict[int, Tuple[int, int, int]] = {}

    def _color(self, class_id: int) -> Tuple[int, int, int]:
        if class_id not in self.colors:
            rng = np.random.default_rng(seed=class_id + 42)
            self.colors[class_id] = tuple(int(v) for v in rng.integers(80, 255, 3))
        return self.colors[class_id]

    def draw(
        self,
        raw_frame: np.ndarray,
        analyzed_frame: np.ndarray,
        tracks: Iterable[TrackResult],
        history_getter,
        fps: float,
        toggles: RenderToggles,
    ) -> np.ndarray:
        left = raw_frame.copy()
        right = analyzed_frame.copy()

        cv2.putText(left, "RAW FEED", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(right, "AI ANALYSIS", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        for tr in tracks:
            color = self._color(tr.class_id)
            x1, y1, x2, y2 = tr.xyxy.astype(int)
            if toggles.show_boxes:
                cv2.rectangle(right, (x1, y1), (x2, y2), color, 2)

            if toggles.show_labels or toggles.show_ids:
                label_parts = []
                if toggles.show_ids:
                    label_parts.append(f"ID {tr.track_id}")
                if toggles.show_labels:
                    label_parts.append(f"{tr.class_name} {tr.confidence * 100:.1f}%")
                label = " | ".join(label_parts)
                cv2.putText(
                    right,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            if toggles.show_trails:
                pts = history_getter(tr.track_id)
                for i in range(1, len(pts)):
                    cv2.line(right, pts[i - 1], pts[i], color, 2)

        if toggles.show_fps:
            cv2.putText(right, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        combined = cv2.hconcat([left, right])
        return combined
