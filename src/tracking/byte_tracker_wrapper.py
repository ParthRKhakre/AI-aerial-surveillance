"""ByteTrack wrapper to produce stable tracking IDs and trajectory history."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Deque, Dict, List, Tuple

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from src.detection.yolo_detector import Detection


@dataclass
class TrackResult:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: np.ndarray
    center: Tuple[int, int]


class ByteTrackerWrapper:
    def __init__(
        self,
        fps: int = 30,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        history_size: int = 30,
    ) -> None:
        args = SimpleNamespace(
            track_thresh=0.25,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=False,
        )
        self.tracker = BYTETracker(args, frame_rate=fps)
        self.history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=history_size))

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackResult]:
        if not detections:
            self.tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            return []

        det_array = []
        for det in detections:
            x1, y1, x2, y2 = det.xyxy.tolist()
            det_array.append([x1, y1, x2, y2, det.confidence, det.class_id])
        det_array_np = np.asarray(det_array, dtype=np.float32)

        tracks = self.tracker.update(det_array_np, frame)

        indexed = {(d.class_id, tuple(np.round(d.xyxy, 1))): d for d in detections}
        results: List[TrackResult] = []
        for tr in tracks:
            box = tr.tlbr
            class_id = int(getattr(tr, "cls", -1))
            matched = None
            key = (class_id, tuple(np.round(box, 1)))
            if key in indexed:
                matched = indexed[key]
            conf = float(getattr(tr, "score", matched.confidence if matched else 0.0))
            class_name = matched.class_name if matched else str(class_id)
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            self.history[int(tr.track_id)].append((cx, cy))

            results.append(
                TrackResult(
                    track_id=int(tr.track_id),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    xyxy=np.array(box),
                    center=(cx, cy),
                )
            )
        return results

    def get_history(self, track_id: int) -> List[Tuple[int, int]]:
        return list(self.history.get(track_id, []))
