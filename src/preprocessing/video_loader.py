"""Video loading and frame iteration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2


@dataclass
class VideoConfig:
    """Configuration for video capture."""

    source: str | int
    target_size: Optional[Tuple[int, int]] = None
    skip_frames: int = 0


class VideoLoader:
    """Loads video frames from file, webcam index, or RTSP URL."""

    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self.cap = cv2.VideoCapture(config.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {config.source}")

    def __iter__(self) -> Generator[tuple[int, any], None, None]:
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            if self.config.target_size:
                frame = cv2.resize(frame, self.config.target_size)

            if self.config.skip_frames > 0 and frame_id % (self.config.skip_frames + 1) != 0:
                frame_id += 1
                continue

            yield frame_id, frame
            frame_id += 1

    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def release(self) -> None:
        self.cap.release()
