"""Main real-time dual-window aerial surveillance pipeline."""

from __future__ import annotations

import argparse
import time

import cv2

from src.detection.yolo_detector import YOLODetector
from src.preprocessing.video_loader import VideoConfig, VideoLoader
from src.tracking.byte_tracker_wrapper import ByteTrackerWrapper
from src.visualization.renderer import RenderToggles, SurveillanceRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-Window AI Aerial Surveillance")
    parser.add_argument("--source", type=str, required=True, help="Path to video file, camera index, or RTSP URL")
    parser.add_argument("--weights", type=str, default="models/yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--skip", type=int, default=0, help="Skip N frames between processed frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    video_loader = VideoLoader(VideoConfig(source=source, skip_frames=args.skip))

    detector = YOLODetector(weights=args.weights, conf_thres=args.conf, iou_thres=args.iou, imgsz=args.imgsz)
    tracker = ByteTrackerWrapper(fps=max(1, int(video_loader.fps() or 30)))
    renderer = SurveillanceRenderer()
    toggles = RenderToggles()

    prev = time.perf_counter()

    for _, frame in video_loader:
        start = time.perf_counter()
        detections = detector.infer(frame)
        tracks = tracker.update(detections, frame)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        canvas = renderer.draw(frame, frame, tracks, tracker.get_history, fps, toggles)
        cv2.imshow("Dual-Window AI Aerial Surveillance", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            toggles.show_boxes = not toggles.show_boxes
        elif key == ord("l"):
            toggles.show_labels = not toggles.show_labels
        elif key == ord("i"):
            toggles.show_ids = not toggles.show_ids
        elif key == ord("t"):
            toggles.show_trails = not toggles.show_trails
        elif key == ord("f"):
            toggles.show_fps = not toggles.show_fps

        _ = time.perf_counter() - start

    video_loader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
