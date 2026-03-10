"""Evaluation utilities for detection (mAP), tracking (MOTA/MOTP), and runtime FPS."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
from ultralytics import YOLO


def eval_map(model_path: str, data_yaml: str) -> None:
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


def eval_fps(model_path: str, video_path: str, max_frames: int = 300) -> None:
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frames = 0
    start = time.perf_counter()

    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        _ = model.predict(frame, verbose=False)
        frames += 1

    elapsed = time.perf_counter() - start
    fps = frames / max(elapsed, 1e-6)
    print(f"Average FPS ({frames} frames): {fps:.2f}")
    cap.release()


def eval_tracking_motp_mota(gt_csv: str, pred_csv: str) -> None:
    """CSV columns: frame,id,x1,y1,x2,y2."""
    acc = mm.MOTAccumulator(auto_id=True)

    gt = np.loadtxt(gt_csv, delimiter=",", skiprows=1)
    pred = np.loadtxt(pred_csv, delimiter=",", skiprows=1)

    frame_ids = sorted(set(gt[:, 0].astype(int)).union(set(pred[:, 0].astype(int))))
    for f in frame_ids:
        gt_f = gt[gt[:, 0] == f]
        pd_f = pred[pred[:, 0] == f]

        gt_ids = gt_f[:, 1].astype(int).tolist() if len(gt_f) else []
        pd_ids = pd_f[:, 1].astype(int).tolist() if len(pd_f) else []

        gt_boxes = gt_f[:, 2:6] if len(gt_f) else np.empty((0, 4))
        pd_boxes = pd_f[:, 2:6] if len(pd_f) else np.empty((0, 4))

        distances = mm.distances.iou_matrix(gt_boxes, pd_boxes, max_iou=0.5)
        acc.update(gt_ids, pd_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "num_switches"], name="seq")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="config/aerial_data.yaml")
    parser.add_argument("--video", default="")
    parser.add_argument("--gt", default="")
    parser.add_argument("--pred", default="")
    args = parser.parse_args()

    eval_map(args.model, args.data)
    if args.video and Path(args.video).exists():
        eval_fps(args.model, args.video)
    if args.gt and args.pred and Path(args.gt).exists() and Path(args.pred).exists():
        eval_tracking_motp_mota(args.gt, args.pred)


if __name__ == "__main__":
    main()
