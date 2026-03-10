"""Convert COCO JSON annotations to YOLO txt labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-out", required=True)
    args = parser.parse_args()

    labels_dir = Path(args.labels_out)
    labels_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.coco_json).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}

    grouped = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    for image_id, anns in grouped.items():
        img = images[image_id]
        w, h = img["width"], img["height"]
        stem = Path(img["file_name"]).stem
        out_path = labels_dir / f"{stem}.txt"

        rows = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cls = int(ann["category_id"]) - 1
            rows.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out_path.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
