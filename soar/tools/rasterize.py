from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
import cv2
import numpy as np
from tqdm import tqdm


def rasterize_coco_image(
    task: Tuple[str, int, int, List[List[float]], str]
) -> Tuple[str, bool]:
    """
    Worker task: rasterize polygons for a single image and save as uint8 PNG.
    task: (file_name, width, height, polygon_list, output_path)
    """
    file_name, width, height, polygons, out_path = task
    mask = np.zeros((height, width), dtype=np.uint8)

    has_foreground = False
    for poly in polygons:
        if len(poly) >= 6:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], color=255)
            has_foreground = True

    # Fast PNG compression level 1 for maximum I/O throughput
    cv2.imwrite(out_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return file_name, has_foreground


def rasterize_coco_dataset(
    annotation_file: str | Path,
    output_dir: str | Path,
    image_dir: Optional[str | Path] = None,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Parse COCO JSON annotations and pre-rasterize binary mask PNGs in parallel.
    """
    ann_path = Path(annotation_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO annotations from {ann_path}...")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image map: id -> (file_name, width, height)
    img_map: Dict[int, Dict[str, Any]] = {}
    for img in coco.get("images", []):
        img_id = img["id"]
        img_map[img_id] = {
            "file_name": img["file_name"],
            "width": img.get("width"),
            "height": img.get("height"),
            "polygons": [],
        }

    # Group polygons by image_id
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in img_map:
            continue
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    img_map[img_id]["polygons"].append(poly)

    # Check for missing width/height from JSON; if missing, inspect image file
    tasks = []
    for img_id, info in img_map.items():
        fname = info["file_name"]
        w = info["width"]
        h = info["height"]
        stem = Path(fname).stem
        out_mask_path = str(out_dir / f"{stem}.png")

        if w is None or h is None:
            if image_dir:
                img_file = Path(image_dir) / fname
                if not img_file.exists():
                    img_file = Path(image_dir) / f"{stem}.png"
                if img_file.exists():
                    im = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
                    if im is not None:
                        h, w = im.shape[:2]
            if w is None or h is None:
                continue

        tasks.append((fname, int(w), int(h), info["polygons"], out_mask_path))

    print(f"Rasterizing {len(tasks)} masks to {out_dir} using {num_workers} workers...")
    n_pos = 0
    if num_workers > 1 and len(tasks) > 10:
        with mp.Pool(processes=num_workers) as pool:
            for _, has_fg in tqdm(pool.imap_unordered(rasterize_coco_image, tasks, chunksize=16), total=len(tasks), desc="Rasterizing"):
                if has_fg:
                    n_pos += 1
    else:
        for t in tqdm(tasks, desc="Rasterizing"):
            _, has_fg = rasterize_coco_image(t)
            if has_fg:
                n_pos += 1

    stats = {
        "total_masks": len(tasks),
        "positive_masks": n_pos,
        "negative_masks": len(tasks) - n_pos,
        "output_dir": str(out_dir),
    }
    print(f"Rasterization complete! Total: {len(tasks)}, Positive: {n_pos}, Negative: {len(tasks) - n_pos}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="SOAR Offline Mask Pre-Rasterizer")
    parser.add_argument("--annotation-file", type=str, required=True, help="Path to COCO JSON annotations")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save pre-rasterized PNG masks")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of source images (if width/height missing from JSON)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    args = parser.parse_args()

    rasterize_coco_dataset(
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
