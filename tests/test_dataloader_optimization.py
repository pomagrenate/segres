from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from soar.tools.rasterize import rasterize_coco_dataset
from soar.data.preprocess import Normalize01, LetterBox, LetterBoxMask
from soar.data.dataset import SegmentationDataset


def test_offline_rasterization_and_dataset_loading():
    """Verify offline rasterization and fast pre-rasterized mask loading in SegmentationDataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_dir = tmp_path / "train" / "train_images"
        train_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = tmp_path / "masks"

        # Create 2 synthetic images
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(train_dir / "img1.png"), img1)
        cv2.imwrite(str(train_dir / "img2.png"), img2)

        # Synthetic COCO annotation: img1 has polygon, img2 has no polygon
        coco = {
            "images": [
                {"id": 1, "file_name": "img1.png", "width": 512, "height": 512},
                {"id": 2, "file_name": "img2.png", "width": 512, "height": 512},
            ],
            "annotations": [
                {
                    "id": 101,
                    "image_id": 1,
                    "segmentation": [[50, 50, 200, 50, 200, 200, 50, 200]],
                }
            ],
        }
        ann_file = tmp_path / "annotations.json"
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(coco, f)

        # 1. Test offline rasterize
        stats = rasterize_coco_dataset(
            annotation_file=ann_file,
            output_dir=masks_dir,
            num_workers=1,
        )
        assert stats["total_masks"] == 2
        assert stats["positive_masks"] == 1
        assert stats["negative_masks"] == 1
        assert (masks_dir / "img1.png").exists()
        assert (masks_dir / "img2.png").exists()

        # 2. Test dataset auto-detecting pre-rasterized masks
        ds = SegmentationDataset(
            data_root=tmp_path,
            split="train",
            img_size=(512, 512),
            in_channels=3,
            mask_dir=str(masks_dir),
        )
        assert len(ds) == 2
        sample1 = ds[0]
        sample2 = ds[1]

        assert sample1["image"].shape == (3, 512, 512)
        assert sample1["mask"].shape == (1, 512, 512)
        assert sample1["has_object"] is True
        assert sample2["has_object"] is False


def test_letterbox_bypass():
    """Verify LetterBox and LetterBoxMask fast path when input matches target size."""
    img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    lb = LetterBox(new_shape=(1024, 1024), auto=False)
    out_img, vmask, meta = lb(img)

    assert out_img.shape[:2] == (1024, 1024)
    assert meta["padding"] == (0, 0, 0, 0)
    assert meta["ratio"] == (1.0, 1.0)
    assert vmask.shape == (1024, 1024)
    assert float(vmask.min()) == 1.0


def test_normalize01_uint8_fastpath():
    """Verify Normalize01 calculates valid [0, 1] range for uint8 arrays using histogram."""
    img = np.random.randint(10, 240, (1024, 1024), dtype=np.uint8)
    norm = Normalize01(percentiles=(1.0, 99.0))
    res, _, meta = norm(img)

    assert res.dtype == np.float32
    assert res.shape == (1024, 1024)
    assert 0.0 <= float(res.min())
    assert float(res.max()) <= 1.0


if __name__ == "__main__":
    print("Running test_offline_rasterization_and_dataset_loading...")
    test_offline_rasterization_and_dataset_loading()
    print("Running test_letterbox_bypass...")
    test_letterbox_bypass()
    print("Running test_normalize01_uint8_fastpath...")
    test_normalize01_uint8_fastpath()
    print("All Phase 1 optimization unit tests passed successfully!")
