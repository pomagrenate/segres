from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def binary_mask_to_rle(mask: np.ndarray) -> str:
    """Convert binary mask to run-length encoding (RLE) format."""
    mask_flat = np.asfortranarray(mask.astype(bool)).flatten()
    if not np.any(mask_flat):
        return ""

    pixels = np.empty(mask_flat.size + 2, dtype=np.int8)
    pixels[0] = 0
    pixels[1:-1] = mask_flat
    pixels[-1] = 0

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_to_binary_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """Convert RLE string to binary mask."""
    if not rle_str or pd.isna(rle_str):
        return np.zeros((height, width), dtype=np.uint8)

    runs = np.array(rle_str.split(), dtype=int)
    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths

    mask_flat = np.zeros(height * width, dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask_flat[s:e] = 1

    return mask_flat.reshape((width, height), order="F").T


def create_submission_csv(
    predictions: Dict[str, List[str]],
    output_path: str,
    id_prefix: str = "image",
) -> None:
    """Create submission CSV from predictions in RLE format."""
    rows = []
    for image_id, rle_list in predictions.items():
        if not rle_list:
            rows.append({f"{id_prefix}_id": f"{image_id}_none", "segmentation_rle": ""})
            continue
        for i, rle_str in enumerate(rle_list):
            rows.append({
                f"{id_prefix}_id": f"{image_id}_{i + 1}",
                "segmentation_rle": rle_str,
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
