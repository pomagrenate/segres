from __future__ import annotations

from typing import Dict, List
import numpy as np


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) for two binary masks."""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / (union + 1e-8))


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient for two binary masks."""
    inter = (mask1 * mask2).sum()
    return float((2.0 * inter) / (mask1.sum() + mask2.sum() + 1e-8))


def compute_panoptic_quality(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Panoptic Quality metric for instance segmentation."""
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "TP": 0, "FP": 0, "FN": 0}
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": len(pred_masks), "FN": len(gt_masks)}

    matched_gt = set()
    iou_sum = 0.0
    tp = 0

    for p in pred_masks:
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(gt_masks):
            if j in matched_gt:
                continue
            curr_iou = compute_iou(p, g)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_idx = j

        if best_iou >= iou_threshold and best_idx != -1:
            matched_gt.add(best_idx)
            iou_sum += best_iou
            tp += 1

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - len(matched_gt)

    sq = iou_sum / (tp + 1e-8) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    return {
        "PQ": sq * rq,
        "SQ": sq,
        "RQ": rq,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
