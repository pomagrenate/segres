from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from ..models import SegmentationModel
from ..data import SegmentationDataset, collate_fn
from ..losses import SegmentationLoss as CompositeSegmentationLoss
from ..losses.structure import soft_skeletonize


def _extract_boundary(mask: torch.Tensor, d: int = 2) -> torch.Tensor:
    """Extract morphological contour of binary mask using min-pooling erosion."""
    kernel_size = 2 * d + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=d)
    return F.relu(mask - eroded)


class BaseValidator:
    """
    Validation engine for high-resolution segmentation models.
    Implements O(1) host memory streaming metric accumulation and multi-rank distributed reduction.
    """

    def __init__(
        self,
        model: nn.Module,
        data_root: str,
        img_size: tuple = (1024, 1024),
        device: str = "cuda",
        num_workers: int = 2,
        save_dir: Optional[str] = None,
        dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.batch_size = 1
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.num_workers = num_workers
        self.save_dir = Path(save_dir) if save_dir else None
        self.dataloader = dataloader

        self.model.to(self.device)
        self.model.eval()

        self.criterion = CompositeSegmentationLoss()
        self.metrics: Dict[str, float] = {}

    @property
    def val_loader(self) -> Optional[DataLoader]:
        return self.dataloader

    @val_loader.setter
    def val_loader(self, loader: Optional[DataLoader]):
        self.dataloader = loader

    def setup_data(self, split: str = "val"):
        """Setup validation data loader if not externally provided."""
        self.dataset = SegmentationDataset(
            data_root=self.data_root,
            split=split,
            img_size=self.img_size,
            augment=False,
            use_cache=True,
            auto=True,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

    @staticmethod
    def _ensure_4d_tensor(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation with streaming O(1) memory metric computation."""
        if self.dataloader is None:
            self.setup_data(split="val")

        self.model.eval()
        # Streaming metric accumulator on-device (12 terms):
        # [0: loss, 1: n_batches, 2: inter, 3: union, 4: pred_card, 5: gt_card,
        #  6: b_inter, 7: b_union, 8: skel_prec_inter, 9: skel_prec_total,
        #  10: skel_sens_inter, 11: skel_sens_total]
        accum = torch.zeros(12, dtype=torch.float64, device=self.device)

        is_rank_zero = (not dist.is_initialized()) or dist.get_rank() == 0
        pbar = (
            tqdm(self.dataloader, desc="Validating", leave=False, bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}")
            if is_rank_zero
            else self.dataloader
        )

        for batch in pbar:
            images = self._ensure_4d_tensor(batch["image"].to(self.device, non_blocking=True))
            valid_masks = self._ensure_4d_tensor(batch["valid_mask"].to(self.device, non_blocking=True))
            masks = self._ensure_4d_tensor(batch["mask"].to(self.device, non_blocking=True))

            preds = self.model(images)
            loss, _ = self.criterion(preds, masks, valid_masks, 0)

            probs = torch.sigmoid(preds)
            bin_preds = (probs >= 0.5).to(dtype=torch.float32)
            gt = masks.to(dtype=torch.float32)

            if valid_masks is not None:
                vmask = valid_masks.to(dtype=torch.float32)
                bin_preds = bin_preds * vmask
                gt = gt * vmask

            inter = (bin_preds * gt).sum()
            union = (bin_preds + gt).clamp_max(1.0).sum()
            pred_card = bin_preds.sum()
            gt_card = gt.sum()

            # Boundary IoU
            b_pred = _extract_boundary(bin_preds, d=2)
            b_gt = _extract_boundary(gt, d=2)
            b_inter = (b_pred * b_gt).sum()
            b_union = (b_pred + b_gt).clamp_max(1.0).sum()

            # clDice (Topological skeleton precision & sensitivity)
            skel_pred = soft_skeletonize(bin_preds, n_iter=2)
            skel_true = soft_skeletonize(gt, n_iter=2)
            skel_prec_inter = (skel_pred * gt).sum()
            skel_prec_total = skel_pred.sum()
            skel_sens_inter = (skel_true * bin_preds).sum()
            skel_sens_total = skel_true.sum()

            accum[0] += loss.detach()
            accum[1] += 1.0
            accum[2] += inter
            accum[3] += union
            accum[4] += pred_card
            accum[5] += gt_card
            accum[6] += b_inter
            accum[7] += b_union
            accum[8] += skel_prec_inter
            accum[9] += skel_prec_total
            accum[10] += skel_sens_inter
            accum[11] += skel_sens_total

        # Synchronize metrics across distributed ranks
        if dist.is_initialized():
            dist.all_reduce(accum, op=dist.ReduceOp.SUM)

        vals = accum.cpu().tolist()
        (
            total_loss,
            n_batches,
            total_inter,
            total_union,
            total_pred,
            total_gt,
            total_b_inter,
            total_b_union,
            total_skel_prec_inter,
            total_skel_prec_total,
            total_skel_sens_inter,
            total_skel_sens_total,
        ) = vals

        iou = total_inter / max(total_union, 1e-7)
        dice = (2.0 * total_inter) / max(total_pred + total_gt, 1e-7)
        prec = total_inter / max(total_pred, 1e-7)
        recall = total_inter / max(total_gt, 1e-7)
        boundary_iou = (
            1.0 if total_b_union == 0 else (total_b_inter / max(total_b_union, 1e-7))
        )

        t_prec = (total_skel_prec_inter + 1e-7) / (total_skel_prec_total + 1e-7)
        t_sens = (total_skel_sens_inter + 1e-7) / (total_skel_sens_total + 1e-7)
        cldice = (
            1.0
            if (total_skel_prec_total == 0 and total_skel_sens_total == 0)
            else (2.0 * t_prec * t_sens) / (t_prec + t_sens + 1e-7)
        )
        avg_loss = total_loss / max(n_batches, 1)

        self.metrics = {
            "loss": float(avg_loss),
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(prec),
            "recall": float(recall),
            "boundary_iou": float(boundary_iou),
            "cldice": float(cldice),
        }
        return self.metrics

    def print_results(self, epoch: Optional[int] = None):
        """Print validation metrics summary in clean tabular format."""
        ep_str = f"Epoch {epoch}" if epoch is not None else "Summary"
        print(f"\n{'-'*95}")
        print(
            f"{'Stage / Metric':<16} {'Samples':<8} {'Loss':<10} {'IoU':<10} {'Dice':<10} {'Prec':<10} {'Recall':<10} {'bIoU':<10} {'clDice':<10}"
        )
        print(f"{'-'*95}")

        num_images = len(self.dataloader.dataset) if self.dataloader is not None else 0
        loss_val = self.metrics.get("loss", 0.0)
        iou_val = self.metrics.get("iou", 0.0)
        dice_val = self.metrics.get("dice", 0.0)
        prec_val = self.metrics.get("precision", 0.0)
        recall_val = self.metrics.get("recall", 0.0)
        biou_val = self.metrics.get("boundary_iou", 0.0)
        cldice_val = self.metrics.get("cldice", 0.0)

        print(
            f"{ep_str:<16} {num_images:<8} {loss_val:<10.4f} {iou_val:<10.4f} {dice_val:<10.4f} {prec_val:<10.4f} {recall_val:<10.4f} {biou_val:<10.4f} {cldice_val:<10.4f}"
        )
        print(f"{'-'*95}\n")

    def save_visualizations(self, num_samples: int = 4):
        """Save sample validation qualitative comparisons."""
        if self.save_dir is None or self.dataloader is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        samples_saved = 0

        with torch.no_grad():
            for batch in self.dataloader:
                if samples_saved >= num_samples:
                    break

                images = self._ensure_4d_tensor(batch["image"].to(self.device, non_blocking=True))
                valid_masks = self._ensure_4d_tensor(batch["valid_mask"].to(self.device, non_blocking=True))
                masks = self._ensure_4d_tensor(batch["mask"].to(self.device, non_blocking=True))

                preds = self.model(images)
                probs = torch.sigmoid(preds)

                for i in range(images.shape[0]):
                    if samples_saved >= num_samples:
                        break

                    img = images[i].cpu().numpy()
                    mask = masks[i].cpu().numpy()
                    valid = valid_masks[i].cpu().numpy()
                    pred = probs[i].cpu().numpy()

                    self._save_sample(img, mask, valid, pred, samples_saved)
                    samples_saved += 1

    def _save_sample(self, img: np.ndarray, mask: np.ndarray, valid: np.ndarray, pred: np.ndarray, idx: int):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img[0] if img.ndim == 3 else img, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(mask[0] if mask.ndim == 3 else mask, cmap="gray")
        axes[1].set_title(f"Ground Truth ({int(mask.sum())} px)")
        axes[1].axis("off")

        bin_pred = (pred[0] if pred.ndim == 3 else pred) >= 0.5
        axes[2].imshow(bin_pred, cmap="gray")
        axes[2].set_title(f"Prediction ({int(bin_pred.sum())} px)")
        axes[2].axis("off")

        axes[3].imshow(valid[0] if valid.ndim == 3 else valid, cmap="gray")
        axes[3].set_title("Valid Mask")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(self.save_dir / f"val_sample_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close()