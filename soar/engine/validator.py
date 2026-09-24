from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from ..models import SegmentationModel
from ..data import SegmentationDataset, collate_fn
from ..losses import SegmentationLoss as CompositeSegmentationLoss


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
        total_loss = 0.0
        n_batches = 0

        # Streaming metric accumulators (zero host-RAM accumulation)
        total_inter = 0.0
        total_union = 0.0
        total_cardinality = 0.0
        total_correct = 0.0
        total_pixels = 0.0

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
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(preds).float()
            bin_preds = (probs >= 0.5).float()
            gt = masks.float()

            if valid_masks is not None:
                vmask = valid_masks.float()
                bin_preds = bin_preds * vmask
                gt = gt * vmask
                active_pixels = vmask.sum().item()
            else:
                active_pixels = float(gt.numel())

            inter = (bin_preds * gt).sum().item()
            union = (bin_preds + gt).clamp_max(1.0).sum().item()
            card = bin_preds.sum().item() + gt.sum().item()
            correct = (bin_preds == gt).float()
            if valid_masks is not None:
                correct = correct * valid_masks.float()
            correct_val = correct.sum().item()

            total_inter += inter
            total_union += union
            total_cardinality += card
            total_correct += correct_val
            total_pixels += active_pixels

        # Synchronize metrics across distributed ranks
        if dist.is_initialized():
            sync_tensor = torch.tensor(
                [total_loss, n_batches, total_inter, total_union, total_cardinality, total_correct, total_pixels],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            vals = sync_tensor.cpu().tolist()
            total_loss, n_batches, total_inter, total_union, total_cardinality, total_correct, total_pixels = vals

        iou = total_inter / max(total_union, 1e-7)
        dice = (2.0 * total_inter) / max(total_cardinality, 1e-7)
        acc = total_correct / max(total_pixels, 1.0)
        avg_loss = total_loss / max(n_batches, 1)

        self.metrics = {
            "loss": float(avg_loss),
            "iou": float(iou),
            "dice": float(dice),
            "accuracy": float(acc),
        }
        return self.metrics

    def print_results(self, epoch: Optional[int] = None):
        """Print validation metrics summary in clean tabular format."""
        ep_str = f"Epoch {epoch}" if epoch is not None else "Summary"
        print(f"\n{'-'*75}")
        print(f"{'Stage / Metric':<20} {'Samples':<10} {'Loss':<12} {'IoU':<12} {'Dice':<12} {'Acc':<10}")
        print(f"{'-'*75}")

        num_images = len(self.dataloader.dataset) if self.dataloader is not None else 0
        loss_val = self.metrics.get("loss", 0.0)
        iou_val = self.metrics.get("iou", 0.0)
        dice_val = self.metrics.get("dice", 0.0)
        acc_val = self.metrics.get("accuracy", 0.0)

        print(f"{ep_str:<20} {num_images:<10} {loss_val:<12.4f} {iou_val:<12.4f} {dice_val:<12.4f} {acc_val:<10.4f}")
        print(f"{'-'*75}\n")

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