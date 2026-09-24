from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from models import SegmentationModel
from data import SegmentationDataset, collate_fn
from losses import SegmentationLoss as CompositeSegmentationLoss


class BaseValidator:
    """
    Base validator class for segmentation models.
    
    Handles validation loop, metrics computation, and visualization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_root: str,
        img_size: tuple = (1024, 1024),
        batch_size: int = 1,
        device: str = "cuda",
        num_workers: int = 2,
        save_dir: Optional[str] = None,
        dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.batch_size = batch_size
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
        """Alias for dataloader."""
        return self.dataloader

    @val_loader.setter
    def val_loader(self, loader: Optional[DataLoader]):
        """Setter to allow assigning val_loader seamlessly."""
        self.dataloader = loader
    
    def setup_data(self, split: str = "val"):
        """Setup validation data loader if not already provided."""
        self.dataset = SegmentationDataset(
            data_root=self.data_root,
            split=split,
            img_size=self.img_size,
            augment=False,
            use_cache=True,
            auto=True,  # Use rectangular inference for efficiency
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

    @staticmethod
    def _ensure_4d_tensor(x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor has exactly 4 dimensions (B, C, H, W)."""
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        if self.dataloader is None:
            self.setup_data(split="val")
            
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.dataloader)
        
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.dataloader, desc="Validating", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            images = self._ensure_4d_tensor(images)
            valid_masks = self._ensure_4d_tensor(valid_masks)
            masks = self._ensure_4d_tensor(masks)
            
            # Forward pass
            preds = self.model(images)
            
            # Compute loss
            loss, _ = self.criterion(preds, masks, valid_masks, 0)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            probs = torch.sigmoid(preds)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / max(n_batches, 1)
        self.metrics = self._compute_metrics(all_preds, all_targets)
        self.metrics["loss"] = avg_loss
        
        return self.metrics
    
    def _compute_metrics(self, preds: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
        """Compute segmentation metrics."""
        if not preds or not targets:
            return {"iou": 0.0, "dice": 0.0, "accuracy": 0.0}

        all_preds = np.concatenate(preds, axis=0)
        all_targets = np.concatenate(targets, axis=0)
        
        binary_preds = (all_preds > 0.5).astype(np.float32)
        
        intersection = (binary_preds * all_targets).sum()
        union = binary_preds.sum() + all_targets.sum() - intersection
        iou = intersection / (union + 1e-8)
        
        dice = (2.0 * intersection) / (binary_preds.sum() + all_targets.sum() + 1e-8)
        accuracy = (binary_preds == all_targets).mean()
        
        return {
            "iou": float(iou),
            "dice": float(dice),
            "accuracy": float(accuracy),
        }
    
    def print_results(self):
        """Print validation results."""
        print("\nValidation Results:")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    def save_visualizations(self, num_samples: int = 4):
        """Save visualization of predictions."""
        if self.save_dir is None or self.dataloader is None:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        samples_saved = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                if samples_saved >= num_samples:
                    break
                
                images = batch["image"].to(self.device, non_blocking=True)
                valid_masks = batch["valid_mask"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                images = self._ensure_4d_tensor(images)
                valid_masks = self._ensure_4d_tensor(valid_masks)
                masks = self._ensure_4d_tensor(masks)
                
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
        """Save a single sample visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Image
        if img.ndim == 3:
            axes[0].imshow(img[0], cmap='gray')
        else:
            axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Image")
        axes[0].axis('off')
        
        # Ground truth
        if mask.ndim == 3:
            axes[1].imshow(mask[0], cmap='gray')
        else:
            axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Prediction
        if pred.ndim == 3:
            axes[2].imshow(pred[0], cmap='gray')
        else:
            axes[2].imshow(pred, cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        # Valid mask
        if valid.ndim == 3:
            axes[3].imshow(valid[0], cmap='gray')
        else:
            axes[3].imshow(valid, cmap='gray')
        axes[3].set_title("Valid Mask")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"val_sample_{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()