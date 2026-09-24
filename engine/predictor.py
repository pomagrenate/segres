# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from models import SegmentationModel
from data import SegmentationDataset, collate_fn


class BasePredictor:
    """
    Base predictor class for segmentation models.
    
    Handles inference, postprocessing, and result export.
    """
    
    def __init__(
        self,
        model: SegmentationModel,
        data_root: str,
        img_size: tuple = (1024, 1024),
        batch_size: int = 1,
        device: str = "cuda",
        num_workers: int = 2,
        threshold: float = 0.5,
        min_area: int = 30,
        close_kernel: int = 3,
    ):
        self.model = model
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.num_workers = num_workers
        self.threshold = threshold
        self.min_area = min_area
        self.close_kernel = close_kernel
        
        self.model.to(self.device)
        self.model.eval()
    
    def setup_data(self, split: str = "test"):
        """Setup inference data loader."""
        self.dataset = SegmentationDataset(
            data_root=self.data_root,
            split=split,
            img_size=self.img_size,
            augment=False,
            use_cache=True,
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
    
    @torch.no_grad()
    def predict(self) -> Dict[str, np.ndarray]:
        """Run inference and return predictions."""
        self.model.eval()
        predictions = {}
        
        for batch in self.dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(self.device, non_blocking=True)
            image_ids = batch["image_id"]
            
            # Forward pass
            preds = self.model(images)
            probs = torch.sigmoid(preds)
            
            # Process each image in batch
            for i, image_id in enumerate(image_ids):
                prob = probs[i].cpu().numpy()
                valid = valid_masks[i].cpu().numpy()
                
                # Apply valid mask
                if valid.ndim == 3:
                    prob = prob * valid[0]
                else:
                    prob = prob * valid
                
                # Postprocess
                binary = self.postprocess_mask(prob)
                predictions[image_id] = binary
        
        return predictions
    
    def postprocess_mask(self, prob_map: np.ndarray) -> np.ndarray:
        """Postprocess probability map to binary mask."""
        if prob_map.ndim == 3:
            prob_map = prob_map[0]
        
        binary = (prob_map >= self.threshold).astype(np.uint8)
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        valid_labels = np.where((areas >= self.min_area) & (np.arange(n_labels) > 0))[0]
        
        return np.isin(labels, valid_labels).astype(np.uint8)
    
    def extract_components(self, binary: np.ndarray) -> List[np.ndarray]:
        """Extract individual components from binary mask."""
        if binary.ndim == 3:
            binary = binary[0]
        
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components = []
        
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= self.min_area:
                component = (labels == lbl).astype(np.uint8)
                components.append(component)
        
        return components
    
    def save_predictions(self, predictions: Dict[str, np.ndarray], output_dir: str):
        """Save predictions as images."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for image_id, mask in predictions.items():
            if mask.ndim == 3:
                mask = mask[0]
            cv2.imwrite(str(output_path / f"{image_id}.png"), (mask * 255).astype(np.uint8))
    
    def save_predictions_rle(self, predictions: Dict[str, np.ndarray], output_file: str):
        """Save predictions as RLE format (for competition submission)."""
        from utils import binary_mask_to_rle
        
        rows = []
        for image_id, binary in predictions.items():
            components = self.extract_components(binary)
            if not components:
                rows.append({"image_id": image_id, "rle": ""})
                continue
            
            for i, component in enumerate(components):
                rle = binary_mask_to_rle(component)
                rows.append({"image_id": f"{image_id}_{i}", "rle": rle})
        
        import pandas as pd
        df = pd.DataFrame(rows)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Predict on a list of images."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for img in images:
                # Preprocess
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
                
                # Forward pass
                pred = self.model(img_tensor)
                prob = torch.sigmoid(pred).cpu().numpy()[0]
                
                # Postprocess
                binary = self.postprocess_mask(prob)
                predictions.append(binary)
        
        return predictions
