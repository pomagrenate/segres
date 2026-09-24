# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import Compose, get_training_augmentation, get_validation_augmentation
from .preprocess import ComposePreprocess, get_training_preprocessor, get_validation_preprocessor

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SegmentationDataset(Dataset):
    """
    General-purpose segmentation dataset.
    
    Supports various image formats and annotation formats (COCO JSON, YOLO, masks).
    Designed to be flexible for any segmentation task, not domain-specific.
    """
    
    SUPPORTED_EXTENSIONS = ('.npy', '.fits', '.fit', '.jpeg', '.jpg', '.png', '.bmp', '.tiff')
    
    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        img_size: Tuple[int, int] = (1024, 1024),
        augment: bool = False,
        use_cache: bool = True,
        cache_limit: int = 8,
        annotation_file: Optional[str] = None,
        mask_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.img_size = img_size
        self.augment = augment and (self.split == 'train')
        self.use_cache = use_cache
        self.cache_limit = max(0, cache_limit)
        self.transform = transform
        
        # Setup preprocessing and augmentation
        if self.augment:
            self.augmentation = get_training_augmentation()
        else:
            self.augmentation = get_validation_augmentation()
        
        self.preprocessor = get_training_preprocessor(img_size) if self.split == 'train' else get_validation_preprocessor(img_size)
        
        # Resolve paths
        self.image_dir = self._resolve_image_dir()
        self.image_files = self._collect_image_files()
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in {self.image_dir}")
        
        # Load annotations
        self.annotations: Dict[str, Any] = {}
        self.img_to_masks: Dict[str, str] = {}
        self._load_annotations(annotation_file, mask_dir)
        
        # Cache
        self._cache_store: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    
    def _resolve_image_dir(self) -> Path:
        """Resolve image directory path."""
        sub = "train" if self.split == "train" else "test"
        candidate_paths = [
            self.data_root / sub / f"{sub}_images",
            self.data_root / f"{sub}_images",
            self.data_root / sub,
            self.data_root / "images",
            self.data_root,
        ]
        for path in candidate_paths:
            if path.is_dir():
                return path
        raise FileNotFoundError(f"Could not locate image directory for split '{self.split}'. Checked: {candidate_paths}")
    
    def _collect_image_files(self) -> List[Path]:
        """Collect all supported image files."""
        files: List[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.image_dir.glob(f"*{ext}"))
            files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted(files)
    
    def _load_annotations(self, annotation_file: Optional[str], mask_dir: Optional[str]) -> None:
        """Load annotations from COCO JSON, YOLO, or mask directory."""
        # Try COCO JSON format
        if annotation_file:
            ann_path = Path(annotation_file)
            if not ann_path.is_absolute():
                ann_path = self.data_root / annotation_file
            if ann_path.exists():
                self._load_coco_annotations(ann_path)
                return
        
        # Try YOLO format
        yolo_labels_dir = self._resolve_yolo_labels_dir()
        if yolo_labels_dir and yolo_labels_dir.is_dir():
            self._load_yolo_annotations(yolo_labels_dir)
            return
        
        # Try mask directory format
        if mask_dir:
            mask_path = Path(mask_dir)
            if not mask_path.is_absolute():
                mask_path = self.data_root / mask_dir
            if mask_path.is_dir():
                self._load_mask_directory(mask_path)
                return
        
        # Try default locations
        candidate_files = [
            self.data_root / "train" / "annotations.json",
            self.data_root / "annotations.json",
            self.data_root / "train.json",
        ]
        for candidate in candidate_files:
            if candidate.exists():
                self._load_coco_annotations(candidate)
                return
        
        candidate_dirs = [
            self.data_root / "train" / "masks",
            self.data_root / "masks",
            self.data_root / f"{self.split}_masks",
        ]
        for candidate in candidate_dirs:
            if candidate.is_dir():
                self._load_mask_directory(candidate)
                return
    
    def _load_coco_annotations(self, ann_path: Path) -> None:
        """Load annotations from COCO JSON format."""
        with open(ann_path, "r", encoding="utf-8") as f:
            coco_payload = json.load(f)
        
        id_to_filename: Dict[int, str] = {}
        for img_info in coco_payload.get("images", []):
            img_id = img_info["id"]
            fname = img_info["file_name"]
            id_to_filename[img_id] = fname
            self.img_to_masks[fname] = "coco"
        
        for ann in coco_payload.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in id_to_filename:
                continue
            fname = id_to_filename[img_id]
            if fname not in self.annotations:
                self.annotations[fname] = []
            self.annotations[fname].append(ann)
    
    def _resolve_yolo_labels_dir(self) -> Optional[Path]:
        """Resolve YOLO labels directory path."""
        sub = "train" if self.split == "train" else "test"
        candidate_paths = [
            self.data_root / sub / "labels",
            self.data_root / "labels",
            self.data_root / f"{sub}_labels",
            self.data_root / sub / f"{sub}_labels",
        ]
        for path in candidate_paths:
            if path.is_dir():
                return path
        return None
    
    def _load_yolo_annotations(self, labels_dir: Path) -> None:
        """Load annotations from YOLO format (segmentation or detection)."""
        for label_file in labels_dir.glob("*.txt"):
            img_name = label_file.stem
            self.img_to_masks[img_name] = "yolo"
            self.annotations[img_name] = str(label_file)
    
    def _parse_yolo_annotation(self, label_file: Path, img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """Parse YOLO annotation file."""
        annotations = []
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                class_id = int(parts[0])
                
                # Check if segmentation (polygon) or detection (bounding box)
                if len(parts) > 5:
                    # Segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                    coords = list(map(float, parts[1:]))
                    # Convert normalized coordinates to absolute
                    polygon = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * img_width)
                        y = int(coords[i + 1] * img_height)
                        polygon.extend([x, y])
                    
                    annotations.append({
                        'class_id': class_id,
                        'type': 'segmentation',
                        'segmentation': [polygon]
                    })
                else:
                    # Detection format: class_id x_center y_center width height
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert to absolute coordinates
                    x_center_abs = int(x_center * img_width)
                    y_center_abs = int(y_center * img_height)
                    width_abs = int(width * img_width)
                    height_abs = int(height * img_height)
                    
                    # Convert to polygon (bounding box as 4 points)
                    x1 = x_center_abs - width_abs // 2
                    y1 = y_center_abs - height_abs // 2
                    x2 = x_center_abs + width_abs // 2
                    y2 = y_center_abs + height_abs // 2
                    
                    polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
                    
                    annotations.append({
                        'class_id': class_id,
                        'type': 'detection',
                        'segmentation': [polygon]
                    })
        
        return annotations
    
    def _load_mask_directory(self, mask_path: Path) -> None:
        """Load annotations from mask directory (one mask per image)."""
        for mask_file in mask_path.glob("*"):
            if mask_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
                img_name = mask_file.stem
                self.img_to_masks[img_name] = str(mask_file)
    
    def _read_image(self, path: Path) -> np.ndarray:
        """Read image from file."""
        ext = path.suffix.lower()
        
        if ext == '.npy':
            arr = np.load(path)
            return np.asarray(arr, dtype=np.float32)
        
        if ext in ('.fits', '.fit'):
            try:
                from astropy.io import fits
                with fits.open(path) as hdul:
                    arr = hdul[0].data.astype(np.float32)
            except ImportError:
                raise ImportError("astropy library is required to read FITS files.")
        else:
            with Image.open(path) as img:
                arr = np.array(img.convert('L'), dtype=np.float32)
        
        # Normalize to [0, 1] if needed
        max_val = arr.max()
        if max_val > 1.0:
            arr /= 255.0 if max_val <= 255.0 else max_val
        
        return arr
    
    def _read_mask(self, img_path: Path) -> Optional[np.ndarray]:
        """Read mask for image."""
        img_name = img_path.name
        
        # Check if mask file exists
        if img_name in self.img_to_masks:
            mask_info = self.img_to_masks[img_name]
            
            if mask_info == "coco":
                # Generate mask from COCO polygons
                return self._generate_coco_mask(img_name)
            elif mask_info == "yolo":
                # Generate mask from YOLO annotations
                return self._generate_yolo_mask(img_name)
            else:
                # Read mask from file
                mask_path = Path(mask_info)
                if mask_path.exists():
                    with Image.open(mask_path) as mask_img:
                        mask = np.array(mask_img.convert('L'), dtype=np.float32)
                        if mask.max() > 1.0:
                            mask /= 255.0
                        return mask
        
        return None
    
    def _generate_coco_mask(self, img_name: str) -> Optional[np.ndarray]:
        """Generate mask from COCO polygon annotations."""
        if img_name not in self.annotations:
            return None
        
        # Get image dimensions
        img_path = next((f for f in self.image_files if f.name == img_name), None)
        if img_path is None:
            return None
        
        img = self._read_image(img_path)
        h, w = img.shape[-2:]
        
        mask = np.zeros((h, w), dtype=np.float32)
        
        for ann in self.annotations[img_name]:
            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                for poly in segmentation:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask, [pts], color=1)
        
        return mask
    
    def _generate_yolo_mask(self, img_name: str) -> Optional[np.ndarray]:
        """Generate mask from YOLO annotations."""
        if img_name not in self.annotations:
            return None
        
        # Get image dimensions
        img_path = next((f for f in self.image_files if f.name == img_name), None)
        if img_path is None:
            return None
        
        img = self._read_image(img_path)
        h, w = img.shape[-2:]
        
        # Parse YOLO annotation
        label_file = Path(self.annotations[img_name])
        annotations = self._parse_yolo_annotation(label_file, w, h)
        
        mask = np.zeros((h, w), dtype=np.float32)
        
        for ann in annotations:
            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                for poly in segmentation:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask, [pts], color=1)
        
        return mask
    
    def _get_processed_data(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Get preprocessed image and mask with caching."""
        key = str(path)
        if key in self._cache_store:
            return self._cache_store[key]
        
        raw_img = self._read_image(path)
        
        # Handle different dimensions
        if raw_img.ndim == 3 and raw_img.shape[0] == 1:
            raw_img = raw_img[0]
        
        # Preprocess
        processed_img, valid_mask = self.preprocessor(raw_img)
        
        # Add channel dimension if needed
        if processed_img.ndim == 2:
            processed_img = processed_img[np.newaxis, ...]
        if valid_mask is not None and valid_mask.ndim == 2:
            valid_mask = valid_mask[np.newaxis, ...]
        
        res = (processed_img, valid_mask)
        if len(self._cache_store) < self.cache_limit:
            self._cache_store[key] = res
        
        return res
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        img, valid_mask = self._get_processed_data(img_path)
        
        sample = {
            'image': torch.from_numpy(np.ascontiguousarray(img)).float(),
            'valid_mask': torch.from_numpy(np.ascontiguousarray(valid_mask)).float() if valid_mask is not None else torch.ones_like(img),
            'image_id': img_path.stem,
        }
        
        # Load mask for training
        if self.split == 'train':
            mask = self._read_mask(img_path)
            if mask is None:
                mask = np.zeros(img.shape[-2:], dtype=np.float32)
            
            # Apply augmentation
            if self.augment:
                img_np = img.transpose(1, 2, 0) if img.ndim == 3 else img
                img_np, mask = self.augmentation(img_np, mask)
                img = img_np.transpose(2, 0, 1) if img_np.ndim == 3 else img_np
            
            sample['mask'] = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
            sample['has_object'] = bool(mask.sum() > 0)
        else:
            sample['mask'] = None
            sample['has_object'] = False
        
        # Apply custom transform if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    if not batch:
        return {}
    
    has_mask = batch[0].get('mask') is not None
    
    if has_mask:
        return {
            'image': torch.stack([b['image'] for b in batch], dim=0),
            'valid_mask': torch.stack([b['valid_mask'] for b in batch], dim=0),
            'mask': torch.stack([b['mask'] for b in batch], dim=0),
            'has_object': torch.tensor([b['has_object'] for b in batch], dtype=torch.bool),
            'image_id': [b['image_id'] for b in batch],
        }
    
    return {
        'image': torch.stack([b['image'] for b in batch], dim=0),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch], dim=0),
        'image_id': [b['image_id'] for b in batch],
    }
