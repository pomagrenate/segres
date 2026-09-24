from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augment import (
    Compose,
    get_training_augmentation,
    get_validation_augmentation,
)
from .preprocess import (
    ComposePreprocess,
    build_preprocessor_from_config,
    get_mask_preprocessor,
    get_training_preprocessor,
    get_validation_preprocessor,
)
from .preprocess_config import PreprocessConfig

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SegmentationDataset(Dataset):
    """General-purpose dense segmentation dataset for high-resolution imagery.

    Supports diverse scientific and raster file formats (.npy, .fits, .png,
    etc.) and parses polygon/RLE annotations from standard COCO and YOLO-style
    structures.
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
        auto: bool = False,
        preprocess_config: Optional[PreprocessConfig] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.is_train = self.split in ('train', 'training')
        self.has_gt = self.split in ('train', 'training', 'val', 'valid', 'validation')
        self.img_size = img_size
        self.augment = augment and self.is_train
        self.use_cache = use_cache
        self.cache_limit = max(0, cache_limit)
        self.transform = transform
        self.auto = auto

        # Configure augmentation pipelines
        if self.augment:
            self.augmentation = get_training_augmentation()
        else:
            self.augmentation = get_validation_augmentation()

        # Build preprocessing pipelines
        if preprocess_config is not None:
            self.preprocessor = build_preprocessor_from_config(preprocess_config)
            if preprocess_config.geometric.letterbox and preprocess_config.geometric.target_size:
                self.mask_preprocessor = get_mask_preprocessor(
                    img_size=preprocess_config.geometric.target_size,
                    auto=preprocess_config.geometric.auto,
                    scaleup=preprocess_config.geometric.scaleup,
                )
            else:
                self.mask_preprocessor = None
        else:
            auto_mode = auto if not self.is_train else False
            self.preprocessor = (
                get_training_preprocessor(img_size, auto=auto_mode)
                if self.is_train
                else get_validation_preprocessor(img_size, auto=auto_mode)
            )
            self.mask_preprocessor = get_mask_preprocessor(
                img_size,
                auto=auto_mode,
                scaleup=self.is_train,
            )

        # File directory indexing
        self.image_dir = self._resolve_image_dir()
        self.image_files = self._collect_image_files()
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in {self.image_dir}")

        # Index label annotations
        self.annotations: Dict[str, Any] = {}
        self.img_to_masks: Dict[str, str] = {}
        self._load_annotations(annotation_file, mask_dir)

        # In-memory processing cache
        self._cache_store: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]] = {}

    def _resolve_image_dir(self) -> Path:
        """Locate root directory containing image targets."""
        sub = "train" if self.has_gt else "test"
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
        raise FileNotFoundError(
            f"Could not locate image directory for split '{self.split}'. Checked: {candidate_paths}"
        )

    def _collect_image_files(self) -> List[Path]:
        """Index all matching image files across supported extensions."""
        files: List[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.image_dir.glob(f"*{ext}"))
            files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted(files)

    def _load_annotations(self, annotation_file: Optional[str], mask_dir: Optional[str]) -> None:
        """Parse annotations across COCO format, YOLO labels, or mask bitmaps."""
        if annotation_file:
            ann_path = Path(annotation_file)
            if not ann_path.is_absolute():
                ann_path = self.data_root / annotation_file
            if ann_path.exists():
                self._load_coco_annotations(ann_path)
                return

        yolo_labels_dir = self._resolve_yolo_labels_dir()
        if yolo_labels_dir and yolo_labels_dir.is_dir():
            self._load_yolo_annotations(yolo_labels_dir)
            return

        if mask_dir:
            mask_path = Path(mask_dir)
            if not mask_path.is_absolute():
                mask_path = self.data_root / mask_dir
            if mask_path.is_dir():
                self._load_mask_directory(mask_path)
                return

        candidate_files = [
            self.data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
            self.data_root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
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
        """Parse COCO polygon structure."""
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
        """Locate YOLO label directory."""
        sub = "train" if self.has_gt else "test"
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
        """Index YOLO annotations."""
        for label_file in labels_dir.glob("*.txt"):
            img_name = label_file.stem
            self.img_to_masks[img_name] = "yolo"
            self.annotations[img_name] = str(label_file)

    def _parse_yolo_annotation(self, label_file: Path, img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """Parse raw coordinates from YOLO text labels."""
        annotations = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])

                if len(parts) > 5:
                    coords = list(map(float, parts[1:]))
                    polygon = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * img_width)
                        y = int(coords[i + 1] * img_height)
                        polygon.extend([x, y])

                    annotations.append({
                        "class_id": class_id,
                        "type": "segmentation",
                        "segmentation": [polygon],
                    })
                else:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x_center_abs = int(x_center * img_width)
                    y_center_abs = int(y_center * img_height)
                    width_abs = int(width * img_width)
                    height_abs = int(height * img_height)

                    x1 = x_center_abs - width_abs // 2
                    y1 = y_center_abs - height_abs // 2
                    x2 = x_center_abs + width_abs // 2
                    y2 = y_center_abs + height_abs // 2
                    polygon = [x1, y1, x2, y1, x2, y2, x1, y2]

                    annotations.append({
                        "class_id": class_id,
                        "type": "detection",
                        "segmentation": [polygon],
                    })

        return annotations

    def _load_mask_directory(self, mask_path: Path) -> None:
        """Index mask bitmaps stored directly in directories."""
        for mask_file in mask_path.glob("*"):
            if mask_file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
                img_name = mask_file.stem
                self.img_to_masks[img_name] = str(mask_file)

    def _read_image(self, path: Path) -> np.ndarray:
        """Load image arrays from disk across formats."""
        ext = path.suffix.lower()

        if ext == '.npy':
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float32)
            if not arr.flags['C_CONTIGUOUS'] or not arr.flags['F_CONTIGUOUS']:
                arr = arr.copy()
            return arr

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

        max_val = arr.max()
        if max_val > 1.0:
            arr /= 255.0 if max_val <= 255.0 else max_val

        return arr

    def _read_mask(self, img_path: Path) -> Optional[np.ndarray]:
        """Retrieve or construct the ground truth mask for a given sample."""
        img_name = img_path.name

        if img_name in self.img_to_masks:
            mask_info = self.img_to_masks[img_name]

            if mask_info == "coco":
                return self._generate_coco_mask(img_name)
            elif mask_info == "yolo":
                return self._generate_yolo_mask(img_name)
            else:
                mask_path = Path(mask_info)
                if mask_path.exists():
                    with Image.open(mask_path) as mask_img:
                        mask = np.array(mask_img.convert('L'), dtype=np.float32)
                        if mask.max() > 1.0:
                            mask /= 255.0
                        if not mask.flags['C_CONTIGUOUS'] or not mask.flags['F_CONTIGUOUS']:
                            mask = mask.copy()
                        return mask

        stem = img_path.stem
        if stem in self.img_to_masks:
            return self._generate_yolo_mask(stem) if self.img_to_masks[stem] == "yolo" else self._generate_coco_mask(stem)

        return None

    def _generate_coco_mask(self, img_name: str) -> Optional[np.ndarray]:
        """Rasterize COCO polygon coordinates into a binary mask."""
        polys = self.annotations.get(img_name)
        if polys is None:
            polys = next((v for k, v in self.annotations.items() if Path(k).stem == Path(img_name).stem), None)

        if not polys:
            return None

        img_path = next((f for f in self.image_files if f.name == img_name or f.stem == Path(img_name).stem), None)
        if img_path is None:
            return None

        img = self._read_image(img_path)
        h, w = img.shape[-2:]

        mask = np.zeros((h, w), dtype=np.float32)
        for ann in polys:
            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                for poly in segmentation:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask, [pts], color=1)

        return np.ascontiguousarray(mask)

    def _generate_yolo_mask(self, img_name: str) -> Optional[np.ndarray]:
        """Rasterize YOLO annotations into a binary mask."""
        if img_name not in self.annotations:
            return None

        img_path = next((f for f in self.image_files if f.name == img_name or f.stem == img_name), None)
        if img_path is None:
            return None

        img = self._read_image(img_path)
        h, w = img.shape[-2:]

        label_file = Path(self.annotations[img_name])
        annotations = self._parse_yolo_annotation(label_file, w, h)

        mask = np.zeros((h, w), dtype=np.float32)
        for ann in annotations:
            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                for poly in segmentation:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask, [pts], color=1)

        return np.ascontiguousarray(mask)

    def _get_processed_data(self, path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Retrieve preprocessed image and geometric mask with caching."""
        key = str(path)
        if key in self._cache_store:
            return self._cache_store[key]

        raw_img = self._read_image(path)

        if raw_img.ndim == 3 and raw_img.shape[0] == 1:
            raw_img = raw_img[0]

        if not raw_img.flags['C_CONTIGUOUS'] or not raw_img.flags['F_CONTIGUOUS']:
            raw_img = raw_img.copy()

        result = self.preprocessor(raw_img)
        if len(result) == 3:
            processed_img, valid_mask, meta = result
        else:
            processed_img, valid_mask = result
            meta = {}

        if processed_img.ndim == 2:
            processed_img = processed_img[np.newaxis, ...]
        if valid_mask is not None and valid_mask.ndim == 2:
            valid_mask = valid_mask[np.newaxis, ...]

        processed_img = np.ascontiguousarray(processed_img)
        if valid_mask is not None:
            valid_mask = np.ascontiguousarray(valid_mask)

        res = (processed_img, valid_mask, meta)
        if len(self._cache_store) < self.cache_limit:
            self._cache_store[key] = res

        return res

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        img, valid_mask, meta = self._get_processed_data(img_path)

        # Handle ground truth masks for train/val splits
        if self.has_gt:
            mask = self._read_mask(img_path)
            target_h, target_w = img.shape[-2:]
            if mask is None:
                mask = np.zeros((target_h, target_w), dtype=np.float32)

            if valid_mask is None:
                valid_mask = np.ones((target_h, target_w), dtype=np.float32)

            # Mask preprocessing via nearest neighbor interpolation
            if self.mask_preprocessor is not None:
                mask_processed, _, mask_meta = self.mask_preprocessor(mask)
                mask = mask_processed.squeeze()

                if mask.ndim != 2:
                    if mask.ndim == 1 and mask.shape[0] == target_h * target_w:
                        mask = mask.reshape(target_h, target_w)
                    else:
                        raise ValueError(
                            f"Cannot reshape mask to 2D. Target: {target_h}x{target_w}, shape: {mask.shape}"
                        )

                if mask.shape != (target_h, target_w):
                    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Synchronized data augmentations across image, mask, and valid_mask
            if self.augment:
                img_np = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
                m_2d = mask.squeeze()
                vm_2d = valid_mask.squeeze()
                stacked_masks = np.stack([m_2d, vm_2d], axis=-1)

                img_np, stacked_masks = self.augmentation(img_np, stacked_masks)
                img = np.transpose(img_np, (2, 0, 1)) if img_np.ndim == 3 else img_np
                mask = stacked_masks[..., 0]
                valid_mask = stacked_masks[..., 1]

            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)
            valid_mask = np.ascontiguousarray(valid_mask)

            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
            if valid_mask.ndim == 2:
                valid_mask = valid_mask[np.newaxis, ...]

            sample = {
                'image': torch.from_numpy(img).float(),
                'mask': torch.from_numpy(mask).float(),
                'valid_mask': torch.from_numpy(valid_mask).float(),
                'image_id': img_path.stem,
                'has_object': bool(mask.sum() > 0),
                'meta': meta or {},
            }
        else:
            img = np.ascontiguousarray(img)
            if valid_mask is None:
                valid_mask = np.ones((1, img.shape[-2], img.shape[-1]), dtype=np.float32)
            else:
                valid_mask = np.ascontiguousarray(valid_mask)
                if valid_mask.ndim == 2:
                    valid_mask = valid_mask[np.newaxis, ...]

            if img.ndim == 2:
                img = img[np.newaxis, ...]

            sample = {
                'image': torch.from_numpy(img).float(),
                'valid_mask': torch.from_numpy(valid_mask).float(),
                'mask': None,
                'image_id': img_path.stem,
                'has_object': False,
                'meta': meta or {},
            }

        # Enforce canonical 3D tensor layout (C, H, W) before batch assembly
        if sample['image'].ndim == 2:
            sample['image'] = sample['image'].unsqueeze(0)
        elif sample['image'].ndim == 3 and sample['image'].shape[-1] in (1, 3):
            sample['image'] = sample['image'].permute(2, 0, 1)

        if sample['valid_mask'].ndim == 2:
            sample['valid_mask'] = sample['valid_mask'].unsqueeze(0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble individual dataset items into uniformly dimensioned batch tensors."""
    if not batch:
        return {}

    def ensure_chw(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        return tensor

    images = [ensure_chw(b['image']) for b in batch]
    valid_masks = [ensure_chw(b['valid_mask']) for b in batch]

    has_mask = batch[0].get('mask') is not None

    res: Dict[str, Any] = {
        'image': torch.stack(images, dim=0),
        'valid_mask': torch.stack(valid_masks, dim=0),
        'image_id': [b['image_id'] for b in batch],
        'meta': [b.get('meta', {}) for b in batch],
    }

    if has_mask:
        masks = [ensure_chw(b['mask']) for b in batch]
        res['mask'] = torch.stack(masks, dim=0)
        res['has_object'] = torch.tensor([b['has_object'] for b in batch], dtype=torch.bool)
    else:
        res['mask'] = None
        res['has_object'] = None

    return res