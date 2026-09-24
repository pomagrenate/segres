# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models import SegmentationModel
from data import SegmentationDataset, collate_fn
from data.preprocess_config import PreprocessConfig
from losses import SegmentationLoss as CompositeSegmentationLoss
from engine.validator import BaseValidator


class BaseTrainer:
    """
    Base trainer class for segmentation models.
    
    Handles training loop, validation, checkpointing, and distributed training.
    """
    
    def __init__(
        self,
        model_cfg: str | Dict,
        data_root: str,
        img_size: tuple = (1024, 1024),
        batch_size: int = 1,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        val_split: float = 0.1,
        num_workers: int = 2,
        use_amp: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        grad_clip: float = 2.0,
        save_interval: int = 5,
        resume: Optional[str] = None,
        in_channels: int = 3,
        num_classes: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        annotation_file: Optional[str] = None,
    ):
        self.model_cfg = model_cfg
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.val_split = val_split
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.save_interval = save_interval
        self.resume = resume
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.preprocess_config = preprocess_config
        self.annotation_file = annotation_file
        
        # Distributed training setup
        self.use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if self.use_ddp:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend="nccl", init_method="env://")
        
        # Initialize
        self._setup_directories()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_data()
        self._setup_validator()
        self._setup_ema()
        self._setup_amp()
        self._load_checkpoint()
    
    def _setup_directories(self):
        """Create checkpoint directories."""
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_model(self):
        """Initialize model."""
        self.model = SegmentationModel(
            cfg=self.model_cfg,
            ch=self.in_channels,
            nc=self.num_classes,
            verbose=(self.rank == 0)
        ).to(self.device)
        
        if self.use_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
    
    def _setup_optimizer(self):
        """Initialize optimizer."""
        raw_model = self.model.module if self.use_ddp else self.model
        self.optimizer = torch.optim.AdamW(raw_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
    
    def _setup_loss(self):
        """Initialize loss function."""
        raw_model = self.model.module if self.use_ddp else self.model
        # Default: Dice + BCE (strong generic baseline)
        self.criterion = CompositeSegmentationLoss()
    
    def _setup_validator(self):
        """Initialize validator for metrics computation."""
        if self.val_loader is not None and self.rank == 0:
            raw_model = self.model.module if self.use_ddp else self.model
            self.validator = BaseValidator(
                model=raw_model,
                data_root=str(self.data_root),
                img_size=self.img_size,
                batch_size=self.batch_size,
                device=str(self.device),
                num_workers=self.num_workers,
                save_dir=str(self.checkpoint_dir / "val_visualizations"),
            )
            self.validator.setup_data(split="val")
        else:
            self.validator = None
    
    def _setup_data(self):
        """Initialize data loaders."""
        # Full dataset
        full_dataset = SegmentationDataset(
            data_root=self.data_root,
            split="train",
            img_size=self.img_size,
            augment=True,
            use_cache=True,
            auto=False,  # Training uses fixed size for consistency
            preprocess_config=self.preprocess_config,
            annotation_file=self.annotation_file,
        )
        
        # Split train/val
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len
        
        generator = torch.Generator().manual_seed(42)
        shuffled_indices = torch.randperm(total_len, generator=generator).tolist()
        train_indices = shuffled_indices[:train_len]
        val_indices = shuffled_indices[train_len:]
        
        train_ds = Subset(full_dataset, train_indices)
        val_ds = Subset(full_dataset, val_indices)
        
        # Samplers for DDP
        train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True) if self.use_ddp else None
        val_sampler = DistributedSampler(val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False) if (self.use_ddp and val_len > 0) else None
        
        # Data loaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )
        
        self.val_loader = (
            DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=collate_fn,
                drop_last=False,
                persistent_workers=(self.num_workers > 0),
            )
            if val_len > 0
            else None
        )
    
    def _setup_ema(self):
        """Setup EMA."""
        from utils import ModelEMA
        raw_model = self.model.module if self.use_ddp else self.model
        self.ema = ModelEMA(raw_model, decay=self.ema_decay, device=self.device) if (self.use_ema and self.rank == 0) else None
    
    def _setup_amp(self):
        """Setup AMP."""
        self.scaler = torch.amp.GradScaler("cuda") if (self.use_amp and self.device.type == "cuda") else None
    
    def _load_checkpoint(self):
        """Load checkpoint if resume is specified."""
        self.start_epoch = 0
        self.best_loss = float("inf")
        
        if self.resume:
            from utils import load_checkpoint
            chk_path = Path(self.resume)
            if chk_path == Path("last"):
                all_chk = list(self.checkpoint_dir.glob("*.pt"))
                chk_path = max(all_chk, key=os.path.getctime) if all_chk else (self.checkpoint_dir / "last.pt")
            
            if chk_path.exists():
                raw_model = self.model.module if self.use_ddp else self.model
                info = load_checkpoint(str(chk_path), raw_model, self.optimizer, self.ema, self.scheduler, str(self.device))
                self.start_epoch = info["epoch"] + 1
                self.best_loss = info["loss"]
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        if self.use_ddp:
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        n_batches = len(self.train_loader)
        
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False) if self.rank == 0 else self.train_loader
        
        for batch in iterator:
            images = batch["image"].to(self.device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.use_amp and self.device.type == "cuda")):
                preds = self.model(images)
                loss, _ = self.criterion(preds, masks, valid_masks, epoch)
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                raw_model = self.model.module if self.use_ddp else self.model
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                raw_model = self.model.module if self.use_ddp else self.model
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
            
            if self.ema is not None:
                raw_model = self.model.module if self.use_ddp else self.model
                self.ema.update(raw_model)
            
            loss_val = loss.item()
            total_loss += loss_val
            if self.rank == 0:
                iterator.set_postfix({"loss": f"{loss_val:.4f}"})
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, dict]:
        """Validate the model and return loss and metrics."""
        if self.val_loader is None:
            return 0.0, {}
        
        # Use validator for metrics (Ultralytics pattern)
        if self.validator is not None:
            # Update validator model with current (or EMA) model
            eval_model = self.ema.shadow_model if (self.ema is not None) else (self.model.module if self.use_ddp else self.model)
            self.validator.model = eval_model
            
            metrics = self.validator.validate()
            val_loss = metrics.get("loss", 0.0)
            return val_loss, metrics
        
        # Fallback: simple loss-based validation
        eval_target = self.ema.shadow_model if (self.ema is not None and self.rank == 0) else (self.model.module if self.use_ddp else self.model)
        eval_target.eval()
        
        val_loss = 0.0
        val_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            
            preds = eval_target(images)
            l_val, _ = self.criterion(preds, masks, valid_masks, epoch)
            val_loss += l_val.item()
        
        val_loss = val_loss / max(val_batches, 1)
        
        if self.use_ddp:
            loss_tensor = torch.tensor([val_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = loss_tensor.item()
        
        return val_loss, {}
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save checkpoint."""
        if self.rank != 0:
            return
        
        from utils import save_checkpoint
        raw_model = self.model.module if self.use_ddp else self.model
        
        save_checkpoint(
            model=raw_model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            filepath=str(self.checkpoint_dir / "last.pt"),
            ema_model=self.ema,
            scheduler=self.scheduler,
        )
        
        if is_best:
            save_checkpoint(
                model=raw_model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss,
                filepath=str(self.checkpoint_dir / "best.pt"),
                ema_model=self.ema,
                scheduler=self.scheduler,
            )
        
        if (epoch + 1) % self.save_interval == 0:
            save_checkpoint(
                model=raw_model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss,
                filepath=str(self.checkpoint_dir / f"epoch_{epoch + 1}.pt"),
                ema_model=self.ema,
                scheduler=self.scheduler,
            )
    
    def train(self):
        """Main training loop with validation at each epoch (Ultralytics pattern)."""
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            eval_loss = val_loss if self.val_loader is not None else train_loss
            
            self.scheduler.step(eval_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]
            
            if self.rank == 0:
                if self.val_loader is not None and metrics:
                    # Ultralytics-style logging with metrics
                    iou = metrics.get("iou", 0.0)
                    dice = metrics.get("dice", 0.0)
                    print(f"[Epoch {epoch + 1}/{self.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} iou={iou:.4f} dice={dice:.4f} lr={lr_now:.2e}", flush=True)
                elif self.val_loader is not None:
                    print(f"[Epoch {epoch + 1}/{self.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr_now:.2e}", flush=True)
                else:
                    print(f"[Epoch {epoch + 1}/{self.epochs}] train_loss={train_loss:.4f} lr={lr_now:.2e}", flush=True)
            
            is_best = eval_loss < self.best_loss
            if is_best:
                self.best_loss = eval_loss
            
            self.save_checkpoint(epoch, eval_loss, is_best)
        
        if self.use_ddp:
            dist.destroy_process_group()
