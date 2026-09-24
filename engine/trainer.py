from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import SegmentationDataset, collate_fn
from data.preprocess_config import PreprocessConfig
from engine.validator import BaseValidator
from losses import SegmentationLoss as CompositeSegmentationLoss
from models import SegmentationModel


class BaseTrainer:
    """
    Base trainer class for segmentation models.

    Handles training loop, validation, checkpointing, visualization, and distributed training.
    """

    def __init__(
        self,
        model_cfg: str | Dict,
        data_root: str,
        img_size: Tuple[int, int] = (1024, 1024),
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

        # Metric history for results.png
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "iou": [],
            "dice": [],
            "lr": [],
        }

        self._setup_directories()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_data()
        self._setup_ema()
        self._setup_validator()
        self._setup_amp()
        self._load_checkpoint()

    def _setup_directories(self):
        """Create checkpoint directories."""
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (self.checkpoint_dir / "val_visualizations").mkdir(parents=True, exist_ok=True)

    def _setup_model(self):
        """Initialize model."""
        self.model = SegmentationModel(
            cfg=self.model_cfg,
            ch=self.in_channels,
            nc=self.num_classes,
            verbose=(self.rank == 0),
        ).to(self.device)

        if self.use_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

    def _setup_optimizer(self):
        raw_model = self.model.module if self.use_ddp else self.model
        self.optimizer = torch.optim.AdamW(
            raw_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )

    def _setup_loss(self):
        self.criterion = CompositeSegmentationLoss()

    def _setup_data(self):
        train_dataset = SegmentationDataset(
            data_root=self.data_root,
            split="train",
            img_size=self.img_size,
            augment=True,
            use_cache=True,
            auto=False,
            preprocess_config=self.preprocess_config,
            annotation_file=self.annotation_file,
        )

        val_dataset = SegmentationDataset(
            data_root=self.data_root,
            split="val",
            img_size=self.img_size,
            augment=False,
            use_cache=True,
            auto=False,
            preprocess_config=self.preprocess_config,
            annotation_file=self.annotation_file,
        )

        total_len = len(train_dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(42)
        shuffled_indices = torch.randperm(total_len, generator=generator).tolist()
        train_indices = shuffled_indices[:train_len]
        val_indices = shuffled_indices[train_len:]

        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(val_dataset, val_indices)

        train_sampler = (
            DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            if self.use_ddp
            else None
        )
        val_sampler = (
            DistributedSampler(val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            if (self.use_ddp and val_len > 0)
            else None
        )

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
        from utils import ModelEMA

        raw_model = self.model.module if self.use_ddp else self.model
        self.ema = (
            ModelEMA(raw_model, decay=self.ema_decay, device=self.device)
            if (self.use_ema and self.rank == 0)
            else None
        )

    def _setup_validator(self):
        if self.val_loader is not None and self.rank == 0:
            eval_model = (
                self.ema.shadow_model
                if (self.ema is not None)
                else (self.model.module if self.use_ddp else self.model)
            )
            self.validator = BaseValidator(
                model=eval_model,
                data_root=str(self.data_root),
                img_size=self.img_size,
                batch_size=self.batch_size,
                device=str(self.device),
                num_workers=self.num_workers,
                save_dir=str(self.checkpoint_dir / "val_visualizations"),
                dataloader=self.val_loader,
            )
            self.validator.dataloader = self.val_loader
            self.validator.val_loader = self.val_loader
        else:
            self.validator = None

    def _setup_amp(self):
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (self.use_amp and self.device.type == "cuda")
            else None
        )

    def _load_checkpoint(self):
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
                info = load_checkpoint(
                    str(chk_path),
                    raw_model,
                    self.optimizer,
                    self.ema,
                    self.scheduler,
                    str(self.device),
                )
                self.start_epoch = info["epoch"] + 1
                self.best_loss = info["loss"]

    @staticmethod
    def _ensure_4d_tensor(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.use_ddp and hasattr(self.train_loader, "sampler") and self.train_loader.sampler is not None:
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        n_batches = len(self.train_loader)

        # Định dạng Header thông tin huấn luyện chuẩn Ultralytics
        if self.rank == 0 and epoch == self.start_epoch:
            print(f"\n{'Epoch':>10} {'GPU_mem':>10} {'total_loss':>12} {'bce_loss':>10} {'dice_loss':>10} {'cldice':>10}")

        pbar_desc = f"{f'{epoch + 1}/{self.epochs}':>10}"
        iterator = (
            tqdm(self.train_loader, desc=pbar_desc, leave=False, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
            if self.rank == 0
            else self.train_loader
        )

        for batch in iterator:
            images = self._ensure_4d_tensor(batch["image"].to(self.device, non_blocking=True))
            valid_masks = self._ensure_4d_tensor(batch["valid_mask"].to(self.device, non_blocking=True))
            masks = self._ensure_4d_tensor(batch["mask"].to(self.device, non_blocking=True))

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(self.use_amp and self.device.type == "cuda"),
            ):
                preds = self.model(images)
                loss, loss_parts = self.criterion(preds, masks, valid_masks, epoch)

            raw_model = self.model.module if self.use_ddp else self.model

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(raw_model)

            loss_val = loss.item()
            total_loss += loss_val

            # Hiển thị VRAM và chi tiết từng loss thành phần trên tqdm
            if self.rank == 0:
                mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if torch.cuda.is_available() else "0G"
                bce = float(loss_parts.get("bce", 0.0))
                dice = float(loss_parts.get("dice", 0.0))
                cldice = float(loss_parts.get("cldice", 0.0))
                iterator.set_postfix({
                    "gpu": mem,
                    "loss": f"{loss_val:.4f}",
                    "bce": f"{bce:.3f}",
                    "dice": f"{dice:.3f}",
                    "cldice": f"{cldice:.3f}",
                })

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        if self.val_loader is None:
            return 0.0, {}

        eval_model = (
            self.ema.shadow_model
            if (self.ema is not None and self.rank == 0)
            else (self.model.module if self.use_ddp else self.model)
        )
        eval_model.eval()

        if self.validator is not None:
            self.validator.model = eval_model
            if self.validator.dataloader is None:
                self.validator.dataloader = self.val_loader

            metrics = self.validator.validate()
            val_loss = metrics.get("loss", 0.0)

            # In bảng tổng kết và lưu ảnh trực quan
            if self.rank == 0:
                self.validator.print_results(epoch=epoch + 1)
                
                epoch_vis_dir = self.checkpoint_dir / "val_visualizations" / f"epoch_{epoch + 1}"
                self.validator.save_dir = epoch_vis_dir
                self.validator.save_visualizations(num_samples=4)
                print(f"Visualizations saved to: {epoch_vis_dir}")

            return val_loss, metrics

        return 0.0, {}

    def _plot_results(self):
        """Plot Ultralytics-style results.png curves."""
        if self.rank != 0 or len(self.history["train_loss"]) == 0:
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(epochs, self.history["train_loss"], "b-", label="Train Loss")
        if any(self.history["val_loss"]):
            axes[0, 0].plot(epochs, self.history["val_loss"], "r-", label="Val Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        if any(self.history["iou"]):
            axes[0, 1].plot(epochs, self.history["iou"], "g-", label="Validation IoU")
            axes[0, 1].set_title("Validation IoU")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("IoU")
            axes[0, 1].legend()
            axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        if any(self.history["dice"]):
            axes[1, 0].plot(epochs, self.history["dice"], "m-", label="Validation Dice")
            axes[1, 0].set_title("Validation Dice")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Dice")
            axes[1, 0].legend()
            axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        axes[1, 1].plot(epochs, self.history["lr"], "k-", label="Learning Rate")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / "results.png", dpi=150)
        plt.close(fig)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
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
        """Main training loop."""
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            eval_loss = val_loss if self.val_loader is not None else train_loss

            self.scheduler.step(eval_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]

            if self.rank == 0:
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["iou"].append(metrics.get("iou", 0.0))
                self.history["dice"].append(metrics.get("dice", 0.0))
                self.history["lr"].append(lr_now)

                self._plot_results()

            is_best = eval_loss < self.best_loss
            if is_best:
                self.best_loss = eval_loss

            self.save_checkpoint(epoch, eval_loss, is_best)

        if self.use_ddp:
            dist.destroy_process_group()