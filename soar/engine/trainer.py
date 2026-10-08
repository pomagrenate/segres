from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

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

from ..data import SegmentationDataset, collate_fn, DatasetConfig
from ..data.config import PreprocessConfig
from .validator import BaseValidator
from ..losses import SegmentationLoss as CompositeSegmentationLoss
from ..models import SegmentationModel, build_model
from ..utils import (
    DeviceSelection,
    ModelEMA,
    announce_device,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    profile_model,
    format_latex_row_table1,
    format_latex_row_table2,
)


class BaseTrainer:
    """
    Base trainer class for segmentation models.

    Handles training loop, validation, checkpointing, visualization, and distributed training.
    Enforces native full-resolution segmentation with physical batch size 1 and gradient accumulation.
    """

    def __init__(
        self,
        model_cfg: str | Dict,
        data_root: str,
        img_size: Tuple[int, int] = (1024, 1024),
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
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
        log_interval: int = 10,
        resume: Optional[str] = None,
        in_channels: int = 3,
        num_classes: int = 1,
        preprocess_config: Optional[PreprocessConfig] = None,
        annotation_file: Optional[str] = None,
        balance_sampler: bool = False,
        positive_ratio: float = 0.7,
        sampler_mode: str = "hybrid",
        loss: str = "soar",
        loss_cfg: Optional[Dict[str, Any]] = None,
        augment: bool = True,
        samples: Optional[int] = None,
        cache_ram: bool = False,
    ):

        self.model_cfg = model_cfg
        if loss_cfg is None and str(loss).lower() == "standard":
            self.loss_cfg = {
                "region": {"type": "dice_bce", "dice_weight": 1.0, "bce_weight": 1.0},
                "boundary": {"enabled": False},
                "structure": {"enabled": False},
            }
        else:
            self.loss_cfg = loss_cfg
        self.loss_type = loss
        self.dataset_cfg = DatasetConfig.resolve(data_root, annotation_file=annotation_file)
        self.data_root = self.dataset_cfg.root_path
        self.img_size = img_size
        if batch_size != 1:
            warnings.warn(
                f"Physical batch size was specified as {batch_size}, but SOAR strictly enforces physical "
                f"batch_size=1 to guarantee full-resolution input and prevent background gradient swamping. "
                f"Setting batch_size=1.",
                UserWarning,
                stacklevel=2,
            )
        self.batch_size = 1
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device_selection = resolve_device(device)
        self.device = self.device_selection.device

        # Automatically determine model_name for checkpointing and metrics
        if isinstance(model_cfg, nn.Module):
            self.model_name = getattr(model_cfg, "name", model_cfg.__class__.__name__)
        elif isinstance(model_cfg, str):
            p = Path(model_cfg)
            self.model_name = p.stem if p.suffix in (".yaml", ".yml") else model_cfg
        else:
            self.model_name = "soar"

        base_ckpt = Path(checkpoint_dir)
        if base_ckpt.name.lower() != self.model_name.lower():
            self.checkpoint_dir = base_ckpt / self.model_name
        else:
            self.checkpoint_dir = base_ckpt

        self.val_split = val_split
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.save_interval = save_interval
        self.log_interval = max(1, int(log_interval))
        self.resume = resume
        self.in_channels = in_channels

        # Automatically resolve num_classes and class_names from DatasetConfig if applicable
        if (self.dataset_cfg.nc > 1 or len(self.dataset_cfg.names) > 1) and num_classes == 1:
            self.num_classes = self.dataset_cfg.nc
        else:
            self.num_classes = num_classes
        self.class_names = self.dataset_cfg.names

        self.preprocess_config = preprocess_config
        self.annotation_file = annotation_file
        self.balance_sampler = balance_sampler
        self.positive_ratio = positive_ratio
        self.sampler_mode = sampler_mode
        self.augment = augment
        self.samples = samples
        self.cache_ram = bool(cache_ram)

        # Distributed training setup
        self.use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.use_ddp:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires an available CUDA device.")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.device_selection = DeviceSelection(
                requested=self.device_selection.requested,
                device=self.device,
                fell_back=False,
            )
            dist.init_process_group(backend="nccl", init_method="env://")

        if self.rank == 0:
            announce_device(self.device_selection)

        # Metric history for results.png
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "iou": [],
            "dice": [],
            "precision": [],
            "recall": [],
            "boundary_iou": [],
            "cldice": [],
            "lr": [],
        }

        self.best_epoch = 0
        self.best_metrics: Dict[str, float] = {}

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
        """Initialize model using universal model factory."""
        if isinstance(self.model_cfg, nn.Module):
            self.model = self.model_cfg.to(self.device)
        else:
            self.model = build_model(
                model=self.model_cfg,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                verbose=(self.rank == 0),
            ).to(self.device)

        if self.use_ddp:
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
        if self.loss_cfg:
            self.criterion = CompositeSegmentationLoss.from_config(self.loss_cfg)
        else:
            self.criterion = CompositeSegmentationLoss()

    def _setup_data(self):
        train_image_dir = self.dataset_cfg.train_images
        val_image_dir = self.dataset_cfg.val_images
        train_image_files = self.dataset_cfg.train_image_list
        val_image_files = self.dataset_cfg.val_image_list

        train_ann = (
            str(self.dataset_cfg.annotation_files["train"])
            if "train" in self.dataset_cfg.annotation_files
            else self.annotation_file
        )
        val_ann = (
            str(self.dataset_cfg.annotation_files["val"])
            if "val" in self.dataset_cfg.annotation_files
            else self.annotation_file
        )

        train_mask_dir = (
            str(self.dataset_cfg.mask_dirs["train"])
            if "train" in self.dataset_cfg.mask_dirs
            else None
        )
        val_mask_dir = (
            str(self.dataset_cfg.mask_dirs["val"])
            if "val" in self.dataset_cfg.mask_dirs
            else None
        )

        train_dataset = SegmentationDataset(
            data_root=self.data_root,
            split="train",
            img_size=self.img_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            names=self.class_names,
            augment=self.augment,
            use_cache=True,
            auto=False,
            preprocess_config=self.preprocess_config,
            annotation_file=train_ann,
            mask_dir=train_mask_dir,
            image_dir=train_image_dir,
            image_files=train_image_files,
            samples=self.samples,
            cache_ram=self.cache_ram,
        )

        if hasattr(train_dataset, "class_names") and train_dataset.class_names:
            self.class_names = train_dataset.class_names

        val_dataset = None
        has_val = False
        try:
            val_dataset = SegmentationDataset(
                data_root=self.data_root,
                split="val",
                img_size=self.img_size,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                names=self.class_names,
                augment=False,
                use_cache=True,
                auto=False,
                preprocess_config=self.preprocess_config,
                annotation_file=val_ann,
                mask_dir=val_mask_dir,
                image_dir=val_image_dir,
                image_files=val_image_files,
                samples=self.samples,
                cache_ram=self.cache_ram,
            )
            if len(val_dataset) > 0 and set(val_dataset.image_files) != set(train_dataset.image_files):
                has_val = True
        except Exception:
            has_val = False

        if has_val and val_dataset is not None:
            train_ds = train_dataset
            val_ds = val_dataset
            val_len = len(val_ds)
        else:
            total_len = len(train_dataset)
            val_len = int(total_len * self.val_split)
            train_len = total_len - val_len

            if self.samples is not None or val_len == 0:
                # Overfit / sample check mode: evaluate directly on sampled train dataset
                train_ds = train_dataset
                val_ds = train_dataset
                val_len = total_len
            else:
                generator = torch.Generator().manual_seed(42)
                shuffled_indices = torch.randperm(total_len, generator=generator).tolist()
                train_indices = shuffled_indices[:train_len]
                val_indices = shuffled_indices[train_len:]

                train_ds = Subset(train_dataset, train_indices)
                val_ds = Subset(train_dataset, val_indices)

        if self.cache_ram:
            from ..data.dataset import preload_dataset_cache
            preload_dataset_cache(train_dataset, verbose=(self.rank == 0))
            if has_val and val_dataset is not None:
                preload_dataset_cache(val_dataset, verbose=(self.rank == 0))

        if self.use_ddp:
            train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        elif self.balance_sampler:
            from ..data.dataset import build_balanced_sampler
            train_sampler = build_balanced_sampler(
                train_ds,
                positive_ratio=self.positive_ratio,
                mode=self.sampler_mode,
            )
        else:
            train_sampler = None
        val_sampler = (
            DistributedSampler(val_ds, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            if (self.use_ddp and val_len > 0 and val_ds is not None)
            else None
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=1,
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
                batch_size=1,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=collate_fn,
                drop_last=False,
                persistent_workers=(self.num_workers > 0),
            )
            if (val_len > 0 and val_ds is not None)
            else None
        )

    def _setup_ema(self):
        raw_model = self.model.module if self.use_ddp else self.model
        self.ema = (
            ModelEMA(raw_model, decay=self.ema_decay, device=self.device)
            if self.use_ema
            else None
        )

    def _setup_validator(self):
        if self.val_loader is not None:
            eval_model = (
                self.ema.shadow_model
                if self.ema is not None
                else (self.model.module if self.use_ddp else self.model)
            )
            self.validator = BaseValidator(
                model=eval_model,
                data_root=str(self.data_root),
                img_size=self.img_size,
                device=str(self.device),
                num_workers=self.num_workers,
                save_dir=str(self.checkpoint_dir / "val_visualizations") if self.rank == 0 else None,
                dataloader=self.val_loader,
                num_classes=self.num_classes,
                class_names=self.class_names,
            )
            self.validator.criterion = self.criterion
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
            chk_path = Path(self.resume)
            if chk_path == Path("last"):
                all_chk = list(self.checkpoint_dir.glob("*.pt"))
                chk_path = max(all_chk, key=os.path.getctime) if all_chk else (self.checkpoint_dir / "last.pt")

            if chk_path.exists():
                raw_model = self.model.module if self.use_ddp else self.model
                info = load_checkpoint(
                    filepath=str(chk_path),
                    model=raw_model,
                    optimizer=self.optimizer,
                    ema_model=self.ema,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    device=str(self.device),
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

        if self.rank == 0 and epoch == self.start_epoch:
            print(f"\n{'Epoch':>10} {'GPU_mem':>10} {'total_loss':>12} {'reg_loss':>10} {'bnd_loss':>10} {'cldice':>10}")

        pbar_desc = f"{f'{epoch + 1}/{self.epochs}':>10}"
        iterator = (
            tqdm(self.train_loader, desc=pbar_desc, leave=True, bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
            if self.rank == 0
            else self.train_loader
        )

        self.optimizer.zero_grad(set_to_none=True)
        raw_model = self.model.module if self.use_ddp else self.model

        for i, batch in enumerate(iterator):
            images = self._ensure_4d_tensor(batch["image"].to(self.device, non_blocking=True))
            valid_masks = self._ensure_4d_tensor(batch["valid_mask"].to(self.device, non_blocking=True))
            masks = self._ensure_4d_tensor(batch["mask"].to(self.device, non_blocking=True))

            is_accumulating = ((i + 1) % self.accumulate_grad_batches != 0) and ((i + 1) != n_batches)
            sync_context = self.model.no_sync() if (self.use_ddp and is_accumulating) else contextlib.nullcontext()

            with sync_context:
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=(self.use_amp and self.device.type == "cuda"),
                ):
                    preds = self.model(images)
                    loss, loss_parts = self.criterion(preds, masks, valid_masks, epoch)
                    scaled_loss = loss / self.accumulate_grad_batches

                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if not is_accumulating:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=self.grad_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update(raw_model)

            loss_val = loss.item()
            total_loss += loss_val

            if self.rank == 0 and (i == 0 or (i + 1) % self.accumulate_grad_batches == 0 or (i + 1) == n_batches):
                mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if torch.cuda.is_available() else "0G"
                region_l = float(loss_parts.get("region", 0.0))
                bnd_l = float(loss_parts.get("boundary", 0.0))
                cldice_l = float(loss_parts.get("cldice", 0.0))
                iterator.set_postfix({
                    "gpu": mem,
                    "loss": f"{loss_val:.4f}",
                    "reg": f"{region_l:.3f}",
                    "bnd": f"{bnd_l:.3f}",
                    "cldice": f"{cldice_l:.3f}",
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

            # Save qualitative validation examples every epoch.
            if self.rank == 0:
                try:
                    epoch_vis_dir = self.checkpoint_dir / "val_visualizations" / f"epoch_{epoch + 1}"
                    self.validator.save_dir = epoch_vis_dir
                    self.validator.save_visualizations(num_samples=4)
                    print(f"Visualizations saved to: {epoch_vis_dir}")
                except Exception as e:
                    print(f"Notice: Skipped visualization export for epoch {epoch + 1} ({e})")

            return val_loss, metrics

        return 0.0, {}

    def _should_log_epoch(self, epoch: int) -> bool:
        """Return whether a zero-based epoch should emit a metric summary."""
        epoch_number = epoch + 1
        return epoch_number % self.log_interval == 0 or epoch_number == self.epochs

    def _print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, Any],
        learning_rate: float,
        is_best: bool,
    ) -> None:
        """Print the metrics needed to monitor a training run."""
        if self.rank != 0:
            return

        epoch_number = epoch + 1
        has_validation = self.val_loader is not None
        metric_label = "mIoU" if len(metrics.get("class_ious", [])) > 1 else "IoU"

        def metric_text(name: str) -> str:
            if not has_validation:
                return "-"
            return f"{float(metrics.get(name, 0.0)):.5f}"

        val_loss_text = f"{val_loss:.5f}" if has_validation else "-"
        best_marker = " (new best)" if is_best else ""

        print(f"\n{'=' * 126}")
        print(
            f"Training summary - epoch {epoch_number}/{self.epochs} "
            f"(console interval: {self.log_interval})"
        )
        print(
            f"{'Train Loss':>11} {'Val Loss':>11} {'LR':>11} "
            f"{metric_label:>9} {'Dice':>9} {'Prec':>9} {'Recall':>9} "
            f"{'bIoU':>9} {'clDice':>9}"
        )
        print(f"{'-' * 126}")
        print(
            f"{train_loss:>11.5f} {val_loss_text:>11} {learning_rate:>11.3e} "
            f"{metric_text('iou'):>9} {metric_text('dice'):>9} "
            f"{metric_text('precision'):>9} {metric_text('recall'):>9} "
            f"{metric_text('boundary_iou'):>9} {metric_text('cldice'):>9}"
        )
        print(
            f"Best so far: epoch {self.best_epoch}/{self.epochs}, "
            f"loss={self.best_loss:.5f}{best_marker}"
        )

        class_ious = metrics.get("class_ious", [])
        if has_validation and len(class_ious) > 1:
            per_class = []
            for class_index, class_iou in enumerate(class_ious):
                class_name = self.class_names.get(class_index, f"Class_{class_index}")
                per_class.append(f"{class_name}={float(class_iou):.5f}")
            print("Per-class IoU: " + " | ".join(per_class))

        print(f"{'=' * 126}\n", flush=True)

    def _plot_results(self):
        """Plot results.png training metric curves."""
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

        raw_model = self.model.module if self.use_ddp else self.model

        save_checkpoint(
            model=raw_model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            filepath=str(self.checkpoint_dir / "last.pt"),
            ema_model=self.ema,
            scheduler=self.scheduler,
            scaler=self.scaler,
            model_name=self.model_name,
            num_classes=self.num_classes,
            class_names=self.class_names,
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
                scaler=self.scaler,
                model_name=self.model_name,
                num_classes=self.num_classes,
                class_names=self.class_names,
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
                scaler=self.scaler,
                model_name=self.model_name,
                num_classes=self.num_classes,
                class_names=self.class_names,
            )

    def train(self):
        """Main training loop."""
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            eval_loss = val_loss if self.val_loader is not None else train_loss

            if self.val_loader is None and self.use_ddp:
                t_loss = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
                eval_loss = t_loss.item() / max(self.world_size, 1)

            self.scheduler.step(eval_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]

            if self.rank == 0:
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["iou"].append(metrics.get("iou", 0.0))
                self.history["dice"].append(metrics.get("dice", 0.0))
                self.history["precision"].append(metrics.get("precision", 0.0))
                self.history["recall"].append(metrics.get("recall", 0.0))
                self.history["boundary_iou"].append(metrics.get("boundary_iou", 0.0))
                self.history["cldice"].append(metrics.get("cldice", 0.0))
                self.history["lr"].append(lr_now)

                self._plot_results()

            is_best = eval_loss < self.best_loss
            if is_best:
                self.best_loss = eval_loss
                self.best_epoch = epoch + 1
                self.best_metrics = dict(metrics) if metrics else {"loss": eval_loss}

            if self.rank == 0 and self._should_log_epoch(epoch):
                self._print_epoch_summary(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=metrics,
                    learning_rate=lr_now,
                    is_best=is_best,
                )

            self.save_checkpoint(epoch, eval_loss, is_best)

        if self.rank == 0:
            self._export_results_summary()

        if self.use_ddp:
            dist.destroy_process_group()

    def _export_results_summary(self):
        """Export epoch history to results.csv, write best_metrics.json, and print LaTeX paper table rows."""
        import csv
        import json

        # 1. Export results.csv
        csv_path = self.checkpoint_dir / "results.csv"
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "iou",
            "dice",
            "precision",
            "recall",
            "boundary_iou",
            "cldice",
            "lr",
        ]
        n_epochs = len(self.history["train_loss"])
        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(n_epochs):
                    writer.writerow({
                        "epoch": i + 1,
                        "train_loss": f"{self.history['train_loss'][i]:.5f}",
                        "val_loss": f"{self.history['val_loss'][i]:.5f}" if i < len(self.history["val_loss"]) else "",
                        "iou": f"{self.history['iou'][i]:.5f}" if i < len(self.history["iou"]) else "",
                        "dice": f"{self.history['dice'][i]:.5f}" if i < len(self.history["dice"]) else "",
                        "precision": f"{self.history['precision'][i]:.5f}" if i < len(self.history["precision"]) else "",
                        "recall": f"{self.history['recall'][i]:.5f}" if i < len(self.history["recall"]) else "",
                        "boundary_iou": f"{self.history['boundary_iou'][i]:.5f}" if i < len(self.history["boundary_iou"]) else "",
                        "cldice": f"{self.history['cldice'][i]:.5f}" if i < len(self.history["cldice"]) else "",
                        "lr": f"{self.history['lr'][i]:.6e}" if i < len(self.history["lr"]) else "",
                    })
        except Exception as e:
            print(f"Notice: Failed to write results.csv ({e})")

        # 2. Profile model params and FLOPs
        raw_model = self.model.module if self.use_ddp else self.model
        params_m, flops_g = profile_model(
            raw_model,
            img_size=self.img_size,
            in_channels=self.in_channels,
            device=str(self.device),
        )

        # 3. Export best_metrics.json
        summary_payload = {
            "model_name": self.model_name,
            "params_m": params_m,
            "flops_g": flops_g,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "metrics": self.best_metrics,
        }
        json_path = self.checkpoint_dir / "best_metrics.json"
        try:
            with open(json_path, "w") as f:
                json.dump(summary_payload, f, indent=2)
        except Exception:
            pass

        # 4. Format LaTeX rows
        row_table1 = format_latex_row_table1(self.model_name, params_m, flops_g, self.best_metrics)
        row_table2 = format_latex_row_table2(self.model_name, params_m, flops_g, self.best_metrics)

        # 5. Print formatted summary banner
        miou_val = self.best_metrics.get("iou", 0.0) * 100.0
        dice_val = self.best_metrics.get("dice", 0.0) * 100.0
        biou_val = self.best_metrics.get("boundary_iou", 0.0) * 100.0
        cldice_val = self.best_metrics.get("cldice", 0.0) * 100.0
        prec_val = self.best_metrics.get("precision", 0.0) * 100.0
        rec_val = self.best_metrics.get("recall", 0.0) * 100.0

        print("\n" + "=" * 90)
        print("SOAR & BENCHMARK SUITE - SCIENTIFIC TRAINING COMPLETE")
        print(f"Model:           {self.model_name}")
        print(f"Best Epoch:      {self.best_epoch} / {self.epochs}")
        print(f"Parameters:      {params_m:.2f} M")
        print(f"FLOPs ({self.img_size[0]}x{self.img_size[1]}): {flops_g:.2f} G")
        print("-" * 90)
        print("Best Validation Metrics:")
        print(f"  mIoU:          {miou_val:.2f}%")
        print(f"  Dice:          {dice_val:.2f}%")
        print(f"  bIoU:          {biou_val:.2f}%")
        print(f"  clDice:        {cldice_val:.2f}%")
        print(f"  Precision:     {prec_val:.2f}%")
        print(f"  Recall:        {rec_val:.2f}%")
        print(f"  Val Loss:      {self.best_loss:.4f}")
        print("-" * 90)
        print("LaTeX Table I (tab:main_benchmark, 1024x1024) Row Ready for Copy-Paste:")
        print(f"  {row_table1}")
        print("\nLaTeX Table II (tab:highres_stress, 2048x2048) Row Ready for Copy-Paste:")
        print(f"  {row_table2}")
        print(f"\nResults history saved to: {csv_path}")
        print(f"Best metrics saved to:    {json_path}")
        print("=" * 90 + "\n")
