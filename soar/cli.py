from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
import torch
import yaml
if __package__ is None or not __package__:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "soar"

from .models import build_model, list_models, SegmentationModel
from .engine.trainer import BaseTrainer
from .engine.validator import BaseValidator
from .engine.predictor import BasePredictor
from .utils import load_checkpoint
from .data.config import PreprocessConfig
from .data import DatasetConfig


def resolve_config_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve config path across both configs/ and legacy cfg/ layouts with validation."""
    if path_str is None:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)
    if "cfg/" in path_str:
        alt = Path(path_str.replace("cfg/", "configs/"))
        if alt.exists():
            return str(alt)
    elif "configs/" in path_str:
        alt = Path(path_str.replace("configs/", "cfg/"))
        if alt.exists():
            return str(alt)
    return str(p)


def parse_img_size(size_arg) -> tuple[int, int]:
    """Parse image size into (height, width) tuple."""
    if isinstance(size_arg, (list, tuple)):
        return (int(size_arg[0]), int(size_arg[1])) if len(size_arg) >= 2 else (int(size_arg[0]), int(size_arg[0]))
    return (int(size_arg), int(size_arg))


def infer_model_name_and_classes(weights_path: str, model_arg: Optional[str] = None, num_classes_arg: Optional[int] = None) -> tuple[str, int]:
    """Inspect checkpoint and path to deduce model name and number of classes if not given."""
    model_name = model_arg
    num_classes = num_classes_arg

    # Try loading checkpoint header
    p = Path(weights_path)
    if p.is_file():
        try:
            ckpt = torch.load(weights_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if not model_name and "model_name" in ckpt and ckpt["model_name"]:
                    model_name = ckpt["model_name"]
                if num_classes is None and "num_classes" in ckpt and ckpt["num_classes"]:
                    num_classes = int(ckpt["num_classes"])
        except Exception:
            pass

    # Infer model name from filepath if still absent
    if not model_name:
        for known in list_models():
            if known.lower().replace("_", "").replace("-", "") in p.as_posix().lower().replace("_", "").replace("-", ""):
                model_name = known
                break

    if not model_name:
        model_name = "soar"

    if num_classes is None:
        num_classes = 1

    return model_name, num_classes


def parse_args(raw_args: Optional[List[str]] = None):
    """Parse command line arguments supporting both unified and model-first invocation."""
    if raw_args is None:
        raw_args = sys.argv[1:]

    # Handle model-first invocation: e.g. "python cli.py unet train --data ..."
    if len(raw_args) >= 2 and raw_args[0].lower() in [m.lower().replace("_", "").replace("-", "") for m in list_models()]:
        model_name = raw_args[0]
        cmd = raw_args[1]
        remaining = raw_args[2:]
        raw_args = [cmd, "--model", model_name] + remaining

    parser = argparse.ArgumentParser(description="SOAR & Benchmark Suite - Unified Scientific Segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # -------------------------------------------------------------
    # Train command
    # -------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train a segmentation model (SOAR or benchmarks)")
    train_parser.add_argument(
        "--model",
        type=str,
        default="soar",
        help="Model architecture: 'soar', 'unet', 'dlinknet', 'csnet', 'bisenetv2', 'ddrnet', 'pidnet', 'isdnet', 'segformer', or path to YAML config",
    )
    train_parser.add_argument("--data", type=str, required=True, help="Dataset root directory or COCO dataset folder")
    train_parser.add_argument("--cfg", type=str, default="configs/default.yaml", help="Default configuration YAML")
    train_parser.add_argument("--imgsz", "--img-size", dest="img_size", type=int, nargs="+", default=[1024, 1024], help="Image resolution (e.g. 1024 or 1024 1024)")
    train_parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=1, help="Physical batch size (default: 1)")
    train_parser.add_argument("--accum-steps", "--accumulate-grad-batches", dest="accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device: cuda, cuda:N, or cpu (default: cuda with an announced CPU fallback)",
    )
    train_parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Base checkpoint directory")
    train_parser.add_argument("--log-interval", type=int, default=10, help="Print a metrics summary every N epochs (default: 10)")
    train_parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio if dataset lacks explicit val set")
    train_parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision")
    train_parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")
    train_parser.add_argument("--ema", action="store_true", help="Use exponential moving average")
    train_parser.add_argument("--resume", "--weights", dest="resume", type=str, default=None, help="Path to checkpoint to resume training from or initialize weights")
    train_parser.add_argument("--in-channels", type=int, default=3, help="Input image channels")
    train_parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (auto-detected from dataset if omitted)")
    train_parser.add_argument("--annotation-file", type=str, default=None, help="Optional COCO annotation file path")
    train_parser.add_argument("--no-augment", action="store_true", help="Disable data augmentations")
    train_parser.add_argument("--samples", type=int, default=None, help="Subsample dataset to first N samples for quick testing")
    train_parser.add_argument("--preprocess-mode", type=str, default="standard", choices=["minimal", "standard", "native"], help="Preprocessing mode")
    train_parser.add_argument("--loss", type=str, default="soar", choices=["soar", "standard"], help="Loss function: 'soar' (composite) or 'standard' (BCE+Dice)")
    train_parser.add_argument("--cache-ram", action="store_true", default=False, help="Enable full in-memory RAM caching for zero-latency training (bypasses disk reads and rasterization)")

    # -------------------------------------------------------------
    # Validate command
    # -------------------------------------------------------------
    val_parser = subparsers.add_parser("val", help="Validate a segmentation model")
    val_parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pt)")
    val_parser.add_argument("--model", type=str, default=None, help="Model architecture (auto-inferred from checkpoint if omitted)")
    val_parser.add_argument("--data", type=str, required=True, help="Dataset root directory")
    val_parser.add_argument("--imgsz", "--img-size", dest="img_size", type=int, nargs="+", default=[1024, 1024], help="Image resolution")
    val_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    val_parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    val_parser.add_argument("--save-dir", type=str, default=None, help="Directory to save validation visualizations")
    val_parser.add_argument("--annotation-file", type=str, default=None, help="Optional COCO annotation file path")
    val_parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    val_parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (auto-inferred from checkpoint or dataset)")
    val_parser.add_argument("--split", type=str, default=None, help="Dataset split ('val', 'test', 'train'). Auto-detected if omitted.")
    val_parser.add_argument("--cache-ram", action="store_true", default=False, help="Enable in-memory RAM caching for validation dataset")
    val_parser.add_argument("--samples", type=int, default=None, help="Limit validation to first N samples for quick checks")
    val_parser.add_argument("--preprocess-mode", type=str, default="standard", choices=["minimal", "standard", "native"], help="Preprocessing mode matching training pipeline")

    # -------------------------------------------------------------
    # Predict command
    # -------------------------------------------------------------
    predict_parser = subparsers.add_parser("predict", help="Run inference on images")
    predict_parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pt)")
    predict_parser.add_argument("--model", type=str, default=None, help="Model architecture (auto-inferred from checkpoint if omitted)")
    predict_parser.add_argument("--data", "--source", dest="data", type=str, required=True, help="Dataset directory or images folder")
    predict_parser.add_argument("--annotation-file", type=str, default=None, help="Optional COCO JSON annotations to evaluate metrics")
    predict_parser.add_argument("--imgsz", "--img-size", dest="img_size", type=int, nargs="+", default=[2048, 2048], help="Inference resolution")
    predict_parser.add_argument("--split", type=str, default="test", help="Dataset split ('test', 'val', 'train')")
    predict_parser.add_argument("--samples", type=int, default=None, help="Limit to first N samples")
    predict_parser.add_argument("--preprocess-mode", type=str, default="standard", choices=["minimal", "standard", "native"], help="Preprocessing mode matching training pipeline")
    predict_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    predict_parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    predict_parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: predictions/<model_name>)")
    predict_parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    predict_parser.add_argument("--num-classes", type=int, default=None, help="Number of classes")
    predict_parser.add_argument("--num-vis", type=int, default=10, help="Number of visual strips to save")
    predict_parser.add_argument("--no-vis", action="store_true", help="Disable visual strip generation")

    return parser.parse_args(raw_args)


def train(args):
    """Execute model training via unified BaseTrainer."""
    img_size = parse_img_size(args.img_size)

    # Resolve dataset to auto-detect number of classes if not given
    cfg = None
    try:
        cfg = DatasetConfig.resolve(args.data, annotation_file=args.annotation_file)
    except Exception:
        pass

    num_classes = args.num_classes
    if num_classes is None:
        if cfg and cfg.nc:
            num_classes = cfg.nc
        else:
            num_classes = 1

    print("=" * 70)
    print(f"Starting Unified Training Run")
    print(f"  - Model:         {args.model}")
    print(f"  - Dataset:       {args.data}")
    print(f"  - Resolution:    {img_size[0]}x{img_size[1]}")
    print(f"  - Classes:       {num_classes}")
    print(f"  - Epochs:        {args.epochs}")
    print(f"  - Batch Size:    {args.batch_size} (Grad Accum: {args.accumulate_grad_batches})")
    print(f"  - Device Req.:   {args.device}")
    print(f"  - Mixed Prec.:   {args.amp}")
    print(f"  - Loss Type:     {args.loss}")
    print(f"  - Log Interval:  Every {max(1, args.log_interval)} epoch(s)")
    print(f"  - Samples:       {args.samples if args.samples is not None else 'All'}")
    print(f"  - RAM Caching:   {getattr(args, 'cache_ram', False)}")
    print("=" * 70)

    # Build preprocessing config
    if args.preprocess_mode == "minimal":
        preprocess_config = PreprocessConfig.minimal()
    elif args.preprocess_mode == "native":
        preprocess_config = PreprocessConfig.native_resolution()
    else:
        preprocess_config = PreprocessConfig.standard_training(img_size=img_size)

    if args.no_augment and preprocess_config is not None:
        preprocess_config.geometric.flip_horizontal = False
        preprocess_config.geometric.flip_vertical = False
        preprocess_config.geometric.rotate = False

    trainer = BaseTrainer(
        model_cfg=args.model,
        data_root=args.data,
        img_size=img_size,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        val_split=args.val_split,
        num_workers=args.workers,
        use_amp=args.amp,
        use_ema=args.ema,
        resume=args.resume,
        in_channels=args.in_channels,
        num_classes=num_classes,
        preprocess_config=preprocess_config,
        annotation_file=args.annotation_file,
        augment=(not args.no_augment),
        samples=args.samples,
        loss=args.loss,
        cache_ram=getattr(args, "cache_ram", False),
    )

    trainer.train()
    print("[Done] Training completed successfully!")


def validate(args):
    """Execute validation evaluation using BaseValidator."""
    img_size = parse_img_size(args.img_size)
    model_name, num_classes = infer_model_name_and_classes(args.weights, args.model, args.num_classes)

    print(f"Validating Model: {model_name} (Classes: {num_classes})")
    print(f"Weights: {args.weights} | Data: {args.data}")

    model = build_model(
        model=model_name,
        in_channels=args.in_channels,
        num_classes=num_classes,
        verbose=True,
    )
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    load_checkpoint(args.weights, model, device=str(device))

    if getattr(args, "preprocess_mode", "standard") == "minimal":
        preprocess_config = PreprocessConfig.minimal()
    elif getattr(args, "preprocess_mode", "standard") == "native":
        preprocess_config = PreprocessConfig.native_resolution()
    else:
        preprocess_config = PreprocessConfig.standard_validation(img_size=img_size)

    validator = BaseValidator(
        model=model,
        data_root=args.data,
        annotation_file=args.annotation_file,
        img_size=img_size,
        device=args.device,
        num_workers=args.workers,
        save_dir=args.save_dir,
        num_classes=num_classes,
        preprocess_config=preprocess_config,
        cache_ram=getattr(args, "cache_ram", False),
        samples=getattr(args, "samples", None),
    )

    split = getattr(args, "split", None)
    if not split:
        data_str = str(args.data).lower()
        ann_str = str(args.annotation_file).lower() if args.annotation_file else ""
        if "test" in ann_str or "test" in data_str:
            split = "test"
        else:
            split = "val"

    validator.setup_data(split=split)
    validator.validate()
    validator.print_results()

    if args.save_dir:
        validator.save_visualizations(num_samples=4)
        print(f"Visualizations saved to {args.save_dir}")


def predict(args):
    """Execute prediction pipeline using BasePredictor."""
    img_size = parse_img_size(args.img_size)
    model_name, num_classes = infer_model_name_and_classes(args.weights, args.model, args.num_classes)

    output_dir = args.output_dir or f"predictions/{model_name}"

    print(f"Running Inference: {model_name} (Classes: {num_classes})")
    print(f"Weights: {args.weights} -> Output: {output_dir}")

    model = build_model(
        model=model_name,
        in_channels=args.in_channels,
        num_classes=num_classes,
        verbose=False,
    )
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    load_checkpoint(args.weights, model, device=str(device))

    if getattr(args, "preprocess_mode", "standard") == "minimal":
        preprocess_config = PreprocessConfig.minimal()
    elif getattr(args, "preprocess_mode", "standard") == "native":
        preprocess_config = PreprocessConfig.native_resolution()
    else:
        preprocess_config = PreprocessConfig.standard_validation(img_size=img_size)

    predictor = BasePredictor(
        model=model,
        data_root=args.data,
        annotation_file=args.annotation_file,
        img_size=img_size,
        num_classes=num_classes,
        device=args.device,
        num_workers=args.workers,
        threshold=args.threshold,
        output_dir=output_dir,
        weights_name=Path(args.weights).stem,
        preprocess_config=preprocess_config,
    )

    predictor.setup_data(split=args.split, samples=args.samples)
    save_vis = not args.no_vis
    summary = predictor.predict(save_vis=save_vis, num_vis=args.num_vis)
    print(f"[Done] Inference finished: Processed {summary['samples_processed']} images in {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "val":
        validate(args)
    elif args.command == "predict":
        predict(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
