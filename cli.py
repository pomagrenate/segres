# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import yaml
from pathlib import Path
import sys

from models import SegmentationModel
from engine.trainer import BaseTrainer
from engine.validator import BaseValidator
from engine.predictor import BasePredictor
from utils import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="High-Resolution Segmentation Framework")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a segmentation model")
    train_parser.add_argument("--model", type=str, default="cfg/models/seg_medium.yaml", help="Model configuration YAML (nano, small, medium, large, xlarge)")
    train_parser.add_argument("--data", type=str, required=True, help="Dataset root directory")
    train_parser.add_argument("--cfg", type=str, default="cfg/default.yaml", help="Default configuration YAML")
    train_parser.add_argument("--img-size", type=int, nargs=2, default=[1024, 1024], help="Image size (height width)")
    train_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    train_parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    train_parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    train_parser.add_argument("--ema", action="store_true", help="Use exponential moving average")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    train_parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    
    # Validate command
    val_parser = subparsers.add_parser("val", help="Validate a segmentation model")
    val_parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    val_parser.add_argument("--model", type=str, default="cfg/models/seg_medium.yaml", help="Model configuration YAML (nano, small, medium, large, xlarge)")
    val_parser.add_argument("--data", type=str, required=True, help="Dataset root directory")
    val_parser.add_argument("--img-size", type=int, nargs=2, default=[1024, 1024], help="Image size (height width)")
    val_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    val_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    val_parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    val_parser.add_argument("--save-dir", type=str, default=None, help="Directory to save visualizations")
    val_parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    val_parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on images")
    predict_parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    predict_parser.add_argument("--model", type=str, default="cfg/models/seg_medium.yaml", help="Model configuration YAML (nano, small, medium, large, xlarge)")
    predict_parser.add_argument("--data", type=str, required=True, help="Dataset root directory")
    predict_parser.add_argument("--img-size", type=int, nargs=2, default=[1024, 1024], help="Image size (height width)")
    predict_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    predict_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    predict_parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    predict_parser.add_argument("--min-area", type=int, default=30, help="Minimum component area")
    predict_parser.add_argument("--close-kernel", type=int, default=3, help="Morphological closing kernel size")
    predict_parser.add_argument("--output-dir", type=str, default="predictions", help="Output directory")
    predict_parser.add_argument("--output-format", type=str, default="image", choices=["image", "rle"], help="Output format")
    predict_parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    predict_parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    
    return parser.parse_args()


def load_config(cfg_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def train(args):
    """Train a segmentation model."""
    print(f"Starting training with model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # Initialize trainer
    trainer = BaseTrainer(
        model_cfg=args.model,
        data_root=args.data,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        val_split=args.val_split,
        num_workers=args.workers,
        use_amp=args.amp,
        use_ema=args.ema,
        resume=args.resume,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )
    
    # Start training
    trainer.train()
    
    print("Training completed!")


def validate(args):
    """Validate a segmentation model."""
    print(f"Validating model: {args.weights}")
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    
    # Load model
    model = SegmentationModel(
        cfg=args.model,
        ch=args.in_channels,
        nc=args.num_classes,
        verbose=True
    )
    
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    load_checkpoint(args.weights, model, device=str(device))
    
    # Initialize validator
    validator = BaseValidator(
        model=model,
        data_root=args.data,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.workers,
        save_dir=args.save_dir,
    )
    
    # Setup data and validate
    validator.setup_data(split="val")
    metrics = validator.validate()
    
    # Print results
    validator.print_results()
    
    # Save visualizations if requested
    if args.save_dir:
        validator.save_visualizations(num_samples=4)
        print(f"Visualizations saved to {args.save_dir}")


def predict(args):
    """Run inference on images."""
    print(f"Running inference with model: {args.weights}")
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Output format: {args.output_format}")
    
    import torch
    
    # Load model
    model = SegmentationModel(
        cfg=args.model,
        ch=args.in_channels,
        nc=args.num_classes,
        verbose=True
    )
    
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    load_checkpoint(args.weights, model, device=str(device))
    
    # Initialize predictor
    predictor = BasePredictor(
        model=model,
        data_root=args.data,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.workers,
        threshold=args.threshold,
        min_area=args.min_area,
        close_kernel=args.close_kernel,
    )
    
    # Setup data and predict
    predictor.setup_data(split="test")
    predictions = predictor.predict()
    
    # Save predictions
    if args.output_format == "image":
        predictor.save_predictions(predictions, args.output_dir)
        print(f"Predictions saved as images to {args.output_dir}")
    else:
        output_file = Path(args.output_dir) / "predictions.csv"
        predictor.save_predictions_rle(predictions, str(output_file))
        print(f"Predictions saved as RLE to {output_file}")
    
    print(f"Processed {len(predictions)} images")


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
