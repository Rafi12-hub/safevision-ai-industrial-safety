"""
Custom YOLO Model Training Script for Mechanical Laboratory Safety Monitoring System.
Trains YOLO11n on the 3-class PPE dataset (0: person, 1: safety_cap, 2: safety_shoes)
and automatically exports the final checkpoint to models/best.pt.
"""

import os
import sys
import shutil
import tempfile
import argparse
from pathlib import Path

# Ensure Matplotlib uses a writable temp cache directory
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for Mechanical Lab PPE Detection")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if CPU memory is limited)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or '0' for CUDA GPU)")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt", help="Pretrained base checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "dataset"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate a resolved data.yaml with absolute paths for Ultralytics
    resolved_yaml = script_dir / "data_resolved.yaml"
    with open(script_dir / "data.yaml", "r") as f:
        content = f.read()
    
    # Ensure path points to absolute dataset_dir
    lines = []
    for line in content.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: '{dataset_dir.as_posix()}'")
        else:
            lines.append(line)
    
    with open(resolved_yaml, "w") as f:
        f.write("\n".join(lines))

    data_yaml = args.data or str(resolved_yaml)

    print("=================================================================")
    print("   Mechanical Lab Safety AI - YOLO Model Training Pipeline       ")
    print("=================================================================")
    print(f"Data config:    {data_yaml}")
    print(f"Base model:     {args.base_model}")
    print(f"Target device:  {args.device}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch}")
    print(f"Image size:     {args.imgsz}")
    print("-----------------------------------------------------------------")

    # Verify dataset exists and contains images
    dataset_dir = project_root / "dataset"
    train_images = list((dataset_dir / "images" / "train").glob("*.*"))
    val_images = list((dataset_dir / "images" / "val").glob("*.*"))

    print(f"Found {len(train_images)} training images and {len(val_images)} validation images.")

    if len(train_images) == 0:
        print("\n[WARNING] No training images found in dataset/images/train/!")
        print("Please place your annotated images into dataset/images/train and annotations in dataset/labels/train.")
        print("You can run 'python training/generate_sample_dataset.py' to generate sample images for testing.")
        sys.exit(1)

    # Initialize model
    print(f"\nLoading base model: {args.base_model}...")
    model = YOLO(args.base_model)

    # Run training
    print("\nStarting training session...")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project_root / "training" / "runs"),
        name="ppe_training",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Locate and copy best.pt to models/best.pt
    run_dir = project_root / "training" / "runs" / "ppe_training"
    trained_best = run_dir / "weights" / "best.pt"

    target_best = models_dir / "best.pt"
    if trained_best.exists():
        shutil.copy(trained_best, target_best)
        print("\n=================================================================")
        print(f"[SUCCESS] Trained model saved to: {target_best}")
        print("The backend will automatically load this model on next startup!")
        print("=================================================================")
    else:
        print(f"\n[NOTICE] Training completed. Check weights in {run_dir / 'weights'}")


if __name__ == "__main__":
    main()
