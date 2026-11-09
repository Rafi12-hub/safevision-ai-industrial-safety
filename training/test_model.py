"""
Model Testing Script for AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
Runs inference on test images using models/best.pt and saves visually annotated output
to training/runs/test_predictions/.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Ensure Matplotlib uses writable temp dir
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test YOLO PPE Detection Model on sample images")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--source", type=str, default=None, help="Path to image, directory, or video")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu')")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_path = args.model or str(project_root / "models" / "best.pt")
    
    # Default source: dataset/images/val
    source_path = args.source or str(project_root / "dataset" / "images" / "val")
    output_dir = project_root / "training" / "runs" / "test_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("   Mechanical Lab Safety AI - Model Testing Pipeline             ")
    print("=================================================================")
    print(f"Model:       {model_path}")
    print(f"Source:      {source_path}")
    print(f"Confidence:  {args.conf}")
    print(f"Output dir:  {output_dir}")
    print("-----------------------------------------------------------------")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}")
        print("Please train a model using 'python training/train.py' first.")
        sys.exit(1)

    if not os.path.exists(source_path):
        print(f"[ERROR] Test source not found at: {source_path}")
        sys.exit(1)

    model = YOLO(model_path)
    print("Running inference and saving visual predictions...")

    results = model.predict(
        source=source_path,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(project_root / "training" / "runs"),
        name="test_predictions",
        exist_ok=True
    )

    print("\n=================================================================")
    print(f"[SUCCESS] Inference complete! Annotated images saved to: {output_dir}")
    print("=================================================================")


if __name__ == "__main__":
    main()

