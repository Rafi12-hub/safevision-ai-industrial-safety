"""
Model Export Script for AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
Exports trained models/best.pt to ONNX format for accelerated CPU inference on Intel Integrated Graphics.
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
    parser = argparse.ArgumentParser(description="Export YOLO PPE Detection Model to ONNX")
    parser.add_argument("--model", type=str, default=None, help="Path to PyTorch weights (.pt)")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for export")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axis batch/dimension")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_path = args.model or str(project_root / "models" / "best.pt")

    print("=================================================================")
    print("   Mechanical Lab Safety AI - Model Export Pipeline              ")
    print("=================================================================")
    print(f"Model:   {model_path}")
    print(f"Format:  {args.format}")
    print(f"Size:    {args.imgsz}")
    print("-----------------------------------------------------------------")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}")
        print("Please train a model using 'python training/train.py' first.")
        sys.exit(1)

    model = YOLO(model_path)
    print(f"Exporting model to {args.format.upper()}...")

    exported_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        dynamic=args.dynamic
    )

    print("\n=================================================================")
    print(f"[SUCCESS] Model exported successfully to: {exported_path}")
    print("=================================================================")


if __name__ == "__main__":
    main()

