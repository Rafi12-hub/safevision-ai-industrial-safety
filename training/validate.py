"""
Model Validation Script for Mechanical Laboratory Safety Monitoring System.
Evaluates models/best.pt against dataset/images/val and outputs validation metrics:
mAP50, mAP50-95, precision, recall per class (person, safety_cap, safety_shoes).
"""

import os
import sys
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


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO PPE Detection Model")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu')")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "dataset"
    model_path = args.model or str(project_root / "models" / "best.pt")

    # Generate a resolved data.yaml with absolute paths for Ultralytics
    resolved_yaml = script_dir / "data_resolved.yaml"
    with open(script_dir / "data.yaml", "r") as f:
        content = f.read()

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
    print("   Mechanical Lab Safety AI - Model Validation Report           ")
    print("=================================================================")
    print(f"Model path:    {model_path}")
    print(f"Data config:   {data_yaml}")
    print(f"Device:        {args.device}")
    print("-----------------------------------------------------------------")

    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model weights not found at: {model_path}")
        print("Please train a model using 'python training/train.py' first.")
        sys.exit(1)

    model = YOLO(model_path)
    print("Running validation on validation dataset split...")

    runs_dir = project_root / "training" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        device=args.device,
        project=str(runs_dir),
        name="val",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    print("\n----------------- SUMMARY VALIDATION METRICS -------------------")
    print(f"Mean Precision:        {metrics.box.mp:.4f}")
    print(f"Mean Recall:           {metrics.box.mr:.4f}")
    print(f"mAP @ 0.50:            {metrics.box.map50:.4f}")
    print(f"mAP @ 0.50:0.95:       {metrics.box.map:.4f}")
    print("=================================================================")


if __name__ == "__main__":
    main()
