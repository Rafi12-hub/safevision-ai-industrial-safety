"""
Sample Dataset Generator for Mechanical Laboratory Safety Monitoring System.
Creates starter synthetic images and standard YOLO format annotation labels
(0: person, 1: safety_cap, 2: safety_shoes) for testing the training pipeline.
"""

import os
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"


def draw_synthetic_student(
    canvas_w=640,
    canvas_h=480,
    has_cap=True,
    has_shoes=True,
    head_cutoff=False,
    feet_cutoff=False
):
    """Draw a stylized human silhouette with optional safety cap and safety shoes."""
    img = np.full((canvas_h, canvas_w, 3), (240, 240, 245), dtype=np.uint8)

    # Add workshop background elements (grid lines/lathe bench)
    cv2.line(img, (0, int(canvas_h * 0.75)), (canvas_w, int(canvas_h * 0.75)), (200, 200, 210), 2)
    cv2.rectangle(img, (20, int(canvas_h * 0.65)), (180, int(canvas_h * 0.85)), (180, 190, 200), -1)

    # Student body dimensions
    pw = 140
    ph = 320

    px1 = (canvas_w - pw) // 2
    if head_cutoff:
        py1 = 5  # Cut off at top frame
    elif feet_cutoff:
        py1 = canvas_h - ph + 25  # Cut off at bottom frame
    else:
        py1 = 80
    px2 = px1 + pw
    py2 = py1 + ph

    # Body torso & legs (person silhouette)
    cv2.rectangle(img, (px1 + 15, py1 + 80), (px2 - 15, py1 + 220), (45, 60, 110), -1)  # Blue lab coat
    cv2.rectangle(img, (px1 + 25, py1 + 220), (px1 + 60, py2 - 25), (60, 60, 70), -1)   # Left leg
    cv2.rectangle(img, (px2 - 60, py1 + 220), (px2 - 25, py2 - 25), (60, 60, 70), -1)   # Right leg

    # Head
    head_cx = (px1 + px2) // 2
    head_cy = py1 + 45
    cv2.circle(img, (head_cx, head_cy), 30, (210, 180, 150), -1)

    labels = []

    # Person label (class 0)
    person_cx = ((px1 + px2) / 2.0) / canvas_w
    person_cy = ((py1 + py2) / 2.0) / canvas_h
    person_nw = (px2 - px1) / canvas_w
    person_nh = (py2 - py1) / canvas_h
    labels.append(f"0 {person_cx:.6f} {person_cy:.6f} {person_nw:.6f} {person_nh:.6f}")

    # Safety Cap (class 1)
    if has_cap and not head_cutoff:
        cap_x1 = head_cx - 36
        cap_y1 = head_cy - 34
        cap_x2 = head_cx + 36
        cap_y2 = head_cy - 8
        cv2.ellipse(img, (head_cx, head_cy - 16), (36, 18), 0, 180, 360, (0, 215, 255), -1)  # Yellow safety helmet
        cv2.rectangle(img, (cap_x1, cap_y1 + 14), (cap_x2, cap_y2), (0, 215, 255), -1)

        cap_cx = ((cap_x1 + cap_x2) / 2.0) / canvas_w
        cap_cy = ((cap_y1 + cap_y2) / 2.0) / canvas_h
        cap_nw = (cap_x2 - cap_x1) / canvas_w
        cap_nh = (cap_y2 - cap_y1) / canvas_h
        labels.append(f"1 {cap_cx:.6f} {cap_cy:.6f} {cap_nw:.6f} {cap_nh:.6f}")

    # Safety Shoes (class 2)
    if has_shoes and not feet_cutoff:
        shoe_x1 = px1 + 15
        shoe_y1 = py2 - 30
        shoe_x2 = px2 - 15
        shoe_y2 = py2
        # Heavy protective steel-toe boots (Dark steel brown with orange accents)
        cv2.rectangle(img, (px1 + 18, shoe_y1), (px1 + 65, shoe_y2), (30, 30, 40), -1)
        cv2.rectangle(img, (px2 - 65, shoe_y1), (px2 - 18, shoe_y2), (30, 30, 40), -1)
        cv2.line(img, (px1 + 18, shoe_y2 - 3), (px1 + 65, shoe_y2 - 3), (0, 140, 255), 3)
        cv2.line(img, (px2 - 65, shoe_y2 - 3), (px2 - 18, shoe_y2 - 3), (0, 140, 255), 3)

        shoe_cx = ((shoe_x1 + shoe_x2) / 2.0) / canvas_w
        shoe_cy = ((shoe_y1 + shoe_y2) / 2.0) / canvas_h
        shoe_nw = (shoe_x2 - shoe_x1) / canvas_w
        shoe_nh = (shoe_y2 - shoe_y1) / canvas_h
        labels.append(f"2 {shoe_cx:.6f} {shoe_cy:.6f} {shoe_nw:.6f} {shoe_nh:.6f}")

    return img, labels


def generate_dataset():
    print("Generating starter dataset in standard YOLO format...")
    splits = [("train", 12), ("val", 4)]

    for split_name, count in splits:
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        scenarios = [
            (True, True, False, False),   # Fully compliant
            (True, False, False, False),  # Missing shoes
            (False, True, False, False),  # Missing cap
            (False, False, False, False), # Missing both
        ]

        for i in range(count):
            cap, shoes, head_cut, feet_cut = scenarios[i % len(scenarios)]
            img, labels = draw_synthetic_student(
                has_cap=cap,
                has_shoes=shoes,
                head_cutoff=head_cut,
                feet_cutoff=feet_cut
            )

            img_path = img_dir / f"student_sample_{i+1:03d}.jpg"
            lbl_path = lbl_dir / f"student_sample_{i+1:03d}.txt"

            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))

        print(f"Generated {count} images and annotations for split '{split_name}'")

    print("\nDataset generation completed successfully!")
    print(f"Location: {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    generate_dataset()
