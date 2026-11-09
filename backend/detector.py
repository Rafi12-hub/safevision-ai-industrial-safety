"""
AI Object Detection Module for AI-Based Mechanical Laboratory Safety Monitoring System.
Loads YOLO11n or custom trained best.pt model, runs CPU-optimized inference,
extracts bounding boxes for Persons, Safety Caps, and Safety Shoes,
and tracks model loading status transparently without faking detections.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


CUSTOM_CLASSES = {
    0: "person",
    1: "safety_cap",
    2: "safety_shoes"
}

BASE_YOLO_PERSON_CLASS = 0  # Class 0 in COCO is person


class SafetyDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        imgsz: int = 480,
        device: str = "cpu"
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.model = None
        self.is_custom_model = False
        self.model_name = "Not Loaded"
        self.model_path = model_path
        self.status_message = "Initializing detection engine..."
        self.last_inference_ms = 0.0
        self.classes = CUSTOM_CLASSES

        # Automatic model resolution
        self.load_model(model_path)

    def load_model(self, custom_path: Optional[str] = None) -> bool:
        """
        Load custom best.pt if available; otherwise load base yolo11n for person tracking
        with clear pending-training status.
        """
        if not ULTRALYTICS_AVAILABLE:
            self.model = None
            self.status_message = "Ultralytics YOLO package is not installed."
            return False

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        best_pt_path = custom_path or os.path.join(root_dir, "models", "best.pt")

        # 1. Try to load custom trained best.pt
        if os.path.exists(best_pt_path):
            try:
                self.model = YOLO(best_pt_path)
                self.model.to(self.device)
                self.model_path = best_pt_path
                self.model_name = "Custom PPE Model (best.pt)"

                # Inspect model classes
                if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                    self.classes = self.model.names
                    # Check if safety classes are present
                    has_cap = any("cap" in str(v).lower() or "hat" in str(v).lower() for v in self.classes.values())
                    has_shoes = any("shoe" in str(v).lower() or "boot" in str(v).lower() for v in self.classes.values())
                    self.is_custom_model = has_cap and has_shoes
                else:
                    self.is_custom_model = True

                self.status_message = "Custom PPE best.pt model loaded successfully (CPU inference active)."
                return True
            except Exception as e:
                self.status_message = f"Error loading custom model from {best_pt_path}: {str(e)}"

        # 2. Fallback to standard YOLO11n / YOLOv8n base model for student person detection
        try:
            # Look for local base model first or download yolo11n.pt
            base_model_path = os.path.join(root_dir, "models", "yolo11n.pt")
            if not os.path.exists(base_model_path):
                base_model_path = "yolo11n.pt"

            self.model = YOLO(base_model_path)
            self.model.to(self.device)
            self.model_name = "YOLO11n Base (Person detection active, custom best.pt PPE model pending training)"
            self.is_custom_model = False
            self.classes = {
                0: "person",
                1: "safety_cap (training pending)",
                2: "safety_shoes (training pending)"
            }
            self.status_message = (
                "Standard YOLO11n active for person detection. "
                "Custom PPE dataset training is ready to produce best.pt."
            )
            return True
        except Exception as e:
            self.model = None
            self.is_custom_model = False
            self.model_name = "Model Unavailable"
            self.status_message = f"Failed to initialize YOLO model: {str(e)}"
            return False

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic model status for settings and API."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path if self.model_path and os.path.exists(self.model_path) else "None",
            "is_custom_model": self.is_custom_model,
            "is_loaded": self.model is not None,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.classes,
            "last_inference_ms": round(self.last_inference_ms, 1),
            "status_message": self.status_message
        }

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single video frame (OpenCV BGR format).
        Returns categorized detection lists:
        - persons: List of person dicts with bbox and confidence
        - caps: List of cap dicts with bbox and confidence
        - shoes: List of shoe dicts with bbox and confidence
        """
        if self.model is None:
            return {
                "persons": [],
                "caps": [],
                "shoes": [],
                "error": "Model not loaded",
                "inference_ms": 0.0
            }

        t0 = time.time()
        try:
            # Run CPU inference with tracking or standard prediction
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
            self.last_inference_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            return {
                "persons": [],
                "caps": [],
                "shoes": [],
                "error": f"Inference failure: {str(e)}",
                "inference_ms": 0.0
            }

        persons: List[Dict[str, Any]] = []
        caps: List[Dict[str, Any]] = []
        shoes: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return {
                "persons": persons,
                "caps": caps,
                "shoes": shoes,
                "inference_ms": self.last_inference_ms
            }

        r = results[0]
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                bbox = [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])]
                conf = float(confs[i])
                cls_id = int(clss[i])

                if self.is_custom_model:
                    # Target classes: 0 = person, 1 = safety_cap, 2 = safety_shoes
                    if cls_id == 0:
                        persons.append({"bbox": bbox, "confidence": conf, "class_id": 0, "name": "person"})
                    elif cls_id == 1:
                        caps.append({"bbox": bbox, "confidence": conf, "class_id": 1, "name": "safety_cap"})
                    elif cls_id == 2:
                        shoes.append({"bbox": bbox, "confidence": conf, "class_id": 2, "name": "safety_shoes"})
                else:
                    # In base COCO model, class 0 is person
                    if cls_id == BASE_YOLO_PERSON_CLASS:
                        persons.append({"bbox": bbox, "confidence": conf, "class_id": 0, "name": "person"})

        return {
            "persons": persons,
            "caps": caps,
            "shoes": shoes,
            "inference_ms": self.last_inference_ms
        }
