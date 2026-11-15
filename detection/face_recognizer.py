"""
Face Recognition and Personnel Identification Engine.
Extracts illumination-invariant HOG feature embeddings from human faces,
matches against registered worker biometric records via cosine similarity,
and outputs real worker names and departments in real-time.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

import config
from database.workers import (
    add_worker,
    get_all_workers_with_encodings,
    get_all_workers,
    delete_worker
)


class FaceRecognizer:
    def __init__(self, match_threshold: float = config.DEFAULT_FACE_MATCH_THRESHOLD):
        self.match_threshold = match_threshold
        # Standard HOG descriptor with safe cloud initialization
        self.hog = None
        try:
            if hasattr(cv2, "HOGDescriptor"):
                self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        except Exception:
            self.hog = None
        self.cached_workers: List[Dict[str, Any]] = []
        self.last_reload = 0.0
        self.reload_workers()

    def reload_workers(self) -> int:
        """Reload registered worker biometrics from SQLite into memory cache."""
        try:
            self.cached_workers = get_all_workers_with_encodings()
            self.last_reload = time.time()
            return len(self.cached_workers)
        except Exception:
            self.cached_workers = []
            return 0

    def extract_encoding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize face crop and compute normalized HOG feature descriptor.
        Returns float32 unit vector of shape (1764,) or None if invalid.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        h, w = face_bgr.shape[:2]
        if h < 20 or w < 20:
            return None

        try:
            # 1. Resize to canonical 64x64 window
            resized = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)

            # 2. Convert to Grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting invariance
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            norm_gray = clahe.apply(gray)

            # 4. Compute HOG gradient descriptor or gradient fallback
            if self.hog is not None and hasattr(self.hog, "compute"):
                desc = self.hog.compute(norm_gray).flatten().astype(np.float32)
            else:
                gx = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 1, ksize=1)
                mag = cv2.magnitude(gx, gy)
                desc = cv2.resize(mag, (42, 42)).flatten().astype(np.float32)

            # 5. L2 unit vector normalization
            norm = np.linalg.norm(desc)
            if norm > 1e-6:
                return desc / norm
            return None
        except Exception:
            return None

    def identify_face(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Compare face features against all registered workers using cosine similarity.
        Returns matched identity dictionary with real worker name or Unregistered fallback.
        """
        encoding = self.extract_encoding(face_bgr)

        unregistered_result = {
            "is_recognized": False,
            "worker_id": None,
            "full_name": "Unregistered Worker",
            "role": "General Personnel / Visitor",
            "department": "Mechanical Lab",
            "confidence": 0.0,
            "photo_path": ""
        }

        if encoding is None or not self.cached_workers:
            return unregistered_result

        best_sim = -1.0
        best_worker: Optional[Dict[str, Any]] = None

        for w in self.cached_workers:
            w_enc = w.get("encoding_array")
            if w_enc is not None and len(w_enc) == len(encoding):
                # Cosine similarity between two unit vectors = dot product
                sim = float(np.dot(encoding, w_enc))
                if sim > best_sim:
                    best_sim = sim
                    best_worker = w

        if best_worker is not None and best_sim >= self.match_threshold:
            return {
                "is_recognized": True,
                "worker_id": best_worker["worker_id"],
                "full_name": best_worker["full_name"],
                "role": best_worker["role"],
                "department": best_worker["department"],
                "confidence": round(best_sim, 3),
                "photo_path": best_worker.get("photo_path", "")
            }

        # Face detected but not matched to registered personnel
        unregistered_result["confidence"] = round(max(0.0, best_sim), 3) if best_sim > 0 else 0.0
        return unregistered_result

    def register_new_worker(
        self,
        worker_id: str,
        full_name: str,
        role: str,
        department: str,
        face_bgr: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Register a new worker with facial photo and extract biometric encoding.
        """
        if not worker_id or not full_name:
            return False, "Worker ID and Full Name are required."

        encoding = self.extract_encoding(face_bgr)
        if encoding is None:
            return False, "Could not extract clear facial features. Please ensure face is centered and clear."

        # Save photo file
        clean_id = worker_id.strip().replace(" ", "_")
        photo_filename = f"{clean_id}.jpg"
        photo_path = config.WORKER_FACES_DIR / photo_filename
        try:
            cv2.imwrite(str(photo_path), face_bgr)
        except Exception as e:
            return False, f"Failed to save profile photo: {str(e)}"

        # Save to database
        try:
            enc_bytes = encoding.tobytes()
            add_worker(
                worker_id=worker_id,
                full_name=full_name,
                role=role or "Technician",
                department=department or "Mechanical Laboratory",
                photo_path=str(photo_path),
                face_encoding=enc_bytes
            )
            self.reload_workers()
            return True, f"Worker '{full_name}' ({worker_id}) registered successfully!"
        except Exception as e:
            return False, f"Database error during registration: {str(e)}"
