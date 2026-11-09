"""
Camera Management and Video Processing Engine for Mechanical Laboratory Safety Monitoring.
Handles webcam capture on Windows 11, frame resizing, FPS tracking, status reporting,
and real-time visualization overlay rendering (bounding boxes, student IDs, PPE badges).
"""

import cv2
import time
import threading
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from safety_checker import STATUS_RGB, SafetyStatus


class CameraManager:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.last_annotated_frame: Optional[np.ndarray] = None
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()
        self.error_message: Optional[str] = None
        self.worker_thread: Optional[threading.Thread] = None

    def start(self, camera_index: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[bool, str]:
        """Start the video capture device."""
        with self.lock:
            if self.is_active and self.cap and self.cap.isOpened():
                return True, "Camera is already running"

            if camera_index is not None:
                self.camera_index = camera_index
            if width is not None:
                self.requested_width = width
            if height is not None:
                self.requested_height = height

            try:
                # Use CAP_DSHOW on Windows for fast webcam initialization
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap.isOpened():
                    self.error_message = f"Unable to open camera device index {self.camera_index}. Please verify webcam connection."
                    self.is_active = False
                    return False, self.error_message

                # Configure resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                self.is_active = True
                self.error_message = None
                self.frame_count = 0
                self.fps_timer = time.time()

                # Start background frame grabber
                self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.worker_thread.start()

                return True, "Camera started successfully"
            except Exception as e:
                self.is_active = False
                self.error_message = f"Camera initialization exception: {str(e)}"
                return False, self.error_message

    def _capture_loop(self) -> None:
        """Background loop to continuously read fresh frames from the webcam."""
        while self.is_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.error_message = "Camera frame read failed. Hardware disconnected or in use by another application."
                self.is_active = False
                break

            with self.lock:
                self.current_frame = frame
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.fps_timer
                if elapsed >= 1.0:
                    self.fps = round(self.frame_count / elapsed, 1)
                    self.frame_count = 0
                    self.fps_timer = now

            time.sleep(0.01)

    def stop(self) -> Tuple[bool, str]:
        """Stop video capture and release hardware resources."""
        with self.lock:
            self.is_active = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.current_frame = None
            self.last_annotated_frame = None
            self.fps = 0.0
            return True, "Camera stopped successfully"

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently acquired frame (thread-safe)."""
        with self.lock:
            if not self.is_active or self.current_frame is None:
                return False, None
            return True, self.current_frame.copy()

    def get_status(self) -> Dict[str, Any]:
        """Return real-time diagnostic status of the camera."""
        return {
            "is_active": self.is_active,
            "camera_index": self.camera_index,
            "resolution": f"{self.requested_width}x{self.requested_height}",
            "fps": self.fps,
            "has_error": self.error_message is not None,
            "error_message": self.error_message
        }

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        students: List[Dict[str, Any]],
        fps: float = 0.0,
        model_name: str = "PPE AI Detector"
    ) -> np.ndarray:
        """
        Draw high-visibility safety verification overlays:
        - Colored bounding box around student body (Green = Safe, Red = Not Safe, Orange = Unknown)
        - Student ID badge, status tag, and individual PPE badges
        - Live HUD showing FPS, active student count, and system status
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for s in students:
            status = s.get("status", SafetyStatus.UNKNOWN)
            bgr_color = STATUS_RGB.get(status, (11, 158, 245))  # Default orange

            box = s["person_box"]
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw main body bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 3)

            # Header badge above head
            tracking_id = s.get("tracking_id", "?")
            status_text = f"STUDENT #{tracking_id}: {status}"
            badge_h = 28
            header_y1 = max(0, y1 - badge_h)
            header_y2 = y1

            # Header background
            cv2.rectangle(annotated, (x1, header_y1), (min(w, x1 + 240), header_y2), bgr_color, -1)
            cv2.putText(
                annotated,
                status_text,
                (x1 + 6, header_y1 + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Cap badge
            cap_ok = s.get("cap_detected", False)
            cap_text = f"CAP: {'YES' if cap_ok else 'NO'}"
            cap_bg = (16, 185, 129) if cap_ok else (68, 68, 239)
            cv2.rectangle(annotated, (x1, y1 + 4), (x1 + 85, y1 + 26), cap_bg, -1)
            cv2.putText(
                annotated,
                cap_text,
                (x1 + 4, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Shoes badge
            shoes_ok = s.get("shoes_detected", False)
            shoes_text = f"SHOES: {'YES' if shoes_ok else 'NO'}"
            shoes_bg = (16, 185, 129) if shoes_ok else (68, 68, 239)
            cv2.rectangle(annotated, (x1 + 90, y1 + 4), (x1 + 190, y1 + 26), shoes_bg, -1)
            cv2.putText(
                annotated,
                shoes_text,
                (x1 + 94, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # If head or feet out of frame, add warning tag
            if not s.get("head_visible", True) or not s.get("feet_visible", True):
                warn_text = "BORDER OCCLUSION"
                cv2.rectangle(annotated, (x1, y2 - 24), (x1 + 175, y2), (11, 158, 245), -1)
                cv2.putText(
                    annotated,
                    warn_text,
                    (x1 + 5, y2 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

        # Draw Top HUD Banner
        hud_bg = (20, 24, 33)
        cv2.rectangle(annotated, (0, 0), (w, 36), hud_bg, -1)

        cv2.putText(
            annotated,
            "MECH LAB SAFETY AI - LIVE MONITOR",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        hud_info = f"FPS: {fps:.1f} | Students: {len(students)}"
        cv2.putText(
            annotated,
            hud_info,
            (w - 210, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (52, 211, 153),
            2,
            cv2.LINE_AA
        )

        return annotated
