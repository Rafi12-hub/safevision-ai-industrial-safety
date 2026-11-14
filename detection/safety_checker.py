"""
Safety Verification and Spatial Association Engine for
AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
Performs:
1. Spatial association: helmets in head zone, shoes in feet zone.
2. Boundary occlusion checks: head/feet truncated by camera boundary -> UNKNOWN (Yellow).
3. Temporal stability (hysteresis) filtering to prevent flicker.
4. Clean visual HUD frame annotation with OpenCV.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import cv2
import numpy as np
import config
from config import SafetyStatus, STATUS_COLORS, STATUS_BGR


class SafetyVerifier:
    def __init__(
        self,
        frame_edge_margin: int = config.FRAME_EDGE_MARGIN,
        hysteresis_window: int = config.HYSTERESIS_WINDOW,
        allow_safety_cap: bool = config.ALLOW_SAFETY_CAP
    ):
        self.frame_edge_margin = frame_edge_margin
        self.hysteresis_window = hysteresis_window
        self.allow_safety_cap = allow_safety_cap
        self.history: Dict[str, deque] = {}

    @staticmethod
    def _box_center(bbox: List[float]) -> Tuple[float, float]:
        """Compute (cx, cy) center of bounding box."""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _is_point_in_box(point: Tuple[float, float], box: List[float]) -> bool:
        """Check if (x, y) point is inside [x1, y1, x2, y2]."""
        x, y = point
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    @staticmethod
    def _compute_overlap_ratio(item_box: List[float], zone_box: List[float]) -> float:
        """Calculate intersection area / item area."""
        ix1 = max(item_box[0], zone_box[0])
        iy1 = max(item_box[1], zone_box[1])
        ix2 = min(item_box[2], zone_box[2])
        iy2 = min(item_box[3], zone_box[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        item_area = max(1.0, (item_box[2] - item_box[0]) * (item_box[3] - item_box[1]))
        return intersection / item_area

    def check_visibility(
        self,
        person_box: List[float],
        frame_width: int,
        frame_height: int
    ) -> Tuple[bool, bool, str]:
        """
        Check if worker's head and feet are reliably visible inside the camera frame.
        If cut off at the border, flags UNKNOWN instead of false SAFE or NOT SAFE.
        """
        px1, py1, px2, py2 = person_box
        head_visible = True
        feet_visible = True
        reasons = []

        # Check top boundary occlusion (head truncated)
        if py1 <= self.frame_edge_margin:
            head_visible = False
            reasons.append("Head partially out of top camera frame")

        # Check bottom boundary occlusion (feet truncated)
        if py2 >= (frame_height - self.frame_edge_margin):
            feet_visible = False
            reasons.append("Feet partially out of bottom camera frame")

        # Check if worker is too small / too distant for reliable PPE evaluation
        ph = py2 - py1
        if ph < 80:
            head_visible = False
            feet_visible = False
            reasons.append("Worker too distant from camera for reliable PPE check")

        reason_str = "; ".join(reasons) if reasons else "Fully in camera view"
        return head_visible, feet_visible, reason_str

    def associate_equipment(
        self,
        person_box: List[float],
        helmet_boxes: List[Dict[str, Any]],
        cap_boxes: List[Dict[str, Any]],
        shoe_boxes: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
        """
        Spatially associate safety equipment with this specific worker.
        - Helmet/Cap must overlap Head Zone (top ~32% of body box).
        - Shoes must overlap Feet Zone (bottom ~32% of body box).
        - Equipment lying elsewhere in the background is rejected.
        """
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1

        # Define Head Zone
        head_zone = [
            max(0.0, px1 - 0.20 * pw),
            max(0.0, py1 - 0.15 * ph),
            min(float(frame_width), px2 + 0.20 * pw),
            py1 + 0.35 * ph
        ]

        # Define Feet Zone
        feet_zone = [
            max(0.0, px1 - 0.25 * pw),
            py2 - 0.35 * ph,
            min(float(frame_width), px2 + 0.25 * pw),
            min(float(frame_height), py2 + 0.15 * ph)
        ]

        # Match Head Protection: hard helmet preferred, or approved safety cap
        matched_head: Optional[Dict[str, Any]] = None
        best_head_score = 0.0
        head_type = "NONE"

        # Check hard helmets first
        for h in helmet_boxes:
            box = h["bbox"]
            center = self._box_center(box)
            overlap = self._compute_overlap_ratio(box, head_zone)
            if self._is_point_in_box(center, head_zone) or overlap > 0.30:
                score = (h.get("confidence", 0.5) * 0.5) + (overlap * 0.5)
                if score > best_head_score:
                    best_head_score = score
                    matched_head = h
                    head_type = "HELMET"

        # Check safety caps if allowed by policy and no hard helmet found
        if matched_head is None and self.allow_safety_cap:
            for c in cap_boxes:
                box = c["bbox"]
                center = self._box_center(box)
                overlap = self._compute_overlap_ratio(box, head_zone)
                if self._is_point_in_box(center, head_zone) or overlap > 0.30:
                    score = (c.get("confidence", 0.5) * 0.5) + (overlap * 0.5)
                    if score > best_head_score:
                        best_head_score = score
                        matched_head = c
                        head_type = "CAP"

        # Match Safety Shoes on feet zone
        matched_shoes: Optional[Dict[str, Any]] = None
        best_shoe_score = 0.0

        for s in shoe_boxes:
            box = s["bbox"]
            center = self._box_center(box)
            overlap = self._compute_overlap_ratio(box, feet_zone)
            if self._is_point_in_box(center, feet_zone) or overlap > 0.30:
                score = (s.get("confidence", 0.5) * 0.5) + (overlap * 0.5)
                if score > best_shoe_score:
                    best_shoe_score = score
                    matched_shoes = s

        return matched_head, matched_shoes, head_type

    def evaluate_worker(
        self,
        worker_id: str,
        person_box: List[float],
        person_confidence: float,
        helmet_boxes: List[Dict[str, Any]],
        cap_boxes: List[Dict[str, Any]],
        shoe_boxes: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        face_box: Optional[List[float]] = None,
        faces: Optional[List[Dict[str, Any]]] = None,
        worker_name: Optional[str] = None,
        role: Optional[str] = None,
        department: Optional[str] = None,
        is_recognized: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate full safety verification logic for a detected worker:
        - Checks boundary visibility.
        - Spatially associates worn equipment.
        - Identifies and associates human face scanning data and real worker identity.
        - Smooths via hysteresis buffer.
        """
        # Associate face box if provided or extract matching face
        matched_face = face_box
        eff_name = worker_name or worker_id
        eff_role = role or "Technician"
        eff_dept = department or "Mechanical Lab"
        eff_is_rec = is_recognized

        if faces:
            px1, py1, px2, py2 = person_box
            for f in faces:
                fb = f["bbox"]
                fcx = (fb[0] + fb[2]) / 2.0
                fcy = (fb[1] + fb[3]) / 2.0
                if px1 <= fcx <= px2 and py1 <= fcy <= (py1 + 0.45 * (py2 - py1)):
                    if matched_face is None:
                        matched_face = fb
                    if not eff_is_rec and f.get("is_recognized"):
                        eff_name = f.get("full_name", eff_name)
                        eff_role = f.get("role", eff_role)
                        eff_dept = f.get("department", eff_dept)
                        eff_is_rec = True
                    break

        # Step 1: Visibility check
        head_visible, feet_visible, vis_reason = self.check_visibility(
            person_box, frame_width, frame_height
        )

        # Step 2: Spatial association
        matched_head, matched_shoes, head_type = self.associate_equipment(
            person_box, helmet_boxes, cap_boxes, shoe_boxes, frame_width, frame_height
        )

        helmet_detected = matched_head is not None
        shoes_detected = matched_shoes is not None

        head_conf = matched_head["confidence"] if matched_head else 0.0
        shoes_conf = matched_shoes["confidence"] if matched_shoes else 0.0

        # Step 3: Raw frame status
        if not head_visible or not feet_visible:
            raw_status = SafetyStatus.UNKNOWN
            helmet_status_label = "UNCLEAR" if not head_visible else ("WORN" if helmet_detected else "MISSING")
            shoes_status_label = "UNCLEAR" if not feet_visible else ("WORN" if shoes_detected else "MISSING")
            details = f"UNKNOWN: {vis_reason}"
        elif helmet_detected and shoes_detected:
            raw_status = SafetyStatus.SAFE
            helmet_status_label = "WORN"
            shoes_status_label = "WORN"
            details = f"SAFE: Both {'Safety Helmet' if head_type == 'HELMET' else 'Safety Cap'} and Safety Shoes verified"
        else:
            raw_status = SafetyStatus.NOT_SAFE
            helmet_status_label = "WORN" if helmet_detected else "MISSING"
            shoes_status_label = "WORN" if shoes_detected else "MISSING"
            missing = []
            if not helmet_detected:
                missing.append("Safety Helmet")
            if not shoes_detected:
                missing.append("Safety Shoes")
            details = f"NOT SAFE: Missing {', '.join(missing)}"

        # Step 4: Temporal Stabilization Buffer
        if worker_id not in self.history:
            self.history[worker_id] = deque(maxlen=self.hysteresis_window)

        self.history[worker_id].append(raw_status)
        window = list(self.history[worker_id])
        total_w = len(window)

        safe_cnt = window.count(SafetyStatus.SAFE)
        unsafe_cnt = window.count(SafetyStatus.NOT_SAFE)
        unknown_cnt = window.count(SafetyStatus.UNKNOWN)

        if total_w < 3:
            stabilized_status = raw_status
            stability = 0.5
        elif (unsafe_cnt / total_w) >= 0.55:
            stabilized_status = SafetyStatus.NOT_SAFE
            stability = unsafe_cnt / total_w
        elif (safe_cnt / total_w) >= 0.55:
            stabilized_status = SafetyStatus.SAFE
            stability = safe_cnt / total_w
        elif (unknown_cnt / total_w) >= 0.5:
            stabilized_status = SafetyStatus.UNKNOWN
            stability = unknown_cnt / total_w
        else:
            stabilized_status = raw_status
            stability = 0.6

        return {
            "worker_id": worker_id,
            "worker_name": eff_name,
            "role": eff_role,
            "department": eff_dept,
            "is_recognized": eff_is_rec,
            "person_box": person_box,
            "person_confidence": round(person_confidence, 3),
            "face_box": matched_face,
            "head_visible": head_visible,
            "feet_visible": feet_visible,
            "helmet_detected": helmet_detected,
            "shoes_detected": shoes_detected,
            "head_type": head_type,
            "helmet_status": helmet_status_label,
            "safety_shoe_status": shoes_status_label,
            "head_details": matched_head,
            "shoe_details": matched_shoes,
            "head_confidence": round(head_conf, 3),
            "shoes_confidence": round(shoes_conf, 3),
            "raw_status": raw_status,
            "status": stabilized_status,
            "color": STATUS_COLORS[stabilized_status],
            "details": details,
            "stability_score": round(stability, 2),
            "frames_analyzed": total_w
        }


def draw_corner_brackets(
    img: np.ndarray,
    box: List[float],
    color: Tuple[int, int, int],
    length: int = 18,
    thickness: int = 2
) -> None:
    """Draw high-tech corner brackets around a bounding box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    l = max(6, min(length, (x2 - x1) // 4, (y2 - y1) // 4))

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)


def draw_face_scanner(
    img: np.ndarray,
    face_box: List[float],
    label: str = "FACE SCAN: IDENTIFIED",
    color: Tuple[int, int, int] = (200, 72, 236)  # High-tech Pink BGR
) -> None:
    """Draw futuristic cybernetic face scanner reticle."""
    fx1, fy1, fx2, fy2 = [int(v) for v in face_box]
    h, w = img.shape[:2]
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(w, fx2), min(h, fy2)

    # 1. Subtle semi-transparent target rectangle
    draw_corner_brackets(img, [fx1, fy1, fx2, fy2], color, length=16, thickness=2)

    # 2. Central crosshairs
    cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
    cv2.circle(img, (cx, cy), 3, color, -1)
    cv2.line(img, (cx - 7, cy), (cx + 7, cy), color, 1)
    cv2.line(img, (cx, cy - 7), (cx, cy + 7), color, 1)

    # 3. Floating Face Scan pill
    pill_w = max(180, int(len(label) * 9.5))
    pill_h = 22
    by1 = max(38, fy1 - pill_h - 4)
    cv2.rectangle(img, (fx1, by1), (fx1 + pill_w, by1 + pill_h), (15, 23, 42), -1)
    cv2.rectangle(img, (fx1, by1), (fx1 + pill_w, by1 + pill_h), color, 1)
    cv2.putText(
        img,
        label,
        (fx1 + 6, by1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def annotate_frame(
    frame: np.ndarray,
    evaluations: List[Dict[str, Any]],
    fps: float = 0.0,
    title: str = config.APP_TITLE,
    faces: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Draw clean, high-contrast industrial overlays onto OpenCV video frame:
    - Bounding boxes colored by safety status:
      GREEN: Safe, RED/PINK: Not Safe, YELLOW: Unknown
    - High-Tech Corner Brackets & Cyber Facial Scanning Overlays
    - Worker Header Tag (Worker 001: SAFE / NOT SAFE / UNKNOWN)
    - Equipment Badges (HELMET: WORN/MISSING, SHOES: WORN/MISSING)
    - Border occlusion / Feet Out of Frame warning
    - Top HUD telemetry bar with White, Pink, Green, and Yellow accents
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Set of face bboxes already drawn to prevent duplicate rendering
    drawn_face_centers = []

    for ev in evaluations:
        status = ev.get("status", SafetyStatus.UNKNOWN)
        # Custom palette: SAFE = Mint Green, NOT_SAFE = Pink/Red, UNKNOWN = Electric Yellow
        if status == SafetyStatus.SAFE:
            bgr = (129, 185, 16)     # Mint green BGR
            hdr_bg = (129, 185, 16)
        elif status == SafetyStatus.NOT_SAFE:
            bgr = (94, 63, 244)      # Electric Pink/Crimson BGR
            hdr_bg = (94, 63, 244)
        else:
            bgr = (21, 204, 250)     # Electric Yellow/Gold BGR
            hdr_bg = (21, 204, 250)

        box = ev["person_box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(38, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 1. Main Worker Body Box + High-tech Corner Brackets
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        draw_corner_brackets(annotated, [x1, y1, x2, y2], (255, 255, 255), length=22, thickness=3)

        # 2. Worker ID & Status Pill Header
        worker_id = ev.get("worker_id", "Worker ???")
        worker_name = ev.get("worker_name") or worker_id
        is_rec = ev.get("is_recognized", False)
        role = ev.get("role", "Staff")
        header_text = f"{worker_name}: {status}"
        badge_h = 28
        badge_w = max(240, int(len(header_text) * 11))
        hy1 = max(38, y1 - badge_h)
        hy2 = max(hy1 + badge_h, y1)

        cv2.rectangle(annotated, (x1, hy1), (min(w, x1 + badge_w), hy2), hdr_bg, -1)
        cv2.putText(
            annotated,
            header_text,
            (x1 + 6, hy1 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 3. PPE Status Pills inside box
        # Helmet Pill
        h_ok = ev.get("helmet_detected", False)
        h_text = f"HELMET: {'WORN' if h_ok else 'MISSING'}"
        h_bg = (129, 185, 16) if h_ok else (94, 63, 244)
        cv2.rectangle(annotated, (x1 + 4, y1 + 4), (x1 + 115, y1 + 25), h_bg, -1)
        cv2.putText(
            annotated,
            h_text,
            (x1 + 7, y1 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Shoes Pill
        s_ok = ev.get("shoes_detected", False)
        feet_in_view = ev.get("feet_visible", True)
        if not feet_in_view:
            s_text = "SHOES: OUT OF FRAME"
            s_bg = (21, 204, 250)
            pill_right = x1 + 270
        else:
            s_text = f"SHOES: {'WORN' if s_ok else 'MISSING'}"
            s_bg = (129, 185, 16) if s_ok else (94, 63, 244)
            pill_right = x1 + 225

        cv2.rectangle(annotated, (x1 + 122, y1 + 4), (min(w, pill_right), y1 + 25), s_bg, -1)
        cv2.putText(
            annotated,
            s_text,
            (x1 + 125, y1 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0) if not feet_in_view else (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 4. Feet / Lower Body Occlusion Warning
        if not ev.get("feet_visible", True):
            warn_tag = "FEET OUT OF FRAME - UNKNOWN"
            cv2.rectangle(annotated, (x1, y2 - 24), (min(w, x1 + 250), y2), (21, 204, 250), -1)
            cv2.putText(
                annotated,
                warn_tag,
                (x1 + 6, y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
        elif not ev.get("head_visible", True):
            warn_tag = "HEAD PARTIALLY OUT OF FRAME"
            cv2.rectangle(annotated, (x1, y2 - 24), (min(w, x1 + 250), y2), (21, 204, 250), -1)
            cv2.putText(
                annotated,
                warn_tag,
                (x1 + 6, y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        # 5. Draw Face Scanner if associated with this worker
        face_b = ev.get("face_box")
        if face_b is not None:
            fcx = (face_b[0] + face_b[2]) / 2.0
            fcy = (face_b[1] + face_b[3]) / 2.0
            drawn_face_centers.append((fcx, fcy))
            if is_rec:
                f_label = f"ID: {worker_name} ({role})"
                f_color = (129, 185, 16)  # Mint Green for verified registered personnel
            else:
                f_label = f"FACE: {worker_name}"
                f_color = (200, 72, 236)  # Cyber Pink for guest/unregistered
            draw_face_scanner(
                annotated,
                face_b,
                label=f_label,
                color=f_color
            )

    # 6. Render any additional unassociated faces in scene
    if faces:
        for f in faces:
            fb = f["bbox"]
            fcx = (fb[0] + fb[2]) / 2.0
            fcy = (fb[1] + fb[3]) / 2.0
            already_drawn = any(abs(fcx - dc[0]) < 25 and abs(fcy - dc[1]) < 25 for dc in drawn_face_centers)
            if not already_drawn:
                if f.get("is_recognized"):
                    f_label = f"ID: {f.get('full_name')} ({f.get('role', 'Staff')})"
                    f_color = (129, 185, 16)
                else:
                    f_label = "HUMAN FACE: SCANNING"
                    f_color = (200, 72, 236)
                draw_face_scanner(
                    annotated,
                    fb,
                    label=f_label,
                    color=f_color
                )

    # 7. Modern Top HUD Bar with Mixed Gradient Accents
    cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
    # Accent top border lines (Pink, Yellow, Green)
    p3 = w // 3
    cv2.line(annotated, (0, 35), (p3, 35), (200, 72, 236), 2)         # Pink
    cv2.line(annotated, (p3, 35), (2 * p3, 35), (21, 204, 250), 2)    # Yellow
    cv2.line(annotated, (2 * p3, 35), (w, 35), (129, 185, 16), 2)     # Green

    cv2.putText(
        annotated,
        "AI SAFETY & FACE MONITORING SYSTEM",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    hud_right = f"FPS: {fps:.1f} | Workers: {len(evaluations)}"
    cv2.putText(
        annotated,
        hud_right,
        (max(10, w - 210), 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (129, 185, 16),
        2,
        cv2.LINE_AA
    )

    return annotated

