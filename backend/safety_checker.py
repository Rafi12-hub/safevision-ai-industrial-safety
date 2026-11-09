"""
Safety Verification Engine for AI-Based Mechanical Laboratory Safety Monitoring System.
Performs spatial association between persons and safety equipment (safety cap and safety shoes),
boundary visibility checks (head and feet out of frame), and temporal stability filtering.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import numpy as np


class SafetyStatus:
    SAFE = "SAFE"
    NOT_SAFE = "NOT_SAFE"
    UNKNOWN = "UNKNOWN"


STATUS_COLORS = {
    SafetyStatus.SAFE: "#10B981",      # Emerald Green
    SafetyStatus.NOT_SAFE: "#EF4444",  # Crimson Red
    SafetyStatus.UNKNOWN: "#F59E0B"   # Amber Orange
}

STATUS_RGB = {
    SafetyStatus.SAFE: (16, 185, 129),     # BGR in OpenCV: (129, 185, 16)
    SafetyStatus.NOT_SAFE: (239, 68, 68),  # BGR: (68, 68, 239)
    SafetyStatus.UNKNOWN: (245, 158, 11)   # BGR: (11, 158, 245)
}


class PersonSafetyVerifier:
    def __init__(
        self,
        frame_edge_margin: int = 15,
        hysteresis_window: int = 10,
        confirmation_threshold: float = 0.6
    ):
        """
        :param frame_edge_margin: Pixel threshold near frame edges to flag head/feet truncation.
        :param hysteresis_window: Number of consecutive frames tracked for temporal smoothing.
        :param confirmation_threshold: Majority fraction needed in window to commit safety status.
        """
        self.frame_edge_margin = frame_edge_margin
        self.hysteresis_window = hysteresis_window
        self.confirmation_threshold = confirmation_threshold
        # History dict: tracking_id -> deque of recent raw status results
        self.history: Dict[int, deque] = {}

    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        """Remove history for persons no longer in the camera frame."""
        for tid in list(self.history.keys()):
            if tid not in active_track_ids:
                del self.history[tid]

    @staticmethod
    def _box_center(bbox: List[float]) -> Tuple[float, float]:
        """Return (cx, cy) of [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _is_point_in_box(point: Tuple[float, float], box: List[float]) -> bool:
        """Check whether (x, y) point falls within [x1, y1, x2, y2]."""
        x, y = point
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    @staticmethod
    def _compute_overlap_ratio(item_box: List[float], zone_box: List[float]) -> float:
        """Compute intersection area / item area."""
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
        Check if student's head and feet are reliably visible within the camera view.
        Returns: (head_visible, feet_visible, reason)
        """
        px1, py1, px2, py2 = person_box
        head_visible = True
        feet_visible = True
        reasons = []

        # Check top boundary truncation (head cut off)
        if py1 <= self.frame_edge_margin:
            head_visible = False
            reasons.append("Head partially out of top frame")

        # Check bottom boundary truncation (feet cut off)
        if py2 >= (frame_height - self.frame_edge_margin):
            feet_visible = False
            reasons.append("Feet partially out of bottom frame")

        # Check person height ratio (if too close or unreasonably small)
        ph = py2 - py1
        if ph < 80:
            head_visible = False
            feet_visible = False
            reasons.append("Student too far from camera for reliable PPE check")

        reason_str = "; ".join(reasons) if reasons else "Fully in frame"
        return head_visible, feet_visible, reason_str

    def associate_equipment(
        self,
        person_box: List[float],
        cap_boxes: List[Dict[str, Any]],
        shoe_boxes: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Spatially associate safety cap and safety shoes with this person.
        - Cap must be in Head Zone (upper ~32% of body box, with horizontal margin)
        - Shoes must be in Feet Zone (lower ~32% of body box, with horizontal margin)
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

        matched_cap: Optional[Dict[str, Any]] = None
        best_cap_score = 0.0

        for cap in cap_boxes:
            cbox = cap["bbox"]
            center = self._box_center(cbox)
            overlap = self._compute_overlap_ratio(cbox, head_zone)
            in_zone = self._is_point_in_box(center, head_zone)

            if in_zone or overlap > 0.30:
                # Score based on confidence and overlap
                score = (cap.get("confidence", 0.5) * 0.5) + (overlap * 0.5)
                if score > best_cap_score:
                    best_cap_score = score
                    matched_cap = cap

        matched_shoe: Optional[Dict[str, Any]] = None
        best_shoe_score = 0.0

        for shoe in shoe_boxes:
            sbox = shoe["bbox"]
            center = self._box_center(sbox)
            overlap = self._compute_overlap_ratio(sbox, feet_zone)
            in_zone = self._is_point_in_box(center, feet_zone)

            if in_zone or overlap > 0.30:
                score = (shoe.get("confidence", 0.5) * 0.5) + (overlap * 0.5)
                if score > best_shoe_score:
                    best_shoe_score = score
                    matched_shoe = shoe

        return matched_cap, matched_shoe

    def evaluate_person(
        self,
        tracking_id: int,
        person_box: List[float],
        person_confidence: float,
        cap_boxes: List[Dict[str, Any]],
        shoe_boxes: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Any]:
        """
        Evaluate full safety verification logic for a detected person.
        Applies frame boundary verification, spatial equipment association,
        and temporal stability smoothing.
        """
        # Step 1: Visibility check
        head_visible, feet_visible, vis_reason = self.check_visibility(
            person_box, frame_width, frame_height
        )

        # Step 2: Spatial association
        matched_cap, matched_shoe = self.associate_equipment(
            person_box, cap_boxes, shoe_boxes, frame_width, frame_height
        )

        cap_detected = matched_cap is not None
        shoes_detected = matched_shoe is not None

        cap_conf = matched_cap["confidence"] if matched_cap else 0.0
        shoes_conf = matched_shoe["confidence"] if matched_shoe else 0.0

        # Step 3: Raw frame decision
        if not head_visible or not feet_visible:
            raw_status = SafetyStatus.UNKNOWN
            details = f"UNKNOWN: {vis_reason}"
        elif cap_detected and shoes_detected:
            raw_status = SafetyStatus.SAFE
            details = "SAFE: Both Safety Cap and Safety Shoes detected"
        else:
            raw_status = SafetyStatus.NOT_SAFE
            missing_items = []
            if not cap_detected:
                missing_items.append("Safety Cap")
            if not shoes_detected:
                missing_items.append("Safety Shoes")
            details = f"NOT SAFE: Missing {', '.join(missing_items)}"

        # Step 4: Temporal Stability (Hysteresis)
        if tracking_id not in self.history:
            self.history[tracking_id] = deque(maxlen=self.hysteresis_window)

        self.history[tracking_id].append(raw_status)
        window = list(self.history[tracking_id])

        # Require majority in temporal buffer to commit stabilized status
        safe_count = window.count(SafetyStatus.SAFE)
        unsafe_count = window.count(SafetyStatus.NOT_SAFE)
        unknown_count = window.count(SafetyStatus.UNKNOWN)
        total_window = len(window)

        # Prioritize warning safety violation if consistently seen
        if total_window < 3:
            stabilized_status = raw_status
            confidence_stability = 0.5
        elif (unsafe_count / total_window) >= self.confirmation_threshold:
            stabilized_status = SafetyStatus.NOT_SAFE
            confidence_stability = unsafe_count / total_window
        elif (safe_count / total_window) >= self.confirmation_threshold:
            stabilized_status = SafetyStatus.SAFE
            confidence_stability = safe_count / total_window
        elif (unknown_count / total_window) >= 0.5:
            stabilized_status = SafetyStatus.UNKNOWN
            confidence_stability = unknown_count / total_window
        else:
            stabilized_status = raw_status
            confidence_stability = 0.6

        return {
            "tracking_id": tracking_id,
            "person_box": person_box,
            "person_confidence": round(person_confidence, 3),
            "head_visible": head_visible,
            "feet_visible": feet_visible,
            "cap_detected": cap_detected,
            "shoes_detected": shoes_detected,
            "cap_details": matched_cap,
            "shoe_details": matched_shoe,
            "cap_confidence": round(cap_conf, 3),
            "shoes_confidence": round(shoes_conf, 3),
            "raw_status": raw_status,
            "status": stabilized_status,
            "color": STATUS_COLORS[stabilized_status],
            "details": details,
            "stability_score": round(confidence_stability, 2),
            "frames_analyzed": total_window
        }
