"""
Multi-Person Tracking and Entry Session Manager for Mechanical Laboratory Safety Monitoring.
Maintains persistent student tracking IDs across video frames, correlates detections,
and prevents database record spam by committing only verified, stabilized entry events.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


class StudentTrackSession:
    def __init__(self, track_id: int, initial_box: List[float]):
        self.track_id = track_id
        self.bbox = initial_box
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_seen = 1
        self.missed_frames = 0
        self.committed = False
        self.committed_status: Optional[str] = None
        self.last_recorded_time = 0.0


class StudentTracker:
    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_missed_frames: int = 30,
        record_cooldown_sec: float = 25.0
    ):
        self.next_track_id = 1
        self.tracks: Dict[int, StudentTrackSession] = {}
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.record_cooldown_sec = record_cooldown_sec

    def update(
        self,
        detected_persons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Correlate detected person boxes with existing tracks.
        Assigns or preserves tracking_id on each detection dictionary.
        """
        now = time.time()
        assigned_tracks: List[Dict[str, Any]] = []

        if not self.tracks:
            # First time seeing people
            for det in detected_persons:
                tid = det.get("track_id") or self.next_track_id
                if not det.get("track_id"):
                    self.next_track_id += 1

                self.tracks[tid] = StudentTrackSession(tid, det["bbox"])
                det["track_id"] = tid
                assigned_tracks.append(det)
            return assigned_tracks

        # Match existing tracks with detected persons via IoU
        existing_ids = list(self.tracks.keys())
        cost_matrix = []

        for tid in existing_ids:
            row = []
            for det in detected_persons:
                iou = compute_iou(self.tracks[tid].bbox, det["bbox"])
                row.append(iou)
            cost_matrix.append(row)

        matched_dets = set()
        matched_tracks = set()

        if cost_matrix and detected_persons:
            matrix = np.array(cost_matrix)
            # Greedy matching for highest IoU
            while True:
                max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
                t_idx, d_idx = max_idx
                max_val = matrix[t_idx, d_idx]

                if max_val < self.iou_threshold:
                    break

                tid = existing_ids[t_idx]
                det = detected_persons[d_idx]

                det["track_id"] = tid
                self.tracks[tid].bbox = det["bbox"]
                self.tracks[tid].last_seen = now
                self.tracks[tid].frames_seen += 1
                self.tracks[tid].missed_frames = 0

                matched_tracks.add(tid)
                matched_dets.add(d_idx)

                # Zero out matched row and column
                matrix[t_idx, :] = -1
                matrix[:, d_idx] = -1

        # Handle unmatched detections (new persons)
        for idx, det in enumerate(detected_persons):
            if idx not in matched_dets:
                # Use detector track_id if available, otherwise new ID
                tid = det.get("track_id")
                if not tid or tid in self.tracks:
                    tid = self.next_track_id
                    self.next_track_id += 1

                self.tracks[tid] = StudentTrackSession(tid, det["bbox"])
                det["track_id"] = tid
                matched_dets.add(idx)

        # Handle tracks with missed detections
        for tid in existing_ids:
            if tid not in matched_tracks:
                self.tracks[tid].missed_frames += 1

        # Purge dead tracks that have left the scene
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        return detected_persons

    def should_record_entry(
        self,
        tracking_id: int,
        status: str,
        stability_score: float,
        frames_seen: int
    ) -> Tuple[bool, str]:
        """
        Decide whether to commit an entry event into the SQLite database.
        Rules:
        1. Must be tracked for at least 5-7 frames to filter out fleeting noise.
        2. Must have high stability score (>= 0.65).
        3. Only record once upon entry, or re-record if status flips (e.g. SAFE -> NOT_SAFE)
           or after session cooldown expires.
        """
        track = self.tracks.get(tracking_id)
        if not track:
            return False, "Track not found"

        now = time.time()

        # Need minimum frames
        if frames_seen < 6:
            return False, "Gathering temporal stability"

        # If not yet committed for this student's visit
        if not track.committed:
            track.committed = True
            track.committed_status = status
            track.last_recorded_time = now
            return True, "Initial confirmed entry"

        # If already committed, check if status changed (e.g., student took off cap)
        if track.committed_status != status and status != "UNKNOWN" and stability_score >= 0.75:
            track.committed_status = status
            track.last_recorded_time = now
            return True, f"Status transition to {status}"

        # If cooldown has elapsed (e.g., re-verifying long-standing session)
        if (now - track.last_recorded_time) > self.record_cooldown_sec:
            track.last_recorded_time = now
            return True, "Periodic session re-verification"

        return False, "Already recorded / in cooldown"
