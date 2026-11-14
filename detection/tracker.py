"""
Worker Identification and Tracking Manager.
Assigns temporary sequential tracking IDs (Worker 001, Worker 002, etc.)
and prevents duplicate database entries using session deduplication.
Strictly does NOT use facial recognition.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import config


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


class WorkerSession:
    def __init__(self, raw_id: int, initial_box: List[float], worker_name: Optional[str] = None):
        self.raw_id = raw_id
        self.worker_id = f"Worker {raw_id:03d}"
        self.worker_name = worker_name
        self.bbox = initial_box
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_seen = 1
        self.missed_frames = 0
        self.committed = False
        self.committed_status: Optional[str] = None
        self.last_recorded_time = 0.0


class WorkerTracker:
    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_missed_frames: int = 25,
        cooldown_sec: float = config.RECORD_COOLDOWN_SECONDS
    ):
        self.next_raw_id = 1
        self.sessions: Dict[int, WorkerSession] = {}
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.cooldown_sec = cooldown_sec

    def update(self, detected_persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Correlate detected person boxes with active worker sessions.
        Assigns formatted worker_id and tracking metadata to each detection.
        """
        now = time.time()
        assigned_workers: List[Dict[str, Any]] = []

        if not self.sessions:
            # First detections in scene
            for det in detected_persons:
                raw_id = det.get("track_id") or self.next_raw_id
                if not det.get("track_id"):
                    self.next_raw_id += 1

                w_name = det.get("worker_name")
                session = WorkerSession(raw_id, det["bbox"], worker_name=w_name)
                self.sessions[raw_id] = session
                det["raw_id"] = raw_id
                det["worker_id"] = session.worker_id
                det["worker_name"] = w_name or session.worker_id
                assigned_workers.append(det)
            return assigned_workers

        existing_ids = list(self.sessions.keys())
        cost_matrix = []

        for rid in existing_ids:
            row = []
            for det in detected_persons:
                iou = compute_iou(self.sessions[rid].bbox, det["bbox"])
                row.append(iou)
            cost_matrix.append(row)

        matched_dets = set()
        matched_sessions = set()

        if cost_matrix and detected_persons:
            matrix = np.array(cost_matrix)
            while True:
                max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
                s_idx, d_idx = max_idx
                max_val = matrix[s_idx, d_idx]

                if max_val < self.iou_threshold:
                    break

                rid = existing_ids[s_idx]
                det = detected_persons[d_idx]

                # Update recognized worker name if newly observed
                if det.get("worker_name"):
                    self.sessions[rid].worker_name = det.get("worker_name")

                det["raw_id"] = rid
                det["worker_id"] = self.sessions[rid].worker_id
                det["worker_name"] = self.sessions[rid].worker_name or self.sessions[rid].worker_id
                self.sessions[rid].bbox = det["bbox"]
                self.sessions[rid].last_seen = now
                self.sessions[rid].frames_seen += 1
                self.sessions[rid].missed_frames = 0

                matched_sessions.add(rid)
                matched_dets.add(d_idx)

                # Invalidate matched row and column
                matrix[s_idx, :] = -1
                matrix[:, d_idx] = -1

        # Handle new unmatched workers
        for idx, det in enumerate(detected_persons):
            if idx not in matched_dets:
                raw_id = self.next_raw_id
                self.next_raw_id += 1

                w_name = det.get("worker_name")
                session = WorkerSession(raw_id, det["bbox"], worker_name=w_name)
                self.sessions[raw_id] = session
                det["raw_id"] = raw_id
                det["worker_id"] = session.worker_id
                det["worker_name"] = w_name or session.worker_id
                matched_dets.add(idx)

        # Increment missed frames on unobserved workers
        for rid in existing_ids:
            if rid not in matched_sessions:
                self.sessions[rid].missed_frames += 1

        # Purge dead sessions that exited the frame
        dead_ids = [
            rid for rid, session in self.sessions.items()
            if session.missed_frames > self.max_missed_frames
        ]
        for rid in dead_ids:
            del self.sessions[rid]

        return detected_persons

    def should_record(
        self,
        raw_id: int,
        status: str,
        stability_score: float,
        frames_seen: int
    ) -> Tuple[bool, str]:
        """
        Anti-spam deduplication engine:
        Only records verified entry events into the database, preventing
        repeated commits for stationary workers in front of the camera.
        """
        session = self.sessions.get(raw_id)
        if not session:
            return False, "Session not found"

        now = time.time()

        # 1. Require minimal consecutive frames for stabilization
        if frames_seen < config.HYSTERESIS_WINDOW:
            return False, "Stabilizing temporal buffer"

        # 2. First-time confirmation
        if not session.committed:
            session.committed = True
            session.committed_status = status
            session.last_recorded_time = now
            return True, "Initial confirmed session verification"

        # 3. Status transition (e.g. Worker removed helmet -> SAFE to NOT_SAFE)
        if session.committed_status != status and status != config.SafetyStatus.UNKNOWN and stability_score >= 0.70:
            session.committed_status = status
            session.last_recorded_time = now
            return True, f"Safety status changed to {status}"

        # 4. Periodic re-verification after session cooldown expires
        if (now - session.last_recorded_time) > self.cooldown_sec:
            session.last_recorded_time = now
            return True, "Periodic session re-verification"

        return False, "Session already recorded / in cooldown"

