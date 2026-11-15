"""
Worker Profiles and Facial Biometrics Database Layer.
Stores worker identities (Name, Role, Department, Photo, Face Embeddings)
and associates detected facial scans with registered personnel records.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import cv2

import config
from .db import get_connection


def add_worker(
    worker_id: str,
    full_name: str,
    role: str,
    department: str,
    photo_path: str = "",
    face_encoding: Optional[bytes] = None
) -> bool:
    """Register a new worker profile with facial encoding."""
    now_str = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO registered_workers (
                worker_id, full_name, role, department,
                photo_path, face_encoding, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            worker_id.strip(),
            full_name.strip(),
            role.strip(),
            department.strip(),
            photo_path,
            face_encoding,
            now_str,
            now_str
        ))
        conn.commit()
    return True


def get_all_workers() -> List[Dict[str, Any]]:
    """Retrieve list of all registered workers."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, worker_id, full_name, role, department, photo_path, created_at, updated_at
            FROM registered_workers
            ORDER BY id ASC
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_all_workers_with_encodings() -> List[Dict[str, Any]]:
    """Retrieve all workers including their serialized facial encodings for real-time matching."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT worker_id, full_name, role, department, photo_path, face_encoding
            FROM registered_workers
            WHERE face_encoding IS NOT NULL
        """)
        rows = cursor.fetchall()
        workers = []
        for r in rows:
            w_dict = dict(r)
            raw_enc = w_dict.get("face_encoding")
            if raw_enc is not None:
                try:
                    w_dict["encoding_array"] = np.frombuffer(raw_enc, dtype=np.float32)
                    workers.append(w_dict)
                except Exception:
                    pass
        return workers


def get_worker_by_id(worker_id: str) -> Optional[Dict[str, Any]]:
    """Find a worker profile by unique Worker ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM registered_workers WHERE worker_id = ?", (worker_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_worker(worker_id: str) -> bool:
    """Delete a worker from the registered database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM registered_workers WHERE worker_id = ?", (worker_id,))
        conn.commit()
        return cursor.rowcount > 0


def count_workers() -> int:
    """Return total number of registered workers."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM registered_workers")
        return cursor.fetchone()[0]


def seed_initial_workers_if_empty() -> None:
    """Populate initial demonstration personnel profiles if empty."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM registered_workers")
        count = cursor.fetchone()[0]
        if count > 0:
            return

    # Seed 3 representative demonstration workers
    demo_workers = [
        ("W-101", "Rajesh Sharma", "Senior Lathe Operator", "Machining & Milling"),
        ("W-102", "Priya Patel", "Lab Safety Incharge", "Mechanical Quality Lab"),
        ("W-103", "Arun Kumar", "CNC Tooling Technician", "Fabrication Workshop")
    ]

    # Pre-generate sample profile avatar images
    hog = None
    try:
        if hasattr(cv2, "HOGDescriptor"):
            hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    except Exception:
        hog = None

    for wid, name, role, dept in demo_workers:
        # Create a stylized avatar image
        avatar = np.zeros((128, 128, 3), dtype=np.uint8)
        avatar[:] = (22, 27, 34)  # dark surface
        # Draw head silhouette
        color_tint = (180, 140, 120) if wid == "W-101" else ((160, 170, 200) if wid == "W-102" else (140, 190, 150))
        cv2.circle(avatar, (64, 50), 30, color_tint, -1)
        cv2.ellipse(avatar, (64, 130), (50, 50), 0, 0, 360, (50, 70, 100), -1)
        cv2.putText(avatar, wid, (28, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Save photo
        photo_file = config.WORKER_FACES_DIR / f"{wid}.jpg"
        cv2.imwrite(str(photo_file), avatar)

        # Compute facial encoding
        gray = cv2.cvtColor(cv2.resize(avatar, (64, 64)), cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        if hog is not None and hasattr(hog, "compute"):
            desc = hog.compute(gray).flatten()
        else:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
            mag = cv2.magnitude(gx, gy)
            desc = cv2.resize(mag, (42, 42)).flatten()
        norm = np.linalg.norm(desc)
        vec = desc / norm if norm > 0 else desc
        enc_bytes = vec.tobytes()

        add_worker(
            worker_id=wid,
            full_name=name,
            role=role,
            department=dept,
            photo_path=str(photo_file),
            face_encoding=enc_bytes
        )
