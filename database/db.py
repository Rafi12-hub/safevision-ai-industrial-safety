"""
Database engine and schema manager for
AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
Uses standard Python sqlite3 with thread-safe connection management.
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Generator
from contextlib import contextmanager
import config

_db_lock = threading.Lock()


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding a thread-safe sqlite3 connection with Row factory.
    Uses busy timeout to ensure concurrent safety across Streamlit threads.
    """
    with _db_lock:
        conn = sqlite3.connect(
            str(config.DATABASE_PATH),
            timeout=20.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def init_db() -> None:
    """Initialize database tables, indexes, and default demonstration records."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Main Safety Verification Records Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                worker_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                helmet_status TEXT NOT NULL CHECK(helmet_status IN ('WORN', 'MISSING', 'UNCLEAR')),
                safety_shoe_status TEXT NOT NULL CHECK(safety_shoe_status IN ('WORN', 'MISSING', 'UNCLEAR')),
                overall_status TEXT NOT NULL CHECK(overall_status IN ('SAFE', 'NOT_SAFE', 'UNKNOWN')),
                confidence REAL NOT NULL,
                violation_details TEXT,
                head_visible INTEGER DEFAULT 1,
                feet_visible INTEGER DEFAULT 1
            )
        """)

        # Indexes for fast querying & analytics
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_date ON safety_records (date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_status ON safety_records (overall_status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_worker ON safety_records (worker_id);")

        # 2. System Settings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Default configuration values
        defaults = [
            ("conf_threshold", str(config.DEFAULT_CONF_THRESHOLD)),
            ("iou_threshold", str(config.DEFAULT_IOU_THRESHOLD)),
            ("imgsz", str(config.DEFAULT_IMGSZ)),
            ("camera_index", str(config.DEFAULT_CAMERA_INDEX)),
            ("allow_safety_cap", "true" if config.ALLOW_SAFETY_CAP else "false"),
            ("hysteresis_frames", str(config.HYSTERESIS_WINDOW)),
            ("cooldown_seconds", str(config.RECORD_COOLDOWN_SECONDS))
        ]

        now_str = datetime.now().isoformat()
        for k, v in defaults:
            cursor.execute("""
                INSERT OR IGNORE INTO system_settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (k, v, now_str))

        conn.commit()

    # Seed baseline records if database is freshly initialized
    seed_initial_records_if_empty()


def seed_initial_records_if_empty() -> None:
    """Populate baseline demonstration records so analytics and charts render immediately."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM safety_records")
        count = cursor.fetchone()[0]

        if count >= 6:
            return

        now = datetime.now()
        samples = [
            # Days back, Worker ID, Helmet, Shoes, Status, Conf, Details
            (4, "Worker 001", "WORN", "WORN", "SAFE", 0.95, "Full PPE compliance: Approved helmet & steel-toe shoes"),
            (4, "Worker 002", "WORN", "MISSING", "NOT_SAFE", 0.91, "Safety violation: Missing required safety shoes"),
            (3, "Worker 003", "WORN", "WORN", "SAFE", 0.94, "Full PPE compliance verified"),
            (3, "Worker 004", "MISSING", "WORN", "NOT_SAFE", 0.88, "Safety violation: Missing safety helmet"),
            (3, "Worker 005", "MISSING", "MISSING", "NOT_SAFE", 0.93, "Critical violation: Missing both helmet and safety shoes"),
            (2, "Worker 001", "WORN", "WORN", "SAFE", 0.96, "Full PPE compliance verified"),
            (2, "Worker 006", "WORN", "WORN", "SAFE", 0.92, "Full PPE compliance verified"),
            (2, "Worker 007", "MISSING", "WORN", "NOT_SAFE", 0.87, "Safety violation: Missing safety helmet"),
            (1, "Worker 002", "WORN", "WORN", "SAFE", 0.94, "Full PPE compliance verified"),
            (1, "Worker 008", "UNCLEAR", "WORN", "UNKNOWN", 0.58, "Head partially truncated by camera boundary"),
            (1, "Worker 009", "WORN", "MISSING", "NOT_SAFE", 0.89, "Safety violation: Missing required safety shoes"),
            (0, "Worker 003", "WORN", "WORN", "SAFE", 0.97, "Full PPE compliance verified"),
            (0, "Worker 010", "WORN", "WORN", "SAFE", 0.95, "Full PPE compliance verified"),
            (0, "Worker 011", "MISSING", "MISSING", "NOT_SAFE", 0.92, "Critical violation: Missing helmet and shoes"),
            (0, "Worker 012", "WORN", "UNCLEAR", "UNKNOWN", 0.52, "Feet partially out of bottom camera frame"),
        ]

        for days_back, worker_id, h_stat, s_stat, overall, conf, details in samples:
            dt = now - timedelta(days=days_back, hours=2, minutes=int(worker_id[-2:]) * 4)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
            ts_str = dt.isoformat()
            rec_id = f"REC-{date_str.replace('-', '')}-{worker_id.replace(' ', '')}"

            cursor.execute("""
                INSERT OR IGNORE INTO safety_records (
                    record_id, worker_id, date, time, timestamp,
                    helmet_status, safety_shoe_status, overall_status,
                    confidence, violation_details, head_visible, feet_visible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec_id, worker_id, date_str, time_str, ts_str,
                h_stat, s_stat, overall, conf, details,
                0 if "Head" in details else 1,
                0 if "Feet" in details else 1
            ))

        conn.commit()


def get_setting(key: str, default: str = "") -> str:
    """Retrieve single configuration value from database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else default


def set_setting(key: str, value: str) -> None:
    """Save or update configuration key-value in database."""
    now_str = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (key, value, now_str))
        conn.commit()

