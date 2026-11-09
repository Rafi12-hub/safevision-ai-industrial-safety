"""
Safety Records Query and Persistence Layer.
Integrates SQLite with Pandas for data retrieval, analytics filtering, and CSV report export.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import config
from .db import get_connection


def add_safety_record(
    worker_id: str,
    helmet_status: str,
    safety_shoe_status: str,
    overall_status: str,
    confidence: float,
    violation_details: str = "",
    head_visible: bool = True,
    feet_visible: bool = True
) -> str:
    """
    Commit a verified safety check event into SQLite and auto-sync records CSV.
    Returns the unique generated record_id.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp_str = now.isoformat()
    clean_worker = worker_id.replace(" ", "").upper()
    random_suffix = now.strftime("%f")[:3]
    record_id = f"REC-{now.strftime('%Y%m%d%H%M%S')}-{clean_worker}-{random_suffix}"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO safety_records (
                record_id, worker_id, date, time, timestamp,
                helmet_status, safety_shoe_status, overall_status,
                confidence, violation_details, head_visible, feet_visible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id, worker_id, date_str, time_str, timestamp_str,
            helmet_status, safety_shoe_status, overall_status,
            round(confidence, 4), violation_details,
            int(head_visible), int(feet_visible)
        ))
        conn.commit()

    # Synchronize to records/safety_records.csv
    try:
        sync_csv_file()
    except Exception:
        pass

    return record_id


def get_all_records_df() -> pd.DataFrame:
    """Fetch all safety verification records as a sorted Pandas DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM safety_records ORDER BY id DESC",
            conn
        )
    return df


def get_filtered_records_df(
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None
) -> pd.DataFrame:
    """Retrieve filtered records matching user criteria for tables and reports."""
    query = "SELECT * FROM safety_records WHERE 1=1"
    params: List[Any] = []

    if status and status.upper() in ("SAFE", "NOT_SAFE", "UNKNOWN"):
        query += " AND overall_status = ?"
        params.append(status.upper())

    if date_from:
        query += " AND date >= ?"
        params.append(str(date_from))

    if date_to:
        query += " AND date <= ?"
        params.append(str(date_to))

    if search:
        query += " AND (worker_id LIKE ? OR record_id LIKE ? OR violation_details LIKE ?)"
        wildcard = f"%{search}%"
        params.extend([wildcard, wildcard, wildcard])

    query += " ORDER BY id DESC"

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def get_safety_summary_stats() -> Dict[str, Any]:
    """Calculate aggregated safety metrics for dashboard summary cards."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    with get_connection() as conn:
        cursor = conn.cursor()

        # Overall Totals
        cursor.execute("SELECT COUNT(*) FROM safety_records")
        total_records = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT worker_id) FROM safety_records")
        total_unique_workers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE overall_status = 'SAFE'")
        safe_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE overall_status = 'NOT_SAFE'")
        unsafe_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE overall_status = 'UNKNOWN'")
        unknown_count = cursor.fetchone()[0]

        # Specific PPE Totals
        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE helmet_status = 'WORN'")
        helmet_worn_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE safety_shoe_status = 'WORN'")
        shoes_worn_count = cursor.fetchone()[0]

        # Today's Specifics
        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE date = ?", (today_str,))
        today_total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE date = ? AND overall_status = 'SAFE'", (today_str,))
        today_safe = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE date = ? AND overall_status = 'NOT_SAFE'", (today_str,))
        today_unsafe = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM safety_records WHERE date = ? AND overall_status = 'UNKNOWN'", (today_str,))
        today_unknown = cursor.fetchone()[0]

    # Percentages
    compliance_pct = round((safe_count / total_records * 100), 1) if total_records > 0 else 0.0
    today_compliance_pct = round((today_safe / today_total * 100), 1) if today_total > 0 else 0.0
    helmet_compliance_pct = round((helmet_worn_count / total_records * 100), 1) if total_records > 0 else 0.0
    shoe_compliance_pct = round((shoes_worn_count / total_records * 100), 1) if total_records > 0 else 0.0

    return {
        "total_records": total_records,
        "total_workers": total_unique_workers,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "unknown_count": unknown_count,
        "compliance_pct": compliance_pct,
        "helmet_compliance_pct": helmet_compliance_pct,
        "shoe_compliance_pct": shoe_compliance_pct,
        "today_total": today_total,
        "today_safe": today_safe,
        "today_unsafe": today_unsafe,
        "today_unknown": today_unknown,
        "today_compliance_pct": today_compliance_pct
    }


def get_recent_violations(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve most recent safety violations for alerts and history table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, record_id, worker_id, date, time,
                   helmet_status, safety_shoe_status, overall_status,
                   confidence, violation_details
            FROM safety_records
            WHERE overall_status = 'NOT_SAFE'
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def sync_csv_file() -> Path:
    """Export complete SQLite safety_records to records/safety_records.csv."""
    df = get_all_records_df()
    config.RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CSV_PATH, index=False)
    return config.CSV_PATH


def export_to_csv_file() -> Path:
    """Public wrapper to ensure updated CSV and return path."""
    return sync_csv_file()

