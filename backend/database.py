"""
Database management module for AI-Based Mechanical Laboratory Safety Monitoring System.
Uses SQLite with aiosqlite for asynchronous operations, maintaining strict safety records,
entry tracking, and analytics aggregation without storing facial or biometric identities.
"""

import os
import aiosqlite
import csv
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "mechanical_safety.db")


@asynccontextmanager
async def get_connection():
    """Asynchronous context manager yielding SQLite connection with Row factory."""
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def init_db() -> None:
    """Initialize database tables with appropriate schemas and indexes."""
    async with get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS safety_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                safety_cap BOOLEAN NOT NULL,
                safety_shoes BOOLEAN NOT NULL,
                final_status TEXT NOT NULL CHECK(final_status IN ('SAFE', 'NOT_SAFE', 'UNKNOWN')),
                confidence REAL NOT NULL,
                verification_status TEXT NOT NULL DEFAULT 'CONFIRMED',
                head_visible BOOLEAN DEFAULT 1,
                feet_visible BOOLEAN DEFAULT 1,
                details TEXT
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_date ON safety_records (date);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_status ON safety_records (final_status);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_records_tracking ON safety_records (tracking_id);
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Default system settings if not already present
        default_settings = [
            ("confidence_threshold", "0.45"),
            ("iou_threshold", "0.45"),
            ("resolution", "640x480"),
            ("camera_index", "0"),
            ("alert_sound", "true"),
            ("hysteresis_frames", "8"),
            ("session_cooldown_sec", "20")
        ]

        now = datetime.now().isoformat()
        for key, val in default_settings:
            await conn.execute("""
                INSERT OR IGNORE INTO system_settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, val, now))

        await conn.commit()

    # Seed sample baseline records if database is empty so analytics and charts render immediately
    await seed_sample_records_if_empty()


async def add_record(
    tracking_id: int,
    safety_cap: bool,
    safety_shoes: bool,
    final_status: str,
    confidence: float,
    verification_status: str = "CONFIRMED",
    head_visible: bool = True,
    feet_visible: bool = True,
    details: str = ""
) -> int:
    """Insert a new safety entry record."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    iso_timestamp = now.isoformat()

    async with get_connection() as conn:
        cursor = await conn.execute("""
            INSERT INTO safety_records (
                tracking_id, timestamp, date, time,
                safety_cap, safety_shoes, final_status,
                confidence, verification_status, head_visible,
                feet_visible, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tracking_id, iso_timestamp, date_str, time_str,
            int(safety_cap), int(safety_shoes), final_status,
            round(confidence, 4), verification_status,
            int(head_visible), int(feet_visible), details
        ))
        await conn.commit()
        return cursor.lastrowid


async def get_records(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve filtered, paginated entry records."""
    query = "SELECT * FROM safety_records WHERE 1=1"
    params: List[Any] = []

    if status and status.upper() in ('SAFE', 'NOT_SAFE', 'UNKNOWN'):
        query += " AND final_status = ?"
        params.append(status.upper())

    if date_from:
        query += " AND date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND date <= ?"
        params.append(date_to)

    if search:
        query += " AND (details LIKE ? OR tracking_id LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    # Count total matching
    count_query = f"SELECT COUNT(*) FROM ({query})"

    async with get_connection() as conn:
        async with conn.execute(count_query, params) as cursor:
            total_records = (await cursor.fetchone())[0]

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        offset = (page - 1) * page_size
        params.extend([page_size, offset])

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            records = [dict(row) for row in rows]

    return {
        "records": records,
        "total": total_records,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_records + page_size - 1) // page_size if total_records > 0 else 1
    }


async def get_dashboard_stats() -> Dict[str, Any]:
    """Calculate aggregated safety statistics for the dashboard overview."""
    today = datetime.now().strftime("%Y-%m-%d")

    async with get_connection() as conn:
        # Total monitored (distinct sessions/entries)
        async with conn.execute("SELECT COUNT(*) FROM safety_records") as c:
            total_monitored = (await c.fetchone())[0]

        async with conn.execute("SELECT COUNT(*) FROM safety_records WHERE final_status = 'SAFE'") as c:
            safe_count = (await c.fetchone())[0]

        async with conn.execute("SELECT COUNT(*) FROM safety_records WHERE final_status = 'NOT_SAFE'") as c:
            unsafe_count = (await c.fetchone())[0]

        async with conn.execute("SELECT COUNT(*) FROM safety_records WHERE final_status = 'UNKNOWN'") as c:
            pending_count = (await c.fetchone())[0]

        # Today specific stats
        async with conn.execute("SELECT COUNT(*) FROM safety_records WHERE date = ?", (today,)) as c:
            today_total = (await c.fetchone())[0]

        async with conn.execute("SELECT COUNT(*) FROM safety_records WHERE date = ? AND final_status = 'SAFE'", (today,)) as c:
            today_safe = (await c.fetchone())[0]

        async with conn.execute("SELECT COUNT(*) FROM safety_records WHERE date = ? AND final_status = 'NOT_SAFE'", (today,)) as c:
            today_unsafe = (await c.fetchone())[0]

        # Recent violations
        async with conn.execute("""
            SELECT id, tracking_id, date, time, safety_cap, safety_shoes, final_status, details, confidence
            FROM safety_records
            WHERE final_status = 'NOT_SAFE'
            ORDER BY id DESC
            LIMIT 6
        """) as c:
            recent_violations = [dict(row) for row in await c.fetchall()]

        # Latest records
        async with conn.execute("""
            SELECT id, tracking_id, date, time, safety_cap, safety_shoes, final_status, details, confidence
            FROM safety_records
            ORDER BY id DESC
            LIMIT 8
        """) as c:
            latest_records = [dict(row) for row in await c.fetchall()]

    safe_percentage = round((safe_count / total_monitored * 100), 1) if total_monitored > 0 else 0.0
    unsafe_percentage = round((unsafe_count / total_monitored * 100), 1) if total_monitored > 0 else 0.0

    return {
        "total_monitored": total_monitored,
        "safe_students": safe_count,
        "unsafe_students": unsafe_count,
        "pending_verification": pending_count,
        "safe_percentage": safe_percentage,
        "unsafe_percentage": unsafe_percentage,
        "today_total": today_total,
        "today_safe": today_safe,
        "today_unsafe": today_unsafe,
        "recent_violations": recent_violations,
        "latest_records": latest_records
    }


async def get_analytics(days: int = 14) -> Dict[str, Any]:
    """Retrieve multi-dimensional time series and distribution analytics for Recharts."""
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    async with get_connection() as conn:
        # Daily safe vs unsafe
        async with conn.execute("""
            SELECT
                date,
                SUM(CASE WHEN final_status = 'SAFE' THEN 1 ELSE 0 END) as safe,
                SUM(CASE WHEN final_status = 'NOT_SAFE' THEN 1 ELSE 0 END) as unsafe,
                SUM(CASE WHEN final_status = 'UNKNOWN' THEN 1 ELSE 0 END) as unknown,
                COUNT(*) as total
            FROM safety_records
            WHERE date >= ?
            GROUP BY date
            ORDER BY date ASC
        """, (start_date,)) as c:
            daily_stats = [dict(row) for row in await c.fetchall()]

        # Violation distribution by missing PPE item
        async with conn.execute("""
            SELECT
                CASE
                    WHEN safety_cap = 0 AND safety_shoes = 0 THEN 'Missing Cap & Shoes'
                    WHEN safety_cap = 0 AND safety_shoes = 1 THEN 'Missing Safety Cap'
                    WHEN safety_cap = 1 AND safety_shoes = 0 THEN 'Missing Safety Shoes'
                    ELSE 'Other Non-compliance'
                END as violation_type,
                COUNT(*) as count
            FROM safety_records
            WHERE final_status = 'NOT_SAFE'
            GROUP BY violation_type
        """) as c:
            violation_distribution = [dict(row) for row in await c.fetchall()]

        # Weekly compliance calculation
        async with conn.execute("""
            SELECT
                strftime('%Y-W%W', date) as week,
                ROUND(100.0 * SUM(CASE WHEN final_status = 'SAFE' THEN 1 ELSE 0 END) / COUNT(*), 1) as compliance_rate,
                COUNT(*) as total_entries
            FROM safety_records
            GROUP BY week
            ORDER BY week ASC
            LIMIT 8
        """) as c:
            weekly_compliance = [dict(row) for row in await c.fetchall()]

        # Monthly compliance calculation
        async with conn.execute("""
            SELECT
                strftime('%Y-%m', date) as month,
                SUM(CASE WHEN final_status = 'SAFE' THEN 1 ELSE 0 END) as safe_count,
                SUM(CASE WHEN final_status = 'NOT_SAFE' THEN 1 ELSE 0 END) as unsafe_count,
                ROUND(100.0 * SUM(CASE WHEN final_status = 'SAFE' THEN 1 ELSE 0 END) / COUNT(*), 1) as compliance_rate,
                COUNT(*) as total_entries
            FROM safety_records
            GROUP BY month
            ORDER BY month ASC
            LIMIT 6
        """) as c:
            monthly_compliance = [dict(row) for row in await c.fetchall()]

    return {
        "daily_stats": daily_stats,
        "violation_distribution": violation_distribution,
        "weekly_compliance": weekly_compliance,
        "monthly_compliance": monthly_compliance
    }


async def get_setting(key: str, default: str = "") -> str:
    """Get single system configuration value."""
    async with get_connection() as conn:
        async with conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)) as c:
            row = await c.fetchone()
            return row["value"] if row else default


async def set_setting(key: str, value: str) -> None:
    """Save or update system configuration value."""
    now = datetime.now().isoformat()
    async with get_connection() as conn:
        await conn.execute("""
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (key, value, now))
        await conn.commit()


async def get_all_settings() -> Dict[str, str]:
    """Retrieve all configuration settings as dictionary."""
    async with get_connection() as conn:
        async with conn.execute("SELECT key, value FROM system_settings") as c:
            rows = await c.fetchall()
            return {row["key"]: row["value"] for row in rows}


async def export_records_csv() -> str:
    """Export all entry records to standard CSV formatted text."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID", "Tracking ID", "Date", "Time", "ISO Timestamp",
        "Safety Cap Detected", "Safety Shoes Detected",
        "Final Safety Status", "Confidence", "Verification Status",
        "Head Visible", "Feet Visible", "Violation Details"
    ])

    async with get_connection() as conn:
        async with conn.execute("SELECT * FROM safety_records ORDER BY id ASC") as c:
            async for row in c:
                writer.writerow([
                    row["id"],
                    row["tracking_id"],
                    row["date"],
                    row["time"],
                    row["timestamp"],
                    "YES" if row["safety_cap"] else "NO",
                    "YES" if row["safety_shoes"] else "NO",
                    row["final_status"],
                    f"{row['confidence']:.2f}",
                    row["verification_status"],
                    "YES" if row["head_visible"] else "NO",
                    "YES" if row["feet_visible"] else "NO",
                    row["details"] or ""
                ])

    return output.getvalue()


async def seed_sample_records_if_empty() -> None:
    """Populate baseline demonstration records if table has fewer than 5 rows."""
    async with get_connection() as conn:
        async with conn.execute("SELECT COUNT(*) FROM safety_records") as c:
            count = (await c.fetchone())[0]

        if count >= 5:
            return

        now = datetime.now()
        samples = [
            # Days back, Tracking ID, Cap, Shoes, Status, Conf, Details
            (3, 101, True, True, "SAFE", 0.94, "Fully compliant entry"),
            (3, 102, True, False, "NOT_SAFE", 0.88, "Missing Safety Shoes"),
            (2, 103, True, True, "SAFE", 0.92, "Fully compliant entry"),
            (2, 104, False, True, "NOT_SAFE", 0.85, "Missing Safety Cap"),
            (2, 105, False, False, "NOT_SAFE", 0.91, "Missing Cap & Shoes"),
            (1, 106, True, True, "SAFE", 0.96, "Fully compliant entry"),
            (1, 107, True, True, "SAFE", 0.89, "Fully compliant entry"),
            (1, 108, False, True, "NOT_SAFE", 0.87, "Missing Safety Cap"),
            (0, 109, True, True, "SAFE", 0.95, "Fully compliant entry"),
            (0, 110, True, False, "NOT_SAFE", 0.90, "Missing Safety Shoes"),
            (0, 111, False, False, "NOT_SAFE", 0.84, "Missing Cap & Shoes"),
            (0, 112, True, True, "SAFE", 0.93, "Fully compliant entry"),
        ]

        for days_back, tid, cap, shoes, status, conf, det in samples:
            ts = now - timedelta(days=days_back, hours=1, minutes=tid % 50)
            date_str = ts.strftime("%Y-%m-%d")
            time_str = ts.strftime("%H:%M:%S")
            iso_str = ts.isoformat()

            await conn.execute("""
                INSERT INTO safety_records (
                    tracking_id, timestamp, date, time,
                    safety_cap, safety_shoes, final_status,
                    confidence, verification_status, head_visible,
                    feet_visible, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tid, iso_str, date_str, time_str,
                int(cap), int(shoes), status,
                conf, "CONFIRMED", 1, 1, det
            ))

        await conn.commit()
