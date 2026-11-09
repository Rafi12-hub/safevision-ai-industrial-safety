"""
Industrial Safety Compliance Report Generator.
Produces daily, weekly, and monthly analytical reports,
summary compliance metrics, and downloadable CSV audit files.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from database.safety_records import get_all_records_df, get_filtered_records_df


def get_report_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate core compliance figures from a filtered DataFrame."""
    if df.empty:
        return {
            "total_entries": 0,
            "unique_workers": 0,
            "safe_entries": 0,
            "unsafe_entries": 0,
            "unknown_entries": 0,
            "compliance_pct": 0.0,
            "helmet_compliance_pct": 0.0,
            "shoe_compliance_pct": 0.0
        }

    total = len(df)
    unique_workers = df["worker_id"].nunique()
    safe = len(df[df["overall_status"] == "SAFE"])
    unsafe = len(df[df["overall_status"] == "NOT_SAFE"])
    unknown = len(df[df["overall_status"] == "UNKNOWN"])

    helmet_worn = len(df[df["helmet_status"] == "WORN"])
    shoes_worn = len(df[df["safety_shoe_status"] == "WORN"])

    return {
        "total_entries": total,
        "unique_workers": unique_workers,
        "safe_entries": safe,
        "unsafe_entries": unsafe,
        "unknown_entries": unknown,
        "compliance_pct": round((safe / total * 100), 1),
        "helmet_compliance_pct": round((helmet_worn / total * 100), 1),
        "shoe_compliance_pct": round((shoes_worn / total * 100), 1)
    }


def generate_daily_report(target_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Generate comprehensive 24-hour daily report."""
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    df = get_filtered_records_df(date_from=target_date, date_to=target_date)
    stats = get_report_summary_stats(df)
    title = f"Daily Safety Compliance Report ({target_date})"
    return df, stats, title


def generate_weekly_report(end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Generate rolling 7-day weekly compliance report."""
    now = datetime.now()
    if end_date:
        try:
            now = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            pass

    start_date = (now - timedelta(days=6)).strftime("%Y-%m-%d")
    end_str = now.strftime("%Y-%m-%d")

    df = get_filtered_records_df(date_from=start_date, date_to=end_str)
    stats = get_report_summary_stats(df)
    title = f"Weekly Safety Compliance Report ({start_date} to {end_str})"
    return df, stats, title


def generate_monthly_report(year_month: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Generate full calendar month compliance report."""
    if not year_month:
        year_month = datetime.now().strftime("%Y-%m")

    start_date = f"{year_month}-01"
    # End date: roughly 31 days forward
    end_date = f"{year_month}-31"

    df = get_filtered_records_df(date_from=start_date, date_to=end_date)
    stats = get_report_summary_stats(df)
    title = f"Monthly Safety Audit Report ({year_month})"
    return df, stats, title


def generate_csv_download(df: pd.DataFrame) -> bytes:
    """Format DataFrame into clean downloadable CSV bytes."""
    clean_df = df.copy()
    columns_map = {
        "record_id": "Record ID",
        "worker_id": "Worker ID",
        "date": "Date",
        "time": "Time",
        "helmet_status": "Safety Helmet",
        "safety_shoe_status": "Safety Shoes",
        "overall_status": "Overall Safety Status",
        "confidence": "Confidence Score",
        "violation_details": "Violation Details"
    }

    keep_cols = [c for c in columns_map.keys() if c in clean_df.columns]
    export_df = clean_df[keep_cols].rename(columns=columns_map)
    return export_df.to_csv(index=False).encode("utf-8")
