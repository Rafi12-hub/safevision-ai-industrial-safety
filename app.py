"""
SafeVision AI: Industrial Safety Monitoring & PPE Compliance Platform.
High-Precision Computer Vision, Facial Biometrics & Automated PPE Verification.
Palette: Warm Industrial Luxury (Sage Green #E3E8D9, Terracotta #C66B48, Warm Ivory #F4EFE6, Cream #FFFCF7).
Left-Side Vertical Navigation Only.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime

# Configure environment before heavy imports
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import config
from config import SafetyStatus, STATUS_COLORS
from database.db import init_db, get_setting, set_setting
from database.safety_records import (
    add_safety_record,
    get_all_records_df,
    get_filtered_records_df,
    get_safety_summary_stats,
    get_recent_violations,
    export_to_csv_file
)
from database.workers import (
    add_worker,
    get_all_workers,
    get_worker_by_id,
    delete_worker,
    count_workers
)
from detection.detector import SafetyDetector
from detection.tracker import WorkerTracker
from detection.safety_checker import SafetyVerifier, annotate_frame
from detection.face_recognizer import FaceRecognizer
from reports.report_generator import (
    generate_daily_report,
    generate_weekly_report,
    generate_monthly_report,
    generate_csv_download
)

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeVision AI | Industrial Safety Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Professional Human-Engineered UI Design System (Warm Industrial Luxury)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* =========================================================================
       1. CORE SYSTEM TYPOGRAPHY & FOUNDATION
       ========================================================================= */
    :root {
        --bg-main: #F4EFE6;
        --sidebar-bg: #E3E8D9;
        --terracotta: #C66B48;
        --terracotta-hover: #B25A38;
        --text-dark: #2C332C;
        --text-muted: #6B736B;
        --divider-soft: #C9D1C1;
        --card-bg: #FFFFFF;
        --card-border: #E2DDD5;
        --safe-green: #2E7D32;
        --safe-bg: #EBF5EE;
        --safe-border: #BCE3C5;
        --alert-rust: #C66B48;
        --alert-bg: #FDF2EE;
        --alert-border: #F1BEAF;
        --amber-warn: #C88F35;
        --amber-bg: #FEF8EC;
        --amber-border: #EEDAB0;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-dark) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    html, body, p, span, label, div, [class*="css"], .stMarkdown {
        color: var(--text-dark);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-dark) !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--divider-soft) !important;
        margin: 1.25rem 0 !important;
    }

    /* =========================================================================
       2. REFINED SIDEBAR (SAGE GREEN #E3E8D9)
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--divider-soft) !important;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        background-color: var(--sidebar-bg) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
        background-color: var(--sidebar-bg) !important;
        padding: 1rem 0.85rem !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        color: var(--text-dark) !important;
        background-color: transparent !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        background-color: #D6DDCB !important;
        border-radius: 6px !important;
    }

    /* =========================================================================
       3. LEFT-SIDE VERTICAL NAVIGATION (CLEAN HUMAN SAAS LIST)
       ========================================================================= */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.3rem !important;
        width: 100% !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        background-color: transparent !important;
        color: var(--text-dark) !important;
        padding: 0.62rem 0.9rem !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.16s ease !important;
    }

    /* Hide standard circular radio dot */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label p,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label span,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label div {
        color: var(--text-dark) !important;
        font-size: 0.91rem !important;
        font-weight: 550 !important;
        margin: 0 !important;
        line-height: 1.35 !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background-color: #D7DECE !important;
        color: var(--text-dark) !important;
    }

    /* Active Navigation State: Terracotta with crisp white font */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked),
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background-color: var(--terracotta) !important;
        box-shadow: 0 2px 6px rgba(198, 107, 72, 0.28) !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) span,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] span {
        color: #FFFFFF !important;
        font-weight: 650 !important;
    }

    /* =========================================================================
       4. ENTERPRISE CARDS & METRIC CONTAINERS
       ========================================================================= */
    .app-card {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 1.25rem 1.4rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .app-card:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06) !important;
    }

    .metric-title {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.35rem !important;
    }
    .metric-value {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: var(--text-dark) !important;
        font-feature-settings: 'tnum' on, 'lnum' on !important;
        line-height: 1.15 !important;
    }
    .metric-caption {
        font-size: 0.8rem !important;
        color: var(--text-muted) !important;
        margin-top: 0.35rem !important;
        font-weight: 500 !important;
    }

    /* Accent Left Stripes */
    .border-accent-safe { border-left: 3px solid var(--safe-green) !important; }
    .border-accent-alert { border-left: 3px solid var(--alert-rust) !important; }
    .border-accent-warn { border-left: 3px solid var(--amber-warn) !important; }
    .border-accent-neutral { border-left: 3px solid var(--text-dark) !important; }

    /* =========================================================================
       5. PROFESSIONAL TAGS & STATUS PILLS
       ========================================================================= */
    .status-pill-safe {
        background-color: var(--safe-bg) !important;
        color: var(--safe-green) !important;
        border: 1px solid var(--safe-border) !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 9999px !important;
        font-weight: 650 !important;
        font-size: 0.76rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
    }
    .status-pill-alert {
        background-color: var(--alert-bg) !important;
        color: var(--alert-rust) !important;
        border: 1px solid var(--alert-border) !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 9999px !important;
        font-weight: 650 !important;
        font-size: 0.76rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
    }
    .status-pill-warn {
        background-color: var(--amber-bg) !important;
        color: var(--amber-warn) !important;
        border: 1px solid var(--amber-border) !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 9999px !important;
        font-weight: 650 !important;
        font-size: 0.76rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
    }
    .status-pill-neutral {
        background-color: #F0ECE4 !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--divider-soft) !important;
        padding: 0.25rem 0.7rem !important;
        border-radius: 9999px !important;
        font-weight: 650 !important;
        font-size: 0.76rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
    }

    /* Buttons: Clean Human SaaS styling */
    .stButton > button {
        background-color: var(--terracotta) !important;
        color: #FFFFFF !important;
        border: 1px solid #B75E3D !important;
        border-radius: 7px !important;
        font-weight: 650 !important;
        font-size: 0.88rem !important;
        padding: 0.48rem 1rem !important;
        transition: all 0.16s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button:hover {
        background-color: var(--terracotta-hover) !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(198, 107, 72, 0.25) !important;
    }

    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--divider-soft) !important;
        border-radius: 7px !important;
        font-weight: 650 !important;
        font-size: 0.88rem !important;
        padding: 0.48rem 1rem !important;
        transition: all 0.16s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: #F5F1E8 !important;
        border-color: var(--text-muted) !important;
    }

    /* Clean form inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--divider-soft) !important;
        color: var(--text-dark) !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--terracotta) !important;
        box-shadow: 0 0 0 2px rgba(198, 107, 72, 0.15) !important;
    }

    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Subsystems Singleton Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_system_subsystems():
    """Initialize database, detector, tracker, verifier, and face recognizer."""
    init_db()
    conf_thresh = float(get_setting("conf_threshold", str(config.DEFAULT_CONF_THRESHOLD)))
    iou_thresh = float(get_setting("iou_threshold", str(config.DEFAULT_IOU_THRESHOLD)))
    imgsz = int(get_setting("imgsz", str(config.DEFAULT_IMGSZ)))
    allow_cap = get_setting("allow_safety_cap", "true").lower() == "true"
    hysteresis = int(get_setting("hysteresis_frames", str(config.HYSTERESIS_WINDOW)))
    cooldown = float(get_setting("cooldown_seconds", str(config.RECORD_COOLDOWN_SECONDS)))
    face_thresh = float(get_setting("face_match_threshold", str(config.DEFAULT_FACE_MATCH_THRESHOLD)))

    detector = SafetyDetector(
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        imgsz=imgsz,
        device="cpu"
    )
    tracker = WorkerTracker(
        iou_threshold=0.35,
        max_missed_frames=25,
        cooldown_sec=cooldown
    )
    verifier = SafetyVerifier(
        frame_edge_margin=config.FRAME_EDGE_MARGIN,
        hysteresis_window=hysteresis,
        allow_safety_cap=allow_cap
    )
    face_recognizer = FaceRecognizer(match_threshold=face_thresh)

    return detector, tracker, verifier, face_recognizer


detector, tracker, verifier, face_recognizer = get_system_subsystems()


# -----------------------------------------------------------------------------
# SIDEBAR: VERTICAL NAVIGATION & SYSTEM TELEMETRY
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. SIDEBAR BRAND HEADER
    st.markdown("""
        <div style="padding:0.4rem 0.2rem 1.1rem 0.2rem; border-bottom:1px solid #C9D1C1; margin-bottom:1.1rem;">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div style="background-color:#C66B48; color:#FFFFFF; width:38px; height:38px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.25rem; box-shadow:0 2px 6px rgba(198, 107, 72, 0.22);">
                    🛡️
                </div>
                <div>
                    <div style="font-weight:800; font-size:1.12rem; color:#2C332C; letter-spacing:-0.02em; line-height:1.2;">SafeVision AI</div>
                    <div style="font-size:0.72rem; color:#6B736B; font-weight:600; letter-spacing:0.02em;">Industrial Lab & Worker Safety</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. VERTICAL NAVIGATION LIST (9 EXACT MODULES)
    nav_options = [
        "📊 Executive Dashboard",
        "📹 Live AI Monitoring",
        "👥 Worker Directory",
        "👤 Face Recognition & Registration",
        "📋 Worker Safety Records",
        "🚨 Safety Violations Log",
        "📈 Safety Compliance Analytics",
        "📑 Audit Reports Generator",
        "⚙️ Settings"
    ]

    selected_module = st.radio(
        label="Workspace Navigation",
        options=nav_options,
        index=0,
        key="safevision_vertical_nav",
        label_visibility="collapsed"
    )

    # 3. SIDEBAR FOOTER & SYSTEM TELEMETRY
    total_enrolled = count_workers()
    diag = detector.get_diagnostics()

    st.markdown(f"""
        <div style="padding-top:1.1rem; margin-top:1.2rem; border-top:1px solid #C9D1C1;">
            <div style="background-color:#FFFFFF; border:1px solid #DED8CE; border-radius:8px; padding:0.8rem 0.9rem; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="font-size:0.72rem; font-weight:700; color:#2C332C; text-transform:uppercase; letter-spacing:0.05em;">SYSTEM TELEMETRY</span>
                    <span style="font-size:0.68rem; font-weight:700; color:#2E7D32; background-color:#EBF5EE; padding:2px 7px; border-radius:5px; border:1px solid #BCE3C5;">ONLINE</span>
                </div>
                <div style="font-size:0.75rem; color:#6B736B; margin-bottom:2px;">Model: <strong style="color:#2C332C;">{'best.pt (Custom PPE)' if diag['is_custom_model'] else 'YOLO11n (Base)'}</strong></div>
                <div style="font-size:0.75rem; color:#6B736B; margin-bottom:2px;">Face AI: <strong style="color:#C66B48;">Active ({total_enrolled} Enrolled)</strong></div>
                <div style="font-size:0.75rem; color:#6B736B; margin-bottom:2px;">Inference: <strong style="color:#2C332C;">Intel CPU (DirectShow)</strong></div>
                <div style="font-size:0.75rem; color:#6B736B;">Latency: <strong style="color:#2E7D32;">{diag['last_inference_ms']} ms</strong></div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.72rem; color:#6B736B; font-weight:550; padding:0 0.2rem;">
                <span>Version {config.APP_VERSION}</span>
                <span title="Use top arrow to collapse sidebar">◀ Sidebar Toggle ▶</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN VIEW: PROFESSIONAL APPLICATION HEADER
# -----------------------------------------------------------------------------

# Clean Human-Engineered Breadcrumb Header Bar
module_name_clean = selected_module.split(" ", 1)[1] if " " in selected_module else selected_module

module_descriptions = {
    "Executive Dashboard": "Real-time facility safety metrics, compliance rate, and recent inspection alerts.",
    "Live AI Monitoring": "Continuous camera feed analyzing personnel PPE compliance and facial identification.",
    "Worker Directory": "Registered laboratory personnel and facial biometric profile records.",
    "Face Recognition & Registration": "Biometric face registration and real-time similarity verification test.",
    "Worker Safety Records": "Historical audit trail of safety verifications, confidence scores, and compliance verdicts.",
    "Safety Violations Log": "Detailed incident log of non-compliant entries missing safety helmets or shoes.",
    "Safety Compliance Analytics": "Trend charts, violation distributions, and PPE adherence performance.",
    "Audit Reports Generator": "Formal safety compliance report export for laboratory audits and inspections.",
    "Settings": "Computer vision thresholds, facial biometrics, and facility safety policies."
}

page_subtitle = module_descriptions.get(module_name_clean, "Industrial Laboratory PPE Compliance & Worker Safety Monitoring")

st.markdown(f"""
<div class="app-card" style="margin-bottom:1.4rem; padding:1.15rem 1.5rem; border-radius:10px;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem;">
                <span style="font-size:0.74rem; font-weight:700; color:#C66B48; text-transform:uppercase; letter-spacing:0.06em;">SafeVision AI</span>
                <span style="color:#C9D1C1; font-weight:600;">/</span>
                <span style="font-size:0.74rem; font-weight:600; color:#6B736B; text-transform:uppercase; letter-spacing:0.03em;">Industrial Safety Monitoring System</span>
            </div>
            <h1 style="margin:0; font-size:1.65rem; color:#2C332C; font-weight:800; letter-spacing:-0.02em; line-height:1.25;">
                {module_name_clean}
            </h1>
            <div style="margin-top:0.25rem; font-size:0.86rem; color:#6B736B; font-weight:500;">
                {page_subtitle}
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;">
            <div class="status-pill-safe">
                <span>●</span>
                <span>System Operational</span>
            </div>
            <div class="status-pill-neutral">
                <span>{count_workers()} Registered Staff</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =============================================================================
if "Executive Dashboard" in selected_module:
    stats = get_safety_summary_stats()

    # 5 Clean Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="app-card border-accent-neutral">
            <div class="metric-title">Total Monitored</div>
            <div class="metric-value">{stats['total_records']}</div>
            <div class="metric-caption">{stats['total_workers']} unique workers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="app-card border-accent-safe">
            <div class="metric-title" style="color:#2E7D32;">Compliant Personnel</div>
            <div class="metric-value" style="color:#2E7D32;">{stats['safe_count']}</div>
            <div class="metric-caption">Today: {stats['today_safe']} verified safe</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="app-card border-accent-alert">
            <div class="metric-title" style="color:#C66B48;">Safety Violations</div>
            <div class="metric-value" style="color:#C66B48;">{stats['unsafe_count']}</div>
            <div class="metric-caption">Today: {stats['today_unsafe']} incidents</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="app-card border-accent-warn">
            <div class="metric-title" style="color:#C88F35;">Boundary Occluded</div>
            <div class="metric-value" style="color:#C88F35;">{stats['unknown_count']}</div>
            <div class="metric-caption">Partial camera visibility</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="app-card border-accent-alert">
            <div class="metric-title" style="color:#C66B48;">Compliance Rate</div>
            <div class="metric-value" style="color:#2C332C;">{stats['compliance_pct']}%</div>
            <div class="metric-caption">Today rate: {stats['today_compliance_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.5, 1.1])

    with c_left:
        st.markdown("##### Recent Safety Violations")
        violations = get_recent_violations(limit=6)
        if violations:
            v_df = pd.DataFrame(violations)
            display_cols = ["worker_id", "date", "time", "helmet_status", "safety_shoe_status", "violation_details"]
            clean_display = v_df[display_cols].rename(columns={
                "worker_id": "Worker ID / Name",
                "date": "Date",
                "time": "Time",
                "helmet_status": "Helmet",
                "safety_shoe_status": "Footwear",
                "violation_details": "Incident Reason"
            })
            st.dataframe(clean_display, use_container_width=True, hide_index=True)
        else:
            st.info("No safety violations recorded. Facility is operating within safety guidelines.")

    with c_right:
        st.markdown("##### Equipment Compliance Breakdown")
        comp_df = pd.DataFrame({
            "Equipment Item": ["Safety Helmet", "Safety Shoes", "Overall Compliance"],
            "Compliance (%)": [
                stats["helmet_compliance_pct"],
                stats["shoe_compliance_pct"],
                stats["compliance_pct"]
            ]
        })
        fig = px.bar(
            comp_df,
            x="Compliance (%)",
            y="Equipment Item",
            orientation="h",
            color="Compliance (%)",
            color_continuous_scale=["#C66B48", "#C88F35", "#2E7D32"],
            range_x=[0, 100],
            text="Compliance (%)"
        )
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#2C332C", family="Segoe UI, sans-serif"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=250,
            coloraxis_showscale=False
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2: LIVE AI MONITORING
# =============================================================================
elif "Live AI Monitoring" in selected_module:
    feed_mode = st.radio(
        "Camera Input Source",
        [
            "📸 Browser Device Camera (Recommended for Cloud / Mobile / Laptop)",
            "💻 Local Hardware Camera (DirectShow / USB Workstation)",
            "📁 Static Photo & Benchmark Inspection"
        ],
        horizontal=True,
        key="live_monitoring_feed_mode"
    )
    st.markdown("---")

    # -------------------------------------------------------------------------
    # MODE 1: BROWSER DEVICE CAMERA (WORKS 100% ON STREAMLIT CLOUD & MOBILE)
    # -------------------------------------------------------------------------
    if feed_mode == "📸 Browser Device Camera (Recommended for Cloud / Mobile / Laptop)":
        st.caption("Captures live video frames directly from your device webcam or phone camera through the web browser.")

        b_col1, b_col2 = st.columns([2.0, 1.1])
        with b_col1:
            cam_snap = st.camera_input("Capture frame from your device camera", key="cloud_browser_camera")

        with b_col2:
            st.markdown("##### Live Personnel & Status")
            b_status = st.empty()

        if cam_snap is not None:
            file_bytes = np.asarray(bytearray(cam_snap.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                h, w = frame.shape[:2]
                with st.spinner("Analyzing PPE Compliance & Facial Biometrics..."):
                    dets = detector.detect(frame, face_recognizer=face_recognizer)
                    tracked = tracker.update(dets["persons"])

                    evaluations = []
                    for p in tracked:
                        wid = p["worker_id"]
                        raw_id = p["raw_id"]
                        w_name = p.get("worker_name", wid)
                        w_role = p.get("role", "Technician")
                        w_dept = p.get("department", "Mechanical Lab")
                        is_rec = p.get("is_recognized", False)

                        evaluation = verifier.evaluate_worker(
                            worker_id=wid,
                            person_box=p["bbox"],
                            person_confidence=p["confidence"],
                            helmet_boxes=dets["helmets"],
                            cap_boxes=dets["caps"],
                            shoe_boxes=dets["shoes"],
                            frame_width=w,
                            frame_height=h,
                            face_box=p.get("face_box"),
                            faces=dets.get("faces", []),
                            worker_name=w_name,
                            role=w_role,
                            department=w_dept,
                            is_recognized=is_rec
                        )

                        should_record, reason = tracker.should_record(
                            raw_id=raw_id,
                            status=evaluation["status"],
                            stability_score=evaluation["stability_score"],
                            frames_seen=evaluation["frames_analyzed"]
                        )

                        if should_record:
                            add_safety_record(
                                worker_id=wid,
                                helmet_status=evaluation["helmet_status"],
                                safety_shoe_status=evaluation["safety_shoe_status"],
                                overall_status=evaluation["status"],
                                confidence=evaluation["stability_score"],
                                violation_details=evaluation["details"],
                                head_visible=evaluation["head_visible"],
                                feet_visible=evaluation["feet_visible"],
                                worker_name=evaluation.get("worker_name", wid)
                            )
                        evaluations.append(evaluation)

                    annotated = annotate_frame(
                        frame,
                        evaluations,
                        fps=30.0,
                        faces=dets.get("faces", [])
                    )

                with b_col1:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="AI Safety & Identification Verdict", use_container_width=True)

                with b_status.container():
                    if evaluations:
                        for ev in evaluations:
                            color_box = STATUS_COLORS[ev["status"]]
                            pill_class = "status-pill-safe" if ev["status"] == "SAFE" else ("status-pill-alert" if ev["status"] == "NOT_SAFE" else "status-pill-warn")
                            w_display_name = ev.get("worker_name", ev["worker_id"])
                            is_recognized_person = ev.get("is_recognized", False)

                            st.markdown(f"""
                                <div class="app-card" style="border-left:3px solid {color_box} !important; padding:0.9rem; margin-bottom:0.65rem;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                                        <strong style="color:#2C332C; font-size:0.98rem; font-weight:750;">{w_display_name}</strong>
                                        <span class="{pill_class}">{ev['status']}</span>
                                    </div>
                                    <div style="margin-bottom:0.35rem;">
                                        {'<span class="status-pill-safe">Verified Staff • ' + ev.get("role", "Staff") + '</span>' if is_recognized_person else '<span class="status-pill-alert">Unregistered Visitor</span>'}
                                    </div>
                                    <div style="font-size:0.78rem; color:#2C332C; font-weight:600; margin-top:0.25rem;">
                                        Helmet: <strong style="color:{'#2E7D32' if ev['helmet_status']=='WORN' else '#C66B48'};">{ev['helmet_status']}</strong> • Footwear: <strong style="color:{'#2E7D32' if ev['safety_shoe_status']=='WORN' else ('#C88F35' if ev['safety_shoe_status']=='UNCLEAR' else '#C66B48')};">{ev['safety_shoe_status']}</strong>
                                    </div>
                                    <div style="font-size:0.75rem; color:#6B736B; margin-top:0.25rem;">
                                        {ev['details']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No worker body detected. Please ensure your upper body and head are visible.")
        else:
            with b_status.container():
                st.info("Click 'Take Photo' above to run live PPE & facial verification.")

    # -------------------------------------------------------------------------
    # MODE 2: LOCAL HARDWARE CAMERA (FOR LOCAL LAB WORKSTATIONS)
    # -------------------------------------------------------------------------
    elif feed_mode == "💻 Local Hardware Camera (DirectShow / USB Workstation)":
        st.info("💡 **Local Lab Camera**: Connects to physical USB/RTSP cameras attached to this computer. When running on Streamlit Cloud, switch to **📸 Browser Device Camera** above.")

        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.3, 1.3, 1.5, 1.9])
        with ctrl1:
            run_camera = st.toggle("Start Camera Stream", value=False, key="camera_toggle_btn")
        with ctrl2:
            capture_btn = st.button("Capture Snapshot", use_container_width=True)
        with ctrl3:
            cam_index = st.selectbox(
                "Camera Device",
                options=[0, 1, 2],
                format_func=lambda x: f"Webcam Device {x}",
                index=int(get_setting("camera_index", "0")),
                label_visibility="collapsed"
            )
        with ctrl4:
            st.markdown(
                f"<div style='padding-top:0.4rem; font-size:0.86rem; color:#2C332C;'>Stream Status: <strong style='color:{'#2E7D32' if run_camera else '#6B736B'};'>{'● STREAMING LIVE' if run_camera else '○ CAMERA STANDBY'}</strong></div>",
                unsafe_allow_html=True
            )

        video_col, status_col = st.columns([2.2, 1.1])
        with video_col:
            video_placeholder = st.empty()
        with status_col:
            st.markdown("##### Live Personnel & Status")
            status_placeholder = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(cam_index)

            if not cap.isOpened():
                st.error(f"Unable to access local camera device index {cam_index}.")
                st.info("💡 Cloud hosting environments do not have physical USB webcams. Please switch to **📸 Browser Device Camera** above.")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                fps_timer = time.time()
                frame_count = 0
                live_fps = 0.0

                try:
                    while run_camera:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Frame acquisition interrupted.")
                            break

                        h, w = frame.shape[:2]
                        dets = detector.detect(frame, face_recognizer=face_recognizer)
                        tracked = tracker.update(dets["persons"])

                        evaluations = []
                        for p in tracked:
                            wid = p["worker_id"]
                            raw_id = p["raw_id"]
                            w_name = p.get("worker_name", wid)
                            w_role = p.get("role", "Technician")
                            w_dept = p.get("department", "Mechanical Lab")
                            is_rec = p.get("is_recognized", False)

                            evaluation = verifier.evaluate_worker(
                                worker_id=wid,
                                person_box=p["bbox"],
                                person_confidence=p["confidence"],
                                helmet_boxes=dets["helmets"],
                                cap_boxes=dets["caps"],
                                shoe_boxes=dets["shoes"],
                                frame_width=w,
                                frame_height=h,
                                face_box=p.get("face_box"),
                                faces=dets.get("faces", []),
                                worker_name=w_name,
                                role=w_role,
                                department=w_dept,
                                is_recognized=is_rec
                            )

                            should_record, reason = tracker.should_record(
                                raw_id=raw_id,
                                status=evaluation["status"],
                                stability_score=evaluation["stability_score"],
                                frames_seen=evaluation["frames_analyzed"]
                            )

                            if should_record:
                                add_safety_record(
                                    worker_id=wid,
                                    helmet_status=evaluation["helmet_status"],
                                    safety_shoe_status=evaluation["safety_shoe_status"],
                                    overall_status=evaluation["status"],
                                    confidence=evaluation["stability_score"],
                                    violation_details=evaluation["details"],
                                    head_visible=evaluation["head_visible"],
                                    feet_visible=evaluation["feet_visible"],
                                    worker_name=evaluation.get("worker_name", wid)
                                )

                            evaluations.append(evaluation)

                        frame_count += 1
                        now_t = time.time()
                        if now_t - fps_timer >= 1.0:
                            live_fps = frame_count / (now_t - fps_timer)
                            frame_count = 0
                            fps_timer = now_t

                        annotated = annotate_frame(
                            frame,
                            evaluations,
                            fps=live_fps,
                            faces=dets.get("faces", [])
                        )

                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        with status_placeholder.container():
                            if evaluations:
                                for ev in evaluations:
                                    color_box = STATUS_COLORS[ev["status"]]
                                    pill_class = "status-pill-safe" if ev["status"] == "SAFE" else ("status-pill-alert" if ev["status"] == "NOT_SAFE" else "status-pill-warn")
                                    w_display_name = ev.get("worker_name", ev["worker_id"])
                                    is_recognized_person = ev.get("is_recognized", False)

                                    st.markdown(f"""
                                        <div class="app-card" style="border-left:3px solid {color_box} !important; padding:0.9rem; margin-bottom:0.65rem;">
                                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                                                <strong style="color:#2C332C; font-size:0.98rem; font-weight:750;">{w_display_name}</strong>
                                                <span class="{pill_class}">{ev['status']}</span>
                                            </div>
                                            <div style="margin-bottom:0.35rem;">
                                                {'<span class="status-pill-safe">Verified Staff • ' + ev.get("role", "Staff") + '</span>' if is_recognized_person else '<span class="status-pill-alert">Unregistered Visitor</span>'}
                                            </div>
                                            <div style="font-size:0.78rem; color:#2C332C; font-weight:600; margin-top:0.25rem;">
                                                Helmet: <strong style="color:{'#2E7D32' if ev['helmet_status']=='WORN' else '#C66B48'};">{ev['helmet_status']}</strong> • Footwear: <strong style="color:{'#2E7D32' if ev['safety_shoe_status']=='WORN' else ('#C88F35' if ev['safety_shoe_status']=='UNCLEAR' else '#C66B48')};">{ev['safety_shoe_status']}</strong>
                                            </div>
                                            <div style="font-size:0.75rem; color:#6B736B; margin-top:0.25rem;">
                                                {ev['details']}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                if dets.get("faces"):
                                    st.markdown(f"""
                                        <div class="app-card border-accent-alert" style="padding:0.85rem;">
                                            <div style="color:#C66B48; font-weight:750; font-size:0.88rem;">{len(dets['faces'])} Face(s) Detected</div>
                                            <div style="font-size:0.78rem; color:#6B736B; margin-top:2px;">Comparing biometrics with registered directory...</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info("No active personnel detected in current camera frame.")

                        if capture_btn:
                            cap_path = config.RECORDS_DIR / f"capture_{int(time.time())}.jpg"
                            cv2.imwrite(str(cap_path), frame)
                            st.sidebar.success(f"Captured: {cap_path.name}")

                        time.sleep(0.02)
                finally:
                    cap.release()
        else:
            standby_img = np.zeros((480, 640, 3), dtype=np.uint8)
            standby_img[:] = (32, 38, 32)
            margin = 40
            corner_len = 30
            c_color = (80, 100, 80)
            cv2.line(standby_img, (margin, margin), (margin + corner_len, margin), c_color, 2)
            cv2.line(standby_img, (margin, margin), (margin, margin + corner_len), c_color, 2)
            cv2.line(standby_img, (640 - margin, margin), (640 - margin - corner_len, margin), c_color, 2)
            cv2.line(standby_img, (640 - margin, margin), (640 - margin, margin + corner_len), c_color, 2)
            cv2.line(standby_img, (margin, 480 - margin), (margin + corner_len, 480 - margin), c_color, 2)
            cv2.line(standby_img, (margin, 480 - margin), (margin, 480 - margin - corner_len), c_color, 2)
            cv2.line(standby_img, (640 - margin, 480 - margin), (640 - margin - corner_len, 480 - margin), c_color, 2)
            cv2.line(standby_img, (640 - margin, 480 - margin), (640 - margin, 480 - margin - corner_len), c_color, 2)

            cv2.putText(
                standby_img,
                "CAMERA FEED INACTIVE",
                (195, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (225, 230, 225),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                standby_img,
                "Toggle 'Start Camera Stream' above to begin real-time monitoring",
                (100, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (150, 165, 150),
                1,
                cv2.LINE_AA
            )
            video_placeholder.image(cv2.cvtColor(standby_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            status_placeholder.info("Camera is on standby. Activate stream to begin monitoring.")

    # -------------------------------------------------------------------------
    # MODE 3: STATIC PHOTO & BENCHMARK INSPECTION
    # -------------------------------------------------------------------------
    else:
        st.caption("Upload image files or run standard benchmarks to verify PPE compliance.")

        up_col1, up_col2 = st.columns([1.5, 1.0])
        with up_col1:
            uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"], key="photo_scanner_uploader")
        with up_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            use_sample = st.button("Load Built-in Test Sample", use_container_width=True)

        test_img = None
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            test_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif use_sample:
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            test_img[:] = (235, 240, 235)
            cv2.circle(test_img, (320, 140), 50, (180, 160, 150), -1)
            cv2.ellipse(test_img, (320, 300), (100, 150), 0, 0, 360, (70, 80, 120), -1)
            cv2.rectangle(test_img, (270, 95), (370, 135), (72, 107, 198), -1)

        if test_img is not None:
            th, tw = test_img.shape[:2]
            with st.spinner("Processing image inspection..."):
                t_dets = detector.detect(test_img, face_recognizer=face_recognizer)
                t_tracked = tracker.update(t_dets["persons"])

                t_evals = []
                for p in t_tracked:
                    wid = p["worker_id"]
                    t_eval = verifier.evaluate_worker(
                        worker_id=wid,
                        person_box=p["bbox"],
                        person_confidence=p["confidence"],
                        helmet_boxes=t_dets["helmets"],
                        cap_boxes=t_dets["caps"],
                        shoe_boxes=t_dets["shoes"],
                        frame_width=tw,
                        frame_height=th,
                        face_box=p.get("face_box"),
                        faces=t_dets.get("faces", []),
                        worker_name=p.get("worker_name", wid),
                        role=p.get("role", "Technician"),
                        department=p.get("department", "Mechanical Lab"),
                        is_recognized=p.get("is_recognized", False)
                    )
                    t_evals.append(t_eval)

                annotated_test = annotate_frame(
                    test_img,
                    t_evals,
                    fps=30.0,
                    faces=t_dets.get("faces", [])
                )

            res_v, res_info = st.columns([2.0, 1.1])
            with res_v:
                st.image(cv2.cvtColor(annotated_test, cv2.COLOR_BGR2RGB), caption="Inspection Result", use_container_width=True)

            with res_info:
                st.markdown("##### Inspection Verdict")
                st.markdown(f"**Faces Detected:** `{len(t_dets.get('faces', []))}`")
                st.markdown(f"**Workers Tracked:** `{len(t_evals)}`")

                for ev in t_evals:
                    color_box = STATUS_COLORS[ev["status"]]
                    pill_class = "status-pill-safe" if ev["status"] == "SAFE" else ("status-pill-alert" if ev["status"] == "NOT_SAFE" else "status-pill-warn")
                    w_name_display = ev.get("worker_name", ev["worker_id"])
                    st.markdown(f"""
                        <div class="app-card" style="border-left:3px solid {color_box} !important; padding:0.85rem; margin-top:0.6rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:#2C332C; font-size:0.95rem;">{w_name_display}</strong>
                                <span class="{pill_class}">{ev['status']}</span>
                            </div>
                            <div style="font-size:0.75rem; color:#6B736B; margin-top:0.25rem;">
                                {ev.get('role', 'Technician')} • {ev.get('department', 'Lab')}
                            </div>
                            <div style="font-size:0.78rem; color:#2C332C; font-weight:600; margin-top:0.25rem;">
                                Helmet: <strong>{ev['helmet_status']}</strong> • Footwear: <strong>{ev['safety_shoe_status']}</strong>
                            </div>
                            <div style="font-size:0.73rem; color:#6B736B; margin-top:0.2rem;">
                                {ev['details']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                if st.button("Save Result to Audit Records"):
                    for ev in t_evals:
                        add_safety_record(
                            worker_id=ev["worker_id"],
                            helmet_status=ev["helmet_status"],
                            safety_shoe_status=ev["safety_shoe_status"],
                            overall_status=ev["status"],
                            confidence=ev["stability_score"],
                            violation_details=ev["details"],
                            head_visible=ev["head_visible"],
                            feet_visible=ev["feet_visible"],
                            worker_name=ev.get("worker_name", ev["worker_id"])
                        )
                    st.success("Successfully committed verification records.")


# =============================================================================
# PAGE 3: WORKER DIRECTORY
# =============================================================================
elif "Worker Directory" in selected_module:
    search_col, count_col = st.columns([2.5, 1.0])
    with search_col:
        search_filter = st.text_input("Search directory by name, badge ID, or department", placeholder="e.g. Rajesh, W-101, Machinist")
    with count_col:
        st.markdown(f"""
            <div class="app-card border-accent-neutral" style="padding:0.7rem 1rem; text-align:center;">
                <div class="metric-title">Enrolled Staff</div>
                <div style="font-size:1.4rem; font-weight:800; color:#2C332C;">{count_workers()}</div>
            </div>
        """, unsafe_allow_html=True)

    workers = get_all_workers()
    if search_filter:
        s_low = search_filter.strip().lower()
        workers = [
            w for w in workers
            if s_low in w.get("full_name", "").lower()
            or s_low in w.get("worker_id", "").lower()
            or s_low in w.get("department", "").lower()
            or s_low in w.get("role", "").lower()
        ]

    st.markdown("---")

    if workers:
        cols = st.columns(3)
        for idx, w in enumerate(workers):
            c = cols[idx % 3]
            with c:
                photo_path = w.get("photo_path")
                avatar_loaded = False
                if photo_path and os.path.exists(photo_path):
                    try:
                        img = Image.open(photo_path)
                        st.image(img, width=100)
                        avatar_loaded = True
                    except Exception:
                        pass

                if not avatar_loaded:
                    st.markdown("<div style='font-size:2.5rem; margin-bottom:0.35rem;'>👤</div>", unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="app-card" style="margin-bottom:0.75rem;">
                        <div style="font-weight:750; font-size:1rem; color:#2C332C;">{w['full_name']}</div>
                        <div style="font-size:0.78rem; color:#C66B48; font-weight:700;">ID: {w['worker_id']}</div>
                        <div style="font-size:0.78rem; color:#2C332C; margin-top:0.25rem;">Role: {w['role']}</div>
                        <div style="font-size:0.75rem; color:#6B736B;">Dept: {w['department']}</div>
                    </div>
                """, unsafe_allow_html=True)

                if st.button(f"Delete {w['worker_id']}", key=f"del_{w['worker_id']}"):
                    delete_worker(w['worker_id'])
                    face_recognizer.reload_workers()
                    st.success(f"Removed {w['full_name']} from registry.")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("No personnel currently found in directory. Use 'Face Recognition & Registration' in the sidebar to enroll personnel.")


# =============================================================================
# PAGE 4: FACE RECOGNITION & REGISTRATION
# =============================================================================
elif "Face Recognition & Registration" in selected_module:
    enroll_col, test_col = st.columns([1.4, 1.2])

    with enroll_col:
        st.markdown("##### 1. Register Worker Biometrics")
        st.caption("Enroll a laboratory worker with photo for automated identification on live video feeds.")

        new_id = st.text_input("Worker Badge / Employee ID*", placeholder="e.g. W-104 or EMP-042", key="reg_wid")
        new_name = st.text_input("Full Name*", placeholder="e.g. Alex Johnson", key="reg_wname")
        new_role = st.text_input("Role / Designation*", placeholder="e.g. Lab Technician, Machinist", key="reg_wrole")
        new_dept = st.text_input("Department / Laboratory*", placeholder="e.g. Mechanical Workshop A", key="reg_wdept")

        photo_mode = st.radio("Photo Capture Method", ["Live Camera Snapshot", "Upload Image File"], horizontal=True, key="photo_mode_selector")

        captured_bgr = None
        if photo_mode == "Live Camera Snapshot":
            cam_snap = st.camera_input("Take facial photograph", key="enroll_camera_snap")
            if cam_snap is not None:
                file_bytes = np.asarray(bytearray(cam_snap.read()), dtype=np.uint8)
                captured_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            uploaded_photo = st.file_uploader("Upload frontal facial photo", type=["jpg", "jpeg", "png"], key="enroll_upload_file")
            if uploaded_photo is not None:
                file_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
                captured_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if captured_bgr is not None:
            gray = cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2GRAY)
            faces = []
            if detector.face_cascade is not None and not detector.face_cascade.empty():
                faces = detector.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                preview_img = captured_bgr.copy()
                cv2.rectangle(preview_img, (fx, fy), (fx+fw, fy+fh), (46, 125, 50), 2)
                cv2.putText(preview_img, "FACE OK", (fx, max(20, fy-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 125, 50), 2)
                st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="Face Detected & Aligned", width=180)
                st.success("Clear facial features verified.")
            else:
                st.warning("No distinct face detected. Please ensure good lighting and frontal alignment.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save Personnel Profile", use_container_width=True, key="enroll_save_btn"):
            if not new_id or not new_name:
                st.error("Employee ID and Full Name are mandatory fields.")
            elif captured_bgr is None:
                st.error("Please provide a facial photograph.")
            else:
                with st.spinner("Extracting facial biometric descriptors and saving..."):
                    success, msg = face_recognizer.register_new_worker(
                        worker_id=new_id,
                        full_name=new_name,
                        role=new_role or "Technician",
                        department=new_dept or "Mechanical Lab",
                        face_bgr=captured_bgr
                    )
                    if success:
                        st.success(f"Successfully registered {new_name}.")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {msg}")

    with test_col:
        st.markdown("##### 2. Biometric Verification Diagnostic")
        st.caption("Upload any face photo to test similarity score against enrolled database records.")

        test_file = st.file_uploader("Upload face photo to test", type=["jpg", "png", "jpeg"], key="test_face_matcher_uploader")
        if test_file is not None:
            f_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
            t_bgr = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)

            st.image(cv2.cvtColor(t_bgr, cv2.COLOR_BGR2RGB), caption="Test Face", width=200)

            match_result = face_recognizer.identify_face(t_bgr)
            st.markdown("###### Biometric Match Verdict:")
            if match_result["is_recognized"]:
                st.success(f"Matched: **{match_result['full_name']}** ({match_result['worker_id']})")
                st.markdown(f"- **Designation:** {match_result['role']}")
                st.markdown(f"- **Department:** {match_result['department']}")
                st.markdown(f"- **Cosine Similarity:** `{match_result['confidence'] * 100:.1f}%` (Threshold: `{face_recognizer.match_threshold * 100:.1f}%`)")
            else:
                st.warning(f"Unregistered Visitor / No Match (Best similarity: `{match_result['confidence'] * 100:.1f}%`)")


# =============================================================================
# PAGE 5: WORKER SAFETY RECORDS
# =============================================================================
elif "Worker Safety Records" in selected_module:
    f1, f2, f3, f4 = st.columns([1.5, 1.2, 1.2, 1.5])
    with f1:
        search_query = st.text_input("Search records by name, ID, or record #", placeholder="e.g. Rajesh or Worker 001")
    with f2:
        status_filter = st.selectbox("Status Filter", ["ALL", "SAFE", "NOT_SAFE", "UNKNOWN"])
    with f3:
        date_from = st.date_input("From Date", value=None)
    with f4:
        date_to = st.date_input("To Date", value=None)

    records_df = get_filtered_records_df(
        status=status_filter if status_filter != "ALL" else None,
        date_from=date_from.strftime("%Y-%m-%d") if date_from else None,
        date_to=date_to.strftime("%Y-%m-%d") if date_to else None,
        search=search_query if search_query else None
    )

    st.markdown(f"**Found {len(records_df)} safety audit records**")

    if not records_df.empty:
        display_df = records_df.copy()
        if "worker_name" not in display_df.columns:
            display_df["worker_name"] = display_df["worker_id"]

        show_cols = ["record_id", "worker_id", "worker_name", "date", "time", "helmet_status", "safety_shoe_status", "overall_status", "confidence", "violation_details"]
        avail_cols = [c for c in show_cols if c in display_df.columns]
        clean_tbl = display_df[avail_cols].rename(columns={
            "record_id": "Record ID",
            "worker_id": "Badge ID",
            "worker_name": "Personnel Name",
            "date": "Date",
            "time": "Time",
            "helmet_status": "Helmet",
            "safety_shoe_status": "Footwear",
            "overall_status": "Compliance Status",
            "confidence": "Confidence",
            "violation_details": "Inspection Details"
        })

        st.dataframe(clean_tbl, use_container_width=True, hide_index=True)

        csv_bytes = generate_csv_download(records_df)
        st.download_button(
            label="Export Filtered Records (CSV)",
            data=csv_bytes,
            file_name=f"worker_safety_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No records matched the selected query parameters.")


# =============================================================================
# PAGE 6: SAFETY VIOLATIONS LOG
# =============================================================================
elif "Safety Violations" in selected_module:
    violations_df = get_filtered_records_df(status="NOT_SAFE")

    if not violations_df.empty:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
                <div class="app-card border-accent-alert">
                    <div class="metric-title" style="color:#C66B48;">Total Recorded Violations</div>
                    <div class="metric-value" style="color:#C66B48;">{len(violations_df)}</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            no_helmets = len(violations_df[violations_df["helmet_status"] == "MISSING"])
            st.markdown(f"""
                <div class="app-card border-accent-alert">
                    <div class="metric-title" style="color:#C66B48;">Missing Helmet Incidents</div>
                    <div class="metric-value" style="color:#C66B48;">{no_helmets}</div>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            no_shoes = len(violations_df[violations_df["safety_shoe_status"] == "MISSING"])
            st.markdown(f"""
                <div class="app-card border-accent-warn">
                    <div class="metric-title" style="color:#C88F35;">Missing Footwear Incidents</div>
                    <div class="metric-value" style="color:#C88F35;">{no_shoes}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        display_viol = violations_df[["record_id", "worker_id", "date", "time", "helmet_status", "safety_shoe_status", "confidence", "violation_details"]].rename(columns={
            "record_id": "Record ID",
            "worker_id": "Worker ID / Name",
            "date": "Date",
            "time": "Time",
            "helmet_status": "Helmet",
            "safety_shoe_status": "Footwear",
            "confidence": "Confidence",
            "violation_details": "Violation Reason"
        })
        st.dataframe(display_viol, use_container_width=True, hide_index=True)

        csv_viol = generate_csv_download(violations_df)
        st.download_button(
            label="Export Violations Log (CSV)",
            data=csv_viol,
            file_name=f"safety_violations_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.success("No safety violations recorded in the audit database.")


# =============================================================================
# PAGE 7: SAFETY COMPLIANCE ANALYTICS
# =============================================================================
elif "Compliance Analytics" in selected_module:
    all_df = get_all_records_df()

    if not all_df.empty:
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("##### Compliant vs Violation Ratio")
            status_counts = all_df["overall_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            fig_donut = px.pie(
                status_counts,
                values="Count",
                names="Status",
                hole=0.45,
                color="Status",
                color_discrete_map={
                    "SAFE": "#2E7D32",
                    "NOT_SAFE": "#C66B48",
                    "UNKNOWN": "#C88F35"
                }
            )
            fig_donut.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#2C332C", family="Segoe UI, sans-serif"),
                height=300,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with ch2:
            st.markdown("##### Violations by PPE Category")
            viol_df = all_df[all_df["overall_status"] == "NOT_SAFE"]
            if not viol_df.empty:
                def categorize_viol(row):
                    if row["helmet_status"] == "MISSING" and row["safety_shoe_status"] == "MISSING":
                        return "Missing Both (Helmet & Shoes)"
                    elif row["helmet_status"] == "MISSING":
                        return "Missing Safety Helmet"
                    elif row["safety_shoe_status"] == "MISSING":
                        return "Missing Safety Footwear"
                    return "Other Non-compliance"

                viol_df["type"] = viol_df.apply(categorize_viol, axis=1)
                type_counts = viol_df["type"].value_counts().reset_index()
                type_counts.columns = ["Violation Type", "Count"]

                fig_vbar = px.bar(
                    type_counts,
                    x="Count",
                    y="Violation Type",
                    orientation="h",
                    color="Violation Type",
                    color_discrete_sequence=["#C66B48", "#C88F35", "#B25A38"]
                )
                fig_vbar.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#2C332C", family="Segoe UI, sans-serif"),
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_vbar, use_container_width=True)
            else:
                st.info("No violations recorded to categorize.")

        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown("##### Daily Safety Violations Trend")
            daily_viol = all_df[all_df["overall_status"] == "NOT_SAFE"].groupby("date").size().reset_index(name="Violations")
            if not daily_viol.empty:
                fig_trend = px.line(
                    daily_viol,
                    x="date",
                    y="Violations",
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=["#C66B48"]
                )
                fig_trend.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#2C332C", family="Segoe UI, sans-serif"),
                    height=300,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No daily violations recorded.")

        with ch4:
            st.markdown("##### Daily Compliance Rate Trend (%)")
            daily_stats = all_df.groupby("date").agg(
                total=("id", "count"),
                safe=("overall_status", lambda s: (s == "SAFE").sum())
            ).reset_index()
            daily_stats["Compliance (%)"] = (daily_stats["safe"] / daily_stats["total"] * 100).round(1)

            fig_comp = px.area(
                daily_stats,
                x="date",
                y="Compliance (%)",
                color_discrete_sequence=["#2E7D32"]
            )
            fig_comp.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#2C332C", family="Segoe UI, sans-serif"),
                height=300,
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No records available to generate analytics.")


# =============================================================================
# PAGE 8: AUDIT REPORTS GENERATOR
# =============================================================================
elif "Audit Reports" in selected_module:
    report_type = st.radio("Audit Report Timeframe", ["Daily Audit", "Weekly Audit (7 Days)", "Monthly Audit"], horizontal=True)

    if report_type == "Daily Audit":
        target_date = st.date_input("Report Date", value=datetime.now())
        df, stats, title = generate_daily_report(target_date.strftime("%Y-%m-%d"))
    elif report_type == "Weekly Audit (7 Days)":
        target_date = st.date_input("Week Ending Date", value=datetime.now())
        df, stats, title = generate_weekly_report(target_date.strftime("%Y-%m-%d"))
    else:
        year_month = st.text_input("Year and Month (YYYY-MM)", value=datetime.now().strftime("%Y-%m"))
        df, stats, title = generate_monthly_report(year_month)

    st.markdown("---")
    st.markdown(f"##### {title}")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
            <div class="app-card border-accent-neutral">
                <div class="metric-title">Monitored Entries</div>
                <div class="metric-value">{stats['total_entries']}</div>
            </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
            <div class="app-card border-accent-safe">
                <div class="metric-title" style="color:#2E7D32;">Unique Workers</div>
                <div class="metric-value" style="color:#2E7D32;">{stats['unique_workers']}</div>
            </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
            <div class="app-card border-accent-alert">
                <div class="metric-title" style="color:#C66B48;">Safe / Unsafe</div>
                <div class="metric-value" style="color:#2C332C; font-size:1.6rem;">{stats['safe_entries']} / {stats['unsafe_entries']}</div>
            </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
            <div class="app-card border-accent-alert">
                <div class="metric-title" style="color:#C66B48;">Compliance Rate</div>
                <div class="metric-value" style="color:#C66B48;">{stats['compliance_pct']}%</div>
            </div>
        """, unsafe_allow_html=True)

    if not df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df[["record_id", "worker_id", "date", "time", "helmet_status", "safety_shoe_status", "overall_status", "violation_details"]], use_container_width=True, hide_index=True)

        csv_data = generate_csv_download(df)
        st.download_button(
            label=f"Export {report_type} (CSV)",
            data=csv_data,
            file_name=f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No entries found for the selected report timeframe.")


# =============================================================================
# PAGE 9: SETTINGS
# =============================================================================
elif "Settings" in selected_module:
    with st.form("settings_form"):
        st.markdown("##### 1. Facial Biometrics & Identification")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            new_face_thresh = st.slider(
                "Face Match Cosine Similarity Threshold",
                min_value=0.30,
                max_value=0.85,
                value=float(get_setting("face_match_threshold", str(config.DEFAULT_FACE_MATCH_THRESHOLD))),
                step=0.05,
                help="Higher threshold requires stricter facial similarity; lower values allow broader tolerance."
            )
        with col_f2:
            enable_face_rec = st.checkbox(
                "Display Recognized Personnel Names on Live Camera",
                value=get_setting("enable_face_recognition", "true").lower() == "true",
                help="When enabled, verified worker names and roles are drawn onto live bounding boxes."
            )

        st.markdown("##### 2. Computer Vision & Inference Parameters")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            new_conf = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.90, value=float(get_setting("conf_threshold", str(config.DEFAULT_CONF_THRESHOLD))), step=0.05)
        with col_s2:
            new_iou = st.slider("IoU NMS Overlap Threshold", min_value=0.20, max_value=0.80, value=float(get_setting("iou_threshold", str(config.DEFAULT_IOU_THRESHOLD))), step=0.05)
        with col_s3:
            new_imgsz = st.selectbox("Inference Resolution (px)", options=[320, 480, 640], index=[320, 480, 640].index(int(get_setting("imgsz", str(config.DEFAULT_IMGSZ)))))

        st.markdown("##### 3. Workplace Safety Policies")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            allow_caps = st.checkbox(
                "Approve Safety Cap as Alternative to Hard Helmet",
                value=get_setting("allow_safety_cap", "true").lower() == "true",
                help="When checked, safety caps satisfy mandatory head protection requirements."
            )
        with col_p2:
            cam_idx = st.number_input("Default Camera Device Index", min_value=0, max_value=5, value=int(get_setting("camera_index", "0")))

        st.markdown("##### 4. Stability & De-bounce Rules")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            hysteresis = st.slider("Temporal Confirmation Window (Frames)", min_value=2, max_value=12, value=int(get_setting("hysteresis_frames", str(config.HYSTERESIS_WINDOW))))
        with col_t2:
            cooldown = st.slider("Worker Re-audit Cooldown (Seconds)", min_value=5, max_value=60, value=int(float(get_setting("cooldown_seconds", str(config.RECORD_COOLDOWN_SECONDS)))))

        submitted = st.form_submit_button("Save Configuration Changes", use_container_width=True)
        if submitted:
            set_setting("conf_threshold", str(new_conf))
            set_setting("iou_threshold", str(new_iou))
            set_setting("imgsz", str(new_imgsz))
            set_setting("allow_safety_cap", "true" if allow_caps else "false")
            set_setting("camera_index", str(cam_idx))
            set_setting("hysteresis_frames", str(hysteresis))
            set_setting("cooldown_seconds", str(cooldown))
            set_setting("face_match_threshold", str(new_face_thresh))
            set_setting("enable_face_recognition", "true" if enable_face_rec else "false")

            # Apply immediately to cached singletons
            detector.conf_threshold = new_conf
            detector.iou_threshold = new_iou
            detector.imgsz = new_imgsz
            verifier.allow_safety_cap = allow_caps
            verifier.hysteresis_window = hysteresis
            tracker.cooldown_sec = float(cooldown)
            face_recognizer.match_threshold = new_face_thresh

            st.success("System configuration saved successfully and applied to live inference pipeline.")

    # Diagnostics
    st.markdown("---")
    st.markdown("##### Model Diagnostics & Custom Training Guide")
    diag = detector.get_diagnostics()

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Active Inference Parameters**")
        st.json(diag)

    with d2:
        st.markdown("**Dataset Training Commands**")
        st.markdown("""
        To retrain a custom PPE detector on facility photography:
        ```bash
        # 1. Place lab images in dataset/images/train and labels in dataset/labels/train
        # 2. Run automated training on CPU:
        python training/train.py --epochs 30 --batch 8 --imgsz 640 --device cpu
        
        # 3. Validate mAP & Recall:
        python training/validate.py
        ```
        Trained model weights are automatically saved to `models/best.pt`.
        """)
