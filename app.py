"""
AI-Based Mechanical Lab & Industrial Worker Safety Monitoring System.
Advanced Computer Vision & Face Scanning Dashboard using YOLO, OpenCV & Streamlit.
Aesthetic Palette: White, Cyber Pink, Mint Green, and Electric Yellow Mixed Gradients.
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
from detection.detector import SafetyDetector
from detection.tracker import WorkerTracker
from detection.safety_checker import SafetyVerifier, annotate_frame
from reports.report_generator import (
    generate_daily_report,
    generate_weekly_report,
    generate_monthly_report,
    generate_csv_download
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Advanced Mixed Gradient Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-tech, Aesthetic UI Styling: White, Pink, Mint Green & Electric Yellow Mixed Gradient
st.markdown("""
<style>
    /* High-Tech Mixed Ambient Mesh Background */
    .stApp {
        background: radial-gradient(circle at 8% 18%, rgba(236, 72, 153, 0.12) 0%, transparent 35%),
                    radial-gradient(circle at 92% 12%, rgba(16, 185, 129, 0.12) 0%, transparent 35%),
                    radial-gradient(circle at 50% 88%, rgba(250, 204, 21, 0.10) 0%, transparent 40%),
                    radial-gradient(circle at 50% 10%, rgba(255, 255, 255, 0.08) 0%, transparent 30%),
                    linear-gradient(135deg, #090d16 0%, #111726 50%, #0a0e1a 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    /* Advanced Glowing Industrial Header */
    .industrial-header {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(250, 204, 21, 0.10) 50%, rgba(16, 185, 129, 0.15) 100%), rgba(17, 24, 39, 0.95);
        border: 1.5px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px -5px rgba(236, 72, 153, 0.15), 0 0 20px rgba(16, 185, 129, 0.10);
    }
    
    /* Header Gradient Text */
    .header-gradient-title {
        background: linear-gradient(90deg, #ffffff 0%, #f43f5e 32%, #facc15 65%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 1.55rem;
        letter-spacing: -0.02em;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.88);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.6);
    }
    
    /* Glowing Accent Card Variants */
    .card-white {
        border-top: 3.5px solid #ffffff;
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.12);
    }
    .card-green {
        border-top: 3.5px solid #10b981;
        box-shadow: 0 4px 22px rgba(16, 185, 129, 0.20);
    }
    .card-pink {
        border-top: 3.5px solid #ec4899;
        box-shadow: 0 4px 22px rgba(236, 72, 153, 0.20);
    }
    .card-yellow {
        border-top: 3.5px solid #facc15;
        box-shadow: 0 4px 22px rgba(250, 204, 21, 0.20);
    }
    .card-gradient {
        border-top: 3.5px solid #ec4899;
        border-image: linear-gradient(90deg, #ec4899, #facc15, #10b981) 1;
        box-shadow: 0 4px 25px rgba(236, 72, 153, 0.18);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.95rem;
        font-weight: 900;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }
    
    /* High-Tech Glowing Status Badges */
    .badge-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(16, 185, 129, 0.08));
        color: #34d399;
        border: 1.5px solid #10b981;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.25);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.76rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .badge-unsafe {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.22), rgba(244, 63, 94, 0.08));
        color: #f43f5e;
        border: 1.5px solid #ec4899;
        box-shadow: 0 0 12px rgba(236, 72, 153, 0.25);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.76rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .badge-unknown {
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.22), rgba(250, 204, 21, 0.08));
        color: #facc15;
        border: 1.5px solid #facc15;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.25);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.76rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .badge-face {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.20), rgba(236, 72, 153, 0.12));
        color: #ffffff;
        border: 1.5px solid rgba(255, 255, 255, 0.45);
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.18);
        padding: 0.22rem 0.68rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.72rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    
    /* Futuristic Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #080c14 !important;
        border-right: 1.5px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Interactive Neon Gradient Buttons */
    .stButton>button {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.18) 0%, rgba(16, 185, 129, 0.18) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        color: #ffffff !important;
        border-radius: 11px !important;
        font-weight: 750 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.22s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ec4899 0%, #facc15 50%, #10b981 100%) !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
        font-weight: 700;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.25), rgba(16, 185, 129, 0.25)) !important;
        border-bottom: 2px solid #ec4899 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Subsystems Singleton Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_system_subsystems():
    """Initialize database, detector, tracker, and safety verifier once."""
    init_db()
    conf_thresh = float(get_setting("conf_threshold", str(config.DEFAULT_CONF_THRESHOLD)))
    iou_thresh = float(get_setting("iou_threshold", str(config.DEFAULT_IOU_THRESHOLD)))
    imgsz = int(get_setting("imgsz", str(config.DEFAULT_IMGSZ)))
    allow_cap = get_setting("allow_safety_cap", "true").lower() == "true"
    hysteresis = int(get_setting("hysteresis_frames", str(config.HYSTERESIS_WINDOW)))
    cooldown = float(get_setting("cooldown_seconds", str(config.RECORD_COOLDOWN_SECONDS)))

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
    return detector, tracker, verifier


detector, tracker, verifier = get_system_subsystems()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.2rem; padding:0.5rem; background:linear-gradient(135deg, rgba(236,72,153,0.15), rgba(16,185,129,0.15)); border:1px solid rgba(255,255,255,0.12); border-radius:14px;">
            <div style="background:linear-gradient(135deg, #ec4899, #facc15, #10b981); width:42px; height:42px; border-radius:11px; display:flex; align-items:center; justify-content:center; font-size:1.35rem; box-shadow:0 0 12px rgba(236,72,153,0.4);">
                🛡️
            </div>
            <div>
                <div style="font-weight:900; font-size:1.05rem; letter-spacing:-0.02em; color:#ffffff;">SAFETY AI</div>
                <div style="font-size:0.68rem; color:#facc15; font-weight:700; letter-spacing:0.04em;">MECHANICAL LAB MONITOR</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation Menu",
        [
            "1. Dashboard",
            "2. Live Monitoring",
            "3. Worker Records",
            "4. Safety Violations",
            "5. Analytics",
            "6. Reports",
            "7. Settings"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick System Status Box in Sidebar
    diag = detector.get_diagnostics()
    st.markdown(f"""
        <div style="background:rgba(17,24,39,0.8); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:0.85rem; font-size:0.75rem;">
            <div style="font-weight:800; color:#cbd5e1; margin-bottom:0.45rem; display:flex; justify-content:space-between; align-items:center;">
                <span style="letter-spacing:0.05em;">AI VISION ENGINE</span>
                <span style="color:#10b981; font-weight:900; background:rgba(16,185,129,0.15); padding:2px 6px; border-radius:6px; border:1px solid #10b981;">ONLINE</span>
            </div>
            <div style="color:#94a3b8; margin-bottom:2px;">Model: <strong style="color:#f1f5f9;">{'best.pt (Custom PPE)' if diag['is_custom_model'] else 'YOLO11n (Base)'}</strong></div>
            <div style="color:#94a3b8; margin-bottom:2px;">Face Scanner: <strong style="color:#ec4899;">ACTIVE</strong></div>
            <div style="color:#94a3b8; margin-bottom:2px;">Device: <strong style="color:#facc15;">CPU (Intel Graphics)</strong></div>
            <div style="color:#94a3b8;">Latency: <strong style="color:#10b981;">{diag['last_inference_ms']} ms</strong></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.7rem; color:#64748b; text-align:center;'>Version {config.APP_VERSION} | Windows 11 CPU Engine</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# High-Tech Header Bar with White, Pink, Yellow, Green Mixed Gradients
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="industrial-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
            <div class="header-gradient-title">
                {config.APP_TITLE}
            </div>
            <div style="font-size:0.82rem; color:#cbd5e1; margin-top:0.25rem;">
                {config.APP_SUBTITLE} • Real-time Facial Scanning & Spatial PPE Verification
            </div>
        </div>
        <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
            <span class="badge-safe">● SAFE: Helmet + Shoes</span>
            <span class="badge-unsafe">● NOT SAFE: Missing PPE</span>
            <span class="badge-unknown">● UNKNOWN: Occluded</span>
            <span class="badge-face">👤 FACE SCAN ACTIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if "1. Dashboard" in menu:
    st.markdown("### Industrial Safety Overview")
    stats = get_safety_summary_stats()

    # 5 Thematic Overview Cards: White, Green, Pink, Yellow, Multi-Gradient
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card card-white">
            <div class="metric-title" style="color:#ffffff;">Total Monitored</div>
            <div class="metric-value">{stats['total_records']}</div>
            <div class="metric-sub">{stats['total_workers']} unique workers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card card-green">
            <div class="metric-title" style="color:#10b981;">Safe Workers</div>
            <div class="metric-value" style="color:#10b981;">{stats['safe_count']}</div>
            <div class="metric-sub">Today safe: {stats['today_safe']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card card-pink">
            <div class="metric-title" style="color:#ec4899;">Unsafe Workers</div>
            <div class="metric-value" style="color:#ec4899;">{stats['unsafe_count']}</div>
            <div class="metric-sub">Today violations: {stats['today_unsafe']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card card-yellow">
            <div class="metric-title" style="color:#facc15;">Unknown / Occluded</div>
            <div class="metric-value" style="color:#facc15;">{stats['unknown_count']}</div>
            <div class="metric-sub">Border occlusions</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card card-gradient">
            <div class="metric-title" style="color:#facc15;">Safety Compliance</div>
            <div class="metric-value" style="color:#ffffff;">{stats['compliance_pct']}%</div>
            <div class="metric-sub">Today rate: {stats['today_compliance_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2-column layout: Recent Violations & Equipment Compliance Rates
    c_left, c_right = st.columns([1.5, 1.1])

    with c_left:
        st.markdown("#### Recent Safety Violations Log")
        violations = get_recent_violations(limit=6)
        if violations:
            v_df = pd.DataFrame(violations)
            display_cols = ["worker_id", "date", "time", "helmet_status", "safety_shoe_status", "violation_details"]
            clean_display = v_df[display_cols].rename(columns={
                "worker_id": "Worker ID",
                "date": "Date",
                "time": "Time",
                "helmet_status": "Helmet",
                "safety_shoe_status": "Shoes",
                "violation_details": "Violation Reason"
            })
            st.dataframe(clean_display, use_container_width=True, hide_index=True)
        else:
            st.info("No safety violations recorded yet. System clean.")

    with c_right:
        st.markdown("#### Equipment Compliance Breakdown")
        comp_df = pd.DataFrame({
            "PPE Item": ["Safety Helmet", "Safety Shoes", "Overall Full Compliance"],
            "Compliance (%)": [
                stats["helmet_compliance_pct"],
                stats["shoe_compliance_pct"],
                stats["compliance_pct"]
            ]
        })
        fig = px.bar(
            comp_df,
            x="Compliance (%)",
            y="PPE Item",
            orientation="h",
            color="Compliance (%)",
            color_continuous_scale=["#ec4899", "#facc15", "#10b981"],
            range_x=[0, 100],
            text="Compliance (%)"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            coloraxis_showscale=False
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2: LIVE MONITORING (Webcam Stream & Photo Scanner)
# =============================================================================
elif "2. Live Monitoring" in menu:
    st.markdown("### Real-Time Live Monitoring & Facial Scanner")
    st.caption("Automated spatial PPE verification with cybernetic facial scanning and boundary occlusion detection")

    tab_webcam, tab_photo = st.tabs(["📹 Live Webcam Feed", "📷 Instant Photo Scanner (Upload & Test)"])

    # -------------------------------------------------------------------------
    # TAB 1: LIVE WEBCAM FEED
    # -------------------------------------------------------------------------
    with tab_webcam:
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.2, 1.5, 2.1])

        with ctrl1:
            run_camera = st.toggle("Start Camera", value=False, key="camera_toggle_btn")

        with ctrl2:
            capture_btn = st.button("📸 Capture Image", use_container_width=True)

        with ctrl3:
            cam_index = st.selectbox(
                "Camera Device",
                options=[0, 1, 2],
                format_func=lambda x: f"Webcam Device {x}",
                index=int(get_setting("camera_index", "0"))
            )

        with ctrl4:
            st.markdown(
                f"<div style='padding-top:0.6rem; font-size:0.8rem; color:#94a3b8;'>Status: <strong style='color:{'#10b981' if run_camera else '#64748b'};'>{'● CAMERA STREAMING' if run_camera else '○ CAMERA STANDBY'}</strong></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        video_col, status_col = st.columns([2.2, 1.0])

        with video_col:
            video_placeholder = st.empty()

        with status_col:
            st.markdown("#### Live Safety Status Panel")
            status_placeholder = st.empty()

        # Camera Capture Streaming Loop
        if run_camera:
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(cam_index)

            if not cap.isOpened():
                st.error(f"Unable to access camera device index {cam_index}. Please check webcam connection.")
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

                        # 1. AI Object Detection & Face Scanning
                        dets = detector.detect(frame)

                        # 2. Worker Multi-Person Tracking
                        tracked = tracker.update(dets["persons"])

                        # 3. Spatial Safety Verification
                        evaluations = []
                        for p in tracked:
                            wid = p["worker_id"]
                            raw_id = p["raw_id"]
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
                                faces=dets.get("faces", [])
                            )

                            # Deduplicated database recording
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
                                    feet_visible=evaluation["feet_visible"]
                                )

                            evaluations.append(evaluation)

                        # 4. Render High-Tech Annotations
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

                        # Render to Streamlit
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        # Update Status Panel with Face Scan & PPE information
                        with status_placeholder.container():
                            if evaluations:
                                for ev in evaluations:
                                    color_box = STATUS_COLORS[ev["status"]]
                                    badge_class = "badge-safe" if ev["status"] == "SAFE" else ("badge-unsafe" if ev["status"] == "NOT_SAFE" else "badge-unknown")
                                    has_face = ev.get("face_box") is not None
                                    st.markdown(f"""
                                        <div style="background:rgba(17,24,39,0.9); border:1px solid rgba(255,255,255,0.12); border-left:4px solid {color_box}; border-radius:12px; padding:0.85rem; margin-bottom:0.75rem; box-shadow:0 4px 15px rgba(0,0,0,0.3);">
                                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                                                <strong style="color:#ffffff; font-size:0.95rem;">{ev['worker_id']}</strong>
                                                <span class="{badge_class}">{ev['status']}</span>
                                            </div>
                                            <div style="margin-bottom:0.4rem;">
                                                {'<span class="badge-face">👤 FACE SCAN CONFIRMED</span>' if has_face else '<span style="font-size:0.7rem; color:#94a3b8;">Face scanning in progress...</span>'}
                                            </div>
                                            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:0.3rem;">
                                                Helmet: <strong style="color:{'#10b981' if ev['helmet_status']=='WORN' else '#ec4899'};">{ev['helmet_status']}</strong> • Shoes: <strong style="color:{'#10b981' if ev['safety_shoe_status']=='WORN' else ('#facc15' if ev['safety_shoe_status']=='UNCLEAR' else '#ec4899')};">{ev['safety_shoe_status']}</strong>
                                            </div>
                                            <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.3rem;">
                                                {ev['details']}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                if dets.get("faces"):
                                    st.markdown(f"""
                                        <div style="background:rgba(236,72,153,0.1); border:1px solid #ec4899; border-radius:10px; padding:0.75rem; color:#ffffff; font-size:0.8rem;">
                                            👤 <strong>{len(dets['faces'])} Human Face(s) Scanned</strong><br>
                                            <span style="font-size:0.72rem; color:#facc15;">Synthesizing worker tracking...</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info("No workers currently in camera view.")

                        # Handle manual capture button
                        if capture_btn:
                            cap_path = config.RECORDS_DIR / f"capture_{int(time.time())}.jpg"
                            cv2.imwrite(str(cap_path), frame)
                            st.sidebar.success(f"Captured: {cap_path.name}")

                        time.sleep(0.02)
                finally:
                    cap.release()
        else:
            # High-Tech Standby Graphic
            standby_img = np.zeros((480, 640, 3), dtype=np.uint8)
            standby_img[:] = (13, 17, 23)
            # Draw decorative cyber grid lines
            for gy in range(60, 480, 60):
                cv2.line(standby_img, (0, gy), (640, gy), (25, 30, 40), 1)
            for gx in range(80, 640, 80):
                cv2.line(standby_img, (gx, 0), (gx, 480), (25, 30, 40), 1)

            cv2.putText(
                standby_img,
                "AI SAFETY & FACIAL SCANNER STANDBY",
                (90, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                standby_img,
                "TOGGLE 'START CAMERA' TO ACTIVATE REAL-TIME MONITORING",
                (60, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (236, 72, 153),
                1,
                cv2.LINE_AA
            )
            video_placeholder.image(cv2.cvtColor(standby_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            status_placeholder.info("Activate camera to view live worker detection & face scanning.")

    # -------------------------------------------------------------------------
    # TAB 2: INSTANT PHOTO SCANNER (Upload & Test)
    # -------------------------------------------------------------------------
    with tab_photo:
        st.markdown("##### Upload Laboratory Worker Image for Immediate Analysis")
        st.caption("Test face scanning and PPE compliance on any photo (JPG, PNG) without needing a live webcam")

        up_col1, up_col2 = st.columns([1.5, 1.0])

        with up_col1:
            uploaded_file = st.file_uploader("Choose a photo to scan", type=["jpg", "jpeg", "png"], key="photo_scanner_uploader")

        with up_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            use_sample = st.button("🧪 Load Built-in Test Sample", use_container_width=True)

        test_img = None
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            test_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif use_sample:
            # Generate synthetic test frame with a worker silhouette and simulated face
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            test_img[:] = (35, 40, 50)
            # Draw person silhouette
            cv2.circle(test_img, (320, 140), 50, (180, 160, 150), -1)  # Head
            cv2.ellipse(test_img, (320, 300), (100, 150), 0, 0, 360, (70, 80, 120), -1) # Body
            cv2.rectangle(test_img, (270, 95), (370, 135), (20, 200, 240), -1) # Safety Cap/Helmet

        if test_img is not None:
            th, tw = test_img.shape[:2]
            with st.spinner("Executing AI Face Scanning & Object Detection..."):
                t_dets = detector.detect(test_img)
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
                        faces=t_dets.get("faces", [])
                    )
                    t_evals.append(t_eval)

                annotated_test = annotate_frame(
                    test_img,
                    t_evals,
                    fps=30.0,
                    faces=t_dets.get("faces", [])
                )

            # Display Results Side by Side
            res_v, res_info = st.columns([2.2, 1.0])
            with res_v:
                st.image(cv2.cvtColor(annotated_test, cv2.COLOR_BGR2RGB), caption="AI Annotated Safety Result", use_container_width=True)

            with res_info:
                st.markdown("#### Detection Verdict")
                st.markdown(f"**Human Faces Scanned:** `{len(t_dets.get('faces', []))}`")
                st.markdown(f"**Workers Detected:** `{len(t_evals)}`")

                for ev in t_evals:
                    color_box = STATUS_COLORS[ev["status"]]
                    badge_class = "badge-safe" if ev["status"] == "SAFE" else ("badge-unsafe" if ev["status"] == "NOT_SAFE" else "badge-unknown")
                    st.markdown(f"""
                        <div style="background:rgba(17,24,39,0.9); border:1px solid rgba(255,255,255,0.12); border-left:4px solid {color_box}; border-radius:12px; padding:0.85rem; margin-top:0.75rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:#ffffff;">{ev['worker_id']}</strong>
                                <span class="{badge_class}">{ev['status']}</span>
                            </div>
                            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:0.4rem;">
                                Helmet: <strong>{ev['helmet_status']}</strong> • Shoes: <strong>{ev['safety_shoe_status']}</strong>
                            </div>
                            <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.3rem;">
                                {ev['details']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                if st.button("💾 Record Evaluation to Safety Audit Log"):
                    for ev in t_evals:
                        add_safety_record(
                            worker_id=ev["worker_id"],
                            helmet_status=ev["helmet_status"],
                            safety_shoe_status=ev["safety_shoe_status"],
                            overall_status=ev["status"],
                            confidence=ev["stability_score"],
                            violation_details=ev["details"],
                            head_visible=ev["head_visible"],
                            feet_visible=ev["feet_visible"]
                        )
                    st.success("Successfully committed verification records to SQLite database!")

    # Manual Scenario Simulator for quick verification of Section 15 tests
    st.markdown("---")
    st.markdown("##### Manual Scenario Testing Injector (Section 15 Specification Test)")
    s1, s2, s3, s4, s5 = st.columns(5)

    with s1:
        if st.button("Test 1: Full PPE (SAFE)", use_container_width=True):
            rec_id = add_safety_record("Worker 001", "WORN", "WORN", "SAFE", 0.96, "Full compliance verified: Safety helmet and safety shoes detected")
            st.success(f"Committed SAFE entry: {rec_id}")

    with s2:
        if st.button("Test 2: Missing Shoes", use_container_width=True):
            rec_id = add_safety_record("Worker 002", "WORN", "MISSING", "NOT_SAFE", 0.92, "Safety violation: Missing required safety shoes")
            st.error(f"Committed NOT SAFE: {rec_id}")

    with s3:
        if st.button("Test 3: Missing Helmet", use_container_width=True):
            rec_id = add_safety_record("Worker 003", "MISSING", "WORN", "NOT_SAFE", 0.89, "Safety violation: Missing required safety helmet")
            st.error(f"Committed NOT SAFE: {rec_id}")

    with s4:
        if st.button("Test 4: Missing Both", use_container_width=True):
            rec_id = add_safety_record("Worker 004", "MISSING", "MISSING", "NOT_SAFE", 0.94, "Critical violation: Missing both safety helmet and safety shoes")
            st.error(f"Committed NOT SAFE: {rec_id}")

    with s5:
        if st.button("Test 5: Out of Frame", use_container_width=True):
            rec_id = add_safety_record("Worker 005", "UNCLEAR", "UNCLEAR", "UNKNOWN", 0.50, "UNKNOWN: Feet partially out of bottom camera frame", feet_visible=False)
            st.warning(f"Committed UNKNOWN: {rec_id}")


# =============================================================================
# PAGE 3: WORKER RECORDS
# =============================================================================
elif "3. Worker Records" in menu:
    st.markdown("### Worker Safety Verification Records")
    st.caption("Complete audit trail of all safety checks, detections, and compliance verdicts")

    # Filters
    f1, f2, f3, f4 = st.columns([1.5, 1.2, 1.2, 1.5])
    with f1:
        search_query = st.text_input("Search Worker / Record ID", placeholder="e.g. Worker 001")
    with f2:
        status_filter = st.selectbox("Filter Status", ["ALL", "SAFE", "NOT_SAFE", "UNKNOWN"])
    with f3:
        date_from = st.date_input("From Date", value=None)
    with f4:
        date_to = st.date_input("To Date", value=None)

    # Query filtered DataFrame
    records_df = get_filtered_records_df(
        status=status_filter if status_filter != "ALL" else None,
        date_from=date_from.strftime("%Y-%m-%d") if date_from else None,
        date_to=date_to.strftime("%Y-%m-%d") if date_to else None,
        search=search_query if search_query else None
    )

    st.markdown(f"**Found {len(records_df)} safety verification records**")

    if not records_df.empty:
        display_df = records_df.copy()
        show_cols = ["record_id", "worker_id", "date", "time", "helmet_status", "safety_shoe_status", "overall_status", "confidence", "violation_details"]
        clean_tbl = display_df[show_cols].rename(columns={
            "record_id": "Record ID",
            "worker_id": "Worker ID",
            "date": "Date",
            "time": "Time",
            "helmet_status": "Helmet",
            "safety_shoe_status": "Shoes",
            "overall_status": "Safety Status",
            "confidence": "Confidence",
            "violation_details": "Details / Violations"
        })

        st.dataframe(clean_tbl, use_container_width=True, hide_index=True)

        csv_bytes = generate_csv_download(records_df)
        st.download_button(
            label="📥 Download Filtered Records (CSV)",
            data=csv_bytes,
            file_name=f"worker_safety_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No records matched the selected criteria.")


# =============================================================================
# PAGE 4: SAFETY VIOLATIONS
# =============================================================================
elif "4. Safety Violations" in menu:
    st.markdown("### Safety Violation History & Incident Logs")
    st.caption("Detailed breakdown of non-compliant entries missing safety helmets or shoes")

    violations_df = get_filtered_records_df(status="NOT_SAFE")

    if not violations_df.empty:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Recorded Violations", len(violations_df))
        with m2:
            no_helmets = len(violations_df[violations_df["helmet_status"] == "MISSING"])
            st.metric("Missing Helmet Incidents", no_helmets)
        with m3:
            no_shoes = len(violations_df[violations_df["safety_shoe_status"] == "MISSING"])
            st.metric("Missing Safety Shoe Incidents", no_shoes)

        st.markdown("<br>", unsafe_allow_html=True)

        display_viol = violations_df[["record_id", "worker_id", "date", "time", "helmet_status", "safety_shoe_status", "confidence", "violation_details"]].rename(columns={
            "record_id": "Record ID",
            "worker_id": "Worker ID",
            "date": "Date",
            "time": "Time",
            "helmet_status": "Helmet",
            "safety_shoe_status": "Shoes",
            "confidence": "Confidence",
            "violation_details": "Violation Reason"
        })
        st.dataframe(display_viol, use_container_width=True, hide_index=True)

        csv_viol = generate_csv_download(violations_df)
        st.download_button(
            label="📥 Export Violations Log (CSV)",
            data=csv_viol,
            file_name=f"safety_violations_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.success("No safety violations recorded in the database.")


# =============================================================================
# PAGE 5: ANALYTICS
# =============================================================================
elif "5. Analytics" in menu:
    st.markdown("### Safety Compliance Analytics")
    st.caption("Visual distribution of safety performance, trends, and violation rates")

    all_df = get_all_records_df()

    if not all_df.empty:
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Safe vs Unsafe Worker Ratio")
            status_counts = all_df["overall_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            fig_donut = px.pie(
                status_counts,
                values="Count",
                names="Status",
                hole=0.45,
                color="Status",
                color_discrete_map={
                    "SAFE": "#10b981",
                    "NOT_SAFE": "#ec4899",
                    "UNKNOWN": "#facc15"
                }
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                height=320,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with ch2:
            st.markdown("#### Violation Breakdown by Missing PPE")
            viol_df = all_df[all_df["overall_status"] == "NOT_SAFE"]
            if not viol_df.empty:
                def categorize_viol(row):
                    if row["helmet_status"] == "MISSING" and row["safety_shoe_status"] == "MISSING":
                        return "Missing Both (Helmet & Shoes)"
                    elif row["helmet_status"] == "MISSING":
                        return "Missing Safety Helmet"
                    elif row["safety_shoe_status"] == "MISSING":
                        return "Missing Safety Shoes"
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
                    color_discrete_sequence=["#ec4899", "#facc15", "#f43f5e"]
                )
                fig_vbar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc",
                    height=320,
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_vbar, use_container_width=True)
            else:
                st.info("No violations to categorize.")

        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown("#### Daily Safety Violations Trend")
            daily_viol = all_df[all_df["overall_status"] == "NOT_SAFE"].groupby("date").size().reset_index(name="Violations")
            if not daily_viol.empty:
                fig_trend = px.line(
                    daily_viol,
                    x="date",
                    y="Violations",
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=["#ec4899"]
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc",
                    height=320,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No daily violations recorded.")

        with ch4:
            st.markdown("#### Daily Compliance Rate (%)")
            daily_stats = all_df.groupby("date").agg(
                total=("id", "count"),
                safe=("overall_status", lambda s: (s == "SAFE").sum())
            ).reset_index()
            daily_stats["Compliance (%)"] = (daily_stats["safe"] / daily_stats["total"] * 100).round(1)

            fig_comp = px.area(
                daily_stats,
                x="date",
                y="Compliance (%)",
                color_discrete_sequence=["#10b981"]
            )
            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                height=320,
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No records available to generate analytics.")


# =============================================================================
# PAGE 6: REPORTS
# =============================================================================
elif "6. Reports" in menu:
    st.markdown("### Safety Compliance Report Generator")
    st.caption("Generate and download formal industrial audit reports (Daily, Weekly, Monthly)")

    report_type = st.radio("Select Report Duration", ["Daily Report", "Weekly Report (7 Days)", "Monthly Report"], horizontal=True)

    if report_type == "Daily Report":
        target_date = st.date_input("Report Date", value=datetime.now())
        df, stats, title = generate_daily_report(target_date.strftime("%Y-%m-%d"))
    elif report_type == "Weekly Report (7 Days)":
        target_date = st.date_input("Week Ending Date", value=datetime.now())
        df, stats, title = generate_weekly_report(target_date.strftime("%Y-%m-%d"))
    else:
        year_month = st.text_input("Year and Month (YYYY-MM)", value=datetime.now().strftime("%Y-%m"))
        df, stats, title = generate_monthly_report(year_month)

    st.markdown("---")
    st.markdown(f"#### {title}")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Total Monitored Entries", stats["total_entries"])
    with r2:
        st.metric("Unique Workers", stats["unique_workers"])
    with r3:
        st.metric("Safe vs Unsafe Entries", f"{stats['safe_entries']} / {stats['unsafe_entries']}")
    with r4:
        st.metric("Overall Compliance Rate", f"{stats['compliance_pct']}%")

    if not df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df[["record_id", "worker_id", "date", "time", "helmet_status", "safety_shoe_status", "overall_status", "violation_details"]], use_container_width=True, hide_index=True)

        csv_data = generate_csv_download(df)
        st.download_button(
            label=f"📥 Download {report_type} (CSV)",
            data=csv_data,
            file_name=f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No entries found for the selected report timeframe.")


# =============================================================================
# PAGE 7: SETTINGS
# =============================================================================
elif "7. Settings" in menu:
    st.markdown("### System & Model Settings")
    st.caption("Configure detection thresholds, workplace safety policy rules, and hardware preferences")

    with st.form("settings_form"):
        st.markdown("#### Detection & Inference Configuration")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            new_conf = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=float(get_setting("conf_threshold", str(config.DEFAULT_CONF_THRESHOLD))), step=0.05)
        with col_s2:
            new_iou = st.slider("IoU NMS Overlap Threshold", min_value=0.20, max_value=0.80, value=float(get_setting("iou_threshold", str(config.DEFAULT_IOU_THRESHOLD))), step=0.05)
        with col_s3:
            new_imgsz = st.selectbox("Inference Resolution (px)", options=[320, 480, 640], index=[320, 480, 640].index(int(get_setting("imgsz", str(config.DEFAULT_IMGSZ)))))

        st.markdown("#### Workplace Safety Policy")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            allow_caps = st.checkbox(
                "Approve Safety Cap as Alternative to Hard Helmet",
                value=get_setting("allow_safety_cap", "true").lower() == "true",
                help="When enabled, workers wearing safety caps satisfy the head protection condition."
            )
        with col_p2:
            cam_idx = st.number_input("Default Webcam Index", min_value=0, max_value=5, value=int(get_setting("camera_index", "0")))

        st.markdown("#### Anti-Spam & Stability Rules")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            hysteresis = st.slider("Consecutive Confirmation Frames", min_value=2, max_value=12, value=int(get_setting("hysteresis_frames", str(config.HYSTERESIS_WINDOW))))
        with col_t2:
            cooldown = st.slider("Worker Entry Cooldown (seconds)", min_value=5, max_value=60, value=int(float(get_setting("cooldown_seconds", str(config.RECORD_COOLDOWN_SECONDS)))))

        submitted = st.form_submit_button("Save Configuration", use_container_width=True)
        if submitted:
            set_setting("conf_threshold", str(new_conf))
            set_setting("iou_threshold", str(new_iou))
            set_setting("imgsz", str(new_imgsz))
            set_setting("allow_safety_cap", "true" if allow_caps else "false")
            set_setting("camera_index", str(cam_idx))
            set_setting("hysteresis_frames", str(hysteresis))
            set_setting("cooldown_seconds", str(cooldown))

            # Apply immediately to cached singletons
            detector.conf_threshold = new_conf
            detector.iou_threshold = new_iou
            detector.imgsz = new_imgsz
            verifier.allow_safety_cap = allow_caps
            verifier.hysteresis_window = hysteresis
            tracker.cooldown_sec = float(cooldown)

            st.success("System configuration saved successfully and applied to live inference!")

    # Model & Diagnostics Information
    st.markdown("---")
    st.markdown("#### AI Model Diagnostics & Dataset Guide")
    diag = detector.get_diagnostics()

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Active Model Parameters**")
        st.json(diag)

    with d2:
        st.markdown("**Dataset & Training Commands**")
        st.markdown("""
        To train a custom PPE model on your laboratory photos:
        ```bash
        # 1. Place lab images in dataset/images/train and annotations in dataset/labels/train
        # 2. Run automated training on CPU:
        python training/train.py --epochs 30 --batch 8 --imgsz 640 --device cpu
        
        # 3. Validate mAP & Recall:
        python training/validate.py
        ```
        The trained weights are automatically saved to `models/best.pt` and loaded by this dashboard.
        """)
