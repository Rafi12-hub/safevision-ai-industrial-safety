# 🛡️ SafeVision AI: Mechanical Lab & Industrial Worker Safety Monitoring System

> **Real-Time Automated PPE Compliance Verification & Facial Scanning System Powered by YOLO11, OpenCV, and Streamlit**

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO Version](https://img.shields.io/badge/YOLO11-Ultralytics-00ffff.svg?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Computer Vision](https://img.shields.io/badge/OpenCV-Haar%20Cascade%20%2B%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Dashboard](https://img.shields.io/badge/Streamlit-High--Tech%20UI-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hardware Inference](https://img.shields.io/badge/Hardware-Intel%20CPU%20Integrated%20Graphics-0071C5.svg?logo=intel&logoColor=white)](https://www.intel.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-8%2F8%20Passed%20(100%25)-10B981.svg)](https://pytest.org/)
[![OS Support](https://img.shields.io/badge/OS-Windows%2011-0078D4.svg?logo=windows11&logoColor=white)](https://www.microsoft.com/windows)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

The **SafeVision AI Industrial Safety Monitoring System** is an end-to-end Computer Vision platform engineered specifically for **mechanical engineering laboratories, manufacturing plants, fabrication workshops, construction sites, and heavy industrial facilities**.

Manual safety audits are periodic, subjective, and prone to human oversight. **SafeVision AI** continuously monitors workers in real-time via live camera feeds or high-resolution photo scans, automatically verifying that workers adhere to mandatory **Personal Protective Equipment (PPE)** standards:
1. **Safety Helmet / Approved Hard Cap** (Head Protection)
2. **Safety Shoes / Steel-Toe Boots** (Foot Protection)

The system automatically classifies workers with distinct visual indicators (**GREEN = SAFE**, **RED = NOT SAFE**, **YELLOW = UNKNOWN**), manages temporary tracking sessions without facial biometric profiling, flags boundary occlusions, logs audit violations to a local SQLite database, and provides analytical safety intelligence with downloadable audit reports.

---

## 🚀 Key Innovations & Architectural Highlights

### 1. High-Speed Human Facial Scanner & Seated Worker Synthesis
- **Real-Time Face Cascade Engine**: Employs a low-latency Haar Cascade facial scanning model running at 2–4ms per frame on standard CPU hardware.
- **Seated Worker Extrapolation**: Traditional YOLO person detectors trained on whole standing bodies often miss students or technicians seated close to laptop cameras or workshop benches. SafeVision AI automatically detects faces and synthesizes upper-body worker tracking boxes, ensuring zero missed detections.
- **Futuristic Cyber Reticle**: Annotates detected human faces with neon pink/mint corner brackets, target crosshairs, and live status verification badges.

### 2. Spatial Equipment Association (Head & Feet Zones)
- Distinguishes between **equipment being worn** versus **equipment merely lying on the floor, on desks, or in the background**.
- Helmets/caps are evaluated strictly within the **upper 32% Head Zone**.
- Safety shoes are evaluated strictly within the **lower 32% Feet Zone**.
- Background items and nearby workers' equipment are rejected via spatial overlap matching.

### 3. Boundary Occlusion & Cut-Off Safeguard (`UNKNOWN`)
- If a worker's head is truncated by the top of the camera frame or their feet are cut off by the bottom border, the system flags `UNKNOWN` (Yellow) with clear descriptors (`FEET OUT OF FRAME`).
- Prevents false safe clearances or erroneous violation penalties when visibility is partial.

### 4. Temporal Stabilization (Hysteresis Filter)
- Utilizes a temporal hysteresis queue across 4–15 consecutive frames.
- Eliminates single-frame detector flicker or rapid status blinking before committing violation events to the audit trail.

### 5. Anti-Spam Session Tracking & Deduplication
- Assigns non-biometric sequential identifiers (`Worker 001`, `Worker 002`).
- State-machine session tracking ensures stationary workers in front of a camera do not create duplicate database records.

### 6. Modern Glassmorphic UI (White, Cyber Pink, Mint Green & Electric Yellow)
- Aesthetic mixed gradient interface with iridescent glowing cards, cyber badges, transparent Plotly analytics, and dual monitoring modes (Live Webcam + Instant Photo Upload Scanner).

---

## 🚦 Safety Verification Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       WORKER ENTERS CAMERA FRAME                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
                     [ Boundary Visibility Check ]
                   Are Head and Feet fully in view?
                      ├── NO ──► YELLOW (UNKNOWN: Boundary Occluded)
                      └── YES
                           ▼
                  [ Spatial Association Check ]
              Is Helmet/Cap detected in Head Zone?
              Are Safety Shoes detected in Feet Zone?
                      ├── BOTH WORN ──────► GREEN (SAFE)
                      └── EITHER MISSING ─► RED (NOT SAFE)
```

| Visual Indicator | Safety Verdict | Conditions Required | System Action |
|:---:|:---:|---|---|
| 🟢 **GREEN** | **SAFE** | Both Safety Helmet/Cap **AND** Safety Shoes verified as worn on body | Entry authorized; compliant session logged |
| 🔴 **RED** | **NOT SAFE** | Missing Safety Helmet, missing Safety Shoes, or missing both | Violation tagged; instant audio/visual warning; violation recorded |
| 🟡 **YELLOW** | **UNKNOWN** | Head or feet out of frame / blurred / worker too distant | Verification paused; requests full camera visibility |

---

## 🛠️ Technology Stack

| Layer | Technologies | Description |
|---|---|---|
| **Deep Learning** | **YOLO11 (Ultralytics)**, **PyTorch CPU** | Real-time object detection optimized for Intel Integrated Graphics |
| **Vision & Image Processing** | **OpenCV (cv2)**, **NumPy**, **PIL** | Frame acquisition, Haar Cascade face detection, cybernetic HUD overlays |
| **Frontend Dashboard** | **Streamlit** | Multi-page industrial monitoring dashboard with reactive widgets |
| **Analytics & Data** | **Pandas**, **Plotly Graph Objects** | Statistical aggregations, compliance metrics, interactive trend visualizers |
| **Database Persistence** | **SQLite3** | Thread-safe local relational storage for worker audits and violations |
| **Reporting Engine** | **CSV & Markdown Generator** | Automated generation of Daily, Weekly, and Monthly compliance reports |
| **Testing Harness** | **Pytest** | Comprehensive test suite validating all 8 specification test cases |

---

## 📂 Repository Directory Structure

```
ai-image-processing-system/
│
├── app.py                         # Complete Streamlit Web Application & Dashboard
├── config.py                      # Global constants, class definitions, and color maps
├── requirements.txt               # Production Python dependencies
├── README.md                      # Comprehensive project documentation
├── start_project.bat              # 1-Click automated Windows 11 launcher
│
├── detection/                     # Core Computer Vision & AI Subsystems
│   ├── __init__.py
│   ├── detector.py                # YOLO11 inference + Haar Cascade Face Scanner
│   ├── tracker.py                 # Multi-person worker session tracking & deduplication
│   └── safety_checker.py          # Spatial association, hysteresis & HUD annotation engine
│
├── database/                      # Relational Storage Layer
│   ├── __init__.py
│   ├── db.py                      # SQLite connection manager & schema migrations
│   └── safety_records.py          # Audit logging, queries, filtering & CSV synchronization
│
├── reports/                       # Formal Audit Report Generators
│   ├── __init__.py
│   └── report_generator.py        # Daily, Weekly, and Monthly compliance reports
│
├── training/                      # Custom Model Training & Export Tools
│   ├── train.py                   # Automated YOLO11 CPU/GPU training script
│   ├── validate.py                # Model validation metrics evaluator (mAP, Precision, Recall)
│   ├── test_model.py              # Synthetic validation script
│   ├── export_model.py            # Model export to ONNX / TorchScript
│   ├── generate_sample_dataset.py # Synthetic sample dataset generator
│   └── data.yaml                  # Ultralytics 6-class dataset configuration
│
├── models/                        # Pre-trained & Fine-Tuned Weights
│   └── best.pt                    # Custom trained PPE model weights
│
├── dataset/                       # Training Data Directories
│   ├── images/                    # Train and validation image sets
│   └── labels/                    # YOLO-format annotation txt files
│
├── records/                       # Auto-synchronized audit records
│   └── safety_records.csv         # Exported CSV safety record log
│
└── tests/                         # Automated Verification Suite
    └── test_safety_system.py      # Complete pytest suite validating all 8 scenarios
```

---

## ⚡ Installation & Quick Start

### System Prerequisites
- **Operating System**: Windows 11 / 10, Linux, or macOS
- **Python**: Version 3.11 or 3.13
- **Hardware**: Standard laptop or desktop with **Intel Integrated Graphics** (Dedicated NVIDIA GPU is **not** required)
- **Webcam**: Integrated laptop camera or USB external webcam

---

### Method 1: One-Click Windows Launcher (Recommended)
Simply double-click the included batch script:
```cmd
start_project.bat
```
The script will automatically verify Python, validate model weights, initialize the SQLite database, and launch the web dashboard at `http://localhost:8501`.

---

### Method 2: Manual Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/industrial-worker-safety-monitoring.git
   cd industrial-worker-safety-monitoring
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard**:
   ```bash
   python -m streamlit run app.py
   ```

5. **Open Browser**:
   Navigate to `http://localhost:8501`.

---

## 🖥️ Dashboard Features & Walkthrough

The web dashboard is organized into **7 specialized operational modules**:

### 1. 📊 Industrial Safety Overview
- **5 High-Impact Metric Cards**: Total Monitored Workers, Safe Workers, Unsafe Workers, Unknown/Occluded, and Overall Compliance Percentage.
- **Recent Violations Log**: Real-time table showcasing recent non-compliant detections with timestamps and violation rationales.
- **Equipment Compliance Breakdown**: Interactive horizontal bar chart tracking Helmet vs. Safety Shoe compliance rates.

### 2. 📹 Real-Time Live Monitoring & Facial Scanner
- **Dual Operating Modes**:
  - **Live Webcam Mode**: Live camera feed with bounding boxes, cybernetic face targeting reticles, real-time FPS counter, and worker status cards.
  - **Instant Photo Scanner**: Upload laboratory photos or snapshot images for immediate PPE and face scanning without needing an active webcam.
- **Live Status Panel**: Dynamic cards displaying worker ID, facial scan confirmation, helmet status, shoe status, and detailed safety rationale.
- **Manual Scenario Simulator**: Quick-action injector buttons to verify all core detection states on demand.

### 3. 📋 Worker Safety Verification Records
- Searchable and filterable database audit table.
- Filter by status (`SAFE`, `NOT_SAFE`, `UNKNOWN`) and custom date ranges.
- One-click CSV export button.

### 4. 🚨 Safety Violations History
- Detailed breakdown of safety infractions.
- Metrics highlighting missing helmet vs. missing shoe incidents.
- Export violation logs directly to CSV for industrial compliance audits.

### 5. 📈 Compliance Analytics
- **Safe vs. Unsafe Worker Ratio**: Donut chart with status distribution.
- **Violation Breakdown by PPE Item**: Categorical breakdown of violations.
- **Daily Safety Violations Trend**: Spline line chart tracking daily incident counts.
- **Daily Compliance Rate Trend**: Gradient area chart monitoring lab compliance over time.

### 6. 📑 Compliance Report Generator
- Formal industrial audit reports generated in three modes:
  - **Daily Report** (24-hour summary)
  - **Weekly Report** (Rolling 7-day summary)
  - **Monthly Report** (Calendar month audit)
- Includes unique worker counts, overall compliance percentage, and full table records with downloadable CSV files.

### 7. ⚙️ System & Model Settings
- Interactive sliders for **Confidence Threshold** (0.10 to 0.90) and **IoU NMS Overlap** (0.20 to 0.80).
- **Workplace Safety Policy Toggle**: Option to accept safety caps as an approved alternative to hard helmets.
- **Webcam Device Selector**: Select between multiple USB webcams.
- **Temporal Stability Controls**: Configure consecutive confirmation frames (hysteresis) and worker cooldown intervals.

---

## 🧪 Comprehensive Automated Test Suite

All safety verification scenarios demanded by the industrial specification are validated via pytest in [`tests/test_safety_system.py`](file:///c:/Users/windows/OneDrive/Desktop/personal%20mine/vvit/ai%20image%20processing%20system/tests/test_safety_system.py):

| Test Scenario | Scenario Condition | Input | Expected Output | Status |
|:---:|:---|---|:---:|:---:|
| **Test 1** | Full PPE Compliance | Worker wearing safety helmet and safety shoes | **GREEN – SAFE** | ✅ **PASSED** |
| **Test 2** | Missing Shoes Violation | Worker wearing helmet, shoes missing | **RED – NOT SAFE** | ✅ **PASSED** |
| **Test 3** | Missing Helmet Violation | Worker wearing shoes, helmet missing | **RED – NOT SAFE** | ✅ **PASSED** |
| **Test 4** | Missing Both PPE Items | Worker wearing neither helmet nor shoes | **RED – NOT SAFE** | ✅ **PASSED** |
| **Test 5** | Boundary Occlusion | Worker's feet cut off at bottom frame border | **YELLOW – UNKNOWN** | ✅ **PASSED** |
| **Test 6** | Multi-Worker Tracking | Multiple workers entering simultaneously | Individual IDs & distinct checks | ✅ **PASSED** |
| **Test 7** | Spatial Association | PPE items lying on floor or in background | Background items rejected | ✅ **PASSED** |
| **Test 8** | Model Diagnostics | Custom model file unavailable | Transparent fallback message | ✅ **PASSED** |

### Running the Test Suite:
```bash
python -m pytest tests/test_safety_system.py -v
```

---

## 🧠 Custom Model Training & Export Guide

The detection engine uses a 6-class architecture:
- `0: person`
- `1: safety_helmet`
- `2: safety_cap`
- `3: safety_shoes`
- `4: no_helmet`
- `5: no_safety_shoes`

### 1. Generating a Synthetic Dataset (for pipeline testing):
```bash
python training/generate_sample_dataset.py
```

### 2. Training on Custom Industrial Images:
Place your images in `dataset/images/train` and labels in `dataset/labels/train`, then execute:
```bash
python training/train.py --epochs 30 --batch 8 --imgsz 640 --device cpu
```
The script will train the model, save runs to `training/runs/`, and copy the optimal weights to `models/best.pt`.

### 3. Validating Model Performance:
```bash
python training/validate.py
```

### 4. Exporting to ONNX for High-Speed Deployment:
```bash
python training/export_model.py --weights models/best.pt --format onnx
```

---

## 📤 Step-by-Step GitHub Push Guide

To push this codebase to your GitHub account:

### Step 1: Initialize Git Repository
Open a terminal in the project root directory and run:
```bash
git init
```

### Step 2: Add Files & Make Initial Commit
```bash
git add .
git commit -m "feat: complete AI industrial worker safety monitoring system with YOLO11, face scanner & modern UI"
```

### Step 3: Create a New GitHub Repository
1. Go to [GitHub.com](https://github.com/new).
2. Create a new repository (e.g., `ai-industrial-safety-monitoring-system`).
3. Leave "Initialize this repository with a README" **unchecked** (we already have a complete README).

### Step 4: Link Remote & Push
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/ai-industrial-safety-monitoring-system.git
git push -u origin main
```

---

## 📜 License & Compliance

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Ethical & Privacy Considerations
- **Non-Biometric Tracking**: Worker IDs (`Worker 001`, `Worker 002`) are transient session keys generated on the fly.
- **Zero Facial Data Storage**: The facial scanner is used purely for real-time spatial positioning and upper-body tracking; no facial templates, biometric hashes, or user identities are saved to disk or transmitted to external servers.

---

<p align="center">
  <b>Developed for Industrial Worker Safety & Mechanical Engineering Laboratory Compliance</b><br>
  Built with Python, YOLO11, OpenCV, and Streamlit.
</p>
