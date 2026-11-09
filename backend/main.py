"""
Main FastAPI Application Entrypoint for AI-Based Mechanical Laboratory Safety Monitoring System.
Integrates live video inference, person tracking, safety compliance verification,
WebSocket broadcasting, REST APIs, and SQLite database persistence.
"""

import asyncio
import logging
import time
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database
from detector import SafetyDetector
from camera import CameraManager
from safety_checker import PersonSafetyVerifier
from tracker import StudentTracker
from api.routes import router as api_router
from api.websocket import ws_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("safety_ai.main")

# Global subsystem singletons
detector: SafetyDetector = None
camera: CameraManager = None
verifier: PersonSafetyVerifier = None
tracker: StudentTracker = None
background_loop_task: asyncio.Task = None
is_running_flag = True


async def background_camera_processor():
    """
    Background asynchronous loop that monitors the camera,
    runs object detection and safety verification, updates student tracks,
    commits verified entry events to the database, and broadcasts updates via WebSocket.
    """
    global is_running_flag, camera, detector, verifier, tracker
    logger.info("Background camera inference and telemetry loop started.")

    while is_running_flag:
        if not camera or not camera.is_active:
            # Idle sleep when camera is stopped
            await asyncio.sleep(0.2)
            continue

        has_frame, frame = camera.get_latest_frame()
        if not has_frame or frame is None:
            await asyncio.sleep(0.04)
            continue

        try:
            h, w = frame.shape[:2]

            # 1. Run AI object detection (Persons, Safety Caps, Safety Shoes)
            dets = detector.detect(frame)
            persons = dets.get("persons", [])
            caps = dets.get("caps", [])
            shoes = dets.get("shoes", [])

            # 2. Update multi-person tracking
            tracked_persons = tracker.update(persons)

            # 3. Verify safety for each tracked person
            evaluated_students = []
            for p in tracked_persons:
                tid = p.get("track_id", 1)
                evaluation = verifier.evaluate_person(
                    tracking_id=tid,
                    person_box=p["bbox"],
                    person_confidence=p["confidence"],
                    cap_boxes=caps,
                    shoe_boxes=shoes,
                    frame_width=w,
                    frame_height=h
                )

                # 4. Entry event deduplication: commit to database if verified
                should_record, reason = tracker.should_record_entry(
                    tracking_id=tid,
                    status=evaluation["status"],
                    stability_score=evaluation["stability_score"],
                    frames_seen=evaluation["frames_analyzed"]
                )

                if should_record:
                    await database.add_record(
                        tracking_id=tid,
                        safety_cap=evaluation["cap_detected"],
                        safety_shoes=evaluation["shoes_detected"],
                        final_status=evaluation["status"],
                        confidence=evaluation["stability_score"],
                        head_visible=evaluation["head_visible"],
                        feet_visible=evaluation["feet_visible"],
                        details=evaluation["details"]
                    )
                    logger.info(f"Committed entry record for Student #{tid}: {evaluation['status']} ({reason})")

                evaluated_students.append(evaluation)

            # 5. Broadcast real-time telemetry to connected WebSocket clients
            if ws_manager.active_connections:
                await ws_manager.broadcast({
                    "type": "telemetry_update",
                    "timestamp": time.time(),
                    "camera_fps": camera.fps,
                    "active_students": len(evaluated_students),
                    "students": evaluated_students,
                    "model_status": {
                        "model_name": detector.model_name,
                        "is_custom": detector.is_custom_model,
                        "inference_ms": round(dets.get("inference_ms", 0.0), 1)
                    }
                })

        except Exception as e:
            logger.error(f"Error in background inference loop: {e}", exc_info=True)

        # Target ~20-25 inferences/sec for smooth CPU execution
        await asyncio.sleep(0.04)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize and teardown subsystems."""
    global detector, camera, verifier, tracker, background_loop_task, is_running_flag

    logger.info("Initializing Mechanical Lab Safety AI subsystems...")

    # Initialize SQLite database
    await database.init_db()

    # Initialize Detector (loads best.pt or base yolo11n for CPU)
    detector = SafetyDetector(device="cpu", imgsz=480)

    # Initialize Safety Verifier and Tracker
    verifier = PersonSafetyVerifier()
    tracker = StudentTracker()

    # Initialize Camera Manager
    camera = CameraManager(camera_index=0, width=640, height=480)

    # Attach subsystem references to router app_state
    api_router.app_state = {
        "detector": detector,
        "camera": camera,
        "verifier": verifier,
        "tracker": tracker
    }

    # Start background camera processor task
    is_running_flag = True
    background_loop_task = asyncio.create_task(background_camera_processor())

    logger.info("Subsystems initialized successfully. Server ready.")
    yield

    # Shutdown
    logger.info("Shutting down subsystems...")
    is_running_flag = False
    if camera:
        camera.stop()
    if background_loop_task:
        background_loop_task.cancel()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Mechanical Laboratory Safety Monitoring AI",
    description="Real-Time AI-Based Safety Monitoring & PPE Verification System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST API endpoints
app.include_router(api_router, prefix="/api")


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for high-frequency live detection telemetry,
    client frame uploads, and camera controls.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client incoming messages (heartbeats, frame pings, etc.)
            try:
                import json
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        ws_manager.disconnect(websocket)


# Mount and serve built React SPA if frontend/dist exists
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("ws") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")

        target_file = os.path.join(frontend_dist, full_path)
        if os.path.isfile(target_file):
            return FileResponse(target_file)

        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
