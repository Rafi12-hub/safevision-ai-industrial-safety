"""
REST API routes for AI-Based Mechanical Laboratory Safety Monitoring System.
Provides endpoints for system health, dashboard statistics, entry records,
analytics, model configuration, camera hardware control, and CSV export.
"""

from fastapi import APIRouter, Query, Response, HTTPException, Body
from fastapi.responses import StreamingResponse, PlainTextResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import cv2
import base64
import numpy as np
import time

import database
from safety_checker import SafetyStatus, STATUS_COLORS

router = APIRouter()


# Pydantic schemas for request validation
class ModelSettingsRequest(BaseModel):
    conf_threshold: Optional[float] = Field(None, ge=0.05, le=0.95)
    iou_threshold: Optional[float] = Field(None, ge=0.10, le=0.90)
    imgsz: Optional[int] = Field(None, ge=160, le=1280)


class CameraControlRequest(BaseModel):
    camera_index: Optional[int] = 0
    resolution: Optional[str] = "640x480"


class ProcessFrameRequest(BaseModel):
    image_base64: str  # Base64 data URL or raw base64 JPEG/PNG


class SimulateEventRequest(BaseModel):
    scenario: str = Field(..., description="Scenario identifier: SAFE, MISSING_SHOES, MISSING_CAP, MISSING_BOTH, OUT_OF_FRAME")
    tracking_id: Optional[int] = None


@router.get("/health")
async def get_health(request: Any = None):
    """System health check endpoint."""
    app_state = getattr(router, "app_state", {})
    detector = app_state.get("detector")
    camera = app_state.get("camera")

    return {
        "status": "healthy",
        "system": "Mechanical Laboratory Safety AI Monitoring System",
        "timestamp": time.time(),
        "database": "connected",
        "detector_loaded": detector.model is not None if detector else False,
        "is_custom_model": detector.is_custom_model if detector else False,
        "camera_active": camera.is_active if camera else False,
        "camera_fps": camera.fps if camera else 0.0
    }


@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Retrieve summarized statistics for dashboard cards and quick metrics."""
    return await database.get_dashboard_stats()


@router.get("/records")
async def get_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    """Retrieve paginated and filtered student entry records."""
    return await database.get_records(
        page=page,
        page_size=page_size,
        status=status,
        date_from=date_from,
        date_to=date_to,
        search=search
    )


@router.get("/analytics")
async def get_analytics(days: int = Query(14, ge=1, le=90)):
    """Retrieve safety compliance and violation analytics for Recharts."""
    return await database.get_analytics(days=days)


@router.get("/model/status")
async def get_model_status():
    """Retrieve current AI detection model status and configuration."""
    app_state = getattr(router, "app_state", {})
    detector = app_state.get("detector")
    if not detector:
        return {"error": "Detector not initialized"}
    return detector.get_status()


@router.post("/model/settings")
async def update_model_settings(settings: ModelSettingsRequest):
    """Update AI detection model thresholds and inference parameters."""
    app_state = getattr(router, "app_state", {})
    detector = app_state.get("detector")
    if not detector:
        raise HTTPException(status_code=500, detail="Detector engine unavailable")

    if settings.conf_threshold is not None:
        detector.conf_threshold = settings.conf_threshold
        await database.set_setting("confidence_threshold", str(settings.conf_threshold))

    if settings.iou_threshold is not None:
        detector.iou_threshold = settings.iou_threshold
        await database.set_setting("iou_threshold", str(settings.iou_threshold))

    if settings.imgsz is not None:
        detector.imgsz = settings.imgsz

    return {
        "success": True,
        "message": "Model settings updated successfully",
        "current_settings": detector.get_status()
    }


@router.get("/settings")
async def get_settings():
    """Retrieve system configuration settings."""
    return await database.get_all_settings()


@router.post("/settings")
async def update_settings(payload: Dict[str, str] = Body(...)):
    """Update system settings."""
    for key, val in payload.items():
        await database.set_setting(key, str(val))
    return {"success": True, "message": "Settings updated"}


@router.get("/camera/status")
async def get_camera_status():
    """Get webcam hardware connection status and current FPS."""
    app_state = getattr(router, "app_state", {})
    camera = app_state.get("camera")
    if not camera:
        return {"is_active": False, "error_message": "Camera manager not configured"}
    return camera.get_status()


@router.post("/camera/start")
async def start_camera(params: CameraControlRequest):
    """Start hardware camera stream."""
    app_state = getattr(router, "app_state", {})
    camera = app_state.get("camera")
    if not camera:
        raise HTTPException(status_code=500, detail="Camera subsystem unavailable")

    w, h = 640, 480
    if params.resolution and "x" in params.resolution:
        parts = params.resolution.split("x")
        try:
            w, h = int(parts[0]), int(parts[1])
        except ValueError:
            pass

    success, msg = camera.start(camera_index=params.camera_index, width=w, height=h)
    if not success:
        return {"success": False, "message": msg, "error": msg}

    return {"success": True, "message": msg, "status": camera.get_status()}


@router.post("/camera/stop")
async def stop_camera():
    """Stop hardware camera stream and release device."""
    app_state = getattr(router, "app_state", {})
    camera = app_state.get("camera")
    if not camera:
        raise HTTPException(status_code=500, detail="Camera subsystem unavailable")

    success, msg = camera.stop()
    return {"success": success, "message": msg, "status": camera.get_status()}


@router.get("/camera/stream")
def get_camera_stream():
    """MJPEG streaming endpoint for low-latency live video visualization."""
    app_state = getattr(router, "app_state", {})
    camera = app_state.get("camera")
    detector = app_state.get("detector")
    verifier = app_state.get("verifier")
    tracker = app_state.get("tracker")

    def frame_generator():
        while True:
            if not camera or not camera.is_active:
                # Render standby frame
                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                standby[:] = (15, 23, 42)  # Dark navy background
                cv2.putText(
                    standby,
                    "CAMERA INACTIVE - CLICK START MONITORING",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (148, 163, 184),
                    2,
                    cv2.LINE_AA
                )
                _, jpeg = cv2.imencode(".jpg", standby)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(0.5)
                continue

            has_frame, frame = camera.get_latest_frame()
            if not has_frame or frame is None:
                time.sleep(0.03)
                continue

            # Run detection and safety analysis
            dets = detector.detect(frame) if detector else {"persons": [], "caps": [], "shoes": []}
            tracked_persons = tracker.update(dets.get("persons", [])) if tracker else dets.get("persons", [])

            h, w = frame.shape[:2]
            evaluations = []
            for p in tracked_persons:
                if verifier:
                    res = verifier.evaluate_person(
                        tracking_id=p.get("track_id", 1),
                        person_box=p["bbox"],
                        person_confidence=p["confidence"],
                        cap_boxes=dets.get("caps", []),
                        shoe_boxes=dets.get("shoes", []),
                        frame_width=w,
                        frame_height=h
                    )
                    evaluations.append(res)

            annotated = camera.annotate_frame(frame, evaluations, fps=camera.fps)
            ret, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            time.sleep(0.03)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/process_frame")
async def process_frame(payload: ProcessFrameRequest):
    """
    Process a single frame uploaded from browser client webcam.
    Allows real-time browser-based webcam monitoring via HTML5 Canvas.
    """
    app_state = getattr(router, "app_state", {})
    detector = app_state.get("detector")
    verifier = app_state.get("verifier")
    tracker = app_state.get("tracker")

    if not detector or not verifier:
        raise HTTPException(status_code=500, detail="Inference pipeline not initialized")

    try:
        data_str = payload.image_base64
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]

        image_bytes = base64.b64decode(data_str)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        h, w = frame.shape[:2]

        # Run AI detection
        dets = detector.detect(frame)
        persons = dets.get("persons", [])
        caps = dets.get("caps", [])
        shoes = dets.get("shoes", [])

        # Track persons
        tracked = tracker.update(persons) if tracker else persons

        # Evaluate safety for each person
        evaluated_students = []
        for p in tracked:
            tid = p.get("track_id", 1)
            eval_res = verifier.evaluate_person(
                tracking_id=tid,
                person_box=p["bbox"],
                person_confidence=p["confidence"],
                cap_boxes=caps,
                shoe_boxes=shoes,
                frame_width=w,
                frame_height=h
            )

            # Check if entry event should be committed to database
            if tracker:
                should_rec, _ = tracker.should_record_entry(
                    tracking_id=tid,
                    status=eval_res["status"],
                    stability_score=eval_res["stability_score"],
                    frames_seen=eval_res["frames_analyzed"]
                )
                if should_rec:
                    await database.add_record(
                        tracking_id=tid,
                        safety_cap=eval_res["cap_detected"],
                        safety_shoes=eval_res["shoes_detected"],
                        final_status=eval_res["status"],
                        confidence=eval_res["stability_score"],
                        head_visible=eval_res["head_visible"],
                        feet_visible=eval_res["feet_visible"],
                        details=eval_res["details"]
                    )

            evaluated_students.append(eval_res)

        return {
            "success": True,
            "timestamp": time.time(),
            "frame_size": {"width": w, "height": h},
            "students": evaluated_students,
            "raw_detections": {
                "persons_count": len(persons),
                "caps_count": len(caps),
                "shoes_count": len(shoes),
                "inference_ms": dets.get("inference_ms", 0.0)
            },
            "model_info": {
                "name": detector.model_name,
                "is_custom": detector.is_custom_model
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame processing error: {str(e)}")


@router.get("/export/csv")
async def export_csv():
    """Export all student safety entry records as downloadable CSV file."""
    csv_content = await database.export_records_csv()
    filename = f"mechanical_safety_records_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/test/simulate_event")
async def simulate_event(req: SimulateEventRequest):
    """
    Test suite endpoint to verify all test scenarios (SAFE, MISSING_CAP, MISSING_SHOES, etc.).
    Inserts a record and returns the verification output.
    """
    tid = req.tracking_id or int(time.time() % 1000)

    scenario_map = {
        "SAFE": (True, True, SafetyStatus.SAFE, True, True, "Fully compliant: Safety cap & shoes detected", 0.95),
        "MISSING_SHOES": (True, False, SafetyStatus.NOT_SAFE, True, True, "NOT SAFE: Missing Safety Shoes", 0.91),
        "MISSING_CAP": (False, True, SafetyStatus.NOT_SAFE, True, True, "NOT SAFE: Missing Safety Cap", 0.89),
        "MISSING_BOTH": (False, False, SafetyStatus.NOT_SAFE, True, True, "NOT SAFE: Missing Safety Cap, Safety Shoes", 0.93),
        "OUT_OF_FRAME": (False, False, SafetyStatus.UNKNOWN, True, False, "UNKNOWN: Feet partially out of bottom frame", 0.45)
    }

    scen = req.scenario.upper()
    if scen not in scenario_map:
        raise HTTPException(status_code=400, detail=f"Invalid scenario. Supported: {list(scenario_map.keys())}")

    cap, shoes, status, head_vis, feet_vis, details, conf = scenario_map[scen]

    record_id = await database.add_record(
        tracking_id=tid,
        safety_cap=cap,
        safety_shoes=shoes,
        final_status=status,
        confidence=conf,
        head_visible=head_vis,
        feet_visible=feet_vis,
        details=details
    )

    return {
        "success": True,
        "record_id": record_id,
        "scenario": scen,
        "tracking_id": tid,
        "status": status,
        "safety_cap": cap,
        "safety_shoes": shoes,
        "details": details,
        "color": STATUS_COLORS[status]
    }
