import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Square, 
  Camera, 
  Settings2, 
  Volume2, 
  VolumeX, 
  AlertOctagon, 
  RefreshCw, 
  CheckCircle2, 
  XCircle, 
  AlertCircle,
  HardHat,
  Footprints,
  Cpu,
  MonitorCheck,
  Sparkles
} from 'lucide-react';
import { SafetyBadge } from '../components/SafetyBadge';
import { ViolationNotification } from '../components/ViolationNotification';
import { StudentDetection, CameraStatus, ModelStatus } from '../types';
import { api } from '../services/api';
import { audioAlert } from '../services/audioAlert';

export const LiveMonitoringPage: React.FC = () => {
  // Streaming modes: 'browser' (Direct Browser Webcam Canvas) or 'backend' (OpenCV Backend Stream)
  const [streamMode, setStreamMode] = useState<'browser' | 'backend'>('browser');
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraIndex, setCameraIndex] = useState<number>(0);
  const [resolution, setResolution] = useState<string>('640x480');
  const [fps, setFps] = useState<number>(0);
  const [students, setStudents] = useState<StudentDetection[]>([]);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
  const [activeAlert, setActiveAlert] = useState<string | null>(null);
  const [alertDetails, setAlertDetails] = useState<string>('');
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loadingCamera, setLoadingCamera] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // References for Browser Webcam
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const isProcessingRef = useRef<boolean>(false);
  const fpsTimerRef = useRef<{ count: number; lastTime: number }>({ count: 0, lastTime: Date.now() });

  // Initial diagnostics
  useEffect(() => {
    fetchDiagnostics();
  }, []);

  const fetchDiagnostics = async () => {
    try {
      const [cam, model] = await Promise.all([
        api.getCameraStatus(),
        api.getModelStatus(),
      ]);
      setCameraStatus(cam);
      setModelStatus(model);
      if (cam.is_active && streamMode === 'backend') {
        setCameraActive(true);
      }
    } catch (e) {
      console.error('Failed to get diagnostics:', e);
    }
  };

  // Connect WebSocket when in backend mode
  useEffect(() => {
    if (streamMode !== 'backend' || !cameraActive) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'telemetry_update') {
          setFps(msg.camera_fps);
          setStudents(msg.students || []);

          // Check for violations
          const violations = (msg.students || []).filter((s: StudentDetection) => s.status === 'NOT_SAFE');
          if (violations.length > 0) {
            const v = violations[0];
            setActiveAlert(`Student #${v.tracking_id} Safety Violation!`);
            setAlertDetails(v.details);
          }
        }
      } catch (e) {
        console.error('WebSocket parse error:', e);
      }
    };

    return () => {
      ws.close();
    };
  }, [streamMode, cameraActive]);

  // Start / Stop Browser Webcam
  const startBrowserWebcam = async () => {
    setLoadingCamera(true);
    setCameraError(null);
    try {
      const [w, h] = resolution.split('x').map(Number);
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: w },
          height: { ideal: h },
          facingMode: 'user',
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraActive(true);
      startProcessingLoop();
    } catch (err: unknown) {
      console.error('Browser webcam access error:', err);
      const msg = err instanceof Error ? err.message : 'Permission denied or webcam unavailable';
      setCameraError(`Camera Error: ${msg}. Please ensure camera access is allowed.`);
      setCameraActive(false);
    } finally {
      setLoadingCamera(false);
    }
  };

  const stopBrowserWebcam = () => {
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setCameraActive(false);
    setStudents([]);
    setFps(0);
  };

  // Start / Stop Backend OpenCV Camera
  const toggleBackendCamera = async () => {
    setLoadingCamera(true);
    setCameraError(null);
    try {
      if (cameraActive) {
        await api.stopCamera();
        setCameraActive(false);
        setStudents([]);
        setFps(0);
      } else {
        const res = await api.startCamera(cameraIndex, resolution);
        if (res.success) {
          setCameraActive(true);
        } else {
          setCameraError(res.message || 'Failed to start OpenCV camera');
        }
      }
      fetchDiagnostics();
    } catch (err) {
      setCameraError('Failed to communicate with camera subsystem');
    } finally {
      setLoadingCamera(false);
    }
  };

  // Start/Stop Handler depending on mode
  const handleToggleCamera = () => {
    if (streamMode === 'browser') {
      if (cameraActive) {
        stopBrowserWebcam();
      } else {
        startBrowserWebcam();
      }
    } else {
      toggleBackendCamera();
    }
  };

  // In-Browser Processing Loop: sends frame to /api/process_frame and draws overlays
  const startProcessingLoop = () => {
    const processFrame = async () => {
      if (!videoRef.current || !canvasRef.current || !streamRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.readyState >= 2 && !isProcessingRef.current) {
        isProcessingRef.current = true;

        // Render video onto canvas
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');

        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

          try {
            const result = await api.processFrame(dataUrl);
            if (result.success) {
              setStudents(result.students);

              // Draw overlays on canvas
              drawOverlays(ctx, result.students, canvas.width, canvas.height);

              // Track violations
              const violations = result.students.filter((s) => s.status === 'NOT_SAFE');
              if (violations.length > 0) {
                const v = violations[0];
                setActiveAlert(`Student #${v.tracking_id} Safety Violation!`);
                setAlertDetails(v.details);
              }
            }
          } catch (e) {
            // frame send error or backend busy
          } finally {
            isProcessingRef.current = false;
          }

          // Compute FPS
          fpsTimerRef.current.count += 1;
          const now = Date.now();
          const elapsed = now - fpsTimerRef.current.lastTime;
          if (elapsed >= 1000) {
            setFps(Math.round((fpsTimerRef.current.count * 1000) / elapsed));
            fpsTimerRef.current.count = 0;
            fpsTimerRef.current.lastTime = now;
          }
        }
      }

      animationFrameId.current = requestAnimationFrame(processFrame);
    };

    animationFrameId.current = requestAnimationFrame(processFrame);
  };

  // Draw clean bounding boxes & PPE badges on canvas
  const drawOverlays = (
    ctx: CanvasRenderingContext2D,
    studentsList: StudentDetection[],
    w: number,
    h: number
  ) => {
    for (const s of studentsList) {
      const [x1, y1, x2, y2] = s.person_box;
      const boxW = x2 - x1;
      const boxH = y2 - y1;

      // Color scheme
      let strokeColor = '#f59e0b'; // orange
      let bgBadgeColor = 'rgba(245, 158, 11, 0.9)';

      if (s.status === 'SAFE') {
        strokeColor = '#10b981'; // green
        bgBadgeColor = 'rgba(16, 185, 129, 0.9)';
      } else if (s.status === 'NOT_SAFE') {
        strokeColor = '#ef4444'; // red
        bgBadgeColor = 'rgba(239, 68, 68, 0.9)';
      }

      // Draw Main Bounding Box
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(x1, y1, boxW, boxH);

      // Student Header Pill above head
      const headerText = `STUDENT #${s.tracking_id}: ${s.status}`;
      ctx.font = 'bold 13px Inter, sans-serif';
      const textWidth = ctx.measureText(headerText).width;

      const badgeY = Math.max(0, y1 - 26);
      ctx.fillStyle = bgBadgeColor;
      ctx.fillRect(x1, badgeY, textWidth + 16, 24);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(headerText, x1 + 8, badgeY + 16);

      // PPE Pills inside bounding box top
      // Cap Pill
      const capText = `CAP: ${s.cap_detected ? 'YES' : 'NO'}`;
      ctx.fillStyle = s.cap_detected ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)';
      ctx.fillRect(x1 + 4, y1 + 6, 75, 20);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.fillText(capText, x1 + 10, y1 + 20);

      // Shoes Pill
      const shoesText = `SHOES: ${s.shoes_detected ? 'YES' : 'NO'}`;
      ctx.fillStyle = s.shoes_detected ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)';
      ctx.fillRect(x1 + 84, y1 + 6, 88, 20);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(shoesText, x1 + 90, y1 + 20);

      // Out of frame warning tag if truncated
      if (!s.head_visible || !s.feet_visible) {
        ctx.fillStyle = 'rgba(245, 158, 11, 0.95)';
        ctx.fillRect(x1, y2 - 22, 140, 20);
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillText('BORDER OCCLUSION', x1 + 6, y2 - 8);
      }
    }
  };

  // Quick Simulation Test Buttons to verify all scenarios
  const handleSimulate = async (scenario: string) => {
    try {
      const res = await api.simulateEvent(scenario);
      if (res.status === 'NOT_SAFE') {
        setActiveAlert(`Test Simulation: ${res.details}`);
        setAlertDetails(res.details);
      }
      // Re-trigger visual feedback
      fetchDiagnostics();
    } catch (e) {
      console.error('Simulation failed:', e);
    }
  };

  // Stop camera on component unmount
  useEffect(() => {
    return () => {
      if (streamMode === 'browser') {
        stopBrowserWebcam();
      }
    };
  }, []);

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Top Banner & Control Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className={`w-3 h-3 rounded-full ${cameraActive ? 'bg-emerald-400 animate-pulse-live' : 'bg-slate-500'}`} />
              <h1 className="text-xl font-black text-white tracking-tight">Real-Time Mechanical Laboratory Safety Monitor</h1>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Live spatial PPE verification for Safety Cap & Safety Shoes | CPU-based AI inference
            </p>
          </div>

          {/* Quick Hardware Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Stream Mode Toggle */}
            <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => {
                  if (cameraActive) handleToggleCamera();
                  setStreamMode('browser');
                }}
                className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                  streamMode === 'browser'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                In-Browser WebCam
              </button>
              <button
                onClick={() => {
                  if (cameraActive) handleToggleCamera();
                  setStreamMode('backend');
                }}
                className={`px-3 py-1.5 rounded-lg font-semibold transition-all ${
                  streamMode === 'backend'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Backend OpenCV
              </button>
            </div>

            {/* Resolution Selector */}
            <select
              value={resolution}
              disabled={cameraActive}
              onChange={(e) => setResolution(e.target.value)}
              className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:ring-1 focus:ring-blue-500 focus:outline-none disabled:opacity-50"
            >
              <option value="640x480">640 x 480 (Recommended for CPU)</option>
              <option value="1280x720">1280 x 720 (HD)</option>
            </select>

            {/* Audio Alarm Toggle */}
            <button
              onClick={() => {
                const next = !soundEnabled;
                setSoundEnabled(next);
                audioAlert.setEnabled(next);
              }}
              title={soundEnabled ? 'Warning Alarm Sound Enabled' : 'Muted'}
              className={`p-2 rounded-xl border transition-colors ${
                soundEnabled
                  ? 'bg-blue-600/20 text-blue-400 border-blue-500/40'
                  : 'bg-slate-800 text-slate-500 border-slate-700'
              }`}
            >
              {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </button>

            {/* Start / Stop Main Button */}
            <button
              onClick={handleToggleCamera}
              disabled={loadingCamera}
              className={`flex items-center gap-2 px-5 py-2 rounded-xl font-bold text-sm shadow-lg transition-all ${
                cameraActive
                  ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/30'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/30'
              }`}
            >
              {loadingCamera ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : cameraActive ? (
                <Square className="w-4 h-4 fill-current" />
              ) : (
                <Play className="w-4 h-4 fill-current" />
              )}
              {loadingCamera ? 'Connecting...' : cameraActive ? 'Stop Camera' : 'Start Camera'}
            </button>
          </div>
        </div>

        {/* Camera Error Alert if camera disconnected (Test 7) */}
        {cameraError && (
          <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{cameraError}</span>
            </div>
            <button onClick={() => setCameraError(null)} className="text-rose-400 hover:text-white font-bold ml-2">
              Dismiss
            </button>
          </div>
        )}
      </div>

      {/* Main Video Screen & Live Telemetry Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Live Video Display (3 cols) */}
        <div className="lg:col-span-3 bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col justify-between">
          <div className="relative w-full aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
            {/* Stream Mode: Browser HTML5 Webcam + Canvas Overlays */}
            {streamMode === 'browser' ? (
              <>
                {/* Hidden video element used as source for canvas */}
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  className="hidden"
                />

                {/* Visible Canvas rendering frame + bounding boxes */}
                <canvas
                  ref={canvasRef}
                  className={`w-full h-full object-contain ${cameraActive ? 'block' : 'hidden'}`}
                />

                {!cameraActive && (
                  <div className="flex flex-col items-center justify-center gap-3 p-6 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-400 shadow-inner">
                      <Camera className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">Browser Webcam Inactive</h3>
                      <p className="text-xs text-slate-400 max-w-sm mt-1">
                        Click "Start Camera" above to activate your laptop webcam and begin live AI verification.
                      </p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              /* Stream Mode: Backend OpenCV MJPEG Stream */
              <>
                {cameraActive ? (
                  <img
                    src="/api/camera/stream"
                    alt="OpenCV Live Video Stream"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center gap-3 p-6 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-400 shadow-inner">
                      <MonitorCheck className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">OpenCV Backend Stream Standby</h3>
                      <p className="text-xs text-slate-400 max-w-sm mt-1">
                        Starts the native OpenCV camera server process on Windows 11.
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* In-Video HUD Bar */}
            {cameraActive && (
              <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                <div className="flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-xs font-semibold text-white">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-live" />
                  <span>LIVE HUD: {fps} FPS</span>
                </div>

                <div className="flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-xs text-slate-300">
                  <span>Detected Students: <strong className="text-white">{students.length}</strong></span>
                </div>
              </div>
            )}
          </div>

          {/* Model Status Bar Below Video (Never Fake Results) */}
          <div className="mt-4 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-400 gap-2 border-t border-slate-800/80 pt-3">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-400" />
              <span>
                Engine: <strong className="text-slate-200">{modelStatus?.model_name || 'Loading...'}</strong>
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span>Device: <strong className="text-slate-200">CPU (Intel Graphics)</strong></span>
              <span>Resolution: <strong className="text-slate-200">{resolution}</strong></span>
            </div>
          </div>
        </div>

        {/* Right Sidebar: Active Detected Students & Rules (1 col) */}
        <div className="space-y-4">
          {/* Active Students Detection Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
            <h3 className="text-sm font-bold text-white flex items-center justify-between mb-3">
              <span>Tracked Students in Frame</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-semibold">
                {students.length} active
              </span>
            </h3>

            {students.length > 0 ? (
              <div className="space-y-3 max-h-[360px] overflow-y-auto pr-1">
                {students.map((student) => (
                  <div
                    key={student.tracking_id}
                    className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white text-xs">STUDENT #{student.tracking_id}</span>
                      <SafetyBadge status={student.status} size="sm" />
                    </div>

                    {/* PPE Breakdown */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className={`p-1.5 rounded-lg border flex items-center gap-1.5 ${
                        student.cap_detected
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                      }`}>
                        <HardHat className="w-3.5 h-3.5" />
                        <span className="font-medium">{student.cap_detected ? 'Cap: YES' : 'Cap: NO'}</span>
                      </div>

                      <div className={`p-1.5 rounded-lg border flex items-center gap-1.5 ${
                        student.shoes_detected
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                      }`}>
                        <Footprints className="w-3.5 h-3.5" />
                        <span className="font-medium">{student.shoes_detected ? 'Shoes: YES' : 'Shoes: NO'}</span>
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-400">{student.details}</p>

                    <div className="text-[10px] text-slate-500 flex justify-between">
                      <span>Stability: {(student.stability_score * 100).toFixed(0)}%</span>
                      <span>Frames: {student.frames_analyzed}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-8 text-center text-xs text-slate-500 space-y-2">
                <AlertCircle className="w-6 h-6 mx-auto text-slate-600" />
                <p>No students currently detected in camera view.</p>
              </div>
            )}
          </div>

          {/* Verification Rules Legend Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl text-xs space-y-3">
            <h4 className="font-bold text-white uppercase tracking-wider text-[11px]">Safety Logic Legend</h4>

            <div className="flex items-start gap-2.5">
              <span className="w-3 h-3 rounded-full bg-emerald-500 shrink-0 mt-0.5" />
              <div>
                <strong className="text-emerald-400">GREEN – SAFE:</strong>
                <p className="text-slate-400 text-[11px]">Both safety cap & shoes detected on student.</p>
              </div>
            </div>

            <div className="flex items-start gap-2.5">
              <span className="w-3 h-3 rounded-full bg-rose-500 shrink-0 mt-0.5" />
              <div>
                <strong className="text-rose-400">RED – NOT SAFE:</strong>
                <p className="text-slate-400 text-[11px]">Either cap or shoes missing while visible.</p>
              </div>
            </div>

            <div className="flex items-start gap-2.5">
              <span className="w-3 h-3 rounded-full bg-amber-500 shrink-0 mt-0.5" />
              <div>
                <strong className="text-amber-400">ORANGE – UNKNOWN:</strong>
                <p className="text-slate-400 text-[11px]">Head or feet out of frame / buffer pending.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Test Scenarios Quick Simulator Bar (Section 15 Testing) */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <h3 className="text-xs font-bold text-white uppercase tracking-wider">
            Automated Scenario Verification Controls (Section 15 Tests)
          </h3>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          <button
            onClick={() => handleSimulate('SAFE')}
            className="p-2.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-xs font-semibold text-center transition-all"
          >
            Test 1: Safe (Cap + Shoes)
          </button>

          <button
            onClick={() => handleSimulate('MISSING_SHOES')}
            className="p-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-400 text-xs font-semibold text-center transition-all"
          >
            Test 2: Missing Shoes
          </button>

          <button
            onClick={() => handleSimulate('MISSING_CAP')}
            className="p-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-400 text-xs font-semibold text-center transition-all"
          >
            Test 3: Missing Cap
          </button>

          <button
            onClick={() => handleSimulate('MISSING_BOTH')}
            className="p-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-400 text-xs font-semibold text-center transition-all"
          >
            Test 4: Missing Both
          </button>

          <button
            onClick={() => handleSimulate('OUT_OF_FRAME')}
            className="p-2.5 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-400 text-xs font-semibold text-center transition-all"
          >
            Test 5: Out of Frame
          </button>
        </div>
      </div>

      {/* Floating Safety Violation Toast Notification */}
      <ViolationNotification
        message={activeAlert}
        details={alertDetails}
        onDismiss={() => setActiveAlert(null)}
        soundEnabled={soundEnabled}
      />
    </div>
  );
};
