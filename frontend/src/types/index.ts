export type SafetyStatusType = 'SAFE' | 'NOT_SAFE' | 'UNKNOWN';

export interface StudentDetection {
  tracking_id: number;
  person_box: [number, number, number, number];
  person_confidence: number;
  head_visible: boolean;
  feet_visible: boolean;
  cap_detected: boolean;
  shoes_detected: boolean;
  cap_confidence: number;
  shoes_confidence: number;
  raw_status: SafetyStatusType;
  status: SafetyStatusType;
  color: string;
  details: string;
  stability_score: number;
  frames_analyzed: number;
}

export interface SafetyRecord {
  id: number;
  tracking_id: number;
  timestamp: string;
  date: string;
  time: string;
  safety_cap: number | boolean;
  safety_shoes: number | boolean;
  final_status: SafetyStatusType;
  confidence: number;
  verification_status: string;
  head_visible?: number | boolean;
  feet_visible?: number | boolean;
  details?: string;
}

export interface DashboardStats {
  total_monitored: number;
  safe_students: number;
  unsafe_students: number;
  pending_verification: number;
  safe_percentage: number;
  unsafe_percentage: number;
  today_total: number;
  today_safe: number;
  today_unsafe: number;
  recent_violations: SafetyRecord[];
  latest_records: SafetyRecord[];
}

export interface AnalyticsData {
  daily_stats: Array<{
    date: string;
    safe: number;
    unsafe: number;
    unknown: number;
    total: number;
  }>;
  violation_distribution: Array<{
    violation_type: string;
    count: number;
  }>;
  weekly_compliance: Array<{
    week: string;
    compliance_rate: number;
    total_entries: number;
  }>;
  monthly_compliance: Array<{
    month: string;
    safe_count: number;
    unsafe_count: number;
    compliance_rate: number;
    total_entries: number;
  }>;
}

export interface ModelStatus {
  model_name: string;
  model_path: string;
  is_custom_model: boolean;
  is_loaded: boolean;
  device: string;
  imgsz: number;
  conf_threshold: number;
  iou_threshold: number;
  classes: Record<string, string>;
  last_inference_ms: number;
  status_message: string;
}

export interface CameraStatus {
  is_active: boolean;
  camera_index: number;
  resolution: string;
  fps: number;
  has_error: boolean;
  error_message?: string | null;
}
