import { DashboardStats, SafetyRecord, AnalyticsData, ModelStatus, CameraStatus, StudentDetection } from '../types';

const API_BASE = '/api';

export const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async getDashboardStats(): Promise<DashboardStats> {
    const res = await fetch(`${API_BASE}/dashboard/stats`);
    if (!res.ok) throw new Error('Failed to fetch dashboard stats');
    return res.json();
  },

  async getRecords(params: {
    page?: number;
    pageSize?: number;
    status?: string;
    dateFrom?: string;
    dateTo?: string;
    search?: string;
  } = {}): Promise<{
    records: SafetyRecord[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  }> {
    const query = new URLSearchParams();
    if (params.page) query.append('page', params.page.toString());
    if (params.pageSize) query.append('page_size', params.pageSize.toString());
    if (params.status && params.status !== 'ALL') query.append('status', params.status);
    if (params.dateFrom) query.append('date_from', params.dateFrom);
    if (params.dateTo) query.append('date_to', params.dateTo);
    if (params.search) query.append('search', params.search);

    const res = await fetch(`${API_BASE}/records?${query.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch records');
    return res.json();
  },

  async getAnalytics(days: number = 14): Promise<AnalyticsData> {
    const res = await fetch(`${API_BASE}/analytics?days=${days}`);
    if (!res.ok) throw new Error('Failed to fetch analytics');
    return res.json();
  },

  async getModelStatus(): Promise<ModelStatus> {
    const res = await fetch(`${API_BASE}/model/status`);
    if (!res.ok) throw new Error('Failed to fetch model status');
    return res.json();
  },

  async updateModelSettings(settings: {
    conf_threshold?: number;
    iou_threshold?: number;
    imgsz?: number;
  }) {
    const res = await fetch(`${API_BASE}/model/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('Failed to update model settings');
    return res.json();
  },

  async getSettings(): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/settings`);
    if (!res.ok) throw new Error('Failed to fetch system settings');
    return res.json();
  },

  async updateSettings(settings: Record<string, string>) {
    const res = await fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('Failed to save settings');
    return res.json();
  },

  async getCameraStatus(): Promise<CameraStatus> {
    const res = await fetch(`${API_BASE}/camera/status`);
    if (!res.ok) throw new Error('Failed to fetch camera status');
    return res.json();
  },

  async startCamera(camera_index: number = 0, resolution: string = '640x480') {
    const res = await fetch(`${API_BASE}/camera/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index, resolution }),
    });
    return res.json();
  },

  async stopCamera() {
    const res = await fetch(`${API_BASE}/camera/stop`, {
      method: 'POST',
    });
    return res.json();
  },

  async processFrame(imageBase64: string): Promise<{
    success: boolean;
    timestamp: number;
    students: StudentDetection[];
    raw_detections: {
      persons_count: number;
      caps_count: number;
      shoes_count: number;
      inference_ms: number;
    };
    model_info: {
      name: string;
      is_custom: boolean;
    };
  }> {
    const res = await fetch(`${API_BASE}/process_frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: imageBase64 }),
    });
    if (!res.ok) throw new Error('Failed to process camera frame');
    return res.json();
  },

  async simulateEvent(scenario: string, tracking_id?: number) {
    const res = await fetch(`${API_BASE}/test/simulate_event`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario, tracking_id }),
    });
    if (!res.ok) throw new Error('Failed to simulate test event');
    return res.json();
  },

  getExportCsvUrl(): string {
    return `${API_BASE}/export/csv`;
  }
};
