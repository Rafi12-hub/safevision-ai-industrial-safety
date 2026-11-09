import React, { useEffect, useState } from 'react';
import { 
  Users, 
  ShieldCheck, 
  ShieldAlert, 
  Clock, 
  Video, 
  ArrowRight, 
  AlertTriangle,
  RefreshCw,
  HardHat,
  Footprints
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { SafetyBadge } from '../components/SafetyBadge';
import { DashboardStats } from '../types';
import { api } from '../services/api';

interface DashboardPageProps {
  onNavigateToMonitoring: () => void;
  onNavigateToRecords: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onNavigateToMonitoring,
  onNavigateToRecords,
}) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStats = async () => {
    try {
      const data = await api.getDashboardStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load dashboard stats:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchStats();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-3">
          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
          <p className="text-sm text-slate-400">Loading safety monitoring telemetry...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight">Mechanical Lab Safety Dashboard</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Real-time automated computer vision verification for safety caps and safety shoes
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium border border-slate-700 transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin text-blue-400' : ''}`} />
            Refresh
          </button>
          <button
            onClick={onNavigateToMonitoring}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-sm font-bold shadow-lg shadow-blue-500/25 transition-all"
          >
            <Video className="w-4 h-4" />
            Open Live Camera
          </button>
        </div>
      </div>

      {/* 4 Dashboard Statistic Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Total Students Monitored"
          value={stats?.total_monitored ?? 0}
          subtext={`Today entries: ${stats?.today_total ?? 0}`}
          icon={Users}
          variant="blue"
        />

        <StatCard
          title="Safe Students"
          value={stats?.safe_students ?? 0}
          percentage={stats?.safe_percentage}
          subtext={`Today compliant: ${stats?.today_safe ?? 0}`}
          icon={ShieldCheck}
          variant="green"
        />

        <StatCard
          title="Unsafe Students"
          value={stats?.unsafe_students ?? 0}
          percentage={stats?.unsafe_percentage}
          subtext={`Today violations: ${stats?.today_unsafe ?? 0}`}
          icon={ShieldAlert}
          variant="red"
        />

        <StatCard
          title="Pending Verification"
          value={stats?.pending_verification ?? 0}
          subtext="Head or feet out of frame / in buffer"
          icon={Clock}
          variant="orange"
        />
      </div>

      {/* Live Monitoring Quick Section & Safety Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Monitoring Quick Feed Preview */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse-live" />
              <h2 className="text-base font-bold text-white">Live Monitoring Stream Preview</h2>
            </div>
            <button
              onClick={onNavigateToMonitoring}
              className="text-xs font-semibold text-blue-400 hover:text-blue-300 flex items-center gap-1 group"
            >
              Full Interactive View
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
            </button>
          </div>

          {/* Video Preview Frame */}
          <div 
            onClick={onNavigateToMonitoring}
            className="relative w-full aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center cursor-pointer group shadow-inner"
          >
            <img
              src="/api/camera/stream"
              alt="Live Laboratory Camera Stream"
              className="w-full h-full object-contain"
              onError={(e) => {
                // If stream is not running yet, show clean standby graphic
                (e.target as HTMLElement).style.display = 'none';
              }}
            />

            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
              <span className="text-sm font-semibold text-white flex items-center gap-2 bg-blue-600/80 px-3 py-1.5 rounded-lg backdrop-blur-sm">
                <Video className="w-4 h-4" /> Click to launch Live Fullscreen Camera
              </span>
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between text-xs text-slate-400 px-1">
            <span>Detection: YOLO11n CPU Inference</span>
            <span>Target: Safety Cap & Safety Shoes</span>
          </div>
        </div>

        {/* Safety Overview Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-white mb-4">Safety Compliance Overview</h2>

            {/* Compliance Circular Gauge & Metrics */}
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-4">
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1.5">
                  <span className="text-emerald-400">Safe Compliance Rate</span>
                  <span className="text-white">{stats?.safe_percentage ?? 0}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full transition-all duration-700"
                    style={{ width: `${stats?.safe_percentage ?? 0}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold mb-1.5">
                  <span className="text-rose-400">Violation Rate</span>
                  <span className="text-white">{stats?.unsafe_percentage ?? 0}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-rose-500 to-amber-500 rounded-full transition-all duration-700"
                    style={{ width: `${stats?.unsafe_percentage ?? 0}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Laboratory Rules Checklist */}
            <div className="mt-5 space-y-2.5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Mandatory Lab PPE Requirements</h3>
              <div className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-800/40 border border-slate-800 text-xs">
                <div className="w-7 h-7 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
                  <HardHat className="w-4 h-4" />
                </div>
                <div>
                  <p className="font-semibold text-slate-200">1. Mechanical Safety Cap</p>
                  <p className="text-slate-400">Required head protection inside machine zones</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-800/40 border border-slate-800 text-xs">
                <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
                  <Footprints className="w-4 h-4" />
                </div>
                <div>
                  <p className="font-semibold text-slate-200">2. Steel-Toe Safety Shoes</p>
                  <p className="text-slate-400">Required foot protection against falling equipment</p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Violations Notice */}
          <div className="mt-5 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
            <span>Missing either item automatically triggers RED NOT_SAFE status.</span>
          </div>
        </div>
      </div>

      {/* Recent Entry Records Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-white">Recent Laboratory Entry Records</h2>
            <p className="text-xs text-slate-400">Latest verified student admissions</p>
          </div>
          <button
            onClick={onNavigateToRecords}
            className="text-xs font-semibold text-blue-400 hover:text-blue-300 flex items-center gap-1 group"
          >
            View All Records
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-950/70 text-slate-400 text-xs uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Student ID</th>
                <th className="py-3 px-4">Date</th>
                <th className="py-3 px-4">Time</th>
                <th className="py-3 px-4">Safety Cap</th>
                <th className="py-3 px-4">Safety Shoes</th>
                <th className="py-3 px-4">Final Safety Status</th>
                <th className="py-3 px-4">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {stats?.latest_records && stats.latest_records.length > 0 ? (
                stats.latest_records.map((rec) => (
                  <tr key={rec.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-bold text-white">
                      STUDENT #{rec.tracking_id}
                    </td>
                    <td className="py-3 px-4 text-xs text-slate-400">{rec.date}</td>
                    <td className="py-3 px-4 text-xs text-slate-400">{rec.time}</td>
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${
                        rec.safety_cap ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                      }`}>
                        {rec.safety_cap ? 'Detected' : 'Missing'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${
                        rec.safety_shoes ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                      }`}>
                        {rec.safety_shoes ? 'Detected' : 'Missing'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <SafetyBadge status={rec.final_status} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-xs font-mono text-slate-400">
                      {(rec.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-500">
                    No student entry records found. Start the camera to begin monitoring.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
