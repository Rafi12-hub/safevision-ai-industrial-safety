import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Volume2, 
  VolumeX, 
  Camera, 
  Clock, 
  ShieldCheck, 
  Save, 
  CheckCircle2, 
  RefreshCw,
  Sparkles,
  Sliders
} from 'lucide-react';
import { api } from '../services/api';
import { audioAlert } from '../services/audioAlert';

export const SystemSettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [saved, setSaved] = useState<boolean>(false);

  // Form states
  const [cameraIndex, setCameraIndex] = useState<string>('0');
  const [resolution, setResolution] = useState<string>('640x480');
  const [alertSound, setAlertSound] = useState<boolean>(true);
  const [hysteresisFrames, setHysteresisFrames] = useState<string>('8');
  const [sessionCooldown, setSessionCooldown] = useState<string>('20');
  const [simStatus, setSimStatus] = useState<string | null>(null);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const data = await api.getSettings();
      setSettings(data);
      if (data.camera_index) setCameraIndex(data.camera_index);
      if (data.resolution) setResolution(data.resolution);
      if (data.alert_sound) setAlertSound(data.alert_sound === 'true');
      if (data.hysteresis_frames) setHysteresisFrames(data.hysteresis_frames);
      if (data.session_cooldown_sec) setSessionCooldown(data.session_cooldown_sec);
    } catch (e) {
      console.error('Failed to load settings:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSettings();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSaved(false);
    try {
      await api.updateSettings({
        camera_index: cameraIndex,
        resolution: resolution,
        alert_sound: alertSound ? 'true' : 'false',
        hysteresis_frames: hysteresisFrames,
        session_cooldown_sec: sessionCooldown,
      });
      audioAlert.setEnabled(alertSound);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e) {
      console.error('Failed to save settings:', e);
    } finally {
      setSaving(false);
    }
  };

  const handleTestAudio = () => {
    audioAlert.playViolationAlarm('Test violation audio check');
  };

  const handleRunSimulation = async (scenario: string) => {
    try {
      const res = await api.simulateEvent(scenario);
      setSimStatus(`Simulation Recorded: #${res.record_id} [${res.scenario}] -> ${res.status}`);
      setTimeout(() => setSimStatus(null), 4000);
    } catch (e) {
      setSimStatus('Simulation failed');
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight">System & Hardware Configuration</h1>
          <p className="text-xs text-slate-400 mt-1">
            Webcam device mapping, audio alarm preferences, and temporal stabilization controls
          </p>
        </div>

        <button
          onClick={fetchSettings}
          className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-400' : ''}`} />
          Reload Settings
        </button>
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Hardware & Camera Settings Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <Camera className="w-4 h-4 text-blue-400" />
            <h2 className="text-base font-bold text-white">Camera Hardware Preferences</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Webcam Device Index
              </label>
              <select
                value={cameraIndex}
                onChange={(e) => setCameraIndex(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="0">Camera 0 (Default Laptop Built-in Webcam)</option>
                <option value="1">Camera 1 (External USB Webcam)</option>
                <option value="2">Camera 2 (Secondary Lab Camera)</option>
              </select>
              <p className="text-[11px] text-slate-500 mt-1">
                DirectShow OpenCV camera device index on Windows 11.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Default Stream Resolution
              </label>
              <select
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="640x480">640 x 480 (Optimal for Intel CPU Inference)</option>
                <option value="1280x720">1280 x 720 (720p HD)</option>
              </select>
              <p className="text-[11px] text-slate-500 mt-1">
                Resolution negotiated with OpenCV video capture backend.
              </p>
            </div>
          </div>
        </div>

        {/* Audio Notification Settings Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-emerald-400" />
              <h2 className="text-base font-bold text-white">Audio Warning Sound System</h2>
            </div>
            <button
              type="button"
              onClick={handleTestAudio}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors flex items-center gap-1.5"
            >
              <Volume2 className="w-3.5 h-3.5" />
              Test Audio Alarm Tone
            </button>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-950 border border-slate-800">
            <div>
              <p className="text-xs font-bold text-white">Enable Real-Time Safety Violation Siren</p>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Emits an alert tone when a student enters missing a safety cap or shoes
              </p>
            </div>

            <button
              type="button"
              onClick={() => setAlertSound(!alertSound)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                alertSound ? 'bg-blue-600' : 'bg-slate-700'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  alertSound ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Temporal Stability & Anti-Spam Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-amber-400" />
            <h2 className="text-base font-bold text-white">Temporal Stability & Session Rules (Anti-Duplicate)</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Consecutive Frame Confirmation Window: <strong className="text-blue-400">{hysteresisFrames} frames</strong>
              </label>
              <input
                type="range"
                min="4"
                max="20"
                step="1"
                value={hysteresisFrames}
                onChange={(e) => setHysteresisFrames(e.target.value)}
                className="w-full accent-blue-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Requires consistent detection across multiple consecutive frames before confirming a violation.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Student Entry Cooldown: <strong className="text-blue-400">{sessionCooldown} seconds</strong>
              </label>
              <input
                type="range"
                min="10"
                max="60"
                step="5"
                value={sessionCooldown}
                onChange={(e) => setSessionCooldown(e.target.value)}
                className="w-full accent-blue-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Prevents spamming the database while a student remains stationary in front of the camera.
              </p>
            </div>
          </div>
        </div>

        {/* Save Bar */}
        <div className="flex items-center justify-between p-4 bg-slate-900 border border-slate-800 rounded-2xl">
          {saved ? (
            <span className="text-xs text-emerald-400 font-semibold flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" /> System settings successfully updated!
            </span>
          ) : (
            <span className="text-xs text-slate-400">Values are stored in the SQLite system_settings table.</span>
          )}

          <button
            type="submit"
            disabled={saving}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-bold shadow-lg shadow-blue-500/20 transition-all"
          >
            <Save className="w-4 h-4" />
            {saving ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </form>

      {/* Scenario Verification Injector */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-bold text-white">Verification Test Injector (Section 15 Requirements)</h3>
          </div>
          {simStatus && (
            <span className="text-xs px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/30 font-semibold">
              {simStatus}
            </span>
          )}
        </div>

        <p className="text-xs text-slate-400">
          Inject test scenarios into the SQLite pipeline to verify status classification, database records, and siren alerts:
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2">
          <button
            type="button"
            onClick={() => handleRunSimulation('SAFE')}
            className="p-2.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-xs font-semibold text-center transition-all"
          >
            Test 1: SAFE
          </button>

          <button
            type="button"
            onClick={() => handleRunSimulation('MISSING_SHOES')}
            className="p-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-400 text-xs font-semibold text-center transition-all"
          >
            Test 2: Missing Shoes
          </button>

          <button
            type="button"
            onClick={() => handleRunSimulation('MISSING_CAP')}
            className="p-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-400 text-xs font-semibold text-center transition-all"
          >
            Test 3: Missing Cap
          </button>

          <button
            type="button"
            onClick={() => handleRunSimulation('MISSING_BOTH')}
            className="p-2.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-400 text-xs font-semibold text-center transition-all"
          >
            Test 4: Missing Both
          </button>

          <button
            type="button"
            onClick={() => handleRunSimulation('OUT_OF_FRAME')}
            className="p-2.5 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-400 text-xs font-semibold text-center transition-all"
          >
            Test 5: Out of Frame
          </button>
        </div>
      </div>
    </div>
  );
};
