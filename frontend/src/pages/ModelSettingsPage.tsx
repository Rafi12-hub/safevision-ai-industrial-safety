import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  CheckCircle2, 
  AlertTriangle, 
  Sliders, 
  Layers, 
  FolderTree, 
  Terminal, 
  Save, 
  RefreshCw,
  HardHat,
  Footprints,
  User,
  ShieldCheck
} from 'lucide-react';
import { ModelStatus } from '../types';
import { api } from '../services/api';

export const ModelSettingsPage: React.FC = () => {
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);

  // Editable parameters
  const [confThreshold, setConfThreshold] = useState<number>(0.40);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [imgSize, setImgSize] = useState<number>(480);

  const fetchModelStatus = async () => {
    setLoading(true);
    try {
      const data = await api.getModelStatus();
      setModel(data);
      if (data.conf_threshold) setConfThreshold(data.conf_threshold);
      if (data.iou_threshold) setIouThreshold(data.iou_threshold);
      if (data.imgsz) setImgSize(data.imgsz);
    } catch (e) {
      console.error('Failed to load model status:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSaveSuccess(false);
    try {
      await api.updateModelSettings({
        conf_threshold: confThreshold,
        iou_threshold: iouThreshold,
        imgsz: imgSize,
      });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
      fetchModelStatus();
    } catch (e) {
      console.error('Failed to update model settings:', e);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tight">AI Model Management & Pipeline</h1>
          <p className="text-xs text-slate-400 mt-1">
            YOLO object detection configuration, dataset instructions, and CPU inference parameters
          </p>
        </div>

        <button
          onClick={fetchModelStatus}
          className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-400' : ''}`} />
          Reload Model Diagnostics
        </button>
      </div>

      {/* Model Diagnostic Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center border shadow-inner ${
              model?.is_custom_model
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30'
            }`}>
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-white">{model?.model_name || 'Loading Model...'}</h3>
                {model?.is_custom_model ? (
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    Custom Trained best.pt Active
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                    Standard YOLO Active
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-0.5">{model?.status_message}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <span className="text-[11px] text-slate-500">Inference Device</span>
              <p className="text-xs font-bold text-slate-200">CPU (Intel Integrated)</p>
            </div>
            <div className="w-px h-8 bg-slate-800" />
            <div className="text-right">
              <span className="text-[11px] text-slate-500">Latency</span>
              <p className="text-xs font-bold text-emerald-400">{model?.last_inference_ms ?? 0} ms</p>
            </div>
          </div>
        </div>

        {/* Target Classes Mapping Table */}
        <div className="mt-5">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Model Classes Mapping</h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/30 flex items-center justify-center font-bold text-xs">
                0
              </div>
              <div>
                <p className="font-bold text-white text-xs">Class 0: Person</p>
                <p className="text-[11px] text-slate-400">Student full body bounding box</p>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center justify-center font-bold text-xs">
                1
              </div>
              <div>
                <p className="font-bold text-white text-xs">Class 1: Safety Cap</p>
                <p className="text-[11px] text-slate-400">Required head protection cap</p>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-center font-bold text-xs">
                2
              </div>
              <div>
                <p className="font-bold text-white text-xs">Class 2: Safety Shoes</p>
                <p className="text-[11px] text-slate-400">Protective steel-toe boots</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Inference Parameter Form */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-center gap-2 mb-4">
          <Sliders className="w-4 h-4 text-blue-400" />
          <h2 className="text-base font-bold text-white">Configurable Inference Thresholds</h2>
        </div>

        <form onSubmit={handleSaveSettings} className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Confidence Threshold Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <label className="font-semibold text-slate-300">Confidence Threshold</label>
                <span className="font-mono text-blue-400 font-bold">{confThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500">Minimum detection confidence before box is accepted</p>
            </div>

            {/* IOU Threshold Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <label className="font-semibold text-slate-300">NMS IoU Threshold</label>
                <span className="font-mono text-blue-400 font-bold">{iouThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.80"
                step="0.05"
                value={iouThreshold}
                onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500">Non-Maximum Suppression overlap threshold</p>
            </div>

            {/* Image Resolution */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold text-slate-300">Inference Input Size</label>
              <select
                value={imgSize}
                onChange={(e) => setImgSize(parseInt(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value={320}>320 x 320 (Ultra Fast CPU)</option>
                <option value={480}>480 x 480 (Recommended Balanced)</option>
                <option value={640}>640 x 640 (High Accuracy)</option>
              </select>
              <p className="text-[11px] text-slate-500">YOLO model tensor input resolution</p>
            </div>
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-slate-800">
            {saveSuccess ? (
              <span className="text-xs text-emerald-400 font-semibold flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" /> Parameters saved successfully!
              </span>
            ) : (
              <span className="text-xs text-slate-500">Changes apply to live webcam stream immediately.</span>
            )}

            <button
              type="submit"
              disabled={saving}
              className="flex items-center gap-2 px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-lg shadow-blue-500/20 transition-all"
            >
              <Save className="w-4 h-4" />
              {saving ? 'Saving...' : 'Save Parameters'}
            </button>
          </div>
        </form>
      </div>

      {/* Dataset & Training Instructions Card (Section 10 Requirements) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dataset Layout */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <FolderTree className="w-4 h-4 text-emerald-400" />
            <h3 className="text-base font-bold text-white">YOLO Dataset Directory Layout</h3>
          </div>

          <p className="text-xs text-slate-400">
            Add laboratory images and YOLO format text annotations into the project structure:
          </p>

          <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-emerald-400 font-mono text-xs overflow-x-auto">
{`dataset/
├── images/
│   ├── train/     # Laboratory student photos
│   └── val/       # Validation split
└── labels/
    ├── train/     # YOLO format .txt annotations
    └── val/       # Validation .txt annotations

Label format: <class_id> <x_center> <y_center> <width> <height>
(0 = person, 1 = safety_cap, 2 = safety_shoes)`}
          </pre>
        </div>

        {/* Training Commands */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-indigo-400" />
            <h3 className="text-base font-bold text-white">Model Training Pipeline Commands</h3>
          </div>

          <p className="text-xs text-slate-400">
            Run the provided training script in PowerShell/Command Prompt on Windows 11:
          </p>

          <div className="space-y-3">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 font-semibold block mb-1">1. Generate Starter Test Dataset:</span>
              <code className="text-xs font-mono text-blue-400">python training/generate_sample_dataset.py</code>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 font-semibold block mb-1">2. Train Custom PPE Model:</span>
              <code className="text-xs font-mono text-blue-400">python training/train.py --epochs 30 --batch 8 --imgsz 640 --device cpu</code>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[11px] text-slate-400 font-semibold block mb-1">3. Validate mAP & Recall:</span>
              <code className="text-xs font-mono text-blue-400">python training/validate.py</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
