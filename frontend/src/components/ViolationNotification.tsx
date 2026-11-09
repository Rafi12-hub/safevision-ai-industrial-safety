import React, { useEffect } from 'react';
import { AlertOctagon, X, Volume2 } from 'lucide-react';
import { audioAlert } from '../services/audioAlert';

interface ViolationNotificationProps {
  message: string | null;
  details?: string;
  onDismiss: () => void;
  soundEnabled: boolean;
}

export const ViolationNotification: React.FC<ViolationNotificationProps> = ({
  message,
  details,
  onDismiss,
  soundEnabled,
}) => {
  useEffect(() => {
    if (message && soundEnabled) {
      audioAlert.playViolationAlarm(details || message);
    }
  }, [message, details, soundEnabled]);

  if (!message) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-md w-full animate-in slide-in-from-bottom-5 duration-300">
      <div className="p-4 rounded-2xl bg-rose-950/95 border border-rose-500/50 shadow-2xl shadow-rose-950/80 backdrop-blur-md flex items-start gap-3.5">
        <div className="w-10 h-10 rounded-xl bg-rose-600/30 border border-rose-500/40 flex items-center justify-center shrink-0 text-rose-400">
          <AlertOctagon className="w-5 h-5 animate-pulse" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-bold text-rose-200 uppercase tracking-wide">Safety Violation Alert</h4>
            {soundEnabled && <Volume2 className="w-3.5 h-3.5 text-rose-400 animate-pulse" />}
          </div>
          <p className="mt-1 text-sm font-semibold text-white">{message}</p>
          {details && <p className="mt-0.5 text-xs text-rose-300/80">{details}</p>}
        </div>

        <button
          onClick={onDismiss}
          className="p-1.5 rounded-lg text-rose-300 hover:text-white hover:bg-rose-900/60 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
