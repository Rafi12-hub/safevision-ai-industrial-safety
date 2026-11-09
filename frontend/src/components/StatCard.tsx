import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  icon: LucideIcon;
  variant?: 'blue' | 'green' | 'red' | 'orange';
  percentage?: number;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtext,
  icon: Icon,
  variant = 'blue',
  percentage,
}) => {
  const variantStyles = {
    blue: {
      bg: 'bg-blue-950/40 border-blue-800/40 hover:border-blue-700/60',
      iconBg: 'bg-blue-600/20 text-blue-400 border-blue-500/30',
      text: 'text-blue-400',
      bar: 'bg-blue-500',
    },
    green: {
      bg: 'bg-emerald-950/40 border-emerald-800/40 hover:border-emerald-700/60',
      iconBg: 'bg-emerald-600/20 text-emerald-400 border-emerald-500/30',
      text: 'text-emerald-400',
      bar: 'bg-emerald-500',
    },
    red: {
      bg: 'bg-rose-950/40 border-rose-800/40 hover:border-rose-700/60',
      iconBg: 'bg-rose-600/20 text-rose-400 border-rose-500/30',
      text: 'text-rose-400',
      bar: 'bg-rose-500',
    },
    orange: {
      bg: 'bg-amber-950/40 border-amber-800/40 hover:border-amber-700/60',
      iconBg: 'bg-amber-600/20 text-amber-400 border-amber-500/30',
      text: 'text-amber-400',
      bar: 'bg-amber-500',
    },
  }[variant];

  return (
    <div className={`p-5 rounded-2xl border transition-all duration-300 shadow-lg backdrop-blur-sm ${variantStyles.bg}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold tracking-wider text-slate-400 uppercase">{title}</p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white">{value}</span>
            {percentage !== undefined && (
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${variantStyles.iconBg}`}>
                {percentage}%
              </span>
            )}
          </div>
        </div>
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center border shadow-inner ${variantStyles.iconBg}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>

      {subtext && <p className="mt-3 text-xs text-slate-400">{subtext}</p>}

      {percentage !== undefined && (
        <div className="mt-3 w-full bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${variantStyles.bar}`}
            style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
          />
        </div>
      )}
    </div>
  );
};
