import React from 'react';
import { CheckCircle2, XCircle, AlertCircle, HelpCircle } from 'lucide-react';
import { SafetyStatusType } from '../types';

interface SafetyBadgeProps {
  status: SafetyStatusType | string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const SafetyBadge: React.FC<SafetyBadgeProps> = ({
  status,
  size = 'md',
  showIcon = true,
}) => {
  const normalized = status.toUpperCase();

  let colorClasses = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
  let label = 'UNKNOWN';
  let Icon = HelpCircle;

  if (normalized === 'SAFE') {
    colorClasses = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    label = 'SAFE';
    Icon = CheckCircle2;
  } else if (normalized === 'NOT_SAFE' || normalized === 'UNSAFE') {
    colorClasses = 'bg-rose-500/10 text-rose-400 border-rose-500/30';
    label = 'NOT SAFE';
    Icon = XCircle;
  } else if (normalized === 'UNKNOWN') {
    colorClasses = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    label = 'UNKNOWN';
    Icon = AlertCircle;
  }

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs',
    lg: 'px-3.5 py-1.5 text-sm font-semibold',
  }[size];

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-3.5 h-3.5',
    lg: 'w-4 h-4',
  }[size];

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full font-medium border ${colorClasses} ${sizeClasses}`}>
      {showIcon && <Icon className={iconSizes} />}
      <span>{label}</span>
    </span>
  );
};
