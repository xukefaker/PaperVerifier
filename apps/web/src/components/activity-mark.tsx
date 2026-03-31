'use client';

import { useEffect, useRef, useState } from 'react';
import { Sparkles } from 'lucide-react';

export type ActivityMarkMode = 'idle' | 'active' | 'done';

type ActivityMarkProps = {
  mode: ActivityMarkMode;
  label?: string | null;
  layout?: 'stacked' | 'inline';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
};

export function useTransientActivityMode(active: boolean, durationMs = 1400): ActivityMarkMode {
  const [mode, setMode] = useState<ActivityMarkMode>(active ? 'active' : 'idle');
  const previousActiveRef = useRef(active);

  useEffect(() => {
    let timeoutId: number | null = null;

    if (active) {
      setMode('active');
      previousActiveRef.current = true;
      return;
    }

    if (previousActiveRef.current) {
      setMode('done');
      timeoutId = window.setTimeout(() => {
        setMode('idle');
      }, durationMs);
      previousActiveRef.current = false;
    } else {
      setMode('idle');
    }

    return () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [active, durationMs]);

  return mode;
}

export function ActivityMark({
  mode,
  label,
  layout = 'inline',
  size = 'md',
  className = '',
}: ActivityMarkProps) {
  const sizeClassName =
    size === 'lg'
      ? 'h-12 w-12 rounded-[1.05rem]'
      : size === 'sm'
        ? 'h-8 w-8 rounded-[0.85rem]'
        : 'h-10 w-10 rounded-[0.95rem]';
  const iconClassName = size === 'lg' ? 'h-5 w-5' : size === 'sm' ? 'h-3.5 w-3.5' : 'h-4.5 w-4.5';
  const wrapperClassName =
    layout === 'stacked'
      ? 'inline-flex flex-col items-center gap-1.5'
      : 'inline-flex items-center gap-2.5';
  const showLabel = Boolean(label) && mode !== 'idle';

  return (
    <div className={`${wrapperClassName} ${className}`.trim()}>
      <div className={`psa-activity-mark psa-activity-mark--${mode}`}>
        <span className={`psa-activity-mark-aura ${sizeClassName}`} />
        <span
          className={`psa-activity-mark-shell ${sizeClassName} inline-flex items-center justify-center border border-white/15 text-white`}
        >
          <Sparkles className={iconClassName} strokeWidth={2.05} />
        </span>
      </div>
      <span
        className={`psa-activity-mark-label ${
          layout === 'stacked'
            ? 'min-h-[0.85rem] text-[0.58rem] tracking-[0.22em]'
            : 'text-[0.64rem] tracking-[0.18em]'
        } ${showLabel ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-1'}`}
      >
        {label ?? ''}
      </span>
    </div>
  );
}
