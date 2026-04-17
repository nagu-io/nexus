import React from 'react'
import { ShieldCheck, ShieldAlert, ShieldOff, Shield } from 'lucide-react'

export default function ReflectMeter({ reflect, stats }) {
  const pct = reflect?.score !== null && reflect?.score !== undefined ? Math.round(reflect.score * 100) : null
  const verdict = reflect?.verdict || 'awaiting'
  const action = reflect?.action || 'idle'
  const color =
    pct === null ? 'var(--text-soft)' : pct < 30 ? 'var(--success)' : pct < 60 ? 'var(--warning)' : 'var(--danger)'

  const ShieldIcon = pct === null ? Shield : pct < 30 ? ShieldCheck : pct < 60 ? ShieldAlert : ShieldOff

  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="section-label">Trust Layer</p>
        <span className="mono text-[10px] uppercase tracking-[0.18em]" style={{ color }}>
          {verdict}
        </span>
      </div>

      {/* Score display */}
      <div className="mt-3 flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[rgba(255,255,255,0.04)]" style={{ color }}>
          <ShieldIcon size={18} />
        </div>
        <div className="flex-1">
          <p className="text-2xl font-semibold tracking-[-0.04em]" style={{ color }}>
            {pct !== null ? `${pct}%` : '--'}
          </p>
          <p className="text-[11px] text-[var(--text-muted)]">{action}</p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]">
        <div
          className="h-1.5 rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct ?? 0}%`, backgroundColor: color }}
        />
      </div>

      {/* Status message */}
      <p className="mt-3 text-[11px] leading-relaxed text-[var(--text-soft)]">
        {pct === null
          ? 'Send a prompt in repo execution mode to activate trust scoring.'
          : `Risk ${pct}% — ${verdict}. Lower is safer.`}
      </p>

      {reflect?.warning && (
        <div className="mt-3 rounded-xl border border-[rgba(240,202,114,0.15)] bg-[rgba(240,202,114,0.06)] px-3 py-2.5 text-[11px] leading-relaxed text-[var(--warning)]">
          {reflect.warning}
        </div>
      )}

      {/* Stats grid */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Metric label="clean" value={stats?.clean ?? 0} tone="var(--success)" />
        <Metric label="warn" value={stats?.warning ?? 0} tone="var(--warning)" />
        <Metric label="blocked" value={stats?.blocked ?? 0} tone="var(--danger)" />
        <Metric label="rerouted" value={stats?.rerouted ?? 0} tone="var(--text-soft)" />
      </div>
    </div>
  )
}

function Metric({ label, value, tone }) {
  return (
    <div className="panel-muted px-3 py-2">
      <div className="flex items-center justify-between">
        <p className="section-label text-[10px]">{label}</p>
        <p className="mono text-sm font-semibold" style={{ color: value > 0 ? tone : 'var(--text-soft)' }}>{value}</p>
      </div>
    </div>
  )
}
