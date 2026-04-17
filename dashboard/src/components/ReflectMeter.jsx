import React from 'react'

export default function ReflectMeter({ reflect, stats }) {
  const pct = reflect?.score !== null && reflect?.score !== undefined ? Math.round(reflect.score * 100) : null
  const verdict = reflect?.verdict || 'awaiting'
  const action = reflect?.action || 'idle'
  const color =
    pct === null ? 'var(--text-soft)' : pct < 30 ? 'var(--success)' : pct < 60 ? 'var(--warning)' : 'var(--danger)'

  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="section-label">Trust Layer</p>
          <p className="mt-2 text-3xl font-semibold tracking-[-0.04em]" style={{ color }}>
            {pct !== null ? `${pct}%` : '--'}
          </p>
        </div>
        <div className="text-right">
          <p className="mono text-xs uppercase tracking-[0.18em]" style={{ color }}>
            {verdict}
          </p>
          <p className="mt-1 text-xs text-[var(--text-soft)]">{action}</p>
        </div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
        <div
          className="h-2 rounded-full transition-all duration-500"
          style={{ width: `${pct ?? 0}%`, backgroundColor: color }}
        />
      </div>

      <p className="mt-3 text-sm leading-relaxed text-[var(--text-soft)]">
        {pct === null
          ? 'ReflectScore wakes up once NEXUS has a response to inspect.'
          : `Verdict ${verdict} with action ${action}. Lower is safer.`}
      </p>

      {reflect?.warning && (
        <div className="panel-muted mt-3 px-3 py-3 text-sm leading-relaxed text-[var(--warning)]">
          {reflect.warning}
        </div>
      )}

      {stats && (
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Metric label="clean" value={stats.clean} />
          <Metric label="warn" value={stats.warning} />
          <Metric label="blocked" value={stats.blocked} />
          <Metric label="rerouted" value={stats.rerouted} />
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="panel-muted px-3 py-3">
      <p className="section-label text-[10px]">{label}</p>
      <p className="mt-2 mono text-sm text-[var(--text)]">{value}</p>
    </div>
  )
}
