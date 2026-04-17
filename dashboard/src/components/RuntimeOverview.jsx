import React from 'react'

const fmtPercent = value => (
  value === null || value === undefined ? '--' : `${Math.round(value * 100)}%`
)

const fmtNumber = value => (
  value === null || value === undefined ? '--' : Number(value).toFixed(1)
)

export default function RuntimeOverview({ overview }) {
  const metrics = overview?.metrics || {}
  const cards = [
    { label: 'Runs', value: metrics.total_runs ?? 0, tone: 'text-[var(--text-strong)]' },
    { label: 'Success', value: fmtPercent(metrics.success_rate), tone: 'text-[var(--success)]' },
    { label: 'Retry', value: fmtNumber(metrics.avg_retries), tone: 'text-[var(--warning)]' },
    { label: 'Conf', value: fmtPercent(metrics.avg_confidence), tone: 'text-[var(--accent-2)]' },
  ]

  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between">
        <p className="section-label">Runtime</p>
        <span className="meta-pill mono text-[11px]">{overview ? 'live' : 'waiting'}</span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        {cards.map(card => (
          <div key={card.label} className="panel-muted px-3 py-3">
            <p className="section-label text-[10px]">{card.label}</p>
            <p className={`mt-2 text-2xl font-semibold tracking-[-0.04em] ${card.tone}`}>{card.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
