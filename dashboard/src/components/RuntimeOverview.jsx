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
    { label: 'Runs', value: metrics.total_runs ?? 0, tone: 'text-cyan-300' },
    { label: 'Success', value: fmtPercent(metrics.success_rate), tone: 'text-green-400' },
    { label: 'Avg Retries', value: fmtNumber(metrics.avg_retries), tone: 'text-yellow-300' },
    { label: 'Confidence', value: fmtPercent(metrics.avg_confidence), tone: 'text-cyan-400' },
    { label: 'Cache Reuse', value: fmtPercent(metrics.cache_reuse_rate), tone: 'text-purple-300' },
    { label: 'Hot Cache', value: metrics.reusable_cache_entries ?? 0, tone: 'text-orange-300' },
  ]

  return (
    <div className="bg-[#071222] border border-[#1e3a5f] rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="mono text-cyan-400 text-xs uppercase tracking-widest">Execution Intel</p>
          <p className="text-sm text-[#7ea6c7] mt-1">
            Recent autonomous runs, confidence, and cache health.
          </p>
        </div>
        <span className="mono text-[11px] text-[#4a7fa5]">
          {overview ? 'live' : 'awaiting runtime data'}
        </span>
      </div>

      <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
        {cards.map(card => (
          <div key={card.label} className="rounded-lg border border-[#18324f] bg-[#0b1a2f] p-3">
            <p className="text-[11px] uppercase tracking-widest text-[#4a7fa5] mono">{card.label}</p>
            <p className={`text-xl font-semibold mt-2 ${card.tone}`}>{card.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
