import React from 'react'

const fmtPercent = value => (
  value === null || value === undefined ? '--' : `${Math.round(value * 100)}%`
)

export default function SkillPatterns({ overview }) {
  const patterns = overview?.patterns || []
  const metrics = overview?.pattern_metrics || {}

  return (
    <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="mono text-cyan-400 text-xs uppercase tracking-widest">Skill Memory</p>
          <p className="text-xs text-[#4a7fa5] mt-1">
            {metrics.total_patterns ?? 0} patterns • avg success {fmtPercent(metrics.avg_success_rate)}
          </p>
        </div>
        <span className="mono text-[11px] text-[#4a7fa5]">
          avg retry {metrics.avg_retries !== undefined ? metrics.avg_retries.toFixed(1) : '--'}
        </span>
      </div>

      <div className="space-y-3">
        {patterns.length === 0 && (
          <p className="text-sm text-[#4a7fa5] leading-relaxed">
            Successful workflows will accumulate here and rank future plans automatically.
          </p>
        )}

        {patterns.map(pattern => (
          <div key={pattern.signature} className="rounded-lg border border-[#18324f] bg-[#0b1a2f] p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="mono text-xs text-cyan-300 uppercase">{pattern.primary_intent}</p>
                <p className="text-sm text-[#d7e7f7] mt-1">
                  {(pattern.examples && pattern.examples[0]) || pattern.signature}
                </p>
              </div>
              <span className="mono text-[11px] text-green-400">
                {fmtPercent(pattern.success_rate)}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-3 text-[11px] mono text-[#7ea6c7]">
              <span>runs {pattern.total_runs}</span>
              <span>retry {pattern.avg_retries.toFixed(1)}</span>
            </div>

            <p className="text-xs text-[#4a7fa5] mt-3 leading-relaxed">
              best path: {(pattern.best_agent_sequence || []).join(' -> ') || 'not learned yet'}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
