import React from 'react'
import { Sparkles, Activity } from 'lucide-react'

const fmtPercent = value => (
  value === null || value === undefined ? '--' : `${Math.round(value * 100)}%`
)

export default function SkillPatterns({ overview }) {
  const patterns = overview?.patterns || []
  const metrics = overview?.pattern_metrics || {}

  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-[var(--border)] bg-[var(--accent-2-soft)] text-[var(--accent-2)] shadow-[0_0_15px_rgba(191,0,255,0.12)]">
            <Sparkles size={16} />
          </div>
          <div>
            <p className="section-label">Skill Memory</p>
            <p className="mt-1 text-[11px] leading-5 text-[var(--text-soft)]">
              {metrics.total_patterns ?? 0} active skills
            </p>
          </div>
        </div>
        <div className="text-right">
          <span className="meta-pill mono text-[11px] text-[var(--success)]">
            avg {fmtPercent(metrics.avg_success_rate)}
          </span>
          <p className="mt-1 text-[10px] text-[var(--text-soft)]">
            retry {metrics.avg_retries !== undefined ? metrics.avg_retries.toFixed(1) : '--'} x/run
          </p>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {patterns.length === 0 && (
          <div className="panel-muted px-4 py-4 text-center text-sm text-[var(--text-soft)]">
            Successful workflows will accumulate here automatically and guide future plans.
          </div>
        )}

        {patterns.map(pattern => (
          <div key={pattern.signature} className="panel-muted px-3 py-3 hover:border-[var(--accent-soft)] transition-colors">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="mono text-[10px] text-[var(--accent)] uppercase tracking-widest">{pattern.primary_intent}</p>
                <p className="mt-1.5 truncate text-sm font-medium text-[var(--text-strong)]">
                  {(pattern.examples && pattern.examples[0]) || pattern.signature}
                </p>
              </div>
              <span className="meta-pill mono text-[10px] text-[var(--success)]">
                {fmtPercent(pattern.success_rate)}
              </span>
            </div>

            <div className="mt-3 flex flex-wrap gap-2 text-[10px] mono text-[var(--text-muted)]">
              <span className="flex items-center gap-1"><Activity size={10} /> {pattern.total_runs} runs</span>
              <span>retry {pattern.avg_retries.toFixed(1)}</span>
            </div>

            <div className="mt-3 rounded-[12px] border border-[var(--border)] bg-[rgba(255,255,255,0.02)] px-3 py-2.5">
              <p className="text-[10px] text-[var(--text-soft)] uppercase tracking-wider mb-1">Learned Sequence</p>
              <p className="text-[11px] font-mono text-[var(--accent-2)] break-words leading-relaxed">
                {(pattern.best_agent_sequence || []).join(' → ') || 'awaiting structural discovery'}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
