import React from 'react'

const fmtPercent = value => (
  value === null || value === undefined ? '--' : `${Math.round(value * 100)}%`
)

export default function RunHistory({ runs = [] }) {
  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between">
        <p className="section-label">Recent Runs</p>
        <span className="meta-pill mono text-[11px]">{runs.length} recent</span>
      </div>

      <div className="mt-3 space-y-2">
        {runs.length === 0 && (
          <div className="panel-muted px-4 py-4 text-sm text-[var(--text-soft)]">
            Recent workflows will appear here.
          </div>
        )}

        {runs.slice(0, 5).map(run => (
          <div key={run.workflow_id} className="panel-muted px-3 py-3">
            <div className="flex items-start justify-between gap-3">
              <p className="line-clamp-2 text-sm leading-relaxed text-[var(--text)]">
                {run.goal || run.workflow_id}
              </p>
              <span className={`mono text-[11px] ${
                run.status === 'completed' ? 'text-[var(--success)]' : 'text-[var(--danger)]'
              }`}>
                {run.status}
              </span>
            </div>

            <div className="mt-2 flex flex-wrap gap-2 text-[11px] mono text-[var(--text-soft)]">
              <span>{run.execution_mode}</span>
              <span>conf {fmtPercent(run.final_confidence)}</span>
              <span>retry {run.retry_count ?? 0}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
