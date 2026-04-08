import React from 'react'

const fmtPercent = value => (
  value === null || value === undefined ? '--' : `${Math.round(value * 100)}%`
)

export default function RunHistory({ runs = [] }) {
  return (
    <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <p className="mono text-cyan-400 text-xs uppercase tracking-widest">Run History</p>
        <span className="mono text-[11px] text-[#4a7fa5]">{runs.length} recent</span>
      </div>

      <div className="space-y-3">
        {runs.length === 0 && (
          <p className="text-sm text-[#4a7fa5] leading-relaxed">
            Runtime traces will appear here after compiler or orchestrator runs complete.
          </p>
        )}

        {runs.map(run => (
          <div key={run.workflow_id} className="rounded-lg border border-[#18324f] bg-[#0b1a2f] p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm text-[#d7e7f7] leading-snug">
                  {run.goal || run.workflow_id}
                </p>
                <p className="mono text-[11px] text-[#4a7fa5] mt-1">
                  {run.execution_mode} • {run.agents?.join(', ') || 'no agents'}
                </p>
              </div>
              <span className={`mono text-[11px] uppercase ${
                run.status === 'completed' ? 'text-green-400' : 'text-red-400'
              }`}>
                {run.status}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 mt-3 text-[11px] mono text-[#7ea6c7]">
              <span>conf {fmtPercent(run.final_confidence)}</span>
              <span>retry {run.retry_count ?? 0}</span>
              <span>{run.parallel_batches ? 'parallel' : 'serial'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
