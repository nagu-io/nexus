import React from 'react'

export default function ModelStatus({ status }) {
  const model = status?.model || 'phi3:mini'
  const ollamaOk = status?.ollama
  const localBackend = status?.local_backend || 'unknown'
  const cloudProvider = status?.cloud_provider || 'none'
  const contextReduction = status?.context_reduction || null

  const rows = [
    { label: 'Model', value: model, tone: 'text-[var(--text-strong)]' },
    { label: 'Backend', value: localBackend, tone: 'text-[var(--accent-2)]' },
    { label: 'Ollama', value: ollamaOk ? 'connected' : 'offline', tone: ollamaOk ? 'text-[var(--success)]' : 'text-[var(--danger)]' },
    { label: 'Cloud', value: cloudProvider, tone: cloudProvider === 'none' ? 'text-[var(--text-soft)]' : 'text-[var(--warning)]' },
    {
      label: 'Reducer',
      value: contextReduction?.enabled ? contextReduction.backend : 'disabled',
      tone: contextReduction?.enabled ? 'text-[var(--accent-2)]' : 'text-[var(--text-soft)]',
    },
    {
      label: 'Budget',
      value: contextReduction?.enabled
        ? `${formatCompactLength(contextReduction.target_chars)} / ${formatCompactLength(contextReduction.threshold_chars)}`
        : '--',
      tone: 'text-[var(--text)]',
    },
  ]

  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between">
        <p className="section-label">Runtime</p>
        <span className="meta-pill mono text-[11px]">{localBackend}</span>
      </div>

      <div className="mt-3 space-y-1.5">
        {rows.map(row => (
          <div key={row.label} className="panel-muted flex items-center justify-between px-3 py-2">
            <span className="text-xs text-[var(--text-soft)]">{row.label}</span>
            <span className={`mono text-xs font-medium ${row.tone}`}>{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function formatCompactLength(value) {
  if (value === null || value === undefined) return '--'
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`
  return String(value)
}
