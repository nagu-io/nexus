import React from 'react'

export default function ModelStatus({ status }) {
  const model = status?.model || 'phi3:mini'
  const ollamaOk = status?.ollama
  const contextReduction = status?.context_reduction || null

  return (
    <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
      <p className="mono text-cyan-400 text-xs mb-3 uppercase tracking-widest">Model</p>
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#4a7fa5]">Active</span>
          <span className="mono text-xs text-white">{model}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#4a7fa5]">Engine</span>
          <span className="mono text-xs text-cyan-400">CompressX</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#4a7fa5]">Ollama</span>
          <div className="flex items-center gap-1">
            <div className={`w-1.5 h-1.5 rounded-full ${ollamaOk ? 'bg-green-400' : 'bg-red-500'}`} />
            <span className={`mono text-xs ${ollamaOk ? 'text-green-400' : 'text-red-400'}`}>
              {ollamaOk ? 'running' : 'offline'}
            </span>
          </div>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#4a7fa5]">Compression</span>
          <span className="mono text-xs text-yellow-400">3.6x</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#4a7fa5]">Context</span>
          <span
            className={`mono text-xs ${
              contextReduction?.enabled ? 'text-cyan-300' : 'text-[#4a7fa5]'
            }`}
          >
            {contextReduction?.enabled ? contextReduction.backend : 'disabled'}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-[#4a7fa5]">Prompt Budget</span>
          <span className="mono text-xs text-white">
            {contextReduction?.enabled
              ? `${formatCompactLength(contextReduction.target_chars)} / ${formatCompactLength(contextReduction.threshold_chars)}`
              : '--'}
          </span>
        </div>
      </div>
    </div>
  )
}

function formatCompactLength(value) {
  if (value === null || value === undefined) return '--'
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`
  return String(value)
}
