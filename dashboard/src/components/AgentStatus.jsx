import React from 'react'

const AGENTS = [
  { name: 'coding', tone: 'bg-[var(--accent)] text-[#201a14]' },
  { name: 'research', tone: 'bg-[var(--accent-2)] text-[#12201c]' },
  { name: 'memory', tone: 'bg-[#d8c3ff] text-[#251538]' },
  { name: 'file', tone: 'bg-[#9ee6b8] text-[#112117]' },
  { name: 'canary', tone: 'bg-[#ffb28b] text-[#2a1610]' },
]

export default function AgentStatus({ active, stats }) {
  return (
    <div className="panel-surface p-4">
      <div className="flex items-center justify-between">
        <p className="section-label">Agents</p>
        <span className="meta-pill mono text-[11px]">
          {active ? `${active} active` : 'idle'}
        </span>
      </div>

      <div className="mt-4 space-y-2">
        {AGENTS.map(agent => {
          const isActive = active === agent.name
          return (
            <div
              key={agent.name}
              className={`panel-muted flex items-center justify-between px-3 py-3 transition-colors ${
                isActive ? 'border border-[var(--accent)] shadow-[0_0_10px_rgba(0,240,255,0.15)] bg-[rgba(0,240,255,0.05)]' : 'border border-transparent'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className={`h-2.5 w-2.5 rounded-full ${isActive ? agent.tone.split(' ')[0] : 'bg-[rgba(255,255,255,0.12)]'}`} />
                <span className={`mono text-sm ${isActive ? 'text-[var(--text-strong)]' : 'text-[var(--text-soft)]'}`}>
                  {agent.name}
                </span>
              </div>
              {isActive ? (
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${agent.tone}`}>
                  live
                </span>
              ) : (
                <span className="text-xs text-[var(--text-soft)]">ready</span>
              )}
            </div>
          )
        })}
      </div>

      {stats && (
        <div className="mt-4 grid grid-cols-3 gap-3">
          <MiniStat label="local" value={stats.local} />
          <MiniStat label="agent" value={stats.agent} />
          <MiniStat label="repo" value={stats.workspace} />
        </div>
      )}
    </div>
  )
}

function MiniStat({ label, value }) {
  return (
    <div className="panel-muted px-3 py-2">
      <p className="section-label text-[10px]">{label}</p>
      <p className="mt-2 mono text-sm text-[var(--text)]">{value}</p>
    </div>
  )
}
