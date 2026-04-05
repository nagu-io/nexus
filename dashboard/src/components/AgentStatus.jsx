const AGENTS = [
  { name: 'coding', color: 'text-cyan-400', dot: 'bg-cyan-400' },
  { name: 'research', color: 'text-blue-400', dot: 'bg-blue-400' },
  { name: 'memory', color: 'text-purple-400', dot: 'bg-purple-400' },
  { name: 'file', color: 'text-green-400', dot: 'bg-green-400' },
  { name: 'canary', color: 'text-orange-400', dot: 'bg-orange-400' },
]

export default function AgentStatus({ active, stats }) {
  return (
    <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
      <p className="mono text-cyan-400 text-xs mb-3 uppercase tracking-widest">Agents</p>
      <div className="space-y-2">
        {AGENTS.map(({ name, color, dot }) => (
          <div key={name} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${active === name ? dot : 'bg-[#1e3a5f]'} transition-colors`} />
              <span className={`mono text-xs ${active === name ? color : 'text-[#4a7fa5]'}`}>{name}</span>
            </div>
            {active === name && (
              <span className={`text-xs ${color} mono`}>active</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
