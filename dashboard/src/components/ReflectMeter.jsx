export default function ReflectMeter({ reflect, stats }) {
  const pct = reflect?.score !== null && reflect?.score !== undefined ? Math.round(reflect.score * 100) : null
  const verdict = reflect?.verdict || 'awaiting'
  const action = reflect?.action || 'idle'
  const color =
    pct === null ? '#2a4a6f' : pct < 30 ? '#22c55e' : pct < 60 ? '#eab308' : '#ef4444'
  const label =
    pct === null ? 'Awaiting...' : verdict === 'clean' ? 'Serve' : verdict === 'warning' ? 'Warn' : 'Block'

  return (
    <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
      <p className="mono text-cyan-400 text-xs mb-3 uppercase tracking-widest">ReflectScore</p>
      <div className="flex items-center justify-between mb-2">
        <span className="text-2xl font-bold mono" style={{ color }}>
          {pct !== null ? `${pct}%` : '--'}
        </span>
        <span className="text-xs mono uppercase" style={{ color }}>
          {label}
        </span>
      </div>
      <div className="h-2 bg-[#1e3a5f] rounded-full overflow-hidden">
        <div
          className="h-2 rounded-full transition-all duration-700"
          style={{ width: `${pct ?? 0}%`, backgroundColor: color }}
        />
      </div>
      <p className="text-xs text-[#2a4a6f] mt-2">
        {pct === null
          ? 'Trust layer is waiting for the next response.'
          : `Action: ${action} | Verdict: ${verdict}`}
      </p>
      {reflect?.warning && (
        <p className="text-xs mt-3 leading-relaxed text-[#e6c46a]">
          {reflect.warning}
        </p>
      )}
      {stats && (
        <div className="grid grid-cols-2 gap-2 mt-4 text-xs mono text-[#4a7fa5]">
          <span>clean: {stats.clean}</span>
          <span>warn: {stats.warning}</span>
          <span>blocked: {stats.blocked}</span>
          <span>rerouted: {stats.rerouted}</span>
        </div>
      )}
    </div>
  )
}
