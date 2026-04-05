export default function ModelStatus({ status }) {
  const model = status?.model || 'phi3:mini'
  const ollamaOk = status?.ollama

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
      </div>
    </div>
  )
}
