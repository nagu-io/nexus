import { useState, useEffect } from 'react'
import Chat from './components/Chat.jsx'
import AgentStatus from './components/AgentStatus.jsx'
import ReflectMeter from './components/ReflectMeter.jsx'
import ModelStatus from './components/ModelStatus.jsx'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null)
  const [reflectState, setReflectState] = useState(null)
  const [activeAgent, setActiveAgent] = useState(null)
  const [routeStats, setRouteStats] = useState({ local: 0, cloud: 0, agent: 0 })

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const r = await fetch(`${API_URL}/status`)
      const data = await r.json()
      setSystemStatus(data)
      setRouteStats(data.route_stats || { local: 0, cloud: 0, agent: 0 })
    } catch {
      // API not running yet
    }
  }

  return (
    <div className="min-h-screen bg-[#020b18] flex flex-col">
      {/* Header */}
      <header className="border-b border-[#1e3a5f] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-cyan-500 glow flex items-center justify-center">
            <span className="mono text-black text-xs font-bold">NX</span>
          </div>
          <span className="mono text-cyan-400 text-lg font-bold tracking-wider">NEXUS</span>
          <span className="text-[#4a7fa5] text-xs font-light">Private AI Developer OS</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${systemStatus ? 'bg-green-400' : 'bg-red-500'}`} />
          <span className="text-xs text-[#4a7fa5] mono">{systemStatus ? 'ONLINE' : 'OFFLINE'}</span>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 border-r border-[#1e3a5f] flex flex-col gap-4 p-4 overflow-y-auto">
          <ModelStatus status={systemStatus} />
          <ReflectMeter reflect={reflectState} stats={systemStatus?.reflect_stats} />
          <AgentStatus active={activeAgent} stats={routeStats} />

          {/* Route Stats */}
          <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
            <p className="mono text-cyan-400 text-xs mb-3 uppercase tracking-widest">AEON Router</p>
            <div className="space-y-2">
              {[
                { label: 'Local (CompressX)', value: routeStats.local, color: 'bg-green-500' },
                { label: 'Cloud (Anthropic)', value: routeStats.cloud, color: 'bg-yellow-500' },
                { label: 'Agents', value: routeStats.agent, color: 'bg-purple-500' },
              ].map(({ label, value, color }) => {
                const total = routeStats.local + routeStats.cloud + routeStats.agent || 1
                const pct = Math.round((value / total) * 100)
                return (
                  <div key={label}>
                    <div className="flex justify-between text-xs text-[#4a7fa5] mb-1">
                      <span>{label}</span>
                      <span className="mono">{value} ({pct}%)</span>
                    </div>
                    <div className="h-1 bg-[#1e3a5f] rounded">
                      <div className={`h-1 ${color} rounded transition-all duration-500`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Projects */}
          <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg p-4">
            <p className="mono text-cyan-400 text-xs mb-3 uppercase tracking-widest">Stack</p>
            <div className="space-y-1">
              {[
                { name: 'CompressX', desc: '3.6x compression', color: 'text-cyan-400' },
                { name: 'AEON', desc: 'Mind Router', color: 'text-purple-400' },
                { name: 'ReflectScore', desc: 'Hallucination guard', color: 'text-yellow-400' },
                { name: 'CanaryRAG', desc: 'RAG leak detection', color: 'text-red-400' },
                { name: 'CanaryVaults', desc: 'Canary seeding', color: 'text-orange-400' },
              ].map(({ name, desc, color }) => (
                <div key={name} className="flex items-center justify-between py-1">
                  <span className={`mono text-xs font-bold ${color}`}>{name}</span>
                  <span className="text-xs text-[#4a7fa5]">{desc}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Chat Main */}
        <main className="flex-1 flex flex-col">
          <Chat
            apiUrl={API_URL}
            onReflectState={setReflectState}
            onAgentChange={setActiveAgent}
            onRouteUpdate={(stats) => setRouteStats(prev => ({
              local: prev.local + (stats.initialRoute === 'local' ? 1 : 0),
              cloud: prev.cloud + (stats.initialRoute === 'cloud' ? 1 : 0),
              agent: prev.agent + (stats.initialRoute === 'agent' ? 1 : 0),
            }))}
          />
        </main>
      </div>
    </div>
  )
}
