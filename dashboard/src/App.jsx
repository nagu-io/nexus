import React, { useState, useEffect } from 'react'
import { MessageSquare, Terminal, Clock } from 'lucide-react'
import Chat from './components/Chat.jsx'
import AgentStatus from './components/AgentStatus.jsx'
import ReflectMeter from './components/ReflectMeter.jsx'
import ModelStatus from './components/ModelStatus.jsx'
import RuntimeOverview from './components/RuntimeOverview.jsx'
import RunHistory from './components/RunHistory.jsx'
import SkillPatterns from './components/SkillPatterns.jsx'
import LiveTerminal from './components/LiveTerminal.jsx'
import ChatHistory from './components/ChatHistory.jsx'
import useWebSocket from './hooks/useWebSocket.js'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = (API_URL.replace('http', 'ws')) + '/ws'

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null)
  const [reflectState, setReflectState] = useState(null)
  const [activeAgent, setActiveAgent] = useState(null)
  const [routeStats, setRouteStats] = useState({ local: 0, cloud: 0, agent: 0 })
  const [runtimeOverview, setRuntimeOverview] = useState(null)
  const [activeTab, setActiveTab] = useState('chat')

  const { events, connected, connectionCount, clearEvents } = useWebSocket(WS_URL)

  useEffect(() => {
    refreshDashboard()
    const interval = setInterval(refreshDashboard, 10000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const latest = events[events.length - 1]
    if (!latest) return
    if (latest.type === 'agent_started' && latest.agent) {
      setActiveAgent(latest.agent)
    }
  }, [events])

  const refreshDashboard = async () => {
    const [statusResult, overviewResult] = await Promise.allSettled([
      fetch(`${API_URL}/status`),
      fetch(`${API_URL}/runtime/overview?limit=6`),
    ])

    if (statusResult.status === 'fulfilled') {
      try {
        const data = await statusResult.value.json()
        setSystemStatus(data)
        setRouteStats(data.route_stats || { local: 0, cloud: 0, agent: 0 })
      } catch {
        // Ignore invalid status payloads
      }
    }

    if (overviewResult.status === 'fulfilled') {
      try {
        const data = await overviewResult.value.json()
        setRuntimeOverview(data)
      } catch {
        // Ignore invalid runtime payloads
      }
    }
  }

  const tabs = [
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'terminal', label: 'Terminal', icon: Terminal, badge: events.length || null },
    { id: 'history', label: 'History', icon: Clock },
  ]

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
        <div className="flex items-center gap-4">
          {/* WebSocket indicator */}
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-cyan-400 animate-pulse' : 'bg-[#4a7fa5]'}`} />
            <span className="text-[10px] text-[#4a7fa5] mono">
              {connected ? `WS:${connectionCount}` : 'WS:OFF'}
            </span>
          </div>
          {/* System status */}
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${systemStatus ? 'bg-green-400' : 'bg-red-500'}`} />
            <span className="text-xs text-[#4a7fa5] mono">{systemStatus ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden flex-col lg:flex-row">
        {/* Sidebar */}
        <aside className="w-full lg:w-72 border-b lg:border-b-0 lg:border-r border-[#1e3a5f] flex flex-col gap-4 p-4 overflow-y-auto">
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

          {/* Stack */}
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

        {/* Main Surface */}
        <main className="flex-1 min-h-0 flex flex-col xl:flex-row">
          <section className="flex-1 min-h-0 flex flex-col border-r-0 xl:border-r border-[#1e3a5f]">
            {/* Runtime Overview */}
            <div className="border-b border-[#1e3a5f] p-4">
              <RuntimeOverview overview={runtimeOverview} />
            </div>

            {/* Tab Navigation */}
            <div className="flex items-center border-b border-[#1e3a5f] bg-[#020b18]">
              {tabs.map(({ id, label, icon: Icon, badge }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`flex items-center gap-1.5 px-4 py-2.5 text-xs mono transition-all border-b-2 ${
                    activeTab === id
                      ? 'border-cyan-400 text-cyan-400 bg-[#0a1628]'
                      : 'border-transparent text-[#4a7fa5] hover:text-cyan-300 hover:bg-[#0a1628]/50'
                  }`}
                >
                  <Icon size={13} />
                  {label}
                  {badge && (
                    <span className="bg-cyan-500/20 text-cyan-400 text-[10px] px-1.5 rounded-full">
                      {badge > 99 ? '99+' : badge}
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 min-h-0">
              {activeTab === 'chat' && (
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
              )}
              {activeTab === 'terminal' && (
                <div className="h-full p-3">
                  <LiveTerminal
                    events={events}
                    connected={connected}
                    connectionCount={connectionCount}
                    onClear={clearEvents}
                  />
                </div>
              )}
              {activeTab === 'history' && (
                <div className="h-full p-3">
                  <ChatHistory />
                </div>
              )}
            </div>
          </section>

          <aside className="w-full xl:w-80 border-t xl:border-t-0 xl:border-l border-[#1e3a5f] p-4 flex flex-col gap-4 overflow-y-auto">
            <RunHistory runs={runtimeOverview?.runs} />
            <SkillPatterns overview={runtimeOverview} />
          </aside>
        </main>
      </div>
    </div>
  )
}
