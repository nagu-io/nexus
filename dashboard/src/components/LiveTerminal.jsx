import React, { useEffect, useRef, useState } from 'react'
import { Terminal, Trash2, Pause, Play, Circle, Wifi, WifiOff } from 'lucide-react'

/**
 * LiveTerminal — real-time execution output viewer.
 * Consumes WebSocket events and renders them as a terminal stream.
 */
export default function LiveTerminal({ events, connected, connectionCount, onClear }) {
  const [paused, setPaused] = useState(false)
  const [filter, setFilter] = useState('all') // all, output, error, chat
  const bottomRef = useRef(null)

  useEffect(() => {
    if (!paused) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [events, paused])

  const filteredEvents = filter === 'all'
    ? events
    : events.filter(e => {
        if (filter === 'output') return ['execution_output', 'tool_executed', 'agent_output'].includes(e.type)
        if (filter === 'error') return e.type === 'execution_output' && e.kind === 'error'
        if (filter === 'chat') return e.type === 'chat_message' || e.type === 'chat_response'
        return true
      })

  return (
    <div className="flex flex-col h-full bg-[#0a1628] border border-[#1e3a5f] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#1e3a5f] bg-[#020b18]">
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-cyan-400" />
          <span className="mono text-cyan-400 text-xs font-bold uppercase tracking-widest">Live Terminal</span>
          <div className="flex items-center gap-1 ml-2">
            {connected
              ? <Wifi size={12} className="text-green-400" />
              : <WifiOff size={12} className="text-red-400" />
            }
            <span className="text-[10px] text-[#4a7fa5] mono">
              {connected ? `${connectionCount} conn` : 'disconnected'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Filter tabs */}
          {['all', 'output', 'error', 'chat'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`text-[10px] mono px-2 py-0.5 rounded transition-colors ${
                filter === f
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-[#4a7fa5] hover:text-cyan-300'
              }`}
            >
              {f.toUpperCase()}
            </button>
          ))}

          {/* Controls */}
          <button
            onClick={() => setPaused(p => !p)}
            className="text-[#4a7fa5] hover:text-cyan-400 transition-colors p-1"
            title={paused ? 'Resume' : 'Pause'}
          >
            {paused ? <Play size={12} /> : <Pause size={12} />}
          </button>
          <button
            onClick={onClear}
            className="text-[#4a7fa5] hover:text-red-400 transition-colors p-1"
            title="Clear"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Terminal output */}
      <div className="flex-1 overflow-y-auto p-3 space-y-0.5 font-mono text-xs">
        {filteredEvents.length === 0 && (
          <div className="flex items-center justify-center h-full text-[#4a7fa5] text-sm">
            <div className="text-center">
              <Terminal size={32} className="mx-auto mb-2 opacity-30" />
              <p>Waiting for events...</p>
              <p className="text-[10px] mt-1">Events will stream here in real-time</p>
            </div>
          </div>
        )}

        {filteredEvents.map((event, i) => (
          <EventLine key={i} event={event} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Status bar */}
      <div className="flex items-center justify-between px-3 py-1 border-t border-[#1e3a5f] bg-[#020b18]">
        <span className="text-[10px] text-[#4a7fa5] mono">
          {filteredEvents.length} events
          {paused && <span className="text-yellow-400 ml-2">⏸ PAUSED</span>}
        </span>
        <div className="flex items-center gap-1">
          <Circle size={6} className={connected ? 'text-green-400 fill-green-400' : 'text-red-400 fill-red-400'} />
          <span className="text-[10px] text-[#4a7fa5] mono">
            {connected ? 'LIVE' : 'RECONNECTING...'}
          </span>
        </div>
      </div>
    </div>
  )
}


function EventLine({ event }) {
  const time = new Date(event._ts).toLocaleTimeString('en-US', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
  })

  const typeStyles = {
    chat_message: { badge: 'bg-blue-500/20 text-blue-400', label: 'USER' },
    chat_response: { badge: 'bg-green-500/20 text-green-400', label: 'NEXUS' },
    agent_started: { badge: 'bg-purple-500/20 text-purple-400', label: 'AGENT' },
    agent_output: { badge: 'bg-indigo-500/20 text-indigo-300', label: 'OUT' },
    tool_executed: { badge: 'bg-yellow-500/20 text-yellow-400', label: 'TOOL' },
    critic_scored: { badge: 'bg-orange-500/20 text-orange-300', label: 'CRITIC' },
    fix_applied: { badge: 'bg-emerald-500/20 text-emerald-300', label: 'FIX' },
    context_reduced: { badge: 'bg-sky-500/20 text-sky-300', label: 'CTX' },
    execution_output: { badge: 'bg-cyan-500/20 text-cyan-400', label: 'EXEC' },
    workflow_complete: { badge: 'bg-green-500/20 text-green-400', label: 'DONE' },
  }

  const style = typeStyles[event.type] || { badge: 'bg-gray-500/20 text-gray-400', label: event.type?.toUpperCase() || 'EVENT' }

  const getMessage = () => {
    switch (event.type) {
      case 'chat_message':
        return event.content?.substring(0, 120) || ''
      case 'chat_response':
        return `[${event.agent || 'unknown'}] ${(event.content || '').substring(0, 120)}`
      case 'agent_started':
        return `Status: ${event.status || 'processing'}`
      case 'agent_output':
        return `[${event.agent || 'agent'}] ${event.summary || ''}`
      case 'tool_executed':
        return `${event.tool || 'tool'}: ${event.summary || 'executed'}`
      case 'critic_scored':
        return `${event.summary || 'critic evaluation complete'}`
      case 'fix_applied':
        return event.summary || `Updated ${event.file || 'workspace'}`
      case 'context_reduced':
        return `chat context reduced ${event.original_length || 0} -> ${event.reduced_length || 0} chars via ${event.backend || 'reducer'}`
      case 'execution_output':
        return event.data || ''
      case 'workflow_complete':
        return event.summary || 'Workflow finished'
      default:
        return JSON.stringify(event).substring(0, 100)
    }
  }

  const isError = event.kind === 'error' || event.type === 'execution_output' && event.kind === 'error'

  return (
    <div className={`flex items-start gap-2 py-0.5 group hover:bg-[#0d1d35] rounded px-1 transition-colors ${
      isError ? 'bg-red-900/10' : ''
    }`}>
      <span className="text-[10px] text-[#4a7fa5] shrink-0 mt-0.5">{time}</span>
      <span className={`text-[10px] px-1.5 py-0 rounded shrink-0 ${style.badge}`}>
        {style.label}
      </span>
      <span className={`text-xs break-all ${isError ? 'text-red-400' : 'text-[#e2f0ff]'}`}>
        {getMessage()}
      </span>
    </div>
  )
}
