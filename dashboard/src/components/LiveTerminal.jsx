import React, { useEffect, useRef, useState } from 'react'
import { Pause, Play, Terminal, Trash2, Wifi, WifiOff } from 'lucide-react'

export default function LiveTerminal({ events, connected, connectionCount, onClear }) {
  const [paused, setPaused] = useState(false)
  const [filter, setFilter] = useState('all')
  const eventsViewportRef = useRef(null)

  useEffect(() => {
    if (paused) return
    const viewport = eventsViewportRef.current
    if (!viewport) return

    const frame = window.requestAnimationFrame(() => {
      viewport.scrollTo({
        top: viewport.scrollHeight,
        behavior: events.length > 4 ? 'smooth' : 'auto',
      })
    })

    return () => window.cancelAnimationFrame(frame)
  }, [events, paused])

  const filteredEvents = filter === 'all'
    ? events
    : events.filter(event => {
        if (filter === 'edits') return isEditEvent(event)
        if (filter === 'output') return ['execution_output', 'tool_executed', 'agent_output'].includes(event.type)
        if (filter === 'error') return event.type === 'execution_output' && event.kind === 'error'
        if (filter === 'chat') return event.type === 'chat_message' || event.type === 'chat_response'
        return true
      })

  return (
    <div className="panel-surface flex h-full flex-col overflow-hidden relative">
      <div className="absolute inset-0 bg-gradient-to-tr from-transparent to-[rgba(0,240,255,0.02)] pointer-events-none" />
      <div className="flex items-center justify-between gap-3 border-b border-[var(--border)] px-4 py-3 relative z-10">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-[rgba(147,199,187,0.14)] text-[var(--accent-2)]">
            <Terminal size={16} />
          </div>
          <div>
            <p className="section-label">Live Activity</p>
            <p className="mt-1 text-sm text-[var(--text-soft)]">
              Runtime events, execution output, and tool traces.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="meta-pill mono text-[11px]">
            {connected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {connected ? `${connectionCount} live` : 'reconnecting'}
          </div>
          {['all', 'edits', 'output', 'error', 'chat'].map(item => (
            <button
              key={item}
              onClick={() => setFilter(item)}
              className={`rounded-full px-3 py-1 text-[11px] mono transition-all ${
                filter === item
                  ? 'bg-gradient-to-r from-[rgba(0,240,255,0.15)] to-[rgba(191,0,255,0.15)] text-[var(--text-strong)] border border-[rgba(255,255,255,0.1)] shadow-[0_0_10px_rgba(0,240,255,0.1)]'
                  : 'text-[var(--text-soft)] hover:bg-[rgba(255,255,255,0.05)] hover:text-[var(--text)] border border-transparent'
              }`}
            >
              {item}
            </button>
          ))}
          <button className="desktop-control" onClick={() => setPaused(value => !value)} title={paused ? 'Resume' : 'Pause'}>
            {paused ? <Play size={12} /> : <Pause size={12} />}
          </button>
          <button className="desktop-control" onClick={onClear} title="Clear">
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      <div ref={eventsViewportRef} className="flex-1 overflow-y-auto px-3 py-3 font-mono text-xs">
        {filteredEvents.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-[var(--text-soft)]">
            <div>
              <Terminal size={32} className="mx-auto mb-3 opacity-40" />
              <p className="text-sm text-[var(--text)]">Waiting for runtime activity</p>
              <p className="mt-1 text-xs">Once NEXUS runs tools or emits workflow events, they land here.</p>
            </div>
          </div>
        ) : (
          filteredEvents.map((event, index) => (
            <EventLine key={`${event._ts || index}-${index}`} event={event} />
          ))
        )}
      </div>

      <div className="border-t border-[var(--border)] bg-black/10 px-4 py-2 relative z-10">
        <div className="flex items-center justify-between text-[11px] mono text-[var(--text-soft)]">
          <span>{filteredEvents.length} events</span>
          <span>{paused ? 'paused' : 'streaming'}</span>
        </div>
      </div>
    </div>
  )
}

function EventLine({ event }) {
  const editEvent = isEditEvent(event)
  const time = event._ts
    ? new Date(event._ts).toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : '--:--:--'

  const typeStyles = {
    chat_message: { badge: 'bg-[rgba(147,199,187,0.16)] text-[var(--accent-2)]', label: 'USER' },
    chat_response: { badge: 'bg-[var(--accent-soft)] text-[var(--accent)]', label: 'NEXUS' },
    agent_started: { badge: 'bg-[rgba(197,193,255,0.18)] text-[#d8c3ff]', label: 'AGENT' },
    agent_output: { badge: 'bg-[rgba(147,199,187,0.14)] text-[var(--accent-2)]', label: 'OUT' },
    tool_executed: { badge: 'bg-[rgba(240,202,114,0.18)] text-[var(--warning)]', label: 'TOOL' },
    critic_scored: { badge: 'bg-[rgba(255,178,139,0.18)] text-[#ffb28b]', label: 'CRITIC' },
    fix_applied: { badge: 'bg-[rgba(137,213,167,0.18)] text-[var(--success)]', label: 'FIX' },
    file_saved: { badge: 'bg-[var(--accent-soft)] text-[var(--accent)]', label: 'FILE' },
    context_reduced: { badge: 'bg-[rgba(147,199,187,0.18)] text-[var(--accent-2)]', label: 'CTX' },
    execution_output: { badge: 'bg-[rgba(255,255,255,0.08)] text-[var(--text)]', label: 'EXEC' },
    workflow_complete: { badge: 'bg-[rgba(137,213,167,0.18)] text-[var(--success)]', label: 'DONE' },
  }

  const style = editEvent ? {
    badge: 'bg-[var(--accent-soft)] text-[var(--accent)]',
    label: 'EDIT',
  } : typeStyles[event.type] || {
    badge: 'bg-[rgba(255,255,255,0.08)] text-[var(--text-muted)]',
    label: event.type?.toUpperCase() || 'EVENT',
  }

  const isError = event.kind === 'error' || (event.type === 'execution_output' && event.kind === 'error')

  return (
    <div className={`mb-2 rounded-2xl border px-3 py-2 transition-all hover:translate-x-1 ${
      isError
        ? 'border-[rgba(255,143,136,0.3)] bg-[rgba(255,143,136,0.08)] shadow-[0_0_15px_rgba(255,143,136,0.1)]'
        : 'border-[var(--border)] panel-glass hover:border-[var(--border-strong)]'
    }`}>
      <div className="mb-2 flex items-center gap-2 text-[10px] text-[var(--text-soft)]">
        <span>{time}</span>
        <span className={`rounded-full px-2 py-1 ${style.badge}`}>{style.label}</span>
      </div>
      <p className={`leading-relaxed ${isError ? 'text-[var(--danger)]' : 'text-[var(--text)]'}`}>
        {messageForEvent(event)}
      </p>
      {editEvent && event.edit_preview?.diff && (
        <div className="mt-3 rounded-xl border border-[var(--border)] bg-black/30 p-3 shadow-inner">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[10px] text-[var(--text-soft)]">
            <span className="meta-pill mono text-[10px]">{event.edit_preview.kind || 'edit'}</span>
            <span className="meta-pill mono text-[10px]">{shortPath(editPathForEvent(event))}</span>
            <span className="meta-pill mono text-[10px]">
              {event.edit_preview.changed_lines ?? 0} line{event.edit_preview.changed_lines === 1 ? '' : 's'} changed
            </span>
            {event.edit_preview.truncated && <span className="meta-pill mono text-[10px]">preview truncated</span>}
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-[var(--text)]">
            {event.edit_preview.diff}
          </pre>
        </div>
      )}
    </div>
  )
}

function messageForEvent(event) {
  switch (event.type) {
    case 'chat_message':
      return event.content?.substring(0, 160) || ''
    case 'chat_response':
      return `[${event.agent || 'unknown'}] ${(event.content || '').substring(0, 160)}`
    case 'agent_started':
      return `Status: ${event.status || 'processing'}`
    case 'agent_output':
      return `[${event.agent || 'agent'}] ${event.summary || ''}`
    case 'tool_executed':
      if (isEditEvent(event)) {
        return `${editActionLabel(event)} ${shortPath(editPathForEvent(event))}`
      }
      return `${event.tool || 'tool'}: ${event.summary || 'executed'}`
    case 'critic_scored':
      return event.summary || 'critic evaluation complete'
    case 'fix_applied':
      return event.summary || `Updated ${shortPath(editPathForEvent(event))}`
    case 'file_saved':
      return event.summary || `Saved ${shortPath(editPathForEvent(event))}`
    case 'context_reduced':
      return `chat context reduced ${event.original_length || 0} -> ${event.reduced_length || 0} chars via ${event.backend || 'reducer'}`
    case 'execution_output':
      return event.data || ''
    case 'workflow_complete':
      return event.summary || 'Workflow finished'
    default:
      return JSON.stringify(event).substring(0, 180)
  }
}

function isEditEvent(event) {
  if (event?.edit_preview?.diff) return true
  if (event?.type === 'file_saved') return true
  if (event?.type === 'fix_applied') return true
  return event?.type === 'tool_executed' && ['edit_file', 'write_file'].includes(event?.action)
}

function editPathForEvent(event) {
  return event?.edit_preview?.path || event?.path || event?.file || 'file'
}

function editActionLabel(event) {
  const kind = event?.edit_preview?.kind
  if (kind === 'create') return 'Created'
  if (kind === 'write') return 'Wrote'
  return 'Edited'
}

function shortPath(value, maxLength = 64) {
  const text = String(value || '')
  if (text.length <= maxLength) return text
  const normalized = text.replaceAll('\\', '/')
  const parts = normalized.split('/')
  if (parts.length <= 2) return text.slice(0, maxLength - 1) + '…'
  return `${parts.slice(0, 2).join('/')}/…/${parts[parts.length - 1]}`
}
