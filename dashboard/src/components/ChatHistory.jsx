import React, { useEffect, useState } from 'react'
import { Clock, MessageSquare, Search, Trash2 } from 'lucide-react'
import { API_URL } from '../lib/runtime.js'

export default function ChatHistory() {
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchSessions()
  }, [])

  useEffect(() => {
    if (activeSession) fetchHistory(activeSession)
  }, [activeSession])

  const fetchSessions = async () => {
    try {
      const response = await fetch(`${API_URL}/chat/sessions`)
      const data = await response.json()
      setSessions(data.sessions || [])
      if (data.sessions?.length > 0 && !activeSession) {
        setActiveSession(data.sessions[0].id)
      }
    } catch {
      // API not available
    }
  }

  const fetchHistory = async sessionId => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/chat/history?session_id=${sessionId}&limit=100`)
      const data = await response.json()
      setMessages(data.messages || [])
    } catch {
      // API not available
    }
    setLoading(false)
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null)
      return
    }
    try {
      const response = await fetch(`${API_URL}/chat/search?q=${encodeURIComponent(searchQuery)}`)
      const data = await response.json()
      setSearchResults(data.results || [])
    } catch {
      // API not available
    }
  }

  const deleteSession = async sessionId => {
    try {
      await fetch(`${API_URL}/chat/sessions/${sessionId}`, { method: 'DELETE' })
      setSessions(prev => prev.filter(session => session.id !== sessionId))
      if (activeSession === sessionId) {
        setActiveSession(null)
        setMessages([])
      }
    } catch {
      // Ignore API failure
    }
  }

  const displayMessages = searchResults || messages

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Compact search bar */}
      <div className="flex items-center gap-2 rounded-2xl border border-[var(--border)] bg-[rgba(17,19,24,0.6)] px-3 py-2.5 mb-3">
        <Search size={13} className="shrink-0 text-[var(--text-muted)]" />
        <input
          type="text"
          value={searchQuery}
          onChange={event => setSearchQuery(event.target.value)}
          onKeyDown={event => event.key === 'Enter' && handleSearch()}
          placeholder="Search conversations..."
          className="w-full bg-transparent text-xs text-[var(--text)] placeholder:text-[var(--text-soft)] focus:outline-none"
        />
        {searchResults && (
          <button
            onClick={() => {
              setSearchResults(null)
              setSearchQuery('')
            }}
            className="shrink-0 text-[10px] mono text-[var(--accent)] hover:underline"
          >
            clear
          </button>
        )}
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto space-y-1.5">
        {sessions.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Clock size={24} className="mb-3 text-[var(--text-muted)] opacity-40" />
            <p className="text-xs text-[var(--text-soft)]">No conversations yet</p>
          </div>
        )}

        {sessions.map(session => {
          const isActive = activeSession === session.id
          return (
            <div
              key={session.id}
              onClick={() => {
                setActiveSession(session.id)
                setSearchResults(null)
              }}
              className={`group flex items-start justify-between gap-2 rounded-2xl px-3 py-3 cursor-pointer transition-all ${
                isActive
                  ? 'bg-[rgba(0,240,255,0.04)] border border-[rgba(0,240,255,0.12)]'
                  : 'border border-transparent hover:bg-[rgba(255,255,255,0.03)]'
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${isActive ? 'bg-[var(--accent)]' : 'bg-[rgba(255,255,255,0.15)]'}`} />
                  <span className="mono text-[11px] text-[var(--text)] truncate">
                    {session.id.substring(0, 8)}
                  </span>
                </div>
                <p className="mt-1 pl-3.5 text-[11px] text-[var(--text-soft)]">
                  {session.message_count} messages
                </p>
              </div>
              <button
                onClick={event => {
                  event.stopPropagation()
                  deleteSession(session.id)
                }}
                className="shrink-0 opacity-0 group-hover:opacity-60 transition-opacity p-1 hover:text-[var(--danger)]"
              >
                <Trash2 size={11} />
              </button>
            </div>
          )
        })}

        {/* Messages view when a session is active */}
        {activeSession && displayMessages.length > 0 && (
          <div className="mt-3 pt-3 border-t border-[var(--border)] space-y-2">
            {displayMessages.map((message, index) => (
              <div
                key={message.id || index}
                className="rounded-2xl border border-[var(--border)] bg-[rgba(255,255,255,0.02)] px-3 py-2.5"
              >
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span className={`mono text-[10px] font-semibold ${
                    message.role === 'user' ? 'text-[var(--accent)]' : 'text-[var(--accent-2)]'
                  }`}>
                    {message.role === 'user' ? 'YOU' : 'NEXUS'}
                    {message.metadata?.agent ? ` · ${message.metadata.agent}` : ''}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {message.created_at ? formatTime(message.created_at) : ''}
                  </span>
                </div>
                <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-[var(--text-soft)]">
                  {message.content?.substring(0, 300)}
                  {message.content?.length > 300 && <span className="text-[var(--text-muted)]">…</span>}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function formatTime(isoString) {
  try {
    const date = new Date(isoString)
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}
