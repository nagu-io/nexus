import React, { useEffect, useState } from 'react'
import { ChevronDown, Clock, MessageSquare, Search, Trash2 } from 'lucide-react'
import { API_URL } from '../lib/runtime.js'

export default function ChatHistory() {
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showSessions, setShowSessions] = useState(false)

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
    <div className="panel-surface flex h-full flex-col overflow-hidden">
      <div className="border-b border-[var(--border)] px-4 py-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
              <MessageSquare size={16} />
            </div>
            <div>
              <p className="section-label">Conversation History</p>
              <p className="mt-1 text-sm text-[var(--text-soft)]">
                Revisit prior sessions, inspect routed answers, and search long-term memory.
              </p>
            </div>
          </div>

          <button
            onClick={() => setShowSessions(value => !value)}
            className="meta-pill mono text-[11px]"
          >
            {activeSession ? activeSession.substring(0, 8) : 'select'}
            <ChevronDown size={12} className={`transition-transform ${showSessions ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {showSessions && (
          <div className="panel-muted mt-4 max-h-48 space-y-2 overflow-y-auto p-2">
            {sessions.map(session => (
              <div
                key={session.id}
                className={`flex cursor-pointer items-center justify-between rounded-2xl px-3 py-3 transition-colors ${
                  activeSession === session.id
                    ? 'bg-[var(--accent-soft)]'
                    : 'hover:bg-[rgba(255,255,255,0.03)]'
                }`}
                onClick={() => {
                  setActiveSession(session.id)
                  setShowSessions(false)
                  setSearchResults(null)
                }}
              >
                <div className="min-w-0">
                  <p className="mono text-xs text-[var(--text)]">{session.id.substring(0, 12)}</p>
                  <p className="mt-1 text-xs text-[var(--text-soft)]">{session.message_count} messages</p>
                </div>
                <button
                  onClick={event => {
                    event.stopPropagation()
                    deleteSession(session.id)
                  }}
                  className="desktop-control"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && (
              <p className="px-2 py-3 text-center text-xs text-[var(--text-soft)]">No sessions yet</p>
            )}
          </div>
        )}

        <div className="panel-muted mt-4 flex items-center gap-2 px-3 py-3">
          <Search size={14} className="text-[var(--text-soft)]" />
          <input
            type="text"
            value={searchQuery}
            onChange={event => setSearchQuery(event.target.value)}
            onKeyDown={event => event.key === 'Enter' && handleSearch()}
            placeholder="Search stored conversations..."
            className="w-full bg-transparent text-sm text-[var(--text)] placeholder:text-[var(--text-soft)] focus:outline-none"
          />
          {searchResults && (
            <button
              onClick={() => {
                setSearchResults(null)
                setSearchQuery('')
              }}
              className="meta-pill mono text-[11px]"
            >
              clear
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {loading && (
          <div className="flex items-center justify-center py-10 text-sm text-[var(--text-soft)]">
            Loading session…
          </div>
        )}

        {!loading && displayMessages.length === 0 && (
          <div className="flex h-full items-center justify-center text-center text-[var(--text-soft)]">
            <div>
              <Clock size={28} className="mx-auto mb-3 opacity-40" />
              <p className="text-sm text-[var(--text)]">{searchResults ? 'No matching results' : 'No messages yet'}</p>
            </div>
          </div>
        )}

        <div className="space-y-3">
          {displayMessages.map((message, index) => (
            <div
              key={message.id || index}
              className={`rounded-3xl border px-4 py-4 ${
                message.role === 'user'
                  ? 'ml-10 chat-bubble-user'
                  : 'mr-10 chat-bubble-system'
              }`}
            >
              <div className="mb-2 flex items-center justify-between gap-3">
                <span className={`mono text-[11px] ${
                  message.role === 'user' ? 'text-[var(--accent)]' : 'text-[var(--accent-2)]'
                }`}>
                  {message.role === 'user' ? 'YOU' : 'NEXUS'}
                  {message.metadata?.agent ? ` via ${message.metadata.agent}` : ''}
                </span>
                <span className="text-xs text-[var(--text-soft)]">
                  {message.created_at ? formatTime(message.created_at) : ''}
                </span>
              </div>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-[var(--text)]">
                {message.content?.substring(0, 600)}
                {message.content?.length > 600 && <span className="text-[var(--text-soft)]">…</span>}
              </p>
            </div>
          ))}
        </div>
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
