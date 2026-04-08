import React, { useEffect, useState } from 'react'
import { MessageSquare, Search, Clock, Trash2, ChevronDown } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * ChatHistory — persistent conversation history with session picker and search.
 * Pulls data from /chat/history, /chat/sessions, /chat/search endpoints.
 */
export default function ChatHistory() {
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showSessions, setShowSessions] = useState(false)

  // Load sessions on mount
  useEffect(() => {
    fetchSessions()
  }, [])

  // Load messages when session changes
  useEffect(() => {
    if (activeSession) fetchHistory(activeSession)
  }, [activeSession])

  const fetchSessions = async () => {
    try {
      const r = await fetch(`${API_URL}/chat/sessions`)
      const data = await r.json()
      setSessions(data.sessions || [])
      if (data.sessions?.length > 0 && !activeSession) {
        setActiveSession(data.sessions[0].id)
      }
    } catch { /* API not available */ }
  }

  const fetchHistory = async (sessionId) => {
    setLoading(true)
    try {
      const r = await fetch(`${API_URL}/chat/history?session_id=${sessionId}&limit=100`)
      const data = await r.json()
      setMessages(data.messages || [])
    } catch { /* API not available */ }
    setLoading(false)
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null)
      return
    }
    try {
      const r = await fetch(`${API_URL}/chat/search?q=${encodeURIComponent(searchQuery)}`)
      const data = await r.json()
      setSearchResults(data.results || [])
    } catch { /* API not available */ }
  }

  const deleteSession = async (sessionId) => {
    try {
      await fetch(`${API_URL}/chat/sessions/${sessionId}`, { method: 'DELETE' })
      setSessions(prev => prev.filter(s => s.id !== sessionId))
      if (activeSession === sessionId) {
        setActiveSession(null)
        setMessages([])
      }
    } catch { /* Ignore */ }
  }

  const displayMessages = searchResults || messages

  return (
    <div className="flex flex-col h-full bg-[#0a1628] border border-[#1e3a5f] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-[#1e3a5f] bg-[#020b18]">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <MessageSquare size={14} className="text-cyan-400" />
            <span className="mono text-cyan-400 text-xs font-bold uppercase tracking-widest">History</span>
            <span className="text-[10px] text-[#4a7fa5] mono">{sessions.length} sessions</span>
          </div>

          {/* Session picker */}
          <button
            onClick={() => setShowSessions(p => !p)}
            className="flex items-center gap-1 text-[10px] mono text-[#4a7fa5] hover:text-cyan-400 transition-colors"
          >
            {activeSession ? activeSession.substring(0, 8) : 'Select'}
            <ChevronDown size={10} className={`transition-transform ${showSessions ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {/* Session dropdown */}
        {showSessions && (
          <div className="bg-[#0d1d35] border border-[#1e3a5f] rounded p-2 mb-2 max-h-32 overflow-y-auto space-y-1">
            {sessions.map(s => (
              <div
                key={s.id}
                className={`flex items-center justify-between p-1.5 rounded cursor-pointer transition-colors ${
                  activeSession === s.id ? 'bg-cyan-500/10 border border-cyan-500/20' : 'hover:bg-[#0a1628]'
                }`}
                onClick={() => { setActiveSession(s.id); setShowSessions(false); setSearchResults(null) }}
              >
                <div>
                  <span className="text-[10px] mono text-cyan-300">{s.id.substring(0, 12)}</span>
                  <span className="text-[10px] text-[#4a7fa5] ml-2">{s.message_count} msgs</span>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteSession(s.id) }}
                  className="text-[#4a7fa5] hover:text-red-400 transition-colors"
                >
                  <Trash2 size={10} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && (
              <p className="text-[10px] text-[#4a7fa5] text-center py-2">No sessions yet</p>
            )}
          </div>
        )}

        {/* Search */}
        <div className="flex gap-1">
          <div className="flex-1 relative">
            <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-[#4a7fa5]" />
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              placeholder="Search conversations..."
              className="w-full bg-[#0d1d35] border border-[#1e3a5f] rounded px-7 py-1.5 text-xs text-[#e2f0ff] placeholder-[#4a7fa5] focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          {searchResults && (
            <button
              onClick={() => { setSearchResults(null); setSearchQuery('') }}
              className="text-[10px] text-[#4a7fa5] hover:text-cyan-400 px-2"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {loading && (
          <div className="flex items-center justify-center py-8 text-[#4a7fa5] text-xs">
            Loading...
          </div>
        )}

        {!loading && displayMessages.length === 0 && (
          <div className="flex items-center justify-center h-full text-[#4a7fa5]">
            <div className="text-center">
              <Clock size={28} className="mx-auto mb-2 opacity-30" />
              <p className="text-xs">{searchResults ? 'No search results' : 'No messages yet'}</p>
            </div>
          </div>
        )}

        {displayMessages.map((msg, i) => (
          <div
            key={msg.id || i}
            className={`rounded-lg p-2.5 text-xs fade-in ${
              msg.role === 'user'
                ? 'bg-[#0d1d35] border border-[#1e3a5f] ml-6'
                : 'bg-[#061020] border border-[#162a45] mr-6'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className={`mono text-[10px] font-bold ${
                msg.role === 'user' ? 'text-blue-400' : 'text-cyan-400'
              }`}>
                {msg.role === 'user' ? 'YOU' : 'NEXUS'}
                {msg.metadata?.agent && (
                  <span className="text-purple-400 ml-1">via {msg.metadata.agent}</span>
                )}
              </span>
              <span className="text-[10px] text-[#4a7fa5]">
                {msg.created_at && formatTime(msg.created_at)}
              </span>
            </div>
            <p className="text-[#e2f0ff] whitespace-pre-wrap leading-relaxed">
              {msg.content?.substring(0, 500)}
              {msg.content?.length > 500 && <span className="text-[#4a7fa5]">... (truncated)</span>}
            </p>
            {msg.metadata?.reflect_score !== undefined && (
              <div className="mt-1.5 flex items-center gap-2">
                <span className="text-[10px] text-[#4a7fa5]">
                  ReflectScore: {msg.metadata.reflect_score?.toFixed(3)}
                </span>
                <span className={`text-[10px] ${
                  msg.metadata.reflect_verdict === 'clean' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                  {msg.metadata.reflect_verdict}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-3 py-1 border-t border-[#1e3a5f] bg-[#020b18] flex items-center justify-between">
        <span className="text-[10px] text-[#4a7fa5] mono">
          {displayMessages.length} messages
          {searchResults && ` (search: "${searchQuery}")`}
        </span>
        {activeSession && (
          <span className="text-[10px] text-[#4a7fa5] mono">
            Session: {activeSession.substring(0, 8)}
          </span>
        )}
      </div>
    </div>
  )
}


function formatTime(isoString) {
  try {
    const d = new Date(isoString)
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}
