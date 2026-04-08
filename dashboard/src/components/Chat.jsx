import React, { useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle, Loader2, Send, ShieldAlert } from 'lucide-react'

const DEFAULT_ASSISTANT_MESSAGE = {
  role: 'assistant',
  content:
    'NEXUS online. ReflectScore will serve, warn on, or block every answer before it reaches you.',
  agent: null,
  reflectScore: 0.1,
  reflectVerdict: 'clean',
  reflectAction: 'serve',
  route: 'local',
  initialRoute: 'local',
  warning: null,
  wasRerouted: false,
  contextReduction: null,
}

const STARTER_PROMPTS = [
  {
    label: 'Explain NEXUS',
    prompt: 'In this repository, what is NEXUS and how do the planner, orchestrator, and dashboard fit together?',
  },
  {
    label: 'Best Demo',
    prompt: 'Based on this repository, what are the best first demo commands or prompts to show NEXUS working locally?',
  },
  {
    label: 'Analyze Architecture',
    prompt: 'Analyze this repository and summarize the compiler, runtime, policy, and dashboard layers.',
  },
  {
    label: 'What Can It Do?',
    prompt: 'Based on this repository, what can NEXUS do right now, and what should I try first in this dashboard?',
  },
]

export default function Chat({ apiUrl, onReflectState, onAgentChange, onRouteUpdate }) {
  const [messages, setMessages] = useState([DEFAULT_ASSISTANT_MESSAGE])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const [sessionId] = useState(() => getOrCreateSessionId())

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    let cancelled = false
    const loadHistory = async () => {
      try {
        const response = await fetch(`${apiUrl}/chat/history?session_id=${sessionId}&limit=100`)
        const data = await response.json()
        if (cancelled) return
        if (Array.isArray(data.messages) && data.messages.length > 0) {
          setMessages(data.messages.map(mapHistoryMessage))
        } else {
          setMessages([DEFAULT_ASSISTANT_MESSAGE])
        }
      } catch {
        if (!cancelled) setMessages([DEFAULT_ASSISTANT_MESSAGE])
      }
    }
    loadHistory()
    return () => {
      cancelled = true
    }
  }, [apiUrl, sessionId])

  const send = async (promptOverride = null) => {
    const text = (promptOverride ?? input).trim()
    if (!text || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      })
      const data = await response.json()

      const message = {
        role: 'assistant',
        content: data.response,
        agent: data.agent,
        reflectScore: data.reflect_score,
        reflectVerdict: data.reflect_verdict,
        reflectAction: data.reflect_action,
        route: data.route,
        initialRoute: data.initial_route,
        warning: data.warning,
        wasRerouted: data.was_rerouted,
        contextReduction: data.context_reduction ?? null,
      }

      setMessages(prev => [...prev, message])
      onReflectState?.({
        score: data.reflect_score,
        verdict: data.reflect_verdict,
        action: data.reflect_action,
        warning: data.warning,
        wasRerouted: data.was_rerouted,
      })
      onAgentChange?.(data.agent)
      onRouteUpdate?.({ initialRoute: data.initial_route, finalRoute: data.route })
    } catch {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'NEXUS API not running. Start with: nexus chat --dashboard',
          agent: null,
          reflectScore: null,
          reflectVerdict: null,
          reflectAction: null,
          route: null,
          initialRoute: null,
          warning: null,
          wasRerouted: false,
          contextReduction: null,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const getScoreColor = score => {
    if (score === null || score === undefined) return 'text-gray-500'
    if (score < 0.3) return 'text-green-400'
    if (score < 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getScoreIcon = score => {
    if (score === null || score === undefined) return null
    if (score < 0.3) return <CheckCircle size={10} />
    if (score < 0.6) return <AlertTriangle size={10} />
    return <ShieldAlert size={10} />
  }

  const showStarterPrompts = messages.length === 1 && !loading

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} fade-in`}>
            <div className={`max-w-2xl ${msg.role === 'user' ? 'order-1' : 'order-2'}`}>
              {msg.role === 'assistant' && (
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="mono text-cyan-400 text-xs font-bold">NEXUS</span>
                  {msg.agent && (
                    <span className="bg-purple-900/50 text-purple-300 text-xs px-2 py-0.5 rounded-full mono">
                      {msg.agent}
                    </span>
                  )}
                  {msg.route && (
                    <span
                      className={`text-xs mono ${
                        msg.route === 'local'
                          ? 'text-green-400'
                          : msg.route === 'cloud'
                            ? 'text-yellow-400'
                            : 'text-purple-400'
                      }`}
                    >
                      via {msg.route}
                      {msg.wasRerouted ? ` (re-routed from ${msg.initialRoute})` : ''}
                    </span>
                  )}
                  {msg.contextReduction?.reduced && (
                    <span className="bg-yellow-950/40 text-yellow-300 text-xs px-2 py-0.5 rounded-full mono">
                      {`ctx ${msg.contextReduction.backend} ${formatCompactLength(msg.contextReduction.original_length)}->${formatCompactLength(msg.contextReduction.reduced_length)}`}
                    </span>
                  )}
                  {msg.reflectScore !== undefined && msg.reflectScore !== null && (
                    <span className={`flex items-center gap-1 text-xs mono ${getScoreColor(msg.reflectScore)}`}>
                      {getScoreIcon(msg.reflectScore)}
                      RS:{msg.reflectScore.toFixed(2)}
                    </span>
                  )}
                </div>
              )}

              {msg.warning && (
                <div
                  className={`rounded-lg px-3 py-2 mb-2 text-xs leading-relaxed border ${
                    msg.reflectAction === 'block'
                      ? 'bg-red-950/40 border-red-700/40 text-red-200'
                      : 'bg-yellow-950/40 border-yellow-700/40 text-yellow-100'
                  }`}
                >
                  {msg.warning}
                </div>
              )}

              <div
                className={`rounded-lg px-4 py-3 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-cyan-900/40 text-cyan-100 border border-cyan-800/50'
                    : 'bg-[#0a1628] text-[#c8e0f4] border border-[#1e3a5f]'
                }`}
              >
                <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>
              </div>
            </div>
          </div>
        ))}

        {showStarterPrompts && (
          <div className="bg-[#071222] border border-[#1e3a5f] rounded-lg p-4 fade-in">
            <p className="mono text-cyan-400 text-xs mb-3 uppercase tracking-widest">Starter Prompts</p>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {STARTER_PROMPTS.map(starter => (
                <button
                  key={starter.label}
                  onClick={() => send(starter.prompt)}
                  className="rounded-lg border border-[#1e3a5f] bg-[#0a1628] px-4 py-3 text-left text-sm text-[#c8e0f4] hover:bg-[#10223a] transition-colors"
                >
                  <span className="mono text-cyan-300 text-xs uppercase tracking-widest block mb-2">
                    {starter.label}
                  </span>
                  <span className="leading-relaxed">{starter.prompt}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="flex justify-start fade-in">
            <div className="bg-[#0a1628] border border-[#1e3a5f] rounded-lg px-4 py-3 flex items-center gap-2">
              <Loader2 size={14} className="animate-spin text-cyan-400" />
              <span className="text-[#4a7fa5] text-sm mono">routing through AEON + ReflectScore...</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-[#1e3a5f] p-4">
        <div className="flex gap-3 items-end">
          <textarea
            value={input}
            onChange={event => setInput(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                send()
              }
            }}
            placeholder="Ask NEXUS anything... (Enter to send, Shift+Enter for newline)"
            className="flex-1 bg-[#0a1628] border border-[#1e3a5f] rounded-lg px-4 py-3 text-sm text-[#c8e0f4] placeholder-[#2a4a6f] resize-none focus:outline-none focus:border-cyan-500 transition-colors"
            rows={2}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="bg-cyan-500 hover:bg-cyan-400 disabled:bg-[#1e3a5f] disabled:text-[#2a4a6f] text-black p-3 rounded-lg transition-colors"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>
        <p className="text-xs text-[#2a4a6f] mono mt-2">
          ReflectScore is the trust layer: serve under 0.3, warn from 0.3 to 0.6, block and re-route above 0.6.
        </p>
      </div>
    </div>
  )
}


function getOrCreateSessionId() {
  try {
    const existing = window.localStorage.getItem('nexus_session_id')
    if (existing) return existing
    const created = window.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
    window.localStorage.setItem('nexus_session_id', created)
    return created
  } catch {
    return 'default'
  }
}


function mapHistoryMessage(message) {
  const metadata = message.metadata || {}
  return {
    role: message.role,
    content: message.content,
    agent: metadata.agent || null,
    reflectScore: metadata.reflect_score ?? null,
    reflectVerdict: metadata.reflect_verdict || null,
    reflectAction: metadata.reflect_action || null,
    route: metadata.route || null,
    initialRoute: metadata.initial_route || null,
    warning: metadata.warning || null,
    wasRerouted: Boolean(metadata.was_rerouted),
    contextReduction: metadata.context_reduction || null,
  }
}


function formatCompactLength(value) {
  if (value === null || value === undefined) return '--'
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`
  return String(value)
}
