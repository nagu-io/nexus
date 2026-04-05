import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle, Loader2, Send, ShieldAlert } from 'lucide-react'

export default function Chat({ apiUrl, onReflectState, onAgentChange, onRouteUpdate }) {
  const [messages, setMessages] = useState([
    {
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
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
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
