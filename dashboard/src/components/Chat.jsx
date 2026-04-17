import React, { useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Loader2, Send, ShieldAlert } from 'lucide-react'

const DEFAULT_ASSISTANT_MESSAGE = {
  role: 'assistant',
  content:
    'Open a repo, describe the task, and NEXUS will work inside the project with traceable runtime steps.',
  agent: null,
  reflectScore: 0.1,
  reflectVerdict: 'clean',
  reflectAction: 'serve',
  route: 'local',
  initialRoute: 'local',
  warning: null,
  wasRerouted: false,
  contextReduction: null,
  execution: null,
  workspaceRoot: null,
}

const STARTER_PROMPTS = [
  'Build an auth page in this repository with solid UX and wire it into the existing app.',
  'Inspect this repo for the current login issue, explain the bug, and fix it in place.',
  'Analyze this repository and explain the architecture, entry points, and likely next work.',
  'Refactor the current dashboard shell to feel more like a desktop coding assistant.',
  '/hive build me a full authentication system',
]

export default function Chat({ apiUrl, events = [], workspaceRoot, onReflectState, onAgentChange, onRouteUpdate }) {
  const [messages, setMessages] = useState([DEFAULT_ASSISTANT_MESSAGE])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [workspaceMode, setWorkspaceMode] = useState(Boolean(workspaceRoot))
  const [executionMode, setExecutionMode] = useState('stable')
  const messagesViewportRef = useRef(null)
  const [sessionId] = useState(() => getOrCreateSessionId())

  useEffect(() => {
    const viewport = messagesViewportRef.current
    if (!viewport) return

    const frame = window.requestAnimationFrame(() => {
      const showEmptyState = isDefaultConversation(messages, loading)
      viewport.scrollTo({
        top: showEmptyState ? 0 : viewport.scrollHeight,
        behavior: !showEmptyState && messages.length > 2 ? 'smooth' : 'auto',
      })
    })

    return () => window.cancelAnimationFrame(frame)
  }, [loading, messages])

  useEffect(() => {
    if (!workspaceRoot) {
      setWorkspaceMode(false)
      return
    }
    setWorkspaceMode(true)
  }, [workspaceRoot])

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
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          workspace_root: workspaceMode ? workspaceRoot : null,
          workspace_mode: Boolean(workspaceMode && workspaceRoot),
          execution_mode: workspaceMode && workspaceRoot ? executionMode : 'stable',
        }),
      })
      let data = null
      try {
        data = await response.json()
      } catch {
        // Non-JSON errors are handled below.
      }

      if (!response.ok) {
        const detail = data?.detail || data?.message || `Chat request failed with status ${response.status}.`
        throw new Error(detail)
      }

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
        execution: data.execution ?? null,
        workspaceRoot: data.workspace_root ?? null,
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
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: error?.message || 'NEXUS could not complete that request.',
          agent: null,
          reflectScore: null,
          reflectVerdict: null,
          reflectAction: null,
          route: null,
          initialRoute: null,
          warning: null,
          wasRerouted: false,
          contextReduction: null,
          execution: null,
          workspaceRoot: null,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const showEmptyState = isDefaultConversation(messages, loading)
  const visibleMessages = showEmptyState ? [] : messages

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {!showEmptyState && (
        <div className="border-b border-[var(--border)] panel-muted px-4 py-3">
          <div className="flex flex-col gap-2 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <span className="section-label">Conversation</span>
              <span className="meta-pill mono text-[11px]">
                {workspaceRoot ? shortPath(workspaceRoot, 46) : 'no repo selected'}
              </span>
              <span className="meta-pill mono text-[11px]">
                {messages.length} message{messages.length === 1 ? '' : 's'}
              </span>
            </div>

            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={() => setWorkspaceMode(value => !value)}
                disabled={!workspaceRoot}
                className={`meta-pill mono text-[11px] ${
                  workspaceMode && workspaceRoot
                    ? 'border-[rgba(0,240,255,0.4)] bg-[rgba(0,240,255,0.05)] text-[var(--accent)] shadow-[0_0_15px_rgba(0,240,255,0.15)] interactive'
                    : 'interactive'
                } ${!workspaceRoot ? 'opacity-50' : ''}`}
              >
                {workspaceMode && workspaceRoot ? 'repo mode' : 'chat mode'}
              </button>
              {workspaceMode && workspaceRoot && ['stable', 'explore'].map(mode => (
                <button
                  key={mode}
                  onClick={() => setExecutionMode(mode)}
                  className={`meta-pill mono text-[11px] ${
                    executionMode === mode
                      ? 'border-[rgba(191,0,255,0.4)] bg-[rgba(191,0,255,0.05)] text-[var(--accent-2)] shadow-[0_0_15px_rgba(191,0,255,0.15)] interactive'
                      : 'interactive'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <div ref={messagesViewportRef} className="flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-4">
          {showEmptyState && (
            <EmptyState
              workspaceRoot={workspaceRoot}
              workspaceMode={workspaceMode}
              executionMode={executionMode}
              onToggleWorkspaceMode={() => setWorkspaceMode(value => !value)}
              onSetExecutionMode={setExecutionMode}
              onPromptSelect={send}
            />
          )}

          {visibleMessages.length > 0 && (
            <div className="space-y-5 pb-4">
              {visibleMessages.map((message, index) => (
            <article key={index} className={`${message.role === 'user' ? 'flex justify-end' : 'flex justify-start'} fade-in`}>
              <div className={`w-full ${
                message.role === 'user'
                  ? 'ml-auto max-w-[52rem]'
                  : 'mr-auto max-w-[66rem]'
              }`}>
                {message.role === 'assistant' && (
                  <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                    <span className="section-label text-[10px]">NEXUS</span>
                    {message.agent && <span className="meta-pill interactive mono text-[11px] font-semibold">{message.agent}</span>}
                    {message.route && <span className={`meta-pill mono text-[11px] ${routeTone(message.route)}`}>via {message.route}</span>}
                    {message.reflectScore !== undefined && message.reflectScore !== null && (
                      <span className={`meta-pill mono text-[11px] ${scoreTone(message.reflectScore)}`}>
                        {scoreIcon(message.reflectScore)}
                        RS {message.reflectScore.toFixed(2)}
                      </span>
                    )}
                  </div>
                )}

                {message.warning && (
                  <div className={`mb-2 rounded-[16px] border px-3.5 py-2.5 text-sm leading-relaxed ${
                    message.reflectAction === 'block'
                      ? 'border-[rgba(255,143,136,0.22)] bg-[rgba(255,143,136,0.08)] text-[var(--danger)] shadow-[0_0_15px_rgba(255,143,136,0.1)]'
                      : 'border-[rgba(240,202,114,0.22)] bg-[rgba(240,202,114,0.08)] text-[var(--warning)] shadow-[0_0_15px_rgba(240,202,114,0.1)]'
                  }`}>
                    {message.warning}
                  </div>
                )}

                {message.execution && (
                  <div className="mb-2 rounded-[16px] border border-[var(--border)] bg-[rgba(255,255,255,0.03)] px-3.5 py-2.5">
                    <div className="flex flex-wrap gap-2 text-[11px] mono text-[var(--text)]">
                      <span className="meta-pill">workflow {shortWorkflow(message.execution.workflow_id)}</span>
                      <span className="meta-pill">{message.execution.status}</span>
                      {message.execution.execution_mode && <span className="meta-pill">{message.execution.execution_mode}</span>}
                      {message.execution.final_confidence !== null && message.execution.final_confidence !== undefined && (
                        <span className="meta-pill">{Math.round(message.execution.final_confidence * 100)}%</span>
                      )}
                    </div>
                    {message.execution.execution_mode === 'hive' && message.execution.selected_nodes?.length > 0 && (
                      <p className="mt-2 text-sm text-[var(--text-soft)]">
                        nodes: {message.execution.selected_nodes.slice(0, 5).join(', ')}
                      </p>
                    )}
                    {message.execution.execution_mode === 'hive' && message.execution.canary_results?.some(item => !item.passed) && (
                      <p className="mt-2 text-sm text-[var(--warning)]">
                        canary failures: {message.execution.canary_results.filter(item => !item.passed).map(item => item.node_id).join(', ')}
                      </p>
                    )}
                    {message.execution.execution_mode === 'hive' && message.execution.assembled_output && (
                      <p className="mt-2 text-sm text-[var(--text-soft)]">
                        assembly ready
                      </p>
                    )}
                    {message.execution.touched_files?.length > 0 && (
                      <p className="mt-2 text-sm text-[var(--text-soft)]">
                        changed: {message.execution.touched_files.slice(0, 5).join(', ')}
                        {message.execution.touched_files.length > 5 ? ` +${message.execution.touched_files.length - 5}` : ''}
                      </p>
                    )}
                  </div>
                )}

                <div className={`px-5 py-4 shadow-lg ${
                  message.role === 'user' ? 'chat-bubble-user text-[var(--text-strong)]' : 'chat-bubble-system text-[var(--text)]'
                }`}>
                  <pre className="whitespace-pre-wrap font-sans text-[14px] leading-7">{message.content}</pre>
                </div>
              </div>
            </article>
              ))}
            </div>
          )}
        </div>

        {loading && (
          <div className="mt-4 flex justify-start fade-in">
            <div className="rounded-[18px] panel-soft px-5 py-3 text-sm text-[var(--text-soft)] shadow-lg inline-flex ring-1 ring-[var(--border)]">
              <div className="flex items-center gap-3">
                <Loader2 size={16} className="animate-spin text-[var(--accent-2)]" />
                {workspaceMode && workspaceRoot ? `Running ${executionMode} repo workflow…` : 'Routing through AEON and ReflectScore…'}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-[var(--border)] bg-transparent px-4 py-4 relative">
        <div className="absolute inset-0 bg-gradient-to-t from-[rgba(12,15,23,0.9)] to-transparent pointer-events-none" />
        <div className="rounded-[20px] panel-surface px-5 py-4 input-glow-focus relative">
          <div className="flex flex-col gap-2.5 md:flex-row md:items-end">
            <div className="min-w-0 flex-1">
              <div className="mb-1.5 flex flex-wrap items-center gap-2">
                <span className="section-label">Prompt</span>
                <span className={`meta-pill mono text-[11px] font-semibold ${
                  workspaceMode && workspaceRoot
                    ? 'border-[rgba(0,240,255,0.3)] bg-[rgba(0,240,255,0.08)] text-[var(--accent)] shadow-[0_0_10px_rgba(0,240,255,0.1)]'
                    : ''
                }`}>
                  {workspaceMode && workspaceRoot ? 'repo execution' : 'chat only'}
                </span>
                {workspaceMode && workspaceRoot && (
                  <span className="meta-pill mono text-[11px] border-[rgba(191,0,255,0.3)] bg-[rgba(191,0,255,0.08)] text-[var(--accent-2)] shadow-[0_0_10px_rgba(191,0,255,0.1)]">
                    policy {executionMode}
                  </span>
                )}
              </div>

              <textarea
                value={input}
                onChange={event => setInput(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    send()
                  }
                }}
                placeholder={workspaceRoot
                  ? 'Describe what to build, fix, or refactor in this repo…'
                  : 'Open a repo in Files, then describe the work you want NEXUS to do…'}
                className="min-h-[54px] w-full resize-none bg-transparent text-[14px] leading-6 text-[var(--text)] placeholder:text-[var(--text-soft)] focus:outline-none"
              />
            </div>

            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-full bg-[var(--text-strong)] px-5 py-2.5 text-sm font-bold text-[#07090f] transition-all hover:bg-[var(--accent)] hover:shadow-[0_0_20px_rgba(0,240,255,0.4)] hover:scale-105 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-[var(--text-strong)] disabled:hover:scale-100 disabled:hover:shadow-none"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function EmptyState({
  workspaceRoot,
  workspaceMode,
  executionMode,
  onToggleWorkspaceMode,
  onSetExecutionMode,
  onPromptSelect,
}) {
  return (
    <div className="space-y-6 pb-6 fade-in pt-4">
      <section className="rounded-[24px] panel-surface px-6 py-6 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 p-32 bg-[radial-gradient(ellipse_at_center,rgba(0,240,255,0.05),transparent_70%)] pointer-events-none" />
        <div className="flex flex-wrap items-center gap-2">
          <span className="section-label">New Task</span>
          <span className="meta-pill mono text-[11px]">
            {workspaceRoot ? shortPath(workspaceRoot, 52) : 'open a repository in Files'}
          </span>
          <button
            onClick={onToggleWorkspaceMode}
            disabled={!workspaceRoot}
            className={`meta-pill interactive mono text-[11px] font-semibold ${
              workspaceMode && workspaceRoot
                ? 'border-[rgba(0,240,255,0.4)] bg-[rgba(0,240,255,0.05)] text-[var(--accent)] shadow-[0_0_15px_rgba(0,240,255,0.15)]'
                : ''
            } ${!workspaceRoot ? 'opacity-50' : ''}`}
          >
            {workspaceMode && workspaceRoot ? 'repo mode' : 'chat mode'}
          </button>
          {workspaceRoot && ['stable', 'explore'].map(mode => (
            <button
              key={mode}
              onClick={() => onSetExecutionMode(mode)}
              className={`meta-pill interactive mono text-[11px] font-semibold ${
                executionMode === mode
                  ? 'border-[rgba(191,0,255,0.4)] bg-[rgba(191,0,255,0.05)] text-[var(--accent-2)] shadow-[0_0_15px_rgba(191,0,255,0.15)]'
                  : ''
              }`}
            >
              {mode}
            </button>
          ))}
        </div>

        <h2 className="mt-8 max-w-[42rem] font-display text-[28px] font-bold leading-tight tracking-tight text-[var(--text-strong)] relative z-10">
          Ask NEXUS to build, fix, refactor, or explain the repo.
        </h2>

        <p className="mt-3 max-w-[42rem] text-[14px] leading-7 text-[var(--text-soft)]">
          Keep the answer in chat. Open Activity only when you want the runtime details.
        </p>
      </section>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {STARTER_PROMPTS.map(prompt => (
          <button
            key={prompt}
            onClick={() => onPromptSelect(prompt)}
            className="rounded-[22px] panel-soft px-5 py-5 text-left text-[14px] leading-relaxed text-[var(--text-soft)] transition-all hover:border-[rgba(0,240,255,0.3)] hover:bg-[rgba(0,240,255,0.02)] hover:text-[var(--text-strong)] hover:shadow-[0_8px_30px_rgba(0,240,255,0.05)] hover:-translate-y-1 block"
          >
            {prompt}
          </button>
        ))}
      </section>
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
    execution: metadata.execution || null,
    workspaceRoot: metadata.workspace_root || null,
  }
}

function isDefaultConversation(messages, loading) {
  return !loading && messages.length === 1 && isDefaultAssistantMessage(messages[0])
}

function isDefaultAssistantMessage(message) {
  return (
    message?.role === DEFAULT_ASSISTANT_MESSAGE.role &&
    message?.content === DEFAULT_ASSISTANT_MESSAGE.content &&
    message?.agent === DEFAULT_ASSISTANT_MESSAGE.agent &&
    message?.route === DEFAULT_ASSISTANT_MESSAGE.route &&
    message?.execution === DEFAULT_ASSISTANT_MESSAGE.execution
  )
}

function shortWorkflow(value) {
  if (!value) return '--'
  return String(value).slice(0, 8)
}

function shortPath(value, maxLength = 48) {
  if (!value) return ''
  if (value.length <= maxLength) return value
  const normalized = value.replaceAll('\\', '/')
  const parts = normalized.split('/')
  if (parts.length <= 2) return value.slice(0, maxLength - 1) + '…'
  return `${parts.slice(0, 2).join('/')}/…/${parts[parts.length - 1]}`
}

function scoreTone(score) {
  if (score < 0.3) return 'text-[var(--success)]'
  if (score < 0.6) return 'text-[var(--warning)]'
  return 'text-[var(--danger)]'
}

function scoreIcon(score) {
  if (score < 0.3) return <CheckCircle2 size={12} />
  if (score < 0.6) return <AlertTriangle size={12} />
  return <ShieldAlert size={12} />
}

function routeTone(route) {
  if (route === 'local') return 'text-[var(--accent-2)]'
  if (route === 'cloud') return 'text-[var(--warning)]'
  if (route === 'workspace') return 'text-[var(--accent)]'
  if (route === 'hive') return 'text-[var(--success)]'
  return 'text-[var(--text-soft)]'
}
