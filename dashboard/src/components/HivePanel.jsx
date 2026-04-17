import React, { useEffect, useState } from 'react'
import { Activity, Cpu, Network, ShieldCheck, Sparkles } from 'lucide-react'

const STARTER_PROMPTS = [
  'build me a full authentication system',
  'design a repo-aware bug triage swarm',
  'research and compare login architecture options',
]

export default function HivePanel({ apiUrl }) {
  const [status, setStatus] = useState(null)
  const [prompt, setPrompt] = useState('build me a full authentication system')
  const [intent, setIntent] = useState('coding')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const loadStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/hive/status`)
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not load Hive status')
      setStatus(data)
      setError('')
    } catch (err) {
      setError(err?.message || 'Could not load Hive status')
    }
  }

  useEffect(() => {
    void loadStatus()
    const timer = window.setInterval(() => {
      void loadStatus()
    }, 20000)
    return () => window.clearInterval(timer)
  }, [apiUrl])

  const runDemo = async () => {
    if (!prompt.trim() || running) return
    setRunning(true)
    setError('')
    try {
      const response = await fetch(`${apiUrl}/hive/demo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, intent }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Hive demo failed')
      setResult(data)
      setStatus(data.status)
      setError('')
    } catch (err) {
      setError(err?.message || 'Hive demo failed')
    } finally {
      setRunning(false)
    }
  }

  const cards = [
    { icon: Network, label: 'Nodes', value: status?.total_nodes ?? '--' },
    { icon: ShieldCheck, label: 'Trusted', value: status?.trusted_nodes ?? '--' },
    { icon: Cpu, label: 'Replication', value: status?.replication_factor ?? '--' },
    { icon: Activity, label: 'Runs', value: status?.demo_runs ?? '--' },
  ]

  return (
    <div className="space-y-3">
      <div className="panel-surface p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="section-label">Hive</p>
            <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
              Experimental distributed search for the desktop app. Nodes are trust-scored, canary-covered, and ranked with ReflectScore.
            </p>
          </div>
          <span className="meta-pill mono text-[11px]">
            {status?.enabled ? 'experimental live' : 'disabled'}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2">
          {cards.map(card => {
            const Icon = card.icon
            return (
              <div key={card.label} className="panel-muted px-3 py-3">
                <div className="flex items-center gap-2 text-[var(--text-soft)]">
                  <Icon size={14} />
                  <span className="section-label text-[10px]">{card.label}</span>
                </div>
                <p className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-[var(--text-strong)]">
                  {card.value}
                </p>
              </div>
            )
          })}
        </div>

        <div className="mt-4 rounded-[18px] panel-muted p-3">
          <div className="flex flex-wrap gap-1.5">
            {STARTER_PROMPTS.map(item => (
              <button
                key={item}
                onClick={() => setPrompt(item)}
                className="meta-pill interactive mono text-[11px]"
              >
                {item}
              </button>
            ))}
          </div>

          <div className="mt-3 flex flex-wrap gap-1.5">
            {['coding', 'research', 'design', 'memory', 'canary'].map(option => (
              <button
                key={option}
                onClick={() => setIntent(option)}
                className={`meta-pill interactive mono text-[11px] ${
                  intent === option
                    ? 'border-[rgba(0,240,255,0.35)] bg-[rgba(0,240,255,0.06)] text-[var(--accent)]'
                    : ''
                }`}
              >
                {option}
              </button>
            ))}
          </div>

          <textarea
            value={prompt}
            onChange={event => setPrompt(event.target.value)}
            className="mt-3 min-h-[110px] w-full resize-none rounded-[16px] border border-[var(--border)] bg-[rgba(255,255,255,0.02)] px-4 py-3 text-[14px] leading-6 text-[var(--text)] placeholder:text-[var(--text-soft)] focus:outline-none"
            placeholder="Describe the task you want Hive to search for..."
          />

          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="text-xs leading-5 text-[var(--text-soft)]">
              Privacy mode: {status?.privacy_mode || 'signature_only'} · envelope {status?.envelope_mode || 'sealed_local'} · trust floor {formatScore(status?.min_trust_score)}
            </p>
            <button
              onClick={() => void runDemo()}
              disabled={running || !prompt.trim()}
              className="inline-flex items-center gap-2 rounded-full bg-[var(--text-strong)] px-4 py-2 text-sm font-bold text-[#07090f] transition-all hover:bg-[var(--accent)] hover:shadow-[0_0_18px_rgba(0,240,255,0.35)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Sparkles size={16} />
              {running ? 'Running Hive...' : 'Run Hive Demo'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-[18px] border border-[rgba(255,143,136,0.22)] bg-[rgba(255,143,136,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
          {error}
        </div>
      )}

      <div className="panel-surface p-4">
        <div className="flex items-center justify-between">
          <p className="section-label">Trusted Nodes</p>
          <span className="meta-pill mono text-[11px]">{status?.strategy || 'parallel_search'}</span>
        </div>
        <div className="mt-3 space-y-2">
          {(status?.top_nodes || []).map(node => (
            <div key={node.node_id} className="panel-muted rounded-[16px] px-3 py-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="mono text-sm text-[var(--text-strong)]">{node.node_id}</p>
                  <p className="mt-1 text-xs text-[var(--text-soft)]">
                    {node.region} · {node.capabilities.join(', ')}
                  </p>
                </div>
                <div className="text-right text-xs text-[var(--text-soft)]">
                  <p>trust {formatScore(node.trust_score)}</p>
                  <p>{Math.round(node.avg_latency_ms)} ms</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {result && (
        <div className="panel-surface p-4">
          <div className="flex items-center justify-between">
            <p className="section-label">Consensus</p>
            <span className="meta-pill mono text-[11px]">
              {result.responded_nodes} responses
            </span>
          </div>

          <div className="mt-3 rounded-[18px] panel-muted p-3">
            <p className="text-sm leading-6 text-[var(--text)]">
              {result.plan.rationale}
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {result.plan.selected_nodes.map(nodeId => (
                <span key={nodeId} className="meta-pill mono text-[11px]">{nodeId}</span>
              ))}
            </div>
            {result.plan.canary_nodes?.length > 0 && (
              <p className="mt-3 text-xs leading-5 text-[var(--text-soft)]">
                canary nodes: {result.plan.canary_nodes.join(', ')}
              </p>
            )}
          </div>

          {result.winner && (
            <div className="mt-3 rounded-[18px] border border-[rgba(0,240,255,0.18)] bg-[rgba(0,240,255,0.05)] px-4 py-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="section-label">Winner</p>
                  <p className="mt-2 mono text-sm text-[var(--text-strong)]">{result.winner.node_id}</p>
                </div>
                <div className="text-right text-xs text-[var(--text-soft)]">
                  <p>network {formatScore(result.winner.network_score)}</p>
                  <p>reflect {formatScore(result.winner.reflect_score)}</p>
                </div>
              </div>
              <p className="mt-3 text-sm leading-6 text-[var(--text)]">{result.winner.output}</p>
            </div>
          )}

          {result.assembled_output && (
            <div className="mt-3 rounded-[18px] border border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.03)] px-4 py-4">
              <div className="flex items-center justify-between gap-3">
                <p className="section-label">Assembly</p>
                <span className="meta-pill mono text-[11px]">
                  {result.assembly_sources?.length || 0} sources
                </span>
              </div>
              <pre className="mt-3 whitespace-pre-wrap font-sans text-sm leading-6 text-[var(--text)]">
                {result.assembled_output}
              </pre>
            </div>
          )}

          {result.canary_results?.length > 0 && (
            <div className="mt-3 rounded-[18px] panel-muted p-3">
              <div className="flex items-center justify-between">
                <p className="section-label">Canary Checks</p>
                <span className="meta-pill mono text-[11px]">
                  {result.canary_results.length}
                </span>
              </div>
              <div className="mt-3 space-y-2">
                {result.canary_results.map(entry => (
                  <div key={entry.challenge_id} className="flex items-center justify-between gap-3 rounded-[14px] bg-[rgba(255,255,255,0.03)] px-3 py-2">
                    <div>
                      <p className="mono text-sm text-[var(--text-strong)]">{entry.node_id}</p>
                      <p className="mt-1 text-xs text-[var(--text-soft)]">{entry.reason}</p>
                    </div>
                    <span className={`meta-pill mono text-[11px] ${entry.passed ? 'text-[var(--success)]' : 'text-[var(--danger)]'}`}>
                      {entry.passed ? 'pass' : 'fail'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.envelopes?.length > 0 && (
            <div className="mt-3 rounded-[18px] panel-muted p-3">
              <div className="flex items-center justify-between">
                <p className="section-label">Envelopes</p>
                <span className="meta-pill mono text-[11px]">
                  {result.envelopes.length}
                </span>
              </div>
              <div className="mt-3 space-y-2">
                {result.envelopes.slice(0, 6).map(envelope => (
                  <div key={envelope.envelope_id} className="rounded-[14px] bg-[rgba(255,255,255,0.03)] px-3 py-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="mono text-sm text-[var(--text-strong)]">{envelope.node_id}</p>
                      <span className="meta-pill mono text-[11px]">
                        {envelope.is_canary ? 'canary' : 'task'}
                      </span>
                    </div>
                    <p className="mt-2 text-xs leading-5 text-[var(--text-soft)]">
                      {envelope.masked_context}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-3 space-y-2">
            {result.candidates.slice(0, 5).map(candidate => (
              <div key={`${candidate.node_id}-${candidate.latency_ms}`} className="panel-muted rounded-[16px] px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="mono text-sm text-[var(--text-strong)]">{candidate.node_id}</p>
                    <p className="mt-1 text-xs text-[var(--text-soft)]">
                      {candidate.reflect_verdict} · trust {formatScore(candidate.trust_score)} · {Math.round(candidate.latency_ms)} ms
                    </p>
                  </div>
                  <span className={`meta-pill mono text-[11px] ${candidate.blocked ? 'text-[var(--danger)]' : 'text-[var(--success)]'}`}>
                    {candidate.blocked ? 'blocked' : `score ${formatScore(candidate.network_score)}`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  return Number(value).toFixed(2)
}
