import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  FolderTree,
  History,
  KeyRound,
  Layers3,
  Network,
  PanelBottom,
  RefreshCw,
  Shield,
  Sparkles,
  Workflow,
} from 'lucide-react'
import AgentStatus from '../components/AgentStatus.jsx'
import Chat from '../components/Chat.jsx'
import ChatHistory from '../components/ChatHistory.jsx'
import DesktopTitleBar from '../components/DesktopTitleBar.jsx'
import HivePanel from '../components/HivePanel.jsx'
import LiveTerminal from '../components/LiveTerminal.jsx'
import ModelStatus from '../components/ModelStatus.jsx'
import ModelControlCenter from '../components/ModelControlCenter.jsx'
import ProviderSettings from '../components/ProviderSettings.jsx'
import ReflectMeter from '../components/ReflectMeter.jsx'
import RunHistory from '../components/RunHistory.jsx'
import RuntimeOverview from '../components/RuntimeOverview.jsx'
import WorkspaceEditor from '../components/WorkspaceEditor.jsx'
import useWebSocket from '../hooks/useWebSocket.js'
import { API_URL, WS_URL, fetchJson, hasOperationalModelPath } from '../lib/runtime.js'

export default function BuildScreen({ initialStatus = null, desktopInfo = null }) {
  const [status, setStatus] = useState(initialStatus)
  const [overview, setOverview] = useState(null)
  const [statusError, setStatusError] = useState('')
  const [overviewError, setOverviewError] = useState('')
  const [reflect, setReflect] = useState({
    score: null,
    verdict: null,
    action: null,
    warning: null,
  })
  const [workspaceRoot, setWorkspaceRoot] = useState(null)
  const [routeInfo, setRouteInfo] = useState({ initialRoute: null, finalRoute: null })
  const [activeAgent, setActiveAgent] = useState(null)
  const [providerSettingsOpen, setProviderSettingsOpen] = useState(false)
  const [leftPanel, setLeftPanel] = useState('files')
  const [showActivity, setShowActivity] = useState(false)

  const { events, connected, connectionCount, clearEvents } = useWebSocket(WS_URL)

  useEffect(() => {
    if (initialStatus) {
      setStatus(initialStatus)
    }
  }, [initialStatus])

  const loadStatus = useCallback(async ({ quiet = false } = {}) => {
    try {
      const data = await fetchJson('/status')
      setStatus(data)
      setStatusError('')
    } catch (error) {
      setStatusError(error?.message || 'Could not load status')
    }
  }, [])

  const loadOverview = useCallback(async ({ quiet = false } = {}) => {
    try {
      const data = await fetchJson('/runtime/overview?limit=8')
      setOverview(data)
      setOverviewError('')
    } catch (error) {
      if (!quiet) {
        setOverviewError(error?.message || 'Could not load runtime overview')
      }
    }
  }, [])

  useEffect(() => {
    void loadStatus({ quiet: Boolean(initialStatus) })
    void loadOverview({ quiet: true })

    const statusInterval = window.setInterval(() => {
      void loadStatus({ quiet: true })
    }, 15000)
    const overviewInterval = window.setInterval(() => {
      void loadOverview({ quiet: true })
    }, 20000)

    return () => {
      window.clearInterval(statusInterval)
      window.clearInterval(overviewInterval)
    }
  }, [initialStatus, loadOverview, loadStatus])

  const flowLabel = useMemo(() => {
    if (activeAgent) return `${capitalize(activeAgent)} active`
    if (routeInfo.finalRoute === 'workspace') return 'repo execution'
    if (routeInfo.finalRoute === 'cloud') return 'cloud fallback'
    if (routeInfo.finalRoute === 'local') return 'local model'
    if (routeInfo.finalRoute === 'hive') return 'hive mesh'
    return 'ready'
  }, [activeAgent, routeInfo])

  const trustLabel =
    reflect.score === null || reflect.score === undefined ? 'trust --' : `trust ${Math.round(reflect.score * 100)}%`

  const workspaceLabel = workspaceRoot || status?.workspace_root || null

  const statusChips = [
    status?.model || 'loading model',
    status?.local_backend || 'backend',
    flowLabel,
    trustLabel,
  ]

  const missionCards = [
    {
      label: 'Routing',
      value: flowLabel,
      detail: `local ${status?.route_stats?.local ?? 0} · repo ${status?.route_stats?.workspace ?? 0}`,
      icon: Workflow,
      tone: 'text-[var(--text-strong)]',
    },
    {
      label: 'Trust Layer',
      value: reflect.score !== null && reflect.score !== undefined ? `${Math.round(reflect.score * 100)}%` : 'awaiting',
      detail: `clean ${status?.reflect_stats?.clean ?? 0} · rerouted ${status?.reflect_stats?.rerouted ?? 0}`,
      icon: Shield,
      tone: trustTone(reflect.score),
    },
    {
      label: 'Hive',
      value: status?.hive ? `${status.hive.trusted_nodes}/${status.hive.total_nodes}` : '--',
      detail: status?.hive?.strategy || 'parallel search',
      icon: Sparkles,
      tone: 'text-[var(--text-strong)]',
    },
    {
      label: 'Trace',
      value: `${overview?.metrics?.total_runs ?? 0} runs`,
      detail: `${status?.conversation_count ?? 0} conversations tracked`,
      icon: Activity,
      tone: 'text-[var(--text-strong)]',
    },
  ]

  const degradedMode = status?.online && !hasOperationalModelPath(status)

  return (
    <div className="app-shell">
      <div className="window-surface flex h-full flex-col">
        <DesktopTitleBar
          desktopInfo={desktopInfo}
          status={status}
          connected={connected}
          workspaceRoot={workspaceLabel}
        />

        <div className="grid min-h-0 flex-1 xl:grid-cols-[296px_minmax(0,1fr)_340px]">
          <aside className="flex min-h-0 flex-col border-r border-[var(--border)] panel-muted">
            <div className="border-b border-[var(--border)] px-4 py-4">
              <div className="flex items-center gap-2">
                <span className="section-label font-display">Workspace</span>
                <span className="text-xs text-[var(--text-soft)]">repo and memory</span>
              </div>
              <div className="mt-4 flex gap-1.5">
                <SidebarTab
                  active={leftPanel === 'files'}
                  icon={FolderTree}
                  label="Files"
                  onClick={() => setLeftPanel('files')}
                />
                <SidebarTab
                  active={leftPanel === 'history'}
                  icon={History}
                  label="History"
                  onClick={() => setLeftPanel('history')}
                />
                <SidebarTab
                  active={leftPanel === 'models'}
                  icon={Layers3}
                  label="Models"
                  onClick={() => setLeftPanel('models')}
                />
                <SidebarTab
                  active={leftPanel === 'hive'}
                  icon={Network}
                  label="Hive"
                  onClick={() => setLeftPanel('hive')}
                />
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-3">
              {leftPanel === 'files' ? (
                <WorkspaceEditor
                  apiUrl={API_URL}
                  defaultRoot={status?.workspace_root}
                  onWorkspaceChange={setWorkspaceRoot}
                  compact
                />
              ) : leftPanel === 'models' ? (
                <ModelControlCenter apiUrl={API_URL} />
              ) : leftPanel === 'hive' ? (
                <HivePanel apiUrl={API_URL} />
              ) : (
                <ChatHistory />
              )}
            </div>
          </aside>

          <main className="relative flex min-h-0 flex-col bg-transparent">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[rgba(255,255,255,0.02)] to-transparent" />

            <header className="relative border-b border-[var(--border)] px-4 py-2.5 bg-[rgba(255,255,255,0.01)]">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  {statusChips.map(chip => (
                    <span key={chip} className="meta-pill mono text-[10px] font-semibold tracking-wider">
                      {chip}
                    </span>
                  ))}
                  {workspaceLabel && (
                    <span className="meta-pill mono text-[10px]">{shortPath(workspaceLabel, 44)}</span>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => setShowActivity(value => !value)}
                    className={`meta-pill interactive mono text-[10px] font-semibold tracking-wider ${
                      showActivity
                        ? 'border-[rgba(255,255,255,0.2)] text-[var(--text-strong)] shadow-[0_0_15px_rgba(255,255,255,0.06)] bg-[rgba(255,255,255,0.06)]'
                        : ''
                    }`}
                  >
                    <PanelBottom size={11} />
                    {showActivity ? 'hide activity' : 'show activity'}
                  </button>
                  <button
                    onClick={() => setProviderSettingsOpen(true)}
                    className="meta-pill interactive mono text-[10px] font-semibold tracking-wider"
                  >
                    <KeyRound size={11} />
                    OpenRouter
                  </button>
                  <button
                    onClick={() => {
                      void loadStatus({ quiet: false })
                      void loadOverview({ quiet: false })
                    }}
                    className="meta-pill interactive mono text-[10px] font-semibold tracking-wider"
                  >
                    <RefreshCw size={11} />
                    refresh
                  </button>
                </div>
              </div>

              {(statusError || degradedMode || overviewError) && (
                <div className="mt-2.5 flex flex-col gap-2">
                  {statusError && <Banner tone="danger">{statusError}</Banner>}
                  {degradedMode && (
                    <Banner tone="warning">
                      Local model access is incomplete. Normal chat execution will be limited.
                    </Banner>
                  )}
                  {!statusError && overviewError && (
                    <Banner tone="info">{overviewError}</Banner>
                  )}
                </div>
              )}
            </header>

            <div className="relative flex min-h-0 flex-1 flex-col">
              <div className="min-h-0 flex-1 px-4 py-4 md:px-5">
                <div className="h-full overflow-hidden rounded-[20px] panel-surface">
                  <Chat
                    key={workspaceRoot || 'global'}
                    apiUrl={API_URL}
                    events={events}
                    workspaceRoot={workspaceRoot}
                    onReflectState={setReflect}
                    onAgentChange={setActiveAgent}
                    onRouteUpdate={info => {
                      setRouteInfo(info)
                      void loadStatus({ quiet: true })
                      void loadOverview({ quiet: true })
                    }}
                  />
                </div>
              </div>

              {showActivity && (
                <div className="border-t border-[var(--border)] px-4 pb-4 pt-0 md:px-5">
                  <div className="h-[220px] overflow-hidden rounded-[20px] panel-surface">
                    <LiveTerminal
                      events={events}
                      connected={connected}
                      connectionCount={connectionCount}
                      onClear={clearEvents}
                    />
                  </div>
                </div>
              )}
            </div>
          </main>

          <aside className="hidden min-h-0 flex-col border-l border-[var(--border)] panel-muted xl:flex">
            <div className="border-b border-[var(--border)] px-4 py-4">
              <div className="flex items-center gap-2">
                <span className="section-label">Mission Rail</span>
                <span className="text-xs text-[var(--text-soft)]">runtime telemetry</span>
              </div>
              <p className="mt-3 text-sm leading-6 text-[var(--text-soft)]">
                Live model health, trust verdicts, agent activity, and recent workflow traces.
              </p>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-3">
              <div className="space-y-3">
                <ModelStatus status={status} />
                <ReflectMeter reflect={reflect} stats={status?.reflect_stats} />
                <AgentStatus active={activeAgent} stats={status?.route_stats} />
                <RuntimeOverview overview={overview} />
                <RunHistory runs={overview?.runs || []} />
              </div>
            </div>
          </aside>
        </div>
      </div>

      <ProviderSettings
        apiUrl={API_URL}
        isOpen={providerSettingsOpen}
        onClose={() => setProviderSettingsOpen(false)}
        onSaved={() => {
          void loadStatus({ quiet: false })
        }}
      />
    </div>
  )
}

function SidebarTab({ active, icon: Icon, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] mono transition-colors ${
        active
          ? 'bg-[rgba(255,255,255,0.1)] text-[var(--text-strong)] border border-[rgba(255,255,255,0.15)] shadow-[0_0_10px_rgba(255,255,255,0.02)]'
          : 'text-[var(--text-soft)] hover:bg-[rgba(255,255,255,0.05)] hover:text-[var(--text)] border border-transparent'
      }`}
    >
      <Icon size={12} />
      {label}
    </button>
  )
}

function MissionCard({ label, value, detail, icon: Icon, tone }) {
  return (
    <div className="panel-surface px-4 py-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="section-label text-[10px]">{label}</p>
          <p className={`mt-3 text-2xl font-semibold tracking-[-0.04em] ${tone}`}>{value}</p>
          <p className="mt-2 text-sm text-[var(--text-soft)]">{detail}</p>
        </div>
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[rgba(255,255,255,0.04)] text-[var(--text-muted)]">
          <Icon size={17} />
        </div>
      </div>
    </div>
  )
}

function Banner({ children, tone = 'info' }) {
  const tones = {
    info: 'border-[rgba(255,255,255,0.12)] bg-[rgba(255,255,255,0.04)] text-[var(--text)]',
    warning: 'border-[rgba(240,202,114,0.2)] bg-[rgba(240,202,114,0.08)] text-[var(--warning)]',
    danger: 'border-[rgba(255,143,136,0.22)] bg-[rgba(255,143,136,0.08)] text-[var(--danger)]',
  }

  return (
    <div className={`rounded-[18px] border px-4 py-3 text-sm leading-6 ${tones[tone] || tones.info}`}>
      {children}
    </div>
  )
}

function trustTone(score) {
  if (score === null || score === undefined) return 'text-[var(--text-soft)]'
  if (score < 0.3) return 'text-[var(--success)]'
  if (score < 0.6) return 'text-[var(--warning)]'
  return 'text-[var(--danger)]'
}

function shortPath(value, maxLength = 42) {
  if (!value) return ''
  if (value.length <= maxLength) return value
  const normalized = value.replaceAll('\\', '/')
  const parts = normalized.split('/')
  if (parts.length <= 2) return value.slice(0, maxLength - 1) + '…'
  return `${parts.slice(0, 2).join('/')}/…/${parts[parts.length - 1]}`
}

function capitalize(value) {
  const text = String(value || '')
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : ''
}
