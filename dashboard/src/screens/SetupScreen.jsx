import React, { useMemo } from 'react'
import {
  AlertTriangle,
  AppWindow,
  ArrowRight,
  Bot,
  CheckCircle2,
  Cloud,
  ExternalLink,
  Loader2,
  RefreshCw,
  Server,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import {
  API_URL,
  configuredModelAvailable,
  desktopAvailable,
  hasOperationalModelPath,
  openExternal,
} from '../lib/runtime.js'

export default function SetupScreen({
  status,
  desktopInfo,
  checking = false,
  error = '',
  checkedAt = null,
  onRetry,
  onComplete,
}) {
  const desktopShellReady = desktopAvailable() || Boolean(desktopInfo)
  const localModelReady = configuredModelAvailable(status)
  const computePathReady = hasOperationalModelPath(status)
  const canEnterWorkspace = Boolean(status?.online)
  const hiveStatus = status?.hive || null

  const checks = useMemo(
    () => [
      {
        id: 'desktop',
        icon: AppWindow,
        label: 'Desktop shell',
        description: desktopShellReady
          ? `${desktopInfo?.packaged ? 'Packaged' : 'Development'} Electron shell is connected.`
          : 'Browser preview is running without the native desktop bridge.',
        passed: desktopShellReady,
        required: false,
      },
      {
        id: 'backend',
        icon: Server,
        label: 'NEXUS backend',
        description: status?.online
          ? `API live at ${desktopInfo?.apiUrl || API_URL}.`
          : 'Backend is still booting or the API is unreachable.',
        passed: Boolean(status?.online),
        required: true,
        actionLabel: 'Retry bootstrap',
        action: () => void onRetry?.(),
      },
      {
        id: 'runtime',
        icon: Bot,
        label: 'Inference path',
        description: computePathReady
          ? localModelReady
            ? `${status?.model || 'Local model'} is ready for local-first execution.`
            : `Cloud fallback is configured through ${status?.cloud_provider}.`
          : 'Add a local model or configure a cloud provider before normal chat workflows.',
        passed: computePathReady,
        required: true,
        actionLabel: localModelReady ? null : 'Open Ollama',
        action: localModelReady ? null : () => void openExternal('https://ollama.com/download'),
      },
      {
        id: 'model',
        icon: Cloud,
        label: 'Launch model',
        description: localModelReady
          ? `${status?.model || 'Launch model'} is installed in Ollama.`
          : status?.ollama
            ? `Install ${status?.model || 'phi3:mini'} with \`ollama pull ${status?.model || 'phi3:mini'}\`.`
            : 'Start Ollama, then install the launch model for offline work.',
        passed: localModelReady,
        required: !status?.cloud_provider || status?.cloud_provider === 'none',
        actionLabel: localModelReady ? null : 'Model library',
        action: localModelReady ? null : () => void openExternal('https://ollama.com/library/phi3'),
      },
      {
        id: 'hive',
        icon: ShieldCheck,
        label: 'Hive trust layer',
        description: hiveStatus?.enabled
          ? `${hiveStatus.trusted_nodes}/${hiveStatus.total_nodes} nodes are trusted and canary-covered.`
          : 'Hive is not ready yet, but the local workspace can still launch.',
        passed: Boolean(hiveStatus?.enabled),
        required: false,
      },
    ],
    [computePathReady, desktopInfo, desktopShellReady, hiveStatus, localModelReady, onRetry, status],
  )

  const blockers = checks.filter(item => item.required && !item.passed)
  const primaryLabel = !canEnterWorkspace
    ? 'Waiting for backend'
    : blockers.length === 0
      ? 'Enter mission control'
      : 'Continue in limited mode'

  const heroCards = [
    {
      label: 'Backend',
      value: status?.online ? 'online' : checking ? 'booting' : 'offline',
      detail: status?.local_backend || 'local runtime',
      tone: status?.online ? 'text-[var(--success)]' : 'text-[var(--warning)]',
    },
    {
      label: 'Inference',
      value: computePathReady ? (localModelReady ? 'local ready' : 'cloud ready') : 'needs setup',
      detail: localModelReady ? status?.model || 'launch model' : status?.cloud_provider || 'Ollama',
      tone: computePathReady ? 'text-[var(--accent)]' : 'text-[var(--warning)]',
    },
    {
      label: 'Hive',
      value: hiveStatus?.enabled ? `${hiveStatus.trusted_nodes} trusted` : 'standby',
      detail: hiveStatus?.strategy || 'parallel search',
      tone: hiveStatus?.enabled ? 'text-[var(--accent-2)]' : 'text-[var(--text-soft)]',
    },
  ]

  return (
    <div className="app-shell">
      <div className="window-surface flex items-center justify-center px-6 py-8">
        <div className="grid w-full max-w-6xl gap-6 xl:grid-cols-[minmax(0,1.05fr)_420px]">
          <section className="panel-surface relative overflow-hidden px-7 py-8 md:px-9">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(0,240,255,0.14),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(191,0,255,0.16),transparent_38%)] pointer-events-none" />

            <div className="relative">
              <div className="flex flex-wrap items-center gap-2">
                <span className="meta-pill mono text-[11px]">
                  NEXUS desktop {desktopInfo?.version || 'preview'}
                </span>
                <span className="meta-pill mono text-[11px]">
                  {desktopInfo?.packaged ? 'packaged shell' : 'developer shell'}
                </span>
                <span className="meta-pill mono text-[11px]">
                  {checkedAt ? `checked ${formatCheckedAt(checkedAt)}` : 'awaiting first handshake'}
                </span>
              </div>

              <div className="mt-6 max-w-3xl">
                <h1 className="font-display text-4xl font-semibold tracking-[-0.04em] text-[var(--text-strong)] md:text-5xl">
                  Bring the desktop runtime fully online.
                </h1>
                <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--text-soft)]">
                  This startup flow now checks the real NEXUS stack: desktop shell, backend reachability, local or cloud inference, and the Hive trust layer.
                </p>
              </div>

              <div className="mt-8 grid gap-3 md:grid-cols-3">
                {heroCards.map(card => (
                  <div key={card.label} className="panel-muted px-4 py-4">
                    <p className="section-label text-[10px]">{card.label}</p>
                    <p className={`mt-3 text-2xl font-semibold tracking-[-0.04em] ${card.tone}`}>{card.value}</p>
                    <p className="mt-2 text-sm text-[var(--text-soft)]">{card.detail}</p>
                  </div>
                ))}
              </div>

              <div className="panel-muted mt-6 px-5 py-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="section-label text-[10px]">
                      {blockers.length === 0 ? 'System healthy' : 'Attention needed'}
                    </p>
                    <p className="mt-2 text-sm leading-7 text-[var(--text)]">
                      {blockers.length === 0
                        ? 'The core path is ready. You can enter the workspace with local-first execution or cloud fallback already configured.'
                        : 'You can still enter the workspace once the backend is live, but normal coding workflows will be limited until the compute path is restored.'}
                    </p>
                  </div>
                  <span className={`meta-pill mono text-[11px] ${blockers.length === 0 ? 'text-[var(--success)]' : 'text-[var(--warning)]'}`}>
                    {blockers.length === 0 ? 'ready' : `${blockers.length} blocker${blockers.length === 1 ? '' : 's'}`}
                  </span>
                </div>

                {(error || blockers.length > 0) && (
                  <div className="mt-4 space-y-2">
                    {error && (
                      <div className="rounded-[16px] border border-[rgba(255,143,136,0.22)] bg-[rgba(255,143,136,0.08)] px-4 py-3 text-sm leading-6 text-[var(--danger)]">
                        {error}
                      </div>
                    )}
                    {blockers.map(blocker => (
                      <div key={blocker.id} className="rounded-[16px] border border-[rgba(240,202,114,0.18)] bg-[rgba(240,202,114,0.08)] px-4 py-3 text-sm leading-6 text-[var(--warning)]">
                        {blocker.label}: {blocker.description}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </section>

          <aside className="panel-surface flex flex-col overflow-hidden px-5 py-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="section-label">Launch Checklist</p>
                <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
                  Resolve anything critical, then move straight into the desktop workspace.
                </p>
              </div>
              {checking && <Loader2 className="mt-1 h-5 w-5 animate-spin text-[var(--accent)]" />}
            </div>

            <div className="mt-5 space-y-3">
              {checks.map(check => (
                <CheckRow key={check.id} check={check} />
              ))}
            </div>

            <div className="mt-5 rounded-[20px] border border-[var(--border)] bg-[rgba(255,255,255,0.03)] px-4 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[rgba(191,0,255,0.12)] text-[var(--accent-2)]">
                  <Sparkles size={18} />
                </div>
                <div>
                  <p className="text-sm font-semibold text-[var(--text-strong)]">Mission Control</p>
                  <p className="mt-1 text-xs leading-5 text-[var(--text-soft)]">
                    Files, chat, runtime traces, and Hive controls now live in one desktop workspace.
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-5 space-y-3">
              <button
                onClick={onComplete}
                disabled={!canEnterWorkspace}
                className={`inline-flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-4 text-sm font-semibold transition-all ${
                  canEnterWorkspace
                    ? 'bg-gradient-to-r from-[var(--accent)] to-[var(--accent-2)] text-[#07090f] shadow-[0_0_24px_rgba(0,240,255,0.28)] hover:shadow-[0_0_36px_rgba(191,0,255,0.34)]'
                    : 'cursor-not-allowed border border-[var(--border)] bg-[rgba(255,255,255,0.03)] text-[var(--text-soft)]'
                }`}
              >
                {canEnterWorkspace ? <ArrowRight size={17} /> : <AlertTriangle size={17} />}
                {primaryLabel}
              </button>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => void onRetry?.()}
                  className="meta-pill interactive app-no-drag justify-center py-3 text-[12px] font-semibold"
                >
                  {checking ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                  Retry bootstrap
                </button>
                <button
                  onClick={() => void openExternal('https://ollama.com/download')}
                  className="meta-pill interactive app-no-drag justify-center py-3 text-[12px] font-semibold"
                >
                  <ExternalLink size={14} />
                  Install Ollama
                </button>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}

function CheckRow({ check }) {
  const Icon = check.icon
  const passed = check.passed

  return (
    <div className="rounded-[20px] border border-[var(--border)] bg-[rgba(255,255,255,0.03)] px-4 py-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 gap-3">
          <div
            className={`mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl ${
              passed
                ? 'bg-[rgba(16,185,129,0.12)] text-[var(--success)]'
                : 'bg-[rgba(240,202,114,0.12)] text-[var(--warning)]'
            }`}
          >
            <Icon size={18} />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-semibold text-[var(--text-strong)]">{check.label}</p>
              {passed ? (
                <span className="meta-pill mono text-[10px] text-[var(--success)]">
                  <CheckCircle2 size={11} />
                  ready
                </span>
              ) : (
                <span className="meta-pill mono text-[10px] text-[var(--warning)]">
                  <AlertTriangle size={11} />
                  {check.required ? 'required' : 'optional'}
                </span>
              )}
            </div>
            <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">{check.description}</p>
          </div>
        </div>

        {!passed && check.action && check.actionLabel && (
          <button
            onClick={check.action}
            className="meta-pill interactive app-no-drag shrink-0 text-[11px] font-semibold"
          >
            {check.actionLabel}
          </button>
        )}
      </div>
    </div>
  )
}

function formatCheckedAt(timestamp) {
  try {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return 'just now'
  }
}
