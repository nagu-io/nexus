import React, { useMemo } from 'react'
import { AppWindow, Cpu, HardDrive, Minus, Square, Wifi, WifiOff, X } from 'lucide-react'
import { getWindowControls } from '../lib/runtime.js'

export default function DesktopTitleBar({ desktopInfo, status, connected, workspaceRoot }) {
  const controls = useMemo(() => getWindowControls(), [])
  const backendOnline = Boolean(status?.online)
  const normalizedWorkspace = workspaceRoot || status?.workspace_root || ''

  return (
    <header data-tauri-drag-region className="app-drag relative border-b border-[var(--border-strong)] bg-black/40 backdrop-blur-3xl">
      <div data-tauri-drag-region className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(255,255,255,0.03),transparent_22%,transparent_78%,rgba(255,255,255,0.015))]" />

      <div data-tauri-drag-region className="relative flex h-[52px] items-center justify-between gap-3 px-4">
        <div className="app-no-drag flex min-w-0 items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-[var(--border)] bg-[rgba(255,255,255,0.02)] text-[var(--text-strong)] shadow-[0_0_20px_rgba(255,255,255,0.06)]">
            <AppWindow size={16} strokeWidth={1.5} />
          </div>

          <div className="min-w-0">
            <div className="flex min-w-0 items-center gap-2">
              <p className="font-display text-sm font-semibold tracking-[0.18em] text-[var(--text-strong)]">
                NEXUS
              </p>
              <span className="meta-pill mono text-[10px]">{desktopInfo?.version ? `v${desktopInfo.version}` : 'preview'}</span>
            </div>
            <p className="text-[11px] text-[var(--text-soft)]">
              Desktop mission control
            </p>
          </div>

          <div className="hidden min-w-0 items-center gap-2 lg:flex">
            <span className={`meta-pill mono text-[10px] ${backendOnline ? 'text-[var(--success)]' : 'text-[var(--warning)]'}`}>
              {backendOnline ? <Cpu size={11} /> : <WifiOff size={11} />}
              {backendOnline ? status?.local_backend || 'backend online' : 'backend offline'}
            </span>
            <span className={`meta-pill mono text-[10px] ${connected ? 'text-[var(--accent)]' : 'text-[var(--text-soft)]'}`}>
              {connected ? <Wifi size={11} /> : <WifiOff size={11} />}
              {connected ? `${status?.ws_connections ?? 0} ws live` : 'ws reconnecting'}
            </span>
            {normalizedWorkspace && (
              <span className="meta-pill mono max-w-[280px] truncate text-[10px]">
                <HardDrive size={11} />
                {shortPath(normalizedWorkspace, 38)}
              </span>
            )}
          </div>
        </div>

        {controls && (
          <div className="app-no-drag flex items-center gap-1.5">
            <button className="desktop-control" onClick={() => void controls.minimize()} title="Minimize">
              <Minus size={14} />
            </button>
            <button className="desktop-control" onClick={() => void controls.toggleMaximize()} title="Maximize">
              <Square size={12} />
            </button>
            <button className="desktop-control destructive" onClick={() => void controls.close()} title="Close">
              <X size={14} />
            </button>
          </div>
        )}
      </div>
    </header>
  )
}

function shortPath(value, maxLength = 38) {
  if (!value) return ''
  if (value.length <= maxLength) return value

  const normalized = value.replaceAll('\\', '/')
  const parts = normalized.split('/')
  if (parts.length <= 2) {
    return `${value.slice(0, maxLength - 1)}…`
  }

  return `${parts.slice(0, 2).join('/')}/…/${parts[parts.length - 1]}`
}
