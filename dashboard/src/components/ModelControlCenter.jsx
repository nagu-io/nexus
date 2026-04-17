import React, { useEffect, useState } from 'react'
import {
  Bot,
  FolderOpen,
  Gauge,
  HardDrive,
  Layers3,
  Loader2,
  Package2,
  RefreshCw,
  Save,
  Sparkles,
} from 'lucide-react'
import { fetchJson } from '../lib/runtime.js'

const EMPTY_OVERVIEW = {
  runtime: {
    backend: 'ollama',
    launch_model: 'phi3:mini',
    resolved_launch_model: '',
    local_model_dir: 'lora_model',
    adapter_ready: false,
    cloud_fallback: 'none',
    single_app_mode: false,
  },
  artifacts: [],
  compressed_models: [],
  packaging: {
    budget_mb: 1024,
    runtime_reserve_mb: 180,
    adapter_pack_mb: 0,
    selected_launch_pack_mb: null,
    selected_launch_pack_name: null,
    estimated_total_mb: null,
    sub_gb_possible: null,
    readiness: 'prototype',
    message: '',
  },
  compressx: {
    manifest_path: '',
    available_outputs: 0,
    launch_alias_supported: true,
  },
}

export default function ModelControlCenter({ apiUrl }) {
  const [overview, setOverview] = useState(EMPTY_OVERVIEW)
  const [form, setForm] = useState({
    backend: 'ollama',
    localModelDir: 'lora_model',
    launchModel: 'phi3:mini',
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [compressing, setCompressing] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    void loadOverview()
  }, [apiUrl])

  const loadOverview = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await fetchJson(`${apiUrl}/models/control-center`)
      setOverview({ ...EMPTY_OVERVIEW, ...data })
      setForm({
        backend: data.runtime?.backend || 'ollama',
        localModelDir: data.runtime?.local_model_dir || 'lora_model',
        launchModel: data.runtime?.launch_model || 'phi3:mini',
      })
    } catch (loadError) {
      setError(loadError.message || 'Could not load model control center')
    } finally {
      setLoading(false)
    }
  }

  const saveRuntime = async () => {
    setSaving(true)
    setError('')
    setMessage('')
    try {
      const data = await fetchJson(`${apiUrl}/settings/local-runtime`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backend: form.backend,
          local_model_dir: form.localModelDir,
          launch_model: form.launchModel,
        }),
      })
      setOverview({ ...EMPTY_OVERVIEW, ...data.overview })
      setMessage(data.summary || 'Local runtime settings saved.')
    } catch (saveError) {
      setError(saveError.message || 'Could not save local runtime settings')
    } finally {
      setSaving(false)
    }
  }

  const runCompression = async () => {
    setCompressing(true)
    setError('')
    setMessage('')
    try {
      const data = await fetchJson(`${apiUrl}/models/compress/launch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bits: 4 }),
      })
      setOverview({ ...EMPTY_OVERVIEW, ...data.overview })
      const mode = data.artifact?.source === 'mock' ? 'mock mode' : 'real pack'
      setMessage(`CompressX finished for ${data.overview.runtime.launch_model} in ${mode}.`)
    } catch (compressError) {
      setError(compressError.message || 'CompressX failed')
    } finally {
      setCompressing(false)
    }
  }

  if (loading) {
    return (
      <div className="panel-surface flex h-full items-center justify-center p-6 text-sm text-[var(--text-soft)]">
        <div className="flex items-center gap-3">
          <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
          Loading model control center…
        </div>
      </div>
    )
  }

  const runtimeCards = [
    {
      label: 'Backend',
      value: overview.runtime.backend,
      detail: overview.runtime.single_app_mode ? 'single-app route' : 'external runtime route',
      icon: Bot,
    },
    {
      label: 'Launch Model',
      value: overview.runtime.launch_model,
      detail: overview.runtime.resolved_launch_model || 'launch alias',
      icon: Sparkles,
    },
    {
      label: 'Cloud',
      value: overview.runtime.cloud_fallback,
      detail: overview.runtime.cloud_fallback === 'none' ? 'offline only' : 'fallback ready',
      icon: Layers3,
    },
  ]

  return (
    <div className="space-y-3">
      <div className="panel-surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="section-label">Model Control</p>
            <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
              Run the local adapter, point NEXUS at the right artifacts, and track whether the desktop app can realistically ship as one installer.
            </p>
          </div>
          <button onClick={() => void loadOverview()} className="meta-pill interactive mono text-[11px]">
            <RefreshCw size={12} />
            refresh
          </button>
        </div>

        <div className="mt-4 space-y-2">
          {runtimeCards.map(card => {
            const Icon = card.icon
            return (
              <div key={card.label} className="panel-muted px-3 py-2.5 flex items-center gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[rgba(255,255,255,0.04)] text-[var(--text-soft)]">
                  <Icon size={14} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="section-label text-[10px]">{card.label}</p>
                  <p className="mt-0.5 text-sm font-semibold tracking-[-0.03em] text-[var(--text-strong)] truncate">{card.value}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="panel-surface p-4">
        <div className="flex items-center justify-between gap-3">
          <p className="section-label">Runtime Switchboard</p>
          <span className="meta-pill mono text-[11px]">
            {overview.runtime.adapter_ready ? 'adapter ready' : 'adapter missing'}
          </span>
        </div>

        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap gap-1.5">
            {['ollama', 'adapter'].map(option => (
              <button
                key={option}
                onClick={() => setForm(prev => ({ ...prev, backend: option }))}
                className={`meta-pill interactive mono text-[11px] ${
                  form.backend === option
                    ? 'border-[rgba(0,240,255,0.35)] bg-[rgba(0,240,255,0.06)] text-[var(--accent)]'
                    : ''
                }`}
              >
                {option}
              </button>
            ))}
          </div>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--text-soft)]">Launch model alias</span>
            <input
              type="text"
              value={form.launchModel}
              onChange={event => setForm(prev => ({ ...prev, launchModel: event.target.value }))}
              placeholder="phi3:mini"
              className="w-full rounded-2xl border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-4 py-3 text-sm text-[var(--text)] outline-none transition-colors focus:border-[var(--accent)]"
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--text-soft)]">Adapter directory</span>
            <div className="flex gap-2">
              <input
                type="text"
                value={form.localModelDir}
                onChange={event => setForm(prev => ({ ...prev, localModelDir: event.target.value }))}
                placeholder="lora_model"
                className="min-w-0 flex-1 rounded-2xl border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-4 py-3 text-sm text-[var(--text)] outline-none transition-colors focus:border-[var(--accent)]"
              />
              {window?.nexusDesktop?.chooseDirectory && (
                <button
                  onClick={async () => {
                    const picked = await window.nexusDesktop.chooseDirectory()
                    if (picked) {
                      setForm(prev => ({ ...prev, localModelDir: picked }))
                    }
                  }}
                  className="meta-pill interactive shrink-0 mono text-[11px]"
                >
                  <FolderOpen size={12} />
                  choose
                </button>
              )}
            </div>
          </label>

          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm leading-6 text-[var(--text-soft)]">
              Adapter mode keeps the desktop experience self-contained. Ollama mode stays compatible with larger local models outside the app.
            </p>
            <button
              onClick={() => void saveRuntime()}
              disabled={saving || !form.launchModel.trim() || !form.localModelDir.trim()}
              className="inline-flex items-center gap-2 rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[#201913] transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              save runtime
            </button>
          </div>
        </div>
      </div>

      <div className="panel-surface p-4">
        <div className="flex items-center justify-between gap-3">
          <p className="section-label">Artifacts</p>
          <span className="meta-pill mono text-[11px]">{overview.artifacts.length} tracked</span>
        </div>
        <div className="mt-3 space-y-2">
          {overview.artifacts.map(item => (
            <ArtifactRow key={item.id} item={item} />
          ))}
        </div>
      </div>

      <div className="panel-surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="section-label">CompressX</p>
            <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
              Create a launch pack for the current local model alias. Real GPTQ output counts toward packaging estimates. Mock output does not.
            </p>
          </div>
          <button
            onClick={() => void runCompression()}
            disabled={compressing}
            className="inline-flex items-center gap-2 rounded-full bg-[var(--text-strong)] px-4 py-2 text-sm font-bold text-[#07090f] transition-all hover:bg-[var(--accent)] hover:shadow-[0_0_18px_rgba(0,240,255,0.35)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {compressing ? <Loader2 size={16} className="animate-spin" /> : <Package2 size={16} />}
            {compressing ? 'compressing' : 'compress launch pack'}
          </button>
        </div>

        <div className="mt-4 space-y-2">
          {overview.compressed_models.length === 0 && (
            <div className="panel-muted px-4 py-4 text-sm text-[var(--text-soft)]">
              No launch packs yet. Run CompressX to create the first packaging artifact.
            </div>
          )}

          {overview.compressed_models.map(item => (
            <div key={`${item.name}-${item.path}`} className="panel-muted px-3 py-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-[var(--text-strong)]">{item.name}</p>
                  <p className="mt-1 break-all text-xs leading-5 text-[var(--text-soft)]">{item.path}</p>
                </div>
                <span className={`meta-pill mono text-[11px] ${item.real_measurement ? 'text-[var(--success)]' : 'text-[var(--warning)]'}`}>
                  {item.real_measurement ? 'real' : item.source}
                </span>
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-[11px] mono text-[var(--text-soft)]">
                <span>{item.bits}-bit</span>
                <span>{item.size_mb} MB</span>
                <span>{item.ratio ? `${Number(item.ratio).toFixed(1)}x` : '--'}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel-surface p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="section-label">Single-App Budget</p>
            <p className="mt-2 text-sm leading-6 text-[var(--text-soft)]">
              This is the honest “can the impossible ship?” readout for the desktop app.
            </p>
          </div>
          <div className={`meta-pill mono text-[11px] ${readinessTone(overview.packaging.readiness)}`}>
            <Gauge size={12} />
            {overview.packaging.readiness}
          </div>
        </div>

        <div className="mt-4 grid gap-2 md:grid-cols-4">
          <BudgetCard label="Budget" value={`${overview.packaging.budget_mb} MB`} />
          <BudgetCard label="Reserve" value={`${overview.packaging.runtime_reserve_mb} MB`} />
          <BudgetCard label="Adapter" value={`${overview.packaging.adapter_pack_mb} MB`} />
          <BudgetCard label="Launch Pack" value={overview.packaging.selected_launch_pack_mb !== null ? `${overview.packaging.selected_launch_pack_mb} MB` : '--'} />
        </div>

        <div className="mt-4 rounded-[18px] border border-[var(--border)] bg-[rgba(255,255,255,0.03)] px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-semibold text-[var(--text-strong)]">Estimated total</p>
            <span className={`meta-pill mono text-[11px] ${
              overview.packaging.sub_gb_possible === null
                ? 'text-[var(--text-soft)]'
                : overview.packaging.sub_gb_possible
                  ? 'text-[var(--success)]'
                  : 'text-[var(--danger)]'
            }`}>
              {overview.packaging.estimated_total_mb !== null ? `${overview.packaging.estimated_total_mb} MB` : 'waiting'}
            </span>
          </div>
          <p className="mt-3 text-sm leading-6 text-[var(--text-soft)]">{overview.packaging.message}</p>
          {overview.packaging.selected_launch_pack_name && (
            <p className="mt-2 text-xs leading-5 text-[var(--text-soft)]">
              measured launch pack: {overview.packaging.selected_launch_pack_name}
            </p>
          )}
        </div>
      </div>

      {(error || message) && (
        <div className={`rounded-[18px] border px-4 py-3 text-sm ${
          error
            ? 'border-[rgba(255,143,136,0.22)] bg-[rgba(255,143,136,0.08)] text-[var(--danger)]'
            : 'border-[rgba(137,213,167,0.2)] bg-[rgba(137,213,167,0.08)] text-[var(--success)]'
        }`}>
          {error || message}
        </div>
      )}
    </div>
  )
}

function ArtifactRow({ item }) {
  return (
    <div className="panel-muted px-3 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <HardDrive size={14} className="text-[var(--text-soft)]" />
            <p className="text-sm font-semibold text-[var(--text-strong)]">{item.label}</p>
          </div>
          <p className="mt-2 text-xs leading-5 text-[var(--text-soft)]">{item.description}</p>
          <p className="mt-2 break-all text-[11px] mono text-[var(--text-soft)]">{item.path}</p>
        </div>
        <span className={`meta-pill mono text-[11px] ${item.exists ? 'text-[var(--success)]' : 'text-[var(--warning)]'}`}>
          {item.exists ? 'ready' : 'missing'}
        </span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-[11px] mono text-[var(--text-soft)]">
        <span>{item.kind}</span>
        <span>{item.size_mb} MB</span>
        <span>{item.entry_count} files</span>
      </div>
    </div>
  )
}

function BudgetCard({ label, value }) {
  return (
    <div className="panel-muted px-3 py-3">
      <p className="section-label text-[10px]">{label}</p>
      <p className="mt-2 text-lg font-semibold tracking-[-0.03em] text-[var(--text-strong)]">{value}</p>
    </div>
  )
}

function readinessTone(value) {
  if (value === 'ready') return 'text-[var(--success)]'
  if (value === 'over_budget') return 'text-[var(--danger)]'
  if (value === 'mock') return 'text-[var(--warning)]'
  return 'text-[var(--text-soft)]'
}
