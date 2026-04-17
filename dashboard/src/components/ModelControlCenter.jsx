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
      setError(loadError.message || 'Could not load')
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
      setMessage(data.summary || 'Saved')
    } catch (saveError) {
      setError(saveError.message || 'Failed to save')
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
      const mode = data.artifact?.source === 'mock' ? 'mock' : 'real'
      setMessage(`Done (${mode})`)
    } catch (compressError) {
      setError(compressError.message || 'CompressX failed')
    } finally {
      setCompressing(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-[var(--text-soft)]">
        <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
      </div>
    )
  }

  const runtimeCards = [
    {
      label: 'Backend',
      value: overview.runtime.backend,
      icon: Bot,
    },
    {
      label: 'Launch Model',
      value: overview.runtime.launch_model,
      icon: Sparkles,
    },
    {
      label: 'Cloud',
      value: overview.runtime.cloud_fallback,
      icon: Layers3,
    },
  ]

  return (
    <div className="space-y-4 pb-4">
      <div className="px-3">
        <div className="flex items-center justify-between pb-2">
          <p className="section-label">Model Status</p>
          <button onClick={() => void loadOverview()} className="text-[var(--text-soft)] hover:text-[var(--text)]">
            <RefreshCw size={12} />
          </button>
        </div>
        <div className="space-y-1.5">
          {runtimeCards.map(card => {
            const Icon = card.icon
            return (
              <div key={card.label} className="panel-muted px-3 py-2 flex items-center gap-3">
                <Icon size={12} className="text-[var(--text-soft)] shrink-0" />
                <div className="min-w-0 flex-1 flex justify-between items-center">
                  <span className="text-[10px] text-[var(--text-soft)]">{card.label}</span>
                  <span className="text-xs font-semibold tracking-[-0.03em] text-[var(--text-strong)] truncate">
                    {card.value}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="border-t border-[var(--border)] pt-4 px-3">
        <div className="flex items-center justify-between pb-2">
          <p className="section-label">Switchboard</p>
          <span className={`h-1.5 w-1.5 rounded-full ${overview.runtime.adapter_ready ? 'bg-[var(--success)]' : 'bg-[var(--danger)]'}`} />
        </div>

        <div className="space-y-3">
          <div className="flex gap-1">
            {['ollama', 'adapter'].map(option => (
              <button
                key={option}
                onClick={() => setForm(prev => ({ ...prev, backend: option }))}
                className={`flex-1 rounded-full py-1 text-[10px] mono uppercase tracking-wider border transition-colors ${
                  form.backend === option
                    ? 'border-[var(--accent)] bg-[rgba(0,240,255,0.06)] text-[var(--accent)]'
                    : 'border-transparent text-[var(--text-soft)] hover:bg-[rgba(255,255,255,0.03)]'
                }`}
              >
                {option}
              </button>
            ))}
          </div>

          <label className="block">
            <span className="mb-1.5 block text-[10px] text-[var(--text-soft)] uppercase tracking-wider">Launch Alias</span>
            <input
              type="text"
              value={form.launchModel}
              onChange={event => setForm(prev => ({ ...prev, launchModel: event.target.value }))}
              placeholder="phi3:mini"
              className="w-full rounded-xl border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-3 py-2 text-xs text-[var(--text)] outline-none focus:border-[var(--accent)]"
            />
          </label>

          <label className="block">
            <span className="mb-1.5 block text-[10px] text-[var(--text-soft)] uppercase tracking-wider">Adapter Path</span>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={form.localModelDir}
                onChange={event => setForm(prev => ({ ...prev, localModelDir: event.target.value }))}
                placeholder="lora_model"
                className="min-w-0 flex-1 rounded-xl border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-3 py-2 text-xs text-[var(--text)] outline-none focus:border-[var(--accent)]"
              />
              {window?.nexusDesktop?.chooseDirectory && (
                <button
                  onClick={async () => {
                    const picked = await window.nexusDesktop.chooseDirectory()
                    if (picked) {
                      setForm(prev => ({ ...prev, localModelDir: picked }))
                    }
                  }}
                  className="rounded-xl bg-[rgba(255,255,255,0.04)] px-2.5 text-[var(--text-soft)] hover:text-[var(--text)] border border-[var(--border)] hover:bg-[rgba(255,255,255,0.08)]"
                >
                  <FolderOpen size={12} />
                </button>
              )}
            </div>
          </label>

          <button
            onClick={() => void saveRuntime()}
            disabled={saving || !form.launchModel.trim() || !form.localModelDir.trim()}
            className="w-full inline-flex justify-center items-center gap-2 rounded-xl bg-[var(--text-strong)] py-2 text-xs font-bold text-[#07090f] transition-all hover:bg-[var(--accent)] hover:shadow-[0_0_15px_rgba(0,240,255,0.3)] disabled:opacity-40"
          >
            {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
            Save
          </button>
        </div>
      </div>

      {overview.artifacts.length > 0 && (
        <div className="border-t border-[var(--border)] pt-4 px-3">
          <p className="section-label pb-2">Artifacts ({overview.artifacts.length})</p>
          <div className="space-y-1.5">
            {overview.artifacts.map(item => (
              <ArtifactRow key={item.id} item={item} />
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-[var(--border)] pt-4 px-3">
        <div className="flex items-center justify-between pb-2">
          <p className="section-label">CompressX</p>
          <button
            onClick={() => void runCompression()}
            disabled={compressing}
            className="text-[var(--text-strong)] hover:text-[var(--accent)] disabled:opacity-40"
          >
            {compressing ? <Loader2 size={12} className="animate-spin" /> : <Package2 size={12} />}
          </button>
        </div>

        <div className="space-y-1.5">
          {overview.compressed_models.length === 0 && (
            <p className="text-xs text-[var(--text-soft)]">No launch packs yet.</p>
          )}
          {overview.compressed_models.map(item => (
            <div key={`${item.name}-${item.path}`} className="panel-muted px-3 py-2 flex justify-between items-center">
              <span className="text-xs text-[var(--text-strong)] truncate mr-2">{item.name}</span>
              <span className="text-[10px] mono text-[var(--text-soft)] whitespace-nowrap">{item.size_mb} MB</span>
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-[var(--border)] pt-4 px-3">
        <div className="flex items-center justify-between pb-2">
          <p className="section-label">Budget constraints</p>
          <div className={`flex items-center gap-1 ${readinessTone(overview.packaging.readiness)}`}>
            <Gauge size={10} />
            <span className="text-[10px] uppercase tracking-wider">{overview.packaging.readiness}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-1.5">
          <BudgetCard label="Cap" value={overview.packaging.budget_mb} />
          <BudgetCard label="Reserve" value={overview.packaging.runtime_reserve_mb} />
          <BudgetCard label="Adapter" value={overview.packaging.adapter_pack_mb} />
          <BudgetCard label="LLM" value={overview.packaging.selected_launch_pack_mb ?? '--'} />
        </div>

        <div className="mt-2 rounded-xl flex items-center justify-between px-3 py-2 border border-[rgba(255,255,255,0.06)] bg-[rgba(255,255,255,0.02)]">
          <span className="text-xs text-[var(--text-soft)]">Total</span>
          <span className={`text-xs font-semibold ${
            overview.packaging.sub_gb_possible === null
              ? 'text-[var(--text-soft)]'
              : overview.packaging.sub_gb_possible ? 'text-[var(--success)]' : 'text-[var(--danger)]'
          }`}>
            {overview.packaging.estimated_total_mb ?? '...'} MB
          </span>
        </div>
      </div>

      {(error || message) && (
        <div className="px-3">
          <div className={`rounded-xl px-3 py-2 text-[11px] ${
            error
              ? 'bg-[rgba(255,143,136,0.08)] text-[var(--danger)] border border-[rgba(255,143,136,0.15)]'
              : 'bg-[rgba(137,213,167,0.08)] text-[var(--success)] border border-[rgba(137,213,167,0.15)]'
          }`}>
            {error || message}
          </div>
        </div>
      )}
    </div>
  )
}

function ArtifactRow({ item }) {
  return (
    <div className="panel-muted px-3 py-2 flex items-center justify-between gap-2">
      <div className="flex items-center gap-2 min-w-0">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${item.exists ? 'bg-[var(--success)]' : 'bg-[var(--warning)]'}`} />
        <span className="text-xs text-[var(--text)] truncate">{item.label}</span>
      </div>
      <span className="text-[10px] mono text-[var(--text-soft)] whitespace-nowrap">{item.size_mb} MB</span>
    </div>
  )
}

function BudgetCard({ label, value }) {
  return (
    <div className="panel-muted px-3 py-2">
      <p className="text-[10px] text-[var(--text-soft)] mb-0.5">{label}</p>
      <p className="text-xs font-medium text-[var(--text-strong)]">{value}</p>
    </div>
  )
}

function readinessTone(value) {
  if (value === 'ready') return 'text-[var(--success)]'
  if (value === 'over_budget') return 'text-[var(--danger)]'
  if (value === 'mock') return 'text-[var(--warning)]'
  return 'text-[var(--text-soft)]'
}
