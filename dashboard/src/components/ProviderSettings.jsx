import React, { useEffect, useState } from 'react'
import { KeyRound, Loader2, Save, Trash2, X } from 'lucide-react'

const EMPTY_STATE = {
  active_provider: 'none',
  active_model: '',
  openrouter: {
    configured: false,
    masked_key: '',
    model: 'openrouter/auto',
    base_url: 'https://openrouter.ai/api/v1',
  },
  anthropic: {
    configured: false,
  },
}

export default function ProviderSettings({ apiUrl, isOpen, onClose, onSaved }) {
  const [settings, setSettings] = useState(EMPTY_STATE)
  const [apiKey, setApiKey] = useState('')
  const [model, setModel] = useState('openrouter/auto')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (!isOpen) return
    void loadSettings()
  }, [apiUrl, isOpen])

  const loadSettings = async () => {
    setLoading(true)
    setError('')
    setMessage('')
    try {
      const response = await fetch(`${apiUrl}/settings/providers`)
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load provider settings')
      }
      setSettings({ ...EMPTY_STATE, ...data })
      setModel(data.openrouter?.model || 'openrouter/auto')
      setApiKey('')
    } catch (loadError) {
      setError(loadError.message || 'Could not load provider settings')
    } finally {
      setLoading(false)
    }
  }

  const saveSettings = async ({ clearApiKey = false } = {}) => {
    setSaving(true)
    setError('')
    setMessage('')
    try {
      const response = await fetch(`${apiUrl}/settings/providers/openrouter`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: clearApiKey ? '' : apiKey,
          model,
          clear_api_key: clearApiKey,
        }),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Could not save settings')
      }
      setSettings({ ...EMPTY_STATE, ...data })
      setApiKey('')
      setModel(data.openrouter?.model || model)
      setMessage(data.summary || 'OpenRouter settings saved.')
      onSaved?.()
    } catch (saveError) {
      setError(saveError.message || 'Could not save settings')
    } finally {
      setSaving(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(9,10,13,0.7)] px-4 py-6 backdrop-blur-sm">
      <div className="panel-surface w-full max-w-[40rem] overflow-hidden">
        <div className="flex items-start justify-between gap-4 border-b border-[var(--border)] px-5 py-4">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
              <KeyRound size={18} />
            </div>
            <div>
              <p className="section-label">Provider Settings</p>
              <h2 className="mt-1 text-xl font-semibold tracking-[-0.03em] text-[var(--text-strong)]">
                OpenRouter
              </h2>
              <p className="mt-2 max-w-[30rem] text-sm leading-6 text-[var(--text-soft)]">
                Save a local OpenRouter key for cloud fallback. When a key is configured, NEXUS will prefer OpenRouter over Anthropic for cloud routing.
              </p>
            </div>
          </div>

          <button className="desktop-control" onClick={onClose} title="Close">
            <X size={14} />
          </button>
        </div>

        <div className="space-y-4 px-5 py-5">
          <div className="grid gap-3 md:grid-cols-3">
            <StatusCard label="Active" value={settings.active_provider || 'none'} />
            <StatusCard label="Cloud Model" value={settings.active_model || '--'} />
            <StatusCard label="Anthropic" value={settings.anthropic?.configured ? 'configured' : 'off'} />
          </div>

          {loading ? (
            <div className="panel-muted flex items-center gap-3 px-4 py-4 text-sm text-[var(--text-soft)]">
              <Loader2 size={16} className="animate-spin text-[var(--accent-2)]" />
              Loading provider settings…
            </div>
          ) : (
            <>
              <div className="panel-muted px-4 py-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="section-label">OpenRouter</span>
                  <span className="meta-pill mono text-[11px]">
                    {settings.openrouter?.configured ? 'configured' : 'not set'}
                  </span>
                  {settings.openrouter?.masked_key && (
                    <span className="meta-pill mono text-[11px]">{settings.openrouter.masked_key}</span>
                  )}
                </div>

                <div className="mt-4 space-y-4">
                  <label className="block">
                    <span className="mb-2 block text-sm text-[var(--text-soft)]">Model</span>
                    <input
                      type="text"
                      value={model}
                      onChange={event => setModel(event.target.value)}
                      placeholder="openrouter/auto"
                      className="w-full rounded-2xl border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-4 py-3 text-sm text-[var(--text)] outline-none transition-colors focus:border-[var(--accent)] focus:bg-[rgba(0,240,255,0.02)] focus:shadow-[0_0_20px_rgba(0,240,255,0.1)_inset]"
                    />
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-sm text-[var(--text-soft)]">API Key</span>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={event => setApiKey(event.target.value)}
                      placeholder={settings.openrouter?.configured ? 'Leave blank to keep existing key' : 'sk-or-v1-...'}
                      className="w-full rounded-2xl border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-4 py-3 text-sm text-[var(--text)] outline-none transition-colors focus:border-[var(--accent)] focus:bg-[rgba(0,240,255,0.02)] focus:shadow-[0_0_20px_rgba(0,240,255,0.1)_inset]"
                    />
                  </label>

                  <p className="text-sm leading-6 text-[var(--text-soft)]">
                    The key is stored in your local <code className="mono text-[var(--text)]">.env</code> file. Leave the field blank when you only want to update the model.
                  </p>
                </div>
              </div>

              {error && (
                <div className="panel-muted border-[rgba(255,143,136,0.2)] bg-[rgba(255,143,136,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
                  {error}
                </div>
              )}

              {message && (
                <div className="panel-muted border-[rgba(137,213,167,0.2)] bg-[rgba(137,213,167,0.08)] px-4 py-3 text-sm text-[var(--success)]">
                  {message}
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--border)] px-5 py-4">
          <button
            onClick={() => saveSettings({ clearApiKey: true })}
            disabled={saving || loading || !settings.openrouter?.configured}
            className="meta-pill mono text-[11px] transition-colors hover:border-[rgba(255,143,136,0.32)] hover:text-[var(--danger)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Trash2 size={12} />
            clear key
          </button>

          <div className="flex items-center gap-2">
            <button className="meta-pill mono text-[11px]" onClick={onClose}>
              close
            </button>
            <button
              onClick={() => saveSettings()}
              disabled={saving || loading || !model.trim()}
              className="inline-flex items-center gap-2 rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[#201913] transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              save
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatusCard({ label, value }) {
  return (
    <div className="panel-muted px-4 py-3">
      <p className="section-label text-[10px]">{label}</p>
      <p className="mt-2 break-all text-sm text-[var(--text)]">{value}</p>
    </div>
  )
}
