const browserHost =
  typeof window !== 'undefined' && window.location?.hostname
    ? window.location.hostname
    : 'localhost'

export const API_URL =
  window?.nexusDesktop?.apiUrl || import.meta.env.VITE_API_URL || `http://${browserHost}:8000`

export const WS_URL = `${API_URL.replace('http', 'ws')}/ws`

export function desktopAvailable() {
  return Boolean(window?.nexusDesktop)
}

export async function getDesktopInfo() {
  if (!desktopAvailable() || typeof window.nexusDesktop.desktopInfo !== 'function') {
    return null
  }

  try {
    return await window.nexusDesktop.desktopInfo()
  } catch {
    return null
  }
}

export async function openExternal(url) {
  const target = String(url || '').trim()
  if (!target) return

  if (desktopAvailable() && typeof window.nexusDesktop.openExternal === 'function') {
    try {
      await window.nexusDesktop.openExternal(target)
      return
    } catch {
      // Fall back to the browser pathway below.
    }
  }

  window.open(target, '_blank', 'noopener,noreferrer')
}

export async function fetchJson(path, options = {}) {
  const target = path.startsWith('http://') || path.startsWith('https://') ? path : `${API_URL}${path}`
  const response = await fetch(target, options)

  let payload = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || `Request failed with status ${response.status}.`)
  }

  return payload
}

export function configuredModelAvailable(status) {
  const models = Array.isArray(status?.ollama_models) ? status.ollama_models : []
  if (!status?.ollama || models.length === 0) {
    return false
  }

  const configured = normalizeModelName(status?.model)
  if (!configured) {
    return models.length > 0
  }

  return models.some(modelName => {
    const value = normalizeModelName(modelName)
    return (
      value === configured ||
      value.startsWith(configured) ||
      configured.startsWith(value) ||
      value.split(':')[0] === configured.split(':')[0]
    )
  })
}

export function hasOperationalModelPath(status) {
  return Boolean(
    configuredModelAvailable(status) ||
      (status?.cloud_provider && status.cloud_provider !== 'none'),
  )
}

function normalizeModelName(value) {
  return String(value || '').trim().toLowerCase()
}
