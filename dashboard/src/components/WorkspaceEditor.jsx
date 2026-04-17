import React, { useEffect, useState } from 'react'
import { ArrowUp, FileCode2, Folder, FolderOpen, HardDrive, RefreshCw, Save } from 'lucide-react'

export default function WorkspaceEditor({ apiUrl, defaultRoot, onWorkspaceChange, compact = false }) {
  const [rootInput, setRootInput] = useState(() => getStoredWorkspaceRoot())
  const [workspaceRoot, setWorkspaceRoot] = useState(null)
  const [rootName, setRootName] = useState('')
  const [currentPath, setCurrentPath] = useState('')
  const [parentPath, setParentPath] = useState(null)
  const [items, setItems] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [content, setContent] = useState('')
  const [originalContent, setOriginalContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingFile, setLoadingFile] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const preferredRoot = getStoredWorkspaceRoot() || defaultRoot
    if (!preferredRoot || workspaceRoot) return
    setRootInput(preferredRoot)
    openWorkspace(preferredRoot)
  }, [defaultRoot, workspaceRoot])

  const isDirty = selectedFile && content !== originalContent

  const openWorkspace = async (requestedRoot = rootInput, nextPath = '') => {
    const targetRoot = String(requestedRoot || '').trim()
    if (!targetRoot) {
      setError('Enter a folder path to open.')
      return
    }

    setLoading(true)
    setError('')
    try {
      const response = await fetch(
        `${apiUrl}/workspace?root=${encodeURIComponent(targetRoot)}&path=${encodeURIComponent(nextPath)}`,
      )
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Could not open folder')
      }

      setWorkspaceRoot(data.root)
      setRootInput(data.root)
      setRootName(data.root_name || '')
      setCurrentPath(data.current_path || '')
      setParentPath(data.parent_path ?? null)
      setItems(Array.isArray(data.items) ? data.items : [])
      persistWorkspaceRoot(data.root)
      onWorkspaceChange?.(data.root)
    } catch (loadError) {
      setError(loadError.message || 'Could not open folder')
    } finally {
      setLoading(false)
    }
  }

  const openFile = async filePath => {
    if (!workspaceRoot) return
    if (isDirty && !window.confirm('Discard unsaved changes in the current file?')) {
      return
    }

    setLoadingFile(true)
    setError('')
    try {
      const response = await fetch(
        `${apiUrl}/workspace/file?root=${encodeURIComponent(workspaceRoot)}&path=${encodeURIComponent(filePath)}`,
      )
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Could not load file')
      }

      setSelectedFile({ path: data.path, name: data.name })
      setContent(data.content || '')
      setOriginalContent(data.content || '')
    } catch (loadError) {
      setError(loadError.message || 'Could not load file')
    } finally {
      setLoadingFile(false)
    }
  }

  const saveFile = async () => {
    if (!workspaceRoot || !selectedFile) return

    setSaving(true)
    setError('')
    try {
      const response = await fetch(`${apiUrl}/workspace/file`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          root: workspaceRoot,
          path: selectedFile.path,
          content,
        }),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Could not save file')
      }

      setOriginalContent(content)
    } catch (saveError) {
      setError(saveError.message || 'Could not save file')
    } finally {
      setSaving(false)
    }
  }

  if (compact) {
    return (
      <div className="panel-surface flex h-full flex-col overflow-hidden">
        <div className="border-b border-[var(--border)] px-3 py-3">
          <div className="flex items-center gap-2">
            <FolderOpen size={15} className="text-[var(--accent)]" />
            <p className="section-label">Repository</p>
          </div>

          <div className="panel-muted mt-3 flex items-center gap-2 px-3 py-2.5">
            <HardDrive size={13} className="text-[var(--text-soft)]" />
            <input
              type="text"
              value={rootInput}
              onChange={event => setRootInput(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  openWorkspace(rootInput, '')
                }
              }}
              placeholder="Open folder"
              className="min-w-0 flex-1 bg-transparent text-sm text-[var(--text)] placeholder:text-[var(--text-soft)] focus:outline-none"
            />
          </div>

          <div className="mt-3 flex flex-wrap gap-1.5">
            <button onClick={() => openWorkspace(rootInput, '')} disabled={loading} className="meta-pill mono text-[11px]">
              open
            </button>
            {window?.nexusDesktop?.chooseDirectory && (
              <button
                onClick={async () => {
                  const picked = await window.nexusDesktop.chooseDirectory()
                  if (picked) {
                    setRootInput(picked)
                    openWorkspace(picked, '')
                  }
                }}
                disabled={loading}
                className="meta-pill mono text-[11px]"
              >
                choose
              </button>
            )}
            <button
              onClick={() => openWorkspace(workspaceRoot || rootInput, currentPath)}
              disabled={loading || !workspaceRoot}
              className="meta-pill mono text-[11px]"
            >
              <RefreshCw size={12} />
              refresh
            </button>
          </div>

          {error && (
            <div className="panel-muted mt-3 border-[rgba(255,143,136,0.2)] bg-[rgba(255,143,136,0.08)] px-3 py-2.5 text-sm text-[var(--danger)]">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-b border-[var(--border)] px-3 py-2.5">
          <div>
            <p className="text-xs text-[var(--text)]">{rootName || 'Files'}</p>
            <p className="mt-1 text-[11px] text-[var(--text-soft)]">{items.length} items</p>
          </div>
          {parentPath !== null && (
            <button onClick={() => openWorkspace(workspaceRoot, parentPath)} className="meta-pill mono text-[11px]">
              <ArrowUp size={12} />
              up
            </button>
          )}
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
          {!workspaceRoot && (
            <div className="flex h-full items-center justify-center px-4 text-center text-sm text-[var(--text-soft)]">
              Open a repo folder to browse files.
            </div>
          )}

          <div className="space-y-1.5">
            {workspaceRoot && items.map(item => (
              <button
                key={`${item.type}:${item.path}`}
                onClick={() => (item.type === 'directory' ? openWorkspace(workspaceRoot, item.path) : openFile(item.path))}
                className={`w-full rounded-[14px] border px-3 py-2.5 text-left transition-colors ${
                  selectedFile?.path === item.path
                    ? 'border-[var(--accent)] shadow-[0_0_15px_rgba(0,240,255,0.15)] bg-[rgba(0,240,255,0.08)]'
                    : 'border-[var(--border)] bg-[rgba(255,255,255,0.02)] hover:bg-[rgba(255,255,255,0.04)] hover:border-[var(--border-strong)]'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  {item.type === 'directory' ? (
                    <Folder size={15} className="shrink-0 text-[var(--warning)]" />
                  ) : (
                    <FileCode2 size={15} className="shrink-0 text-[var(--accent-2)]" />
                  )}
                  <span className="truncate text-sm text-[var(--text)]">{item.name}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {selectedFile && (
          <div className="border-t border-[var(--border)] px-3 py-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="truncate text-sm text-[var(--text-strong)]">{selectedFile.name}</p>
              <span className="meta-pill mono text-[10px]">{formatLines(content)} lines</span>
            </div>
            <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words rounded-[14px] border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-3 py-3 text-[11px] leading-5 text-[var(--text)]">
              {loadingFile ? 'Loading file…' : content.slice(0, 2400)}
            </pre>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="panel-surface flex h-full flex-col overflow-hidden">
      <div className="border-b border-[var(--border)] px-4 py-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
              <FolderOpen size={18} />
            </div>
            <div>
              <p className="section-label">Repository Files</p>
              <p className="mt-1 text-sm leading-relaxed text-[var(--text-soft)]">
                Open any local repo, navigate folders, and edit files in the same window.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => openWorkspace(rootInput, '')}
              disabled={loading}
              className="meta-pill mono text-[11px] transition-colors hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
            >
              open
            </button>
            {window?.nexusDesktop?.chooseDirectory && (
              <button
                onClick={async () => {
                  const picked = await window.nexusDesktop.chooseDirectory()
                  if (picked) {
                    setRootInput(picked)
                    openWorkspace(picked, '')
                  }
                }}
                disabled={loading}
                className="meta-pill mono text-[11px] transition-colors hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
              >
                choose folder
              </button>
            )}
            <button
              onClick={() => {
                if (defaultRoot) {
                  setRootInput(defaultRoot)
                  openWorkspace(defaultRoot, '')
                }
              }}
              disabled={!defaultRoot || loading}
              className="meta-pill mono text-[11px] transition-colors hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
            >
              repo root
            </button>
            <button
              onClick={() => openWorkspace(workspaceRoot || rootInput, currentPath)}
              disabled={loading || !workspaceRoot}
              className="meta-pill mono text-[11px] transition-colors hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
            >
              <RefreshCw size={12} />
              refresh
            </button>
            <button
              onClick={saveFile}
              disabled={!selectedFile || !isDirty || saving}
              className="meta-pill mono text-[11px] transition-colors hover:border-[var(--success)] hover:text-[var(--success)]"
            >
              <Save size={12} />
              {saving ? 'saving' : 'save'}
            </button>
          </div>
        </div>

        <div className="panel-muted mt-4 flex flex-wrap items-center gap-3 px-4 py-3">
          <HardDrive size={14} className="text-[var(--text-soft)]" />
          <input
            type="text"
            value={rootInput}
            onChange={event => setRootInput(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                openWorkspace(rootInput, '')
              }
            }}
            placeholder="Enter a folder path, for example D:\project"
            className="min-w-[220px] flex-1 bg-transparent text-sm text-[var(--text)] placeholder:text-[var(--text-soft)] focus:outline-none"
          />
          <span className="meta-pill mono text-[11px]">
            {workspaceRoot ? shortPath(workspaceRoot, 40) : 'no folder'}
          </span>
        </div>

        {error && (
          <div className="panel-muted mt-3 border-[rgba(255,143,136,0.2)] bg-[rgba(255,143,136,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
            {error}
          </div>
        )}
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[340px_minmax(0,1fr)]">
        <div className="border-b border-[var(--border)] xl:border-b-0 xl:border-r xl:border-[var(--border)]">
          <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
            <div>
              <p className="section-label">{rootName || 'Workspace'}</p>
              <p className="mt-1 text-sm text-[var(--text-soft)]">{items.length} items</p>
            </div>
            {parentPath !== null && (
              <button
                onClick={() => openWorkspace(workspaceRoot, parentPath)}
                className="meta-pill mono text-[11px]"
              >
                <ArrowUp size={12} />
                up
              </button>
            )}
          </div>

          <div className="max-h-full overflow-y-auto px-3 py-3">
            {!workspaceRoot && (
              <div className="flex min-h-[260px] items-center justify-center text-center text-[var(--text-soft)]">
                Open a folder path to browse and edit project files.
              </div>
            )}

            <div className="space-y-2">
              {workspaceRoot && items.map(item => (
                <button
                  key={`${item.type}:${item.path}`}
                  onClick={() => (item.type === 'directory' ? openWorkspace(workspaceRoot, item.path) : openFile(item.path))}
                  className={`w-full rounded-2xl border px-3 py-3 text-left transition-colors ${
                    selectedFile?.path === item.path
                      ? 'border-[var(--accent)] shadow-[0_0_15px_rgba(0,240,255,0.15)] bg-[rgba(0,240,255,0.08)]'
                      : 'border-[var(--border)] bg-[rgba(255,255,255,0.02)] hover:bg-[rgba(255,255,255,0.04)] hover:border-[var(--border-strong)]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {item.type === 'directory' ? (
                      <Folder size={16} className="shrink-0 text-[var(--warning)]" />
                    ) : (
                      <FileCode2 size={16} className="shrink-0 text-[var(--accent-2)]" />
                    )}
                    <span className="truncate text-sm text-[var(--text)]">{item.name}</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-[var(--text-soft)]">
                    <span className="mono truncate">{item.path || '.'}</span>
                    {item.type === 'file' && item.size !== null && <span className="mono shrink-0">{formatSize(item.size)}</span>}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex min-h-0 flex-col">
          <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
            <div>
              <p className="section-label">Editor</p>
              <p className="mt-1 text-sm text-[var(--text-soft)]">
                {selectedFile ? selectedFile.path : 'Select a file from the explorer'}
              </p>
            </div>
            {selectedFile && (
              <span className="meta-pill mono text-[11px]">
                {formatLines(content)} lines • {content.length} chars
              </span>
            )}
          </div>

          {!selectedFile ? (
            <div className="flex flex-1 items-center justify-center px-10 text-center text-[var(--text-soft)]">
              Choose a file to load it into the editor. Folder rows navigate, file rows open content.
            </div>
          ) : (
            <div className="flex-1 p-4">
              <textarea
                value={content}
                onChange={event => setContent(event.target.value)}
                disabled={loadingFile}
                spellCheck={false}
                className="h-full w-full resize-none rounded-[24px] border border-[var(--border)] bg-[rgba(17,19,24,0.8)] px-5 py-4 font-mono text-sm leading-7 text-[var(--text)] outline-none transition-colors focus:border-[var(--accent)] focus:bg-[rgba(0,240,255,0.02)] focus:shadow-[0_0_20px_rgba(0,240,255,0.1)_inset]"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function getStoredWorkspaceRoot() {
  try {
    return window.localStorage.getItem('nexus_workspace_root') || ''
  } catch {
    return ''
  }
}

function persistWorkspaceRoot(root) {
  try {
    window.localStorage.setItem('nexus_workspace_root', root)
  } catch {
    // Ignore storage failures
  }
}

function formatLines(value) {
  return String(value || '').split('\n').length
}

function formatSize(value) {
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${value} B`
}

function shortPath(value, maxLength = 44) {
  if (!value) return ''
  if (value.length <= maxLength) return value
  const normalized = value.replaceAll('\\', '/')
  const parts = normalized.split('/')
  if (parts.length <= 2) return value.slice(0, maxLength - 1) + '…'
  return `${parts.slice(0, 2).join('/')}/…/${parts[parts.length - 1]}`
}
