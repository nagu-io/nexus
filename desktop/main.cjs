const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

let mainWindow = null
let backendProcess = null

const backendPort = process.env.NEXUS_BACKEND_PORT || '8000'
const apiUrl = process.env.NEXUS_API_URL || `http://127.0.0.1:${backendPort}`
const projectRoot = path.join(__dirname, '..')

function isDev() {
  return !app.isPackaged || process.env.NEXUS_DESKTOP_DEV_URL
}

function dashboardEntry() {
  if (isDev()) {
    return process.env.NEXUS_DESKTOP_DEV_URL || 'http://127.0.0.1:5173'
  }
  return `file://${path.join(process.resourcesPath, 'dashboard', 'index.html')}`
}

function bundledBackendExecutable() {
  const executable = process.platform === 'win32' ? 'nexus-backend.exe' : 'nexus-backend'
  const candidate = path.join(process.resourcesPath, 'backend', executable)
  return fs.existsSync(candidate) ? candidate : null
}

function bundledAdapterPack() {
  const candidate = path.join(process.resourcesPath, 'model-packs', 'default-adapter')
  return fs.existsSync(candidate) ? candidate : null
}

function backendWorkingDirectory() {
  return app.isPackaged ? app.getPath('userData') : projectRoot
}

function backendEnvironment() {
  const env = {
    ...process.env,
    NEXUS_DESKTOP: '1',
    NEXUS_BACKEND_PORT: backendPort,
    NEXUS_ENV_PATH: path.join(app.getPath('userData'), 'nexus.env'),
    NEXUS_APP_ROOT: app.isPackaged ? process.resourcesPath : projectRoot,
  }

  const adapterPack = bundledAdapterPack()
  if (adapterPack && !env.NEXUS_LOCAL_MODEL_DIR) {
    env.NEXUS_LOCAL_MODEL_DIR = adapterPack
    if (!env.NEXUS_LOCAL_BACKEND) {
      env.NEXUS_LOCAL_BACKEND = 'adapter'
    }
  }

  return env
}

function backendCommand() {
  const bundled = app.isPackaged ? bundledBackendExecutable() : null
  if (bundled) {
    return { command: bundled, args: [] }
  }

  const python = process.env.NEXUS_PYTHON || (process.platform === 'win32' ? 'py' : 'python3')
  const args = process.platform === 'win32'
    ? ['-3.11', '-m', 'uvicorn', 'nexus.api:app', '--host', '127.0.0.1', '--port', backendPort]
    : ['-m', 'uvicorn', 'nexus.api:app', '--host', '127.0.0.1', '--port', backendPort]
  return { command: python, args }
}

function startBackend() {
  if (backendProcess || process.env.NEXUS_SKIP_BACKEND === '1') {
    return
  }

  const { command, args } = backendCommand()
  backendProcess = spawn(command, args, {
    cwd: backendWorkingDirectory(),
    env: backendEnvironment(),
    stdio: 'ignore',
    windowsHide: true,
  })

  backendProcess.on('exit', () => {
    backendProcess = null
  })
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#17191d',
    autoHideMenuBar: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'hidden',
    titleBarOverlay: process.platform === 'darwin'
      ? false
      : {
          color: '#17191d',
          symbolColor: '#f2eee8',
          height: 44,
        },
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  })

  const entry = dashboardEntry()
  if (entry.startsWith('file://')) {
    mainWindow.loadURL(entry)
  } else {
    mainWindow.loadURL(entry)
    if (process.env.NEXUS_OPEN_DEVTOOLS === '1') {
      mainWindow.webContents.openDevTools({ mode: 'detach' })
    }
  }
}

ipcMain.handle('nexus:pick-directory', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('nexus:desktop-info', async () => ({
  name: app.getName(),
  version: app.getVersion(),
  platform: process.platform,
  arch: process.arch,
  packaged: app.isPackaged,
  apiUrl,
  backendPort,
  backendRunning: Boolean(backendProcess),
}))

ipcMain.handle('nexus:open-external', async (_event, targetUrl) => {
  const value = String(targetUrl || '').trim()
  if (!/^https?:\/\//i.test(value)) {
    return { ok: false, error: 'Only http and https links are allowed.' }
  }

  await shell.openExternal(value)
  return { ok: true }
})

ipcMain.handle('nexus:window-action', async (_event, action) => {
  if (!mainWindow) {
    return null
  }

  switch (action) {
    case 'minimize':
      mainWindow.minimize()
      return { ok: true }
    case 'toggleMaximize':
      if (mainWindow.isMaximized()) {
        mainWindow.unmaximize()
      } else {
        mainWindow.maximize()
      }
      return { ok: true, maximized: mainWindow.isMaximized() }
    case 'close':
      mainWindow.close()
      return { ok: true }
    default:
      return { ok: false, error: `Unknown action: ${action}` }
  }
})

app.whenReady().then(() => {
  startBackend()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  if (backendProcess) {
    backendProcess.kill()
    backendProcess = null
  }
})
