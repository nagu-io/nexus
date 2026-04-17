const { contextBridge, ipcRenderer } = require('electron')

const apiUrl = process.env.NEXUS_API_URL || `http://127.0.0.1:${process.env.NEXUS_BACKEND_PORT || '8000'}`

contextBridge.exposeInMainWorld('nexusDesktop', {
  apiUrl,
  desktopInfo: () => ipcRenderer.invoke('nexus:desktop-info'),
  chooseDirectory: () => ipcRenderer.invoke('nexus:pick-directory'),
  openExternal: url => ipcRenderer.invoke('nexus:open-external', url),
  windowControls: {
    minimize: () => ipcRenderer.invoke('nexus:window-action', 'minimize'),
    toggleMaximize: () => ipcRenderer.invoke('nexus:window-action', 'toggleMaximize'),
    close: () => ipcRenderer.invoke('nexus:window-action', 'close'),
  },
})
