import React, { useCallback, useState } from 'react'
import AmbientBackground from './components/AmbientBackground.jsx'
import { fetchJson, getDesktopInfo, hasOperationalModelPath } from './lib/runtime.js'
import BuildScreen from './screens/BuildScreen.jsx'
import LoadingScreen from './screens/LoadingScreen.jsx'
import SetupScreen from './screens/SetupScreen.jsx'

const INITIAL_BOOTSTRAP = {
  status: null,
  desktopInfo: null,
  error: '',
  checkedAt: null,
  checking: false,
}

export default function App() {
  const [screen, setScreen] = useState('loading')
  const [bootstrap, setBootstrap] = useState(INITIAL_BOOTSTRAP)

  const runBootstrap = useCallback(async () => {
    setBootstrap(prev => ({ ...prev, checking: true, error: '' }))

    const desktopInfoPromise = getDesktopInfo()
    let status = null
    let latestError = ''

    for (let attempt = 0; attempt < 4; attempt += 1) {
      try {
        status = await fetchJson('/status')
        latestError = ''
        break
      } catch (error) {
        latestError = error?.message || 'NEXUS backend did not respond.'
        if (attempt < 3) {
          await wait(550 * (attempt + 1))
        }
      }
    }

    const desktopInfo = await desktopInfoPromise

    const nextBootstrap = {
      status,
      desktopInfo,
      error: latestError,
      checkedAt: Date.now(),
      checking: false,
    }

    setBootstrap(nextBootstrap)

    if (status?.online && hasOperationalModelPath(status)) {
      setScreen('build')
    } else {
      setScreen('setup')
    }
  }, [])

  return (
    <>
      <AmbientBackground />
      <div className="relative z-10 h-full">
        {screen === 'loading' && <LoadingScreen onReady={runBootstrap} />}
        {screen === 'setup' && (
          <SetupScreen
            status={bootstrap.status}
            desktopInfo={bootstrap.desktopInfo}
            checking={bootstrap.checking}
            error={bootstrap.error}
            checkedAt={bootstrap.checkedAt}
            onRetry={runBootstrap}
            onComplete={() => setScreen('build')}
          />
        )}
        {screen === 'build' && (
          <BuildScreen
            initialStatus={bootstrap.status}
            desktopInfo={bootstrap.desktopInfo}
          />
        )}
      </div>
    </>
  )
}

function wait(ms) {
  return new Promise(resolve => {
    window.setTimeout(resolve, ms)
  })
}
