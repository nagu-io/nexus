import { useEffect, useRef, useState, useCallback } from 'react'

/**
 * Reusable WebSocket hook for NEXUS dashboard.
 * Auto-reconnects with exponential backoff.
 *
 * Usage:
 *   const { events, send, connected, connectionCount } = useWebSocket(wsUrl)
 */
export default function useWebSocket(url) {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const [connectionCount, setConnectionCount] = useState(0)
  const wsRef = useRef(null)
  const retryRef = useRef(0)
  const maxEvents = 500

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      retryRef.current = 0
      // Send ping to get connection count
      ws.send(JSON.stringify({ type: 'ping' }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'pong') {
          setConnectionCount(data.connections || 0)
          return
        }
        setEvents(prev => {
          const next = [...prev, { ...data, _ts: Date.now() }]
          return next.length > maxEvents ? next.slice(-maxEvents) : next
        })
      } catch {
        // Ignore non-JSON messages
      }
    }

    ws.onclose = () => {
      setConnected(false)
      // Exponential backoff: 1s, 2s, 4s, 8s, max 30s
      const delay = Math.min(1000 * Math.pow(2, retryRef.current), 30000)
      retryRef.current += 1
      setTimeout(connect, delay)
    }

    ws.onerror = () => ws.close()
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }, [])

  const clearEvents = useCallback(() => setEvents([]), [])

  return { events, send, connected, connectionCount, clearEvents }
}
