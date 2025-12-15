import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// API base URL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)

  // Check API health on mount
  useEffect(() => {
    checkHealth()
    fetchStats()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/health`)
      setHealth(response.data)
    } catch (error) {
      console.error('Health check failed:', error)
      setHealth({ status: 'error', message: error.message })
    }
  }

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/stats`)
      setStats(response.data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    // Add user message
    const userMessage = { type: 'user', content: query }
    setMessages(prev => [...prev, userMessage])
    setLoading(true)

    try {
      const response = await axios.post(`${API_BASE}/api/query`, {
        query: query,
        top_k: 3,
        include_sources: true
      })

      const data = response.data

      // Add assistant message
      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        sources: data.sources,
        performance: data.performance,
        timestamp: data.timestamp
      }
      setMessages(prev => [...prev, assistantMessage])
      
      // Update stats
      fetchStats()
    } catch (error) {
      console.error('Query failed:', error)
      const errorMessage = {
        type: 'error',
        content: `Error: ${error.response?.data?.detail || error.message}`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
      setQuery('')
    }
  }

  const formatTime = (ms) => {
    if (ms < 1000) return `${Math.round(ms)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="container">
          <h1>🍬 Caramella RAG System</h1>
          <div className="header-info">
            {health && (
              <div className={`status ${health.status}`}>
                <span className="status-dot"></span>
                {health.status === 'healthy' ? (
                  <>
                    {health.gpu_enabled ? '⚡ GPU' : '🔧 CPU'} | 
                    {health.vector_db?.split('/').pop()}
                  </>
                ) : (
                  'API Offline'
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      {stats && stats.total_queries > 0 && (
        <div className="stats-bar">
          <div className="container">
            <div className="stat">
              <span className="stat-label">Queries</span>
              <span className="stat-value">{stats.total_queries}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Avg Latency</span>
              <span className="stat-value">{formatTime(stats.avg_latency_ms)}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Retrieval</span>
              <span className="stat-value">{formatTime(stats.avg_retrieval_ms)}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Generation</span>
              <span className="stat-value">{formatTime(stats.avg_generation_ms)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="main">
        <div className="container">
          {/* Chat Messages */}
          <div className="messages">
            {messages.length === 0 ? (
              <div className="welcome">
                <h2>Welcome to Caramella RAG! 👋</h2>
                <p>Ask me anything</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.type}`}>
                  <div className="message-content">
                    <div className="message-text">{msg.content}</div>
                    
                    {msg.performance && (
                      <div className="message-meta">
                        <span className="time-badge total">
                          ⚡ Total: {formatTime(msg.performance.total_ms)}
                        </span>
                        <span className="time-badge retrieval">
                          📚 Retrieval: {formatTime(msg.performance.retrieval_ms)}
                        </span>
                        <span className="time-badge generation">
                          🤖 Generation: {formatTime(msg.performance.generation_ms)}
                        </span>
                      </div>
                    )}

                    {msg.sources && msg.sources.length > 0 && (
                      <details className="sources">
                        <summary>📚 {msg.sources.length} Sources</summary>
                        {msg.sources.map((src, i) => (
                          <div key={i} className="source">
                            <div className="source-score">
                              Score: {(src.score * 100).toFixed(1)}%
                            </div>
                            <div className="source-content">
                              {src.content.slice(0, 200)}...
                            </div>
                            {src.metadata?.file && (
                              <div className="source-file">
                                📄 {src.metadata.file}
                              </div>
                            )}
                          </div>
                        ))}
                      </details>
                    )}
                  </div>
                </div>
              ))
            )}

            {loading && (
              <div className="message assistant">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              disabled={loading || health?.status !== 'healthy'}
              className="query-input"
            />
            <button 
              type="submit" 
              disabled={loading || !query.trim() || health?.status !== 'healthy'}
              className="submit-button"
            >
              {loading ? '⏳' : '🚀'}
            </button>
          </form>
        </div>
      </main>
    </div>
  )
}

export default App
