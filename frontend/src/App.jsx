import { useEffect, useState } from 'react'
import './App.css'
import {
  API_BASE,
  checkBackendHealth,
  fetchModelMetadata,
  analyzeAudioFile,
  ApiError,
} from './services/apiClient'
import { Masthead } from './components/Masthead'
import { UploadZone } from './components/UploadZone'
import { InferenceReport } from './components/InferenceReport'
import { ModelInfoCard } from './components/ModelInfoCard'
import { BenchmarkPanel } from './components/BenchmarkPanel'
import { RoadmapSection } from './components/RoadmapSection'

export function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState({
    status: 'checking',
    online: false,
    modelLoaded: false,
    startupError: null,
  })
  const [modelInfo, setModelInfo] = useState(null)

  // Probe backend on mount and set polling interval
  useEffect(() => {
    let active = true

    async function probe() {
      const h = await checkBackendHealth()
      if (!active) return
      setHealth(h)

      if (h.online && h.modelLoaded) {
        try {
          const meta = await fetchModelMetadata()
          if (active) setModelInfo(meta)
        } catch {
          // Non-fatal if metadata fetch fails while health is online
        }
      }
    }

    probe()
    const interval = setInterval(probe, 15000)
    return () => {
      active = false
      clearInterval(interval)
    }
  }, [])

  function handleSelectFile(selectedFile, validationError = null) {
    if (validationError) {
      setFile(null)
      setError(validationError)
      return
    }
    setFile(selectedFile)
    setError('')
    setResult(null)
  }

  function handleClearFile() {
    setFile(null)
    setResult(null)
    setError('')
  }

  async function handleAnalyze() {
    if (!file || busy) return

    setBusy(true)
    setError('')
    setResult(null)

    try {
      const response = await analyzeAudioFile(file)
      setResult(response)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError(err.message || 'Analysis could not be completed. Check server logs.')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="editorial-app-shell">
      <div className="editorial-container">
        <Masthead health={health} modelInfo={modelInfo} apiBase={API_BASE} />

        <main className="editorial-main">
          <UploadZone
            file={file}
            onSelectFile={handleSelectFile}
            onClearFile={handleClearFile}
            onAnalyze={handleAnalyze}
            busy={busy}
            health={health}
            errorMessage={error}
            onDismissError={() => setError('')}
          />

          {result && <InferenceReport result={result} />}

          <ModelInfoCard modelInfo={modelInfo} />

          <BenchmarkPanel referencePerformance={modelInfo?.reference_performance} />

          <RoadmapSection />
        </main>

        <footer className="editorial-colophon">
          <div className="colophon-top">
            <div className="colophon-brand">
              <strong>AEGIS FORENSIC MEDIA LABORATORY</strong>
              <span>AI-Enabled Generative-Media Integrity & Security</span>
            </div>
            <div className="colophon-meta">
              <span>Primary Benchmark: ASVspoof 2019 LA Evaluation Set (71,237 Utterances)</span>
              <span>Model Architecture: 4L / 4H / 128D Spectral Audio Transformer</span>
            </div>
          </div>
          <div className="colophon-bottom">
            <span>Local Execution · Zero Remote Data Retention · Academic Research Prototype</span>
            <span>Not a certification instrument · Requires corroborating forensic evidence</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App
