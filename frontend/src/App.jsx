
import { useEffect, useState } from 'react'
import './App.css'

import {
  API_BASE,
  checkBackendHealth,
  fetchModelMetadata,
  analyzeAudioFile,
  analyzeVideoFile,
  ApiError,
} from './services/apiClient'

import { WorkspaceShell } from './components/WorkspaceShell'
import { UploadZone } from './components/UploadZone'
import { InferenceReport } from './components/InferenceReport'
import { VideoInferenceReport } from './components/VideoInferenceReport'
import { ModelInfoCard } from './components/ModelInfoCard'
import { BenchmarkPanel } from './components/BenchmarkPanel'

const VIDEO_EXTENSIONS = /\.(mp4|mov|mkv|avi|webm|m4v)$/i

function PageHeading({ eyebrow, title, description }) {
  return (
    <div className="workspace-page-heading">
      <div className="workspace-page-eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  )
}

function Overview({ health, modelInfo, onNavigate }) {
  const audioReady = health.status === 'online'

  return (
    <div className="workspace-page">
      <PageHeading
        eyebrow="FORENSIC MEDIA ANALYSIS / RESEARCH WORKSTATION"
        title="Evidence deserves more than an algorithm."
        description="AEGIS is an academic research prototype for examining potentially manipulated audio and video through machine-learning-based forensic analysis."
      />

      <div className="workspace-overview-status">
        <div className="workspace-overview-status-main">
          <span
            className={`workspace-status-dot ${
              audioReady ? 'online' : 'offline'
            }`}
          />
          <div>
            <div className="workspace-overview-status-label">
              AUDIO INFERENCE ENGINE
            </div>
            <div className="workspace-overview-status-value">
              {health.status === 'checking'
                ? 'Checking connection'
                : audioReady
                  ? 'Connected and model loaded'
                  : 'Currently unavailable'}
            </div>
          </div>
        </div>
        <div className="workspace-overview-api">
          <span>LOCAL API</span>
          <strong>{API_BASE}</strong>
        </div>
      </div>

      <div className="workspace-capability-heading">
        <span>01 / ANALYSIS CAPABILITIES</span>
        <span>AVAILABLE WORKFLOWS</span>
      </div>

      <div className="workspace-capability-grid">
        <article className="workspace-capability-card">
          <div className="workspace-card-number">01</div>
          <div className="workspace-card-category">AUDIO FORENSICS</div>
          <h2>Synthetic Voice Screening</h2>
          <p>
            Examine audio recordings using a spectral audio transformer.
            Review window-level spoof scores, aggregate decisions and
            technical evidence.
          </p>
          <button
            className="workspace-text-button"
            onClick={() => onNavigate('analysis')}
          >
            Start audio analysis <span>↗</span>
          </button>
        </article>

        <article className="workspace-capability-card">
          <div className="workspace-card-number">02</div>
          <div className="workspace-card-category">VIDEO FORENSICS</div>
          <h2>Video Manipulation Analysis</h2>
          <p>
            Examine video using face-focused frame sampling and
            temporal model analysis. Review the predicted manipulation
            score and frame-level evidence.
          </p>
          <button
            className="workspace-text-button"
            onClick={() => onNavigate('analysis')}
          >
            Start video analysis <span>↗</span>
          </button>
        </article>
      </div>

      <div className="workspace-overview-lower">
        <div className="workspace-overview-note">
          <div className="workspace-card-category">RESEARCH PRINCIPLE</div>
          <h2>Analysis is not proof.</h2>
          <p>
            Model scores are indicators produced under specific
            evaluation conditions. They are not calibrated probabilities
            or definitive evidence of authenticity or manipulation.
            Independent corroboration remains essential.
          </p>
        </div>

        <div className="workspace-overview-model">
          <div className="workspace-card-category">AUDIO MODEL</div>
          <h2>
            {modelInfo?.model?.name || 'Awaiting model metadata'}
          </h2>
          <div className="workspace-model-detail">
            {modelInfo?.model?.parameter_count
              ? `${modelInfo.model.parameter_count.toLocaleString()} parameters`
              : 'Model specifications unavailable'}
          </div>
          <button
            className="workspace-text-button"
            onClick={() => onNavigate('models')}
          >
            Explore model intelligence <span>↗</span>
          </button>
        </div>
      </div>

      <div className="workspace-overview-disclaimer">
        ACADEMIC RESEARCH PROTOTYPE · NOT A CERTIFICATION INSTRUMENT
      </div>
    </div>
  )
}

function SystemStatus({ health, modelInfo }) {
  const runtime = modelInfo?.runtime

  const rows = [
    {
      label: 'Backend connection',
      value:
        health.status === 'checking'
          ? 'Checking'
          : health.online
            ? 'Connected'
            : 'Unavailable',
    },
    {
      label: 'Audio model',
      value: health.modelLoaded ? 'Loaded' : 'Not loaded',
    },
    {
      label: 'Metadata service',
      value: modelInfo ? 'Available' : 'Unavailable',
    },
    {
      label: 'Inference device',
      value: runtime?.device || 'Not reported',
    },
    {
      label: 'Python runtime',
      value: runtime?.python || 'Not reported',
    },
    {
      label: 'PyTorch',
      value: runtime?.torch || 'Not reported',
    },
  ]

  return (
    <div className="workspace-page">
      <PageHeading
        eyebrow="INFRASTRUCTURE / DIAGNOSTICS"
        title="System status"
        description="A transparent view of the services and runtime information currently reported by the AEGIS backend."
      />

      <div className="workspace-system-banner">
        <div>
          <div className="workspace-card-category">
            AUDIO BACKEND STATUS
          </div>
          <h2>
            {health.status === 'checking'
              ? 'Checking connection'
              : health.online
                ? 'Backend connected'
                : 'Backend unavailable'}
          </h2>
        </div>
        <span
          className={`workspace-status-dot ${
            health.online ? 'online' : 'offline'
          }`}
        />
      </div>

      <div className="workspace-system-table">
        {rows.map((row) => (
          <div className="workspace-system-row" key={row.label}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
          </div>
        ))}
      </div>

      <div className="workspace-system-endpoint">
        <div className="workspace-card-category">API ENDPOINT</div>
        <code>{API_BASE}</code>
      </div>

      {health.startupError && (
        <div className="workspace-system-error">
          <div className="workspace-card-category">STARTUP ERROR</div>
          <p>{health.startupError}</p>
        </div>
      )}

      <div className="workspace-system-note">
        <strong>Monitoring limitation</strong>
        <p>
          The current health endpoint reports the audio inference
          service. It does not independently report video-engine
          readiness. Video inference is available through its separate
          analysis endpoint.
        </p>
      </div>
    </div>
  )
}

export function App() {
  const [activeView, setActiveView] = useState('overview')
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

  useEffect(() => {
    let active = true

    async function probe() {
      const currentHealth = await checkBackendHealth()

      if (!active) return

      setHealth(currentHealth)

      if (currentHealth.online && currentHealth.modelLoaded) {
        try {
          const metadata = await fetchModelMetadata()

          if (active) {
            setModelInfo(metadata)
          }
        } catch {
          // Metadata failure does not invalidate backend connectivity.
        }
      } else if (active) {
        setModelInfo(null)
      }
    }

    probe()

    const interval = setInterval(probe, 15000)

    return () => {
      active = false
      clearInterval(interval)
    }
  }, [])

  function handleNavigate(view) {
    setActiveView(view)
    setError('')
  }

  function handleSelectFile(selectedFile, validationError = null) {
    if (validationError) {
      setFile(null)
      setResult(null)
      setError(validationError)
      return
    }

    setFile(selectedFile)
    setResult(null)
    setError('')
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
      const isVideo = VIDEO_EXTENSIONS.test(file.name)

      const response = isVideo
        ? await analyzeVideoFile(file)
        : await analyzeAudioFile(file)

      setResult(response)
      setActiveView('analysis')
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError(
          err.message || 'Analysis could not be completed. Check server logs.'
        )
      }
    } finally {
      setBusy(false)
    }
  }

  function renderAnalysis() {
    return (
      <div className="workspace-page">
        <PageHeading
          eyebrow="EVIDENCE INGESTION / INFERENCE"
          title="New analysis"
          description="Submit an audio or video recording for local forensic screening. The resulting report includes model outputs, technical metadata and relevant limitations."
        />

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

        {result && (
          VIDEO_EXTENSIONS.test(file?.name || '')
            ? <VideoInferenceReport result={result} />
            : <InferenceReport result={result} />
        )}
      </div>
    )
  }

  function renderModels() {
    return (
      <div className="workspace-page">
        <PageHeading
          eyebrow="MODEL REGISTRY / EVALUATION"
          title="Model intelligence"
          description="Inspect the audio model architecture, inference policy and benchmark evidence reported by the backend."
        />

        <ModelInfoCard modelInfo={modelInfo} />

        <BenchmarkPanel
          referencePerformance={modelInfo?.reference_performance}
        />

        <div className="workspace-model-caveat">
          <strong>Interpretation matters.</strong>
          <p>
            The benchmark results describe performance on the ASVspoof
            2019 LA evaluation set. They do not establish performance on
            arbitrary real-world recordings. Model scores are not
            calibrated probabilities.
          </p>
        </div>
      </div>
    )
  }

  return (
    <WorkspaceShell
      activeView={activeView}
      onNavigate={handleNavigate}
      health={health}
    >
      {activeView === 'overview' && (
        <Overview
          health={health}
          modelInfo={modelInfo}
          onNavigate={handleNavigate}
        />
      )}

      {activeView === 'analysis' && renderAnalysis()}

      {activeView === 'models' && renderModels()}

      {activeView === 'system' && (
        <SystemStatus health={health} modelInfo={modelInfo} />
      )}
    </WorkspaceShell>
  )
}

export default App
