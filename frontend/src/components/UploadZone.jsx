import { useRef, useState, useEffect } from 'react'

const ALLOWED_EXTENSIONS = ['wav', 'flac', 'mp3', 'ogg', 'm4a', 'aiff', 'aif']
const MAX_BYTES = 50 * 1024 * 1024 // 50 MiB

export function UploadZone({
  file,
  onSelectFile,
  onClearFile,
  onAnalyze,
  busy,
  health,
  errorMessage,
  onDismissError,
}) {
  const fileInputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [elapsedSec, setElapsedSec] = useState(0)

  useEffect(() => {
    if (!busy) return
    const startTime = Date.now()
    const timer = setInterval(() => {
      setElapsedSec((Date.now() - startTime) / 1000)
    }, 100)
    return () => clearInterval(timer)
  }, [busy])

  function handleFileValidation(candidate) {
    if (!candidate) return

    const ext = candidate.name.split('.').pop()?.toLowerCase() || ''
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      onSelectFile(
        null,
        `Unsupported audio format (.${ext || 'unknown'}). Supported formats: ${ALLOWED_EXTENSIONS.join(', ').toUpperCase()}`
      )
      return
    }

    if (candidate.size === 0) {
      onSelectFile(null, 'Selected file is empty (0 bytes).')
      return
    }

    if (candidate.size > MAX_BYTES) {
      onSelectFile(
        null,
        `File size (${(candidate.size / (1024 * 1024)).toFixed(1)} MiB) exceeds the 50 MiB upload limit.`
      )
      return
    }

    onSelectFile(candidate, null)
  }

  function handleDragOver(e) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  function handleDragLeave(e) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  function handleDrop(e) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const dropped = e.dataTransfer.files?.[0]
    handleFileValidation(dropped)
  }

  function handleTriggerAnalyze() {
    setElapsedSec(0)
    onAnalyze()
  }

  const isOnline = health.status === 'online'
  const canAnalyze = Boolean(file) && !busy && isOnline

  return (
    <section className="editorial-section" id="section-ingestion" aria-labelledby="upload-heading">
      <div className="section-header">
        <span className="section-number">01</span>
        <div className="section-header-text">
          <span className="section-kicker">EVIDENCE INGESTION & SCREENING</span>
          <h2 id="upload-heading">Submit Audio Recording for Analysis</h2>
          <p className="section-description">
            Provide a speech recording to screen for synthetic manipulation, voice conversion, or acoustic neural vocoder artifacts.
            Audio is decoded to 16 kHz mono float32 and evaluated across 2.56-second sliding spectral windows.
          </p>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        className="visually-hidden"
        accept=".wav,.flac,.mp3,.ogg,.m4a,.aiff,.aif,audio/*"
        onChange={(e) => handleFileValidation(e.target.files?.[0])}
        tabIndex={-1}
      />

      <div
        className={`dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !busy && fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !busy) {
            e.preventDefault()
            fileInputRef.current?.click()
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload audio evidence: drag file here or click to browse"
      >
        <div className="dropzone-inner">
          <div className="dropzone-icon">
            {file ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
                <path d="M12 3v13M17 8l-5-5-5 5" />
                <path d="M20 21H4a2 2 0 0 1-2-2v-5" />
              </svg>
            )}
          </div>

          {file ? (
            <div className="file-preview-card">
              <span className="file-preview-label">SELECTED EVIDENCE FILE</span>
              <strong className="drop-title">{file.name}</strong>
              <div className="file-meta-row">
                <span>{(file.size / (1024 * 1024)).toFixed(2)} MiB</span>
                <span className="meta-separator">·</span>
                <span>{file.name.split('.').pop()?.toUpperCase()} audio</span>
                <span className="meta-separator">·</span>
                <span className="status-badge-ready">Ready for analysis</span>
              </div>
            </div>
          ) : (
            <div className="drop-prompt">
              <strong className="drop-title">Select or drag an audio evidence file to inspect</strong>
              <p className="drop-caption">
                Click to browse files or drop directly into this canvas.
              </p>
              <div className="format-pills">
                {ALLOWED_EXTENSIONS.map((ext) => (
                  <span key={ext} className="format-pill">
                    {ext.toUpperCase()}
                  </span>
                ))}
                <span className="format-limit">Max 50 MiB</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {errorMessage && (
        <div className="alert-box alert-error" role="alert">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div className="alert-content">
            <strong>Input Rejected</strong>
            <p>{errorMessage}</p>
          </div>
          {onDismissError && (
            <button
              type="button"
              className="alert-dismiss"
              onClick={onDismissError}
              aria-label="Dismiss error"
            >
              ×
            </button>
          )}
        </div>
      )}

      <div className="upload-actions-bar">
        <div className="actions-left">
          <button
            className="editorial-button button-primary"
            type="button"
            onClick={handleTriggerAnalyze}
            disabled={!canAnalyze}
            aria-busy={busy}
          >
            {busy ? (
              <>
                <span className="editorial-spinner" />
                <span>Screening Audio… ({elapsedSec.toFixed(1)}s)</span>
              </>
            ) : (
              <>
                <span>Run Forensic Screening</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </>
            )}
          </button>

          {file && (
            <button
              className="editorial-button button-subtle"
              type="button"
              onClick={() => {
                if (fileInputRef.current) fileInputRef.current.value = ''
                onClearFile()
              }}
              disabled={busy}
            >
              Clear File
            </button>
          )}
        </div>

        <div className="privacy-disclosure">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" />
          </svg>
          <span>Local In-Memory Processing · Audio Buffers Unlinked Immediately</span>
        </div>
      </div>

      {!isOnline && (
        <p className="offline-notice">
          {health.status === 'checking'
            ? 'Checking connection to local AEGIS FastAPI backend…'
            : 'FastAPI backend service is offline at 127.0.0.1:8000. Start the server (uvicorn src.api:app) to enable inference.'}
        </p>
      )}
    </section>
  )
}
