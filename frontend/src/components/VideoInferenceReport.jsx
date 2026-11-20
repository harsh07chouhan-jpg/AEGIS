import { useState } from 'react'
import { exportVideoEvidence } from '../utils/videoEvidenceReport'
import './VideoInferenceReport.css'

function formatSeconds(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return `${value.toFixed(2)}s`
}

function formatPercent(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

export function VideoInferenceReport({ result }) {
  const [exportNotice, setExportNotice] = useState(null)
  const [exportError, setExportError] = useState(null)

  function handleExport(format) {
    try {
      setExportError(null)
      const filename = exportVideoEvidence(result, format)
      setExportNotice(`Download initiated: ${filename}`)
    } catch (err) {
      setExportNotice(null)
      setExportError(err.message || 'Video report export failed.')
    }
  }

  const metadata = result?.metadata || {}
  const isFake = String(result?.verdict || '').toUpperCase() === 'FAKE'
  const predictions = metadata.frame_predictions || []

  return (
    <section className="video-report">
      <div className="video-report-header">
        <div>
          <span className="video-eyebrow">VIDEO FORENSIC ANALYSIS</span>
          <h2>Inference Report</h2>
        </div>
        <span className={`video-verdict ${isFake ? 'fake' : 'real'}`}>
          {isFake ? 'FAKE' : 'REAL'}
        </span>
      </div>

      <div className="video-score-grid">
        <div className="video-score-card">
          <span>Fake-class score</span>
          <strong>{formatPercent(result?.prob_fake)}</strong>
          <small>Uncalibrated model score</small>
        </div>

        <div className="video-score-card">
          <span>Model confidence (uncalibrated)</span>
            <strong>{formatPercent(result?.confidence)}</strong>
            <small>Model-derived score, not a calibrated probability</small>
        </div>
      </div>

      <div className="video-section">
        <h3>Video Metadata</h3>
        <div className="video-metadata-grid">
          <div><span>Duration</span><strong>{formatSeconds(metadata.duration_seconds)}</strong></div>
          <div><span>Frame rate</span><strong>{metadata.fps ?? '—'} FPS</strong></div>
          <div><span>Frames sampled</span><strong>{metadata.frames_sampled ?? '—'}</strong></div>
          <div><span>Faces detected</span><strong>{metadata.faces_detected ?? '—'}</strong></div>
        </div>
      </div>

      {predictions.length > 0 && (
        <div className="video-section">
          <h3>Temporal Attention</h3>
          <p className="video-note">
            Relative model attention across sampled frames. These are not
            independent authenticity scores.
          </p>

          <div className="video-timeline">
            {predictions.map((item, index) => {
              const attention = Math.max(
                0,
                Math.min(1, Number(item.attention) || 0)
              )

              return (
                <div className="video-frame-row" key={index}>
                  <span>{formatSeconds(item.timestamp)}</span>
                  <div className="video-bar-track">
                    <div
                      className="video-bar-fill"
                      style={{ width: `${attention * 100}%` }}
                    />
                  </div>
                  <strong>{formatPercent(attention)}</strong>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <div className="video-model-info">
        <span>MODEL VERSION</span>
        <strong>{result?.model_version || 'Video detector'}</strong>
      </div>
            {/* Video forensic report export */}
      <div className="export-toolbar">
        <div className="export-buttons">
          <button
            type="button"
            className="editorial-button button-secondary"
            onClick={() => handleExport('markdown')}
            title="Download video forensic evidence dossier"
          >
            <span>Export Evidence Dossier (.md)</span>
          </button>

          <button
            type="button"
            className="editorial-button button-secondary"
            onClick={() => handleExport('json')}
            title="Download structured video forensic audit payload"
          >
            <span>Export Audit Payload (.json)</span>
          </button>
        </div>

        {exportNotice && (
          <span className="export-notice-tag" role="status" aria-live="polite">
            {exportNotice}
          </span>
        )}

        {exportError && (
          <span className="export-error-tag" role="alert">
            {exportError}
          </span>
        )}
      </div>

      <div className="video-disclaimer">
        <strong>Research prototype disclaimer</strong>
        <p>
          This result is an AI-generated forensic assessment, not definitive
          proof of manipulation or authenticity. Do not use it as the sole
          basis for legal, disciplinary, or other high-stakes decisions.
        </p>
      </div>
    </section>
  )
}

export default VideoInferenceReport