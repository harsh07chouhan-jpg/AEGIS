import { useState } from 'react'
import { exportEvidence } from '../utils/evidenceReport'

function formatSeconds(sec) {
  if (!Number.isFinite(sec)) return '—'
  if (sec < 60) return `${sec.toFixed(2)}s`
  const minutes = Math.floor(sec / 60)
  const remaining = (sec % 60).toFixed(2)
  return `${minutes}m ${remaining}s`
}

export function InferenceReport({ result }) {
  const [copiedHash, setCopiedHash] = useState(false)
  const [showWindows, setShowWindows] = useState(false)
  const [exportNotice, setExportNotice] = useState(null)
  const [exportError, setExportError] = useState(null)

  if (!result) return null

  const isRejected = result.status === 'rejected'

  function copySha256() {
    if (result.input?.inputSha256) {
      navigator.clipboard.writeText(result.input.inputSha256)
      setCopiedHash(true)
      setTimeout(() => setCopiedHash(false), 2000)
    }
  }

  function handleExport(format) {
    try {
      setExportError(null)
      const filename = exportEvidence(result, format)
      setExportNotice(`Downloaded ${filename}`)
      setTimeout(() => setExportNotice(null), 3500)
    } catch (err) {
      setExportError(err.message || 'Export failed')
      setTimeout(() => setExportError(null), 4000)
    }
  }

  if (isRejected) {
    return (
      <section className="editorial-section" id="section-evaluation" aria-live="polite">
        <div className="section-header">
          <span className="section-number">02</span>
          <div className="section-header-text">
            <span className="section-kicker">EVIDENCE EVALUATION REPORT</span>
            <h2>Input Validation Failure</h2>
          </div>
        </div>

        <div className="alert-box alert-error">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div className="alert-content">
            <strong>The audio recording was rejected by the forensic service</strong>
            <p>
              Rejection Code: <code>{result.error?.code || 'rejected'}</code>
            </p>
            <p className="alert-detail-msg">{result.error?.message || 'Audio input is invalid or unreadable.'}</p>
          </div>
        </div>
      </section>
    )
  }

  const meanScore = result.aggregate?.meanSpoofScore ?? 0
  const maxScore = result.aggregate?.maxSpoofScore ?? 0
  const isSpoof = result.decision?.label === 'spoof'
  const scorePercent = (meanScore * 100).toFixed(1)
  const maxScorePercent = (maxScore * 100).toFixed(1)
  const threshold = 0.50

  return (
    <section className="editorial-section" id="section-evaluation" aria-live="polite">
      <div className="section-header">
        <span className="section-number">02</span>
        <div className="section-header-text">
          <span className="section-kicker">FORENSIC EVIDENCE ASSESSMENT</span>
          <h2>Acoustic Manipulation Screening Report</h2>
          <p className="section-description">
            Analysis result produced by the verified AEGIS Audio Transformer based on spectral Log-Mel and Constant-Q Transform window logits.
          </p>
        </div>
      </div>

      {/* Decision Banner */}
      <div className={`editorial-decision-card ${isSpoof ? 'decision-spoof' : 'decision-bonafide'}`}>
        <div className="decision-header">
          <div className="decision-titles">
            <span className="decision-kicker">MODEL CLASSIFICATION</span>
            <h3 className="decision-heading">
              {isSpoof ? 'Flagged: Likely Synthetic or Manipulated Speech' : 'Unflagged: Acoustic Profile Consistent with Bona Fide Speech'}
            </h3>
            <p className="decision-description">
              {result.decision?.displayText ||
                `Model mean spoof score ${meanScore.toFixed(4)} evaluated against threshold 0.50.`}
            </p>
            <div className="decision-rule-line">
              <span>Decision Rule:</span>
              <code>{result.decision?.rule || `mean_spoof_score >= ${threshold.toFixed(2)}`}</code>
            </div>
          </div>

          <div className="decision-badge-container">
            <span className={`editorial-stamp ${isSpoof ? 'stamp-spoof' : 'stamp-bonafide'}`}>
              {isSpoof ? 'FLAGGED / SPOOF' : 'UNFLAGGED / BONA FIDE'}
            </span>
          </div>
        </div>

        {/* Linear Calibrated Scale */}
        <div className="scale-container">
          <div className="scale-label-row">
            <span className="scale-title">Mean Softmax Spoof Score</span>
            <strong className="scale-numeral">{scorePercent}%</strong>
          </div>

          <div className="editorial-scale-track" role="meter" aria-valuenow={scorePercent} aria-valuemin="0" aria-valuemax="100">
            <div
              className={`editorial-scale-fill ${isSpoof ? 'fill-spoof' : 'fill-bonafide'}`}
              style={{ width: `${Math.max(2, Math.min(100, meanScore * 100))}%` }}
            />
            <div className="scale-threshold-pin" style={{ left: '50%' }} title="Decision Threshold (0.50)">
              <div className="pin-tick" />
              <span className="pin-text">0.50 THRESHOLD</span>
            </div>
          </div>

          <div className="scale-annotations">
            <span>0.00 (Low excitation)</span>
            <span className="scale-semantics-note">Uncalibrated softmax model score · Not a posterior probability</span>
            <span>1.00 (High excitation)</span>
          </div>
        </div>

        {/* Key Metrics Row */}
        <div className="editorial-metrics-grid">
          <div className="metric-box">
            <span className="metric-label">Mean Spoof Score</span>
            <strong className="metric-value">{scorePercent}%</strong>
            <span className="metric-caption">{meanScore.toFixed(6)}</span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Maximum Window Score</span>
            <strong className="metric-value">{maxScorePercent}%</strong>
            <span className="metric-caption">{maxScore.toFixed(6)}</span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Decoded Duration</span>
            <strong className="metric-value">{formatSeconds(result.input?.durationSec)}</strong>
            <span className="metric-caption">{result.input?.durationSec?.toFixed(4)}s decoded</span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Analysis Windows</span>
            <strong className="metric-value">{result.aggregate?.numWindows ?? '—'}</strong>
            <span className="metric-caption">2.56s window / 1.28s hop</span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Inference Latency</span>
            <strong className="metric-value">{result.elapsedSec.toFixed(3)}s</strong>
            <span className="metric-caption">PyTorch CPU inference</span>
          </div>
        </div>

        {/* Evidence Hash */}
        {result.input?.inputSha256 && (
          <div className="hash-block">
            <div className="hash-header">
              <span className="hash-title">INPUT ARTIFACT SHA-256 DIGEST</span>
              <button
                type="button"
                className="copy-hash-btn"
                onClick={copySha256}
                aria-label="Copy input SHA-256 digest"
              >
                {copiedHash ? '✓ Copied' : 'Copy Digest'}
              </button>
            </div>
            <code className="hash-code">{result.input.inputSha256}</code>
          </div>
        )}

        {/* Export Toolbar */}
        <div className="export-toolbar">
          <div className="export-buttons">
            <button
              type="button"
              className="editorial-button button-secondary"
              onClick={() => handleExport('markdown')}
              title="Download human-readable Markdown evidence dossier"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              <span>Export Evidence Dossier (.md)</span>
            </button>

            <button
              type="button"
              className="editorial-button button-secondary"
              onClick={() => handleExport('json')}
              title="Download machine-readable JSON forensic audit envelope"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span>Export Audit Payload (.json)</span>
            </button>
          </div>

          {exportNotice && (
            <span className="export-notice-tag" role="status" aria-live="polite">
              ✓ {exportNotice}
            </span>
          )}

          {exportError && (
            <span className="export-error-tag" role="alert">
              ✕ {exportError}
            </span>
          )}
        </div>
      </div>

      {/* Pipeline Warnings */}
      {result.warnings?.length > 0 && (
        <div className="warnings-container">
          {result.warnings.map((w, index) => (
            <div key={index} className="alert-box alert-warning">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              <div className="alert-content">
                <strong>Pipeline Notice: {w.code}</strong>
                <p>{w.message}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Scientific Caution Callout */}
      <div className="editorial-callout">
        <div className="callout-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <strong>Scientific Interpretation Standards & Error Rate Notice</strong>
        </div>
        <p>
          The AEGIS Audio Transformer produces uncalibrated window-mean scores. Benchmark evaluation on the
          ASVspoof 2019 LA dataset demonstrated a ~43.28% false-alarm rate on bona fide human speech at the 0.50
          threshold (bona fide recall: 56.72%). A <em>bonafide</em> label must never be presented as proof of
          authenticity, nor should a <em>spoof</em> label alone be treated as definitive legal proof.
        </p>
      </div>

      {/* Temporal Window Breakdown */}
      {result.windows?.length > 0 && (
        <div className="editorial-table-block">
          <div className="table-header-row">
            <div>
              <span className="table-kicker">TEMPORAL ANALYSIS</span>
              <h4 className="table-title">Per-Window Score Breakdown</h4>
            </div>
            <button
              type="button"
              className="editorial-button button-subtle"
              onClick={() => setShowWindows(!showWindows)}
              aria-expanded={showWindows}
            >
              {showWindows ? '▲ Hide Window Details' : `▼ View ${result.windows.length} Window${result.windows.length > 1 ? 's' : ''}`}
            </button>
          </div>

          {showWindows && (
            <div className="table-wrapper">
              <table className="editorial-data-table">
                <thead>
                  <tr>
                    <th>Window #</th>
                    <th>Time Span</th>
                    <th>Audio Samples</th>
                    <th>Zero-Padded</th>
                    <th>Raw Spoof Score</th>
                    <th>Classification</th>
                  </tr>
                </thead>
                <tbody>
                  {result.windows.map((win) => {
                    const winScore = Number(win.spoofScore) || 0
                    const winIsSpoof = winScore >= threshold
                    return (
                      <tr key={win.index}>
                        <td>{win.index}</td>
                        <td>
                          {win.startSec.toFixed(2)}s – {win.endSec.toFixed(2)}s
                        </td>
                        <td>{win.audioSamples.toLocaleString()}</td>
                        <td>{(win.paddedFraction * 100).toFixed(0)}%</td>
                        <td>
                          <div className="table-score-cell">
                            <strong>{(winScore * 100).toFixed(1)}%</strong>
                            <span className="table-score-raw">({winScore.toFixed(4)})</span>
                          </div>
                        </td>
                        <td>
                          <span className={`table-badge ${winIsSpoof ? 'badge-spoof' : 'badge-bonafide'}`}>
                            {winIsSpoof ? 'SPOOF' : 'BONAFIDE'}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
