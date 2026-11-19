import { useEffect } from 'react'

export function RoadmapModal({ modality = 'video', onClose }) {
  const isVideo = modality === 'video'

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-card panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-text">
            <span className="section-kicker">
              RESEARCH ARCHITECTURE <span>{isVideo ? 'MOD-02' : 'MOD-03'}</span>
            </span>
            <h2>{isVideo ? 'Video Manipulation Forensics' : 'Cross-Modal Fusion & Synchronization'}</h2>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="status-disclaimer-pill">
            <span className="status-dot degraded" />
            <span>RESEARCH ROADMAP · UNTRAINED / PENDING DATASET VALIDATION</span>
          </div>

          <p className="modal-lead">
            {isVideo
              ? 'AEGIS is architected to perform spatial and temporal video manipulation screening. Video inference is currently in the engineering roadmap stage to ensure rigorous scientific validation before deployment.'
              : 'The cross-modal branch will analyze temporal synchronization between facial articulation and speech audio to detect lip-sync manipulation and voice-swapped media.'}
          </p>

          <div className="pipeline-stages">
            <h3>Planned Technical Pipeline</h3>
            <div className="stage-steps">
              <div className="stage-step">
                <span className="step-num">01</span>
                <div>
                  <strong>Video Ingest & Decoupling</strong>
                  <p>Extract audio track at 16 kHz mono; sample video frames at uniform cadence (e.g. 10–25 fps).</p>
                </div>
              </div>

              <div className="stage-step">
                <span className="step-num">02</span>
                <div>
                  <strong>Facial Landmark & Crop Localization</strong>
                  <p>Apply MTCNN / RetinaFace to detect and crop face bounding boxes with margin normalization.</p>
                </div>
              </div>

              <div className="stage-step">
                <span className="step-num">03</span>
                <div>
                  <strong>Spatio-Temporal Video Transformer</strong>
                  <p>
                    Frame sequence encoding through 3D CNN or Video Transformer to detect boundary artifacts,
                    blending seams, and temporal flicker.
                  </p>
                </div>
              </div>

              <div className="stage-step">
                <span className="step-num">04</span>
                <div>
                  <strong>{isVideo ? 'Sequence Aggregation' : 'Bidirectional Cross-Attention'}</strong>
                  <p>
                    {isVideo
                      ? 'Aggregate frame-level manipulation probabilities across temporal clips.'
                      : 'Align visual lip landmarks with acoustic phoneme embeddings to measure sync discrepancies.'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="dataset-prerequisites">
            <h3>Dataset & Licensing Prerequisites</h3>
            <ul>
              <li>
                <strong>FakeAVCeleb:</strong> Primary multimodal dataset containing audio and video manipulations.
                Access request pending institutional review.
              </li>
              <li>
                <strong>FaceForensics++:</strong> DeepFakes, Face2Face, FaceSwap, and NeuralTextures benchmark.
              </li>
            </ul>
          </div>

          <div className="message caution-message modal-caution">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            <div>
              <strong>Scientific Transparency Standard</strong>
              <p>
                AEGIS does not output simulated, random, or hardcoded video predictions. Video analysis will only be
                enabled when a verified, evaluated PyTorch checkpoint is integrated into the inference service.
              </p>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="secondary-button" onClick={onClose}>
            Close Roadmap
          </button>
        </div>
      </div>
    </div>
  )
}
