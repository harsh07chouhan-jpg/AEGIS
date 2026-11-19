import { useState } from 'react'

export function RoadmapSection() {
  const [activeTab, setActiveTab] = useState('video')

  return (
    <section className="editorial-section" id="section-roadmap" aria-labelledby="roadmap-heading">
      <div className="section-header">
        <span className="section-number">05</span>
        <div className="section-header-text">
          <span className="section-kicker">RESEARCH ROADMAP · UNTRAINED CAPABILITY</span>
          <h2 id="roadmap-heading">Video & Multimodal Forensics Architecture</h2>
          <p className="section-description">
            AEGIS is architected to perform spatial and temporal video screening and cross-modal audio-visual synchronization analysis.
            Video analysis is currently a documented engineering blueprint. In accordance with forensic standards, no simulated or random video predictions are returned.
          </p>
        </div>
      </div>

      <div className="roadmap-tab-nav">
        <button
          type="button"
          className={`roadmap-tab-btn ${activeTab === 'video' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('video')}
        >
          <span>Phase 2: Video Manipulation Forensics</span>
        </button>
        <button
          type="button"
          className={`roadmap-tab-btn ${activeTab === 'multimodal' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('multimodal')}
        >
          <span>Phase 3: Cross-Modal Fusion & Synchronization</span>
        </button>
      </div>

      <div className="roadmap-card">
        <div className="roadmap-badge-line">
          <span className="roadmap-status-pill">STATUS: ARCHITECTURAL BLUEPRINT (PENDING DATASET ACCESS)</span>
        </div>

        {activeTab === 'video' ? (
          <div className="roadmap-content">
            <h3 className="roadmap-title">Spatio-Temporal Face Manipulation Detection</h3>
            <p className="roadmap-summary">
              Screening video streams for face swaps, facial reenactment, and boundary blending artifacts.
              Requires institutional access to <strong>FaceForensics++</strong> and <strong>FakeAVCeleb</strong>.
            </p>

            <div className="pipeline-grid">
              <div className="pipeline-item">
                <span className="pipeline-step">01</span>
                <div>
                  <strong>Video Ingest & Temporal Sampling</strong>
                  <p>FFmpeg demuxing, extracting 16 kHz audio while uniformly sampling video frames at 15–25 fps to capture motion dynamics without frame redundancy.</p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-step">02</span>
                <div>
                  <strong>Face Detection & Crop Normalization</strong>
                  <p>MTCNN or RetinaFace landmark localization with Kalman bounding-box temporal smoothing. Extracts 224×224 normalized face crops with a 1.3× expanded margin to evaluate boundary seams.</p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-step">03</span>
                <div>
                  <strong>Spatial-Temporal Video Transformer</strong>
                  <p>Frame-sequence encoding via 3D CNN or Spatio-Temporal Vision Transformer (ViT-B/16 or TimeSformer) to identify frequency artifacts, spatial warping, and temporal inter-frame jitter.</p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-step">04</span>
                <div>
                  <strong>Clip-Level Temporal Aggregation</strong>
                  <p>Aggregating frame-level manipulation evidence across temporal clips to localize exact manipulation timestamps.</p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="roadmap-content">
            <h3 className="roadmap-title">Cross-Modal Synchronization & Audio-Visual Fusion</h3>
            <p className="roadmap-summary">
              Detecting voice dubbing, lip-sync replacement, and identity mismatches by evaluating temporal coherence between facial articulation and speech acoustics.
            </p>

            <div className="pipeline-grid">
              <div className="pipeline-item">
                <span className="pipeline-step">01</span>
                <div>
                  <strong>Visual Lip Landmark Tracking</strong>
                  <p>Tracking 3D mouth aperture, lip corner displacement, and jaw velocity to map visual phoneme sequences.</p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-step">02</span>
                <div>
                  <strong>Acoustic Formant Extraction</strong>
                  <p>Extracting spectral formant transitions and energy envelopes from the accompanying audio stream.</p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-step">03</span>
                <div>
                  <strong>Bidirectional Cross-Attention</strong>
                  <p>Cross-attention transformer layer allowing visual tokens to attend to acoustic tokens, measuring cross-modal temporal discrepancies in milliseconds.</p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-step">04</span>
                <div>
                  <strong>Calibrated Decision Fusion</strong>
                  <p>Modality-aware gating combining audio spoof score, video manipulation score, and sync mismatch with temperature scaling for calibrated multi-signal reporting.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="editorial-callout callout-subtle">
          <strong>Dataset Prerequisite Status:</strong>
          <p>
            The FakeAVCeleb access request remains pending institutional compliance review. Video inference will only be activated when a trained, verified, and benchmarked PyTorch model checkpoint is added to the repository.
          </p>
        </div>
      </div>
    </section>
  )
}
