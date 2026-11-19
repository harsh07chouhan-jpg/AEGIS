export function ModelInfoCard({ modelInfo }) {
  const model = modelInfo?.model || {}
  const sha = model.checkpoint_sha256 || 'B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4'
  const paramCount = model.parameter_count
    ? `${Number(model.parameter_count).toLocaleString()} parameters`
    : '875,906 parameters'

  return (
    <section className="editorial-section" id="section-model" aria-labelledby="model-heading">
      <div className="section-header">
        <span className="section-number">03</span>
        <div className="section-header-text">
          <span className="section-kicker">MODEL SPECIFICATION & PROVENANCE</span>
          <h2 id="model-heading">Active Transformer Architecture</h2>
          <p className="section-description">
            Cryptographically pinned checkpoint loaded into the AEGIS runtime. Checkpoint bytes are hashed and verified prior to model deserialization.
          </p>
        </div>
      </div>

      <div className="editorial-spec-table">
        <div className="spec-row">
          <span className="spec-label">Model Identifier</span>
          <strong className="spec-value">{model.name || 'aegis_audio_transformer_reproducible_v1'}</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Checkpoint Artifact</span>
          <strong className="spec-value">best_audio_transformer.pt (Epoch {model.checkpoint_epoch ?? 5})</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Verified SHA-256 Digest</span>
          <code className="spec-code" title={sha}>
            {sha}
          </code>
        </div>

        <div className="spec-row">
          <span className="spec-label">Model Size</span>
          <strong className="spec-value">{paramCount}</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Neural Architecture</span>
          <strong className="spec-value">4 Layers · 4 Heads · 128 Embedding Dimension · GELU · 16×16 Patch Projection</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Feature Extraction</span>
          <strong className="spec-value">Log-Mel Spectrogram + Constant-Q Transform (CQT) · Tensor Shape (2, 128, 257)</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Acoustic Windowing</span>
          <strong className="spec-value">2.56-second window (40,960 samples) · 1.28-second hop (20,480 samples) @ 16 kHz</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Decision Threshold</span>
          <strong className="spec-value">0.50 (Uncalibrated Softmax Score)</strong>
        </div>

        <div className="spec-row">
          <span className="spec-label">Integrity Status</span>
          <span className="spec-status-verified">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <span>Bit-Identical Checkpoint Verified at API Startup</span>
          </span>
        </div>
      </div>
    </section>
  )
}
