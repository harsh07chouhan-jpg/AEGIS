export function Masthead({ health, modelInfo, apiBase }) {
  const isOnline = health.status === 'online'
  const model = modelInfo?.model || {}
  const sha = model.checkpoint_sha256 || 'B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4'
  const displaySha = `${sha.slice(0, 6)}…${sha.slice(-4)}`

  return (
    <header className="editorial-masthead">
      {/* Top Telemetry Strip */}
      <div className="masthead-top-strip">
        <div className="top-strip-left">
          <span className="strip-label">AEGIS FORENSIC MEDIA LAB</span>
          <span className="strip-divider">/</span>
          <span className="strip-sub">REPRODUCIBLE RESEARCH PROTOCOL</span>
        </div>

        <div className="top-strip-right">
          <span className="strip-item">
            API: <code>{apiBase.replace(/^https?:\/\//, '')}</code>
          </span>
          <span className="strip-divider">·</span>
          <span className="strip-item" title={sha}>
            SHA: <code>{displaySha}</code>
          </span>
          <span className="strip-divider">·</span>
          <span className={`api-health-pill ${isOnline ? 'pill-online' : 'pill-offline'}`}>
            <span className="health-dot" />
            {isOnline ? 'ONLINE' : health.status === 'checking' ? 'PROBING…' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* Main Title Banner */}
      <div className="masthead-main">
        <div className="masthead-title-group">
          <h1 className="masthead-title">AEGIS</h1>
          <span className="masthead-tagline">AI Audio Deepfake & Synthetic Voice Screening</span>
        </div>
        <p className="masthead-lead">
          An applied AI media-forensics workstation designed to detect synthetic speech, cloned voices, and acoustic manipulations.
          Built with verifiable model provenance, strict zero-retention in-memory processing, and offline benchmark disclosures.
        </p>
      </div>

      {/* Editorial Jump Navigation */}
      <nav className="masthead-nav" aria-label="Section navigation">
        <a href="#section-ingestion" className="nav-link">
          <span className="nav-num">01</span>
          <span>Ingest Audio</span>
        </a>
        <a href="#section-evaluation" className="nav-link">
          <span className="nav-num">02</span>
          <span>Screening Report</span>
        </a>
        <a href="#section-model" className="nav-link">
          <span className="nav-num">03</span>
          <span>Model Architecture</span>
        </a>
        <a href="#section-benchmark" className="nav-link">
          <span className="nav-num">04</span>
          <span>Benchmark Audit</span>
        </a>
        <a href="#section-roadmap" className="nav-link">
          <span className="nav-num">05</span>
          <span>Video Roadmap</span>
        </a>
      </nav>
    </header>
  )
}
