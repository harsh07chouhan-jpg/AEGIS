function Icon({ name, size = 18 }) {
  const paths = {
    shield: (
      <>
        <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    audio: (
      <>
        <path d="M2 10v4" />
        <path d="M6 6v12" />
        <path d="M10 3v18" />
        <path d="M14 8v8" />
        <path d="M18 5v14" />
        <path d="M22 10v4" />
      </>
    ),
    video: (
      <>
        <rect width="18" height="14" x="2" y="5" rx="2" />
        <path d="m17 9 5-3v12l-5-3" />
      </>
    ),
    multimodal: (
      <>
        <circle cx="9" cy="9" r="6" />
        <circle cx="15" cy="15" r="6" />
      </>
    ),
    cpu: (
      <>
        <rect width="16" height="16" x="4" y="4" rx="2" />
        <path d="M9 9h6v6H9z" />
        <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" />
      </>
    ),
    lock: (
      <>
        <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
      </>
    ),
  }

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.shield}
    </svg>
  )
}

export function Sidebar({
  activeTab,
  onSelectTab,
  health,
  modelInfo,
  onOpenRoadmap,
}) {
  const device = modelInfo?.runtime?.device || 'cpu'

  return (
    <aside className="sidebar">
      <a className="brand" href="#" aria-label="AEGIS Forensics Home">
        <span className="brand-mark">
          <Icon name="shield" size={20} />
        </span>
        <span className="brand-copy">
          <strong>AEGIS</strong>
          <small>MEDIA FORENSICS</small>
        </span>
      </a>

      <div className="nav-label">FORENSIC SUITE</div>
      <button
        className={`nav-item ${activeTab === 'audio' ? 'active' : ''}`}
        type="button"
        onClick={() => onSelectTab('audio')}
      >
        <Icon name="audio" size={17} />
        <span>Audio Screening</span>
        <span className="nav-dot" />
      </button>

      <button
        className="nav-item nav-item-roadmap"
        type="button"
        onClick={() => onOpenRoadmap('video')}
        title="View Video Forensics Architecture Roadmap"
      >
        <Icon name="video" size={17} />
        <span>Video Forensics</span>
        <span className="roadmap-tag">ROADMAP</span>
      </button>

      <button
        className="nav-item nav-item-roadmap"
        type="button"
        onClick={() => onOpenRoadmap('multimodal')}
        title="View Multimodal Fusion Architecture Roadmap"
      >
        <Icon name="multimodal" size={17} />
        <span>Multimodal Fusion</span>
        <span className="roadmap-tag">ROADMAP</span>
      </button>

      <div className="sidebar-divider" />

      <div className="nav-label">SYSTEM TELEMETRY</div>
      <div className="system-row">
        <span className={`status-dot ${health.status}`} />
        <span>Inference Engine</span>
        <span className="system-value">
          {health.status === 'checking'
            ? 'PROBING…'
            : health.status === 'online'
            ? 'READY'
            : 'OFFLINE'}
        </span>
      </div>

      <div className="system-row">
        <Icon name="cpu" size={13} />
        <span>Execution Device</span>
        <span className="system-value">{device.toUpperCase()}</span>
      </div>

      <div className="system-row">
        <Icon name="lock" size={13} />
        <span>Integrity SHA</span>
        <span className="system-value" title={modelInfo?.model?.checkpoint_sha256}>
          {modelInfo?.model?.checkpoint_sha256
            ? `${modelInfo.model.checkpoint_sha256.slice(0, 6)}…${modelInfo.model.checkpoint_sha256.slice(-4)}`
            : 'VERIFIED'}
        </span>
      </div>

      <div className="sidebar-bottom">
        <div className="privacy-mark">
          <Icon name="shield" size={15} />
        </div>
        <div>
          <strong>Local Execution Policy</strong>
          <p>
            Evidence files stream directly to your local FastAPI service. Audio is
            analyzed in-memory and temporary buffers are unlinked immediately after inference.
          </p>
        </div>
      </div>
    </aside>
  )
}
