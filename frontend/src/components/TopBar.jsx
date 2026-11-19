export function TopBar({ health, apiBase, activeTab }) {
  const tabTitles = {
    audio: 'Audio Evidence Screening',
    video: 'Video Manipulation Forensics (Roadmap)',
    multimodal: 'Cross-Modal Fusion & Synchronization (Roadmap)',
  }

  const currentTitle = tabTitles[activeTab] || 'Audio Evidence Screening'

  return (
    <header className="topbar">
      <div className="breadcrumbs">
        <span>Workspace</span>
        <span className="crumb-slash">/</span>
        <strong>{currentTitle}</strong>
      </div>

      <div className="topbar-right">
        <span className="api-url-chip" title={`Connected to backend at ${apiBase}`}>
          API: <code>{apiBase.replace(/^https?:\/\//, '')}</code>
        </span>
        <span className="version-chip">FORENSIC v1.0</span>
        <span className={`api-indicator ${health.status}`}>
          <span className="status-dot" />
          {health.status === 'online'
            ? 'API Connected'
            : health.status === 'checking'
            ? 'Connecting…'
            : health.status === 'degraded'
            ? 'Model Unready'
            : 'Backend Offline'}
        </span>
      </div>
    </header>
  )
}
