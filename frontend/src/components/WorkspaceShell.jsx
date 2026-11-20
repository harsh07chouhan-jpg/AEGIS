
import './WorkspaceShell.css'

const navigation = [
  {
    group: 'WORKSPACE',
    items: [
      { id: 'overview', number: '01', label: 'Overview' },
      { id: 'analysis', number: '02', label: 'New Analysis' },
    ],
  },
  {
    group: 'INTELLIGENCE',
    items: [
      { id: 'models', number: '03', label: 'Model Intelligence' },
      { id: 'system', number: '04', label: 'System Status' },
    ],
  },
]

export function WorkspaceShell({
  activeView,
  onNavigate,
  health,
  children,
}) {
  const isOnline = health?.status === 'online'
  const pageTitles = {
    overview: 'Overview',
    analysis: 'New Analysis',
    models: 'Model Intelligence',
    system: 'System Status',
  }

  return (
    <div className="workspace-shell">
      <aside className="workspace-sidebar">
        <div className="workspace-brand">
          <div className="workspace-brand-mark">A</div>
          <div>
            <div className="workspace-brand-name">AEGIS</div>
            <div className="workspace-brand-caption">FORENSIC MEDIA LAB</div>
          </div>
        </div>

        <div className="workspace-nav">
          {navigation.map((group) => (
            <div className="workspace-nav-group" key={group.group}>
              <div className="workspace-nav-heading">{group.group}</div>

              {group.items.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={`workspace-nav-item ${
                    activeView === item.id ? 'active' : ''
                  }`}
                  onClick={() => onNavigate(item.id)}
                  aria-current={activeView === item.id ? 'page' : undefined}
                >
                  <span className="workspace-nav-number">{item.number}</span>
                  <span>{item.label}</span>
                  {activeView === item.id && (
                    <span className="workspace-nav-indicator" />
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>

        <div className="workspace-sidebar-bottom">
          <div className="workspace-connection">
            <span
              className={`workspace-status-dot ${
                isOnline ? 'online' : 'offline'
              }`}
            />
            <div>
              <div className="workspace-connection-label">AUDIO ENGINE</div>
              <div className="workspace-connection-value">
                {health?.status === 'checking'
                  ? 'Checking connection'
                  : isOnline
                    ? 'Connected'
                    : 'Unavailable'}
              </div>
            </div>
          </div>
          <div className="workspace-sidebar-version">
            ACADEMIC RESEARCH PROTOTYPE
          </div>
        </div>
      </aside>

      <div className="workspace-main">
        <header className="workspace-topbar">
          <div className="workspace-breadcrumb">
            <span>AEGIS</span>
            <span className="workspace-breadcrumb-divider">/</span>
            <strong>{pageTitles[activeView] || 'Overview'}</strong>
          </div>

          <div className="workspace-topbar-meta">
            <span className="workspace-api-label">LOCAL API</span>
            <span
              className={`workspace-status-dot ${
                isOnline ? 'online' : 'offline'
              }`}
            />
            <span>{isOnline ? 'OPERATIONAL' : 'NOT CONNECTED'}</span>
          </div>
        </header>

        <main className="workspace-content">{children}</main>

        <footer className="workspace-footer">
          <span>AEGIS FORENSIC MEDIA LABORATORY</span>
          <span>RESEARCH USE · NOT A CERTIFICATION INSTRUMENT</span>
        </footer>
      </div>
    </div>
  )
}
