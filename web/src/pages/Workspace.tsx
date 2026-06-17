import { useEffect, useRef, useState } from 'react'
import { api, type Project } from '../api'
import Chat from './Chat'
import Step1Register from './steps/Step1Register'
import Step2Scan from './steps/Step2Scan'
import Step3Profile from './steps/Step3Profile'
import Step4Verify from './steps/Step4Verify'
import Step5Generate from './steps/Step5Generate'
import Step6Validate from './steps/Step6Validate'
import Step7Commit from './steps/Step7Commit'

// ─── Stage → step mapping ─────────────────────────────────────────────────────
const STAGE_STEP: Record<string, number> = {
  configured:       2,
  scanned:          3,
  evaluated:        3,
  generated:        6,
  validation_failed:6,
  validated_local:  7,
  committed:        7,
  operational:      7,
}

function stageToStep(stage: string): number {
  return STAGE_STEP[stage] ?? 2
}

function stepDone(step: number, currentStage: string): boolean {
  const stageNum = STAGE_STEP[currentStage] ?? 1
  return step < stageNum
}

// ─── Step definitions ─────────────────────────────────────────────────────────
const STEPS = [
  { n: 1, label: 'Register',   sub: 'Connect repo' },
  { n: 2, label: 'Scan',       sub: 'Build graph' },
  { n: 3, label: 'Profile',    sub: 'Metrics & targets' },
  { n: 4, label: 'Verify',     sub: 'Script sign-off' },
  { n: 5, label: 'Build',      sub: 'Fill gaps' },
  { n: 6, label: 'Validate',   sub: 'Test locally' },
  { n: 7, label: 'Commit',     sub: 'Ship it' },
]

export default function Workspace() {
  const [projects,      setProjects]      = useState<Project[]>([])
  const [selectedId,    setSelectedId]    = useState<string>('__new__')
  const [activeStep,    setActiveStep]    = useState(1)
  const [chatOpen,      setChatOpen]      = useState(false)
  const [healthy,       setHealthy]       = useState<boolean | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const project = projects.find(p => p.id === selectedId) ?? null

  // Backend health
  useEffect(() => {
    fetch('/api/health').then(r => setHealthy(r.ok)).catch(() => setHealthy(false))
  }, [])

  // Load projects
  const loadProjects = async () => {
    try {
      const list = await api.listProjects()
      setProjects(list)
      return list
    } catch { return [] }
  }

  useEffect(() => {
    loadProjects()
  }, [])

  // When a project is selected, jump to its current step
  useEffect(() => {
    if (project) {
      setActiveStep(stageToStep(project.stage))
    } else {
      setActiveStep(1)
    }
  }, [selectedId]) // eslint-disable-line react-hooks/exhaustive-deps

  // Scroll panel to top on step change
  useEffect(() => {
    panelRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
  }, [activeStep])

  // ── Handlers ─────────────────────────────────────────────────────────────
  function handleCreated(p: Project) {
    setProjects(prev => [p, ...prev])
    setSelectedId(p.id)
    setActiveStep(2)
  }

  function handleProjectUpdate(updated: Project) {
    setProjects(prev => prev.map(p => p.id === updated.id ? updated : p))
    // advance if needed
    const newStep = stageToStep(updated.stage)
    if (newStep > activeStep) setActiveStep(newStep)
  }

  function advanceTo(step: number, p?: Project) {
    if (p) handleProjectUpdate(p)
    setActiveStep(step)
  }

  async function handleDeleteProject() {
    if (!project) return
    if (!confirm(`Are you sure you want to delete project "${project.name}" and all its data?`)) return
    try {
      await api.deleteProject(project.id)
      setProjects(prev => prev.filter(p => p.id !== project.id))
      setSelectedId('__new__')
      setActiveStep(1)
      setChatOpen(false)
    } catch (e) {
      alert('Failed to delete project: ' + (e as Error).message)
    }
  }

  // ── Sidebar step dot state ─────────────────────────────────────────────
  function dotState(stepN: number): 'done' | 'active' | 'todo' {
    if (stepN === activeStep) return 'active'
    if (project && stepDone(stepN, project.stage)) return 'done'
    if (stepN < activeStep) return 'done'
    return 'todo'
  }

  // ── Render active step ─────────────────────────────────────────────────
  function renderStep() {
    if (activeStep === 1 || !project) {
      return <Step1Register onCreated={handleCreated} />
    }
    switch (activeStep) {
      case 2: return <Step2Scan project={project} onScanned={p => advanceTo(3, p)} />
      case 3: return <Step3Profile project={project} onSaved={p => { handleProjectUpdate(p); setActiveStep(4) }} />
      case 4: return <Step4Verify project={project} onSignedOff={p => advanceTo(5, p)} />
      case 5: return <Step5Generate project={project} onGenerated={p => advanceTo(6, p)} />
      case 6: return <Step6Validate project={project} onValidated={p => advanceTo(7, p)} />
      case 7: return <Step7Commit project={project} onCommitted={p => handleProjectUpdate(p)} onOpenChat={() => setChatOpen(true)} />
      default: return null
    }
  }

  return (
    <div className="workspace">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="ws-header">
        <div className="ws-logo">
          <div className="ws-logo-icon">M</div>
          <span className="ws-logo-text">MLOps Platform</span>
        </div>

        {/* Project selector */}
        <div className="project-selector" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <select
            id="project-select"
            value={selectedId}
            onChange={e => { setSelectedId(e.target.value); setChatOpen(false) }}
          >
            <option value="__new__">＋ New project…</option>
            {projects.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}  [{p.stage}]
              </option>
            ))}
          </select>
          {project && (
            <button 
              className="btn btn-sm btn-danger" 
              onClick={handleDeleteProject}
              title="Delete project"
              style={{ padding: '0.3rem 0.6rem', fontSize: '11px', background: 'rgba(248,113,113,0.1)', color: '#f87171', border: '1px solid rgba(248,113,113,0.3)' }}
            >
              🗑 Delete
            </button>
          )}
        </div>

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '12px', color: 'var(--text-muted)' }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: healthy === null ? 'var(--text-muted)' : healthy ? 'var(--success)' : 'var(--danger)',
            boxShadow: healthy ? '0 0 6px var(--success)' : 'none',
            display: 'inline-block',
          }} />
          {healthy === null ? 'Connecting…' : healthy ? 'Backend connected' : 'Offline'}
        </div>
      </header>

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="ws-sidebar">
        <div className="sidebar-section-label">Pipeline</div>

        {STEPS.map((s, idx) => {
          const state = dotState(s.n)
          const isClickable = s.n === 1 || !!project
          return (
            <div key={s.n}>
              <button
                className={`step-nav-item ${s.n === activeStep ? 'active' : ''}`}
                id={`step-nav-${s.n}`}
                onClick={() => isClickable && (s.n === 1 || project) && setActiveStep(s.n)}
                disabled={!isClickable}
                style={{ opacity: isClickable ? 1 : 0.4, cursor: isClickable ? 'pointer' : 'not-allowed' }}
              >
                <span className={`step-dot ${state}`}>
                  {state === 'done' ? '✓' : s.n}
                </span>
                <div>
                  <div className="step-label">{s.label}</div>
                  <div className="step-sublabel">{s.sub}</div>
                </div>
              </button>
              {idx < STEPS.length - 1 && <div className="step-connector" />}
            </div>
          )
        })}

        {/* Project info footer */}
        {project && (
          <div style={{ marginTop: 'auto', padding: '1rem 1.1rem', borderTop: '1px solid var(--border)' }}>
            <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Current project</div>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)', marginBottom: '0.2rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{project.name}</div>
            <div style={{ fontSize: '10px', fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent-hover)' }}>{project.stage}</div>
          </div>
        )}
      </aside>

      {/* ── Main panel ───────────────────────────────────────────────────── */}
      <main className="ws-panel" ref={panelRef}>
        {renderStep()}
      </main>

      {/* ── Chat FAB (hidden while the drawer is open — the header ✕ closes it) ── */}
      {project && !chatOpen && (
        <button
          id="chat-fab"
          className="chat-fab"
          onClick={() => setChatOpen(true)}
          title="Open agent chat"
        >
          💬
        </button>
      )}

      {/* ── Chat drawer ──────────────────────────────────────────────────── */}
      {chatOpen && project && (
        <>
          <div className="chat-overlay" onClick={() => setChatOpen(false)} />
          <div className="chat-drawer">
            <div style={{ padding: '0.9rem 1rem', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--surface)' }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: '14px' }}>Agent Chat</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{project.name}</div>
              </div>
              <button
                className="btn btn-sm btn-icon"
                onClick={() => setChatOpen(false)}
                style={{ fontSize: '16px' }}
              >
                ✕
              </button>
            </div>
            <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <Chat embeddedProjectId={project.id} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}
