import { useState } from 'react'
import { api, type Project } from '../../api'

interface Props {
  project: Project
  onCommitted: (p: Project) => void
  onOpenChat: () => void
}

export default function Step7Commit({ project, onCommitted, onOpenChat }: Props) {
  const [commitMsg,   setCommitMsg]   = useState('feat(mlops): add generated pipeline assets')
  const [committing,  setCommitting]  = useState(false)
  const [commitDone,  setCommitDone]  = useState(false)
  const [registering, setRegistering] = useState(false)
  const [regDone,     setRegDone]     = useState(false)
  const [msg,         setMsg]         = useState('')
  const [error,       setError]       = useState('')

  const alreadyCommitted = ['committed', 'operational'].includes(project.stage)

  async function commit() {
    if (!commitMsg.trim()) { setError('Commit message is required'); return }
    setCommitting(true); setError(''); setMsg('')
    try {
      await api.commitProject(project.id, commitMsg)
      setCommitDone(true)
      setMsg('✓ Changes committed to your repository.')
      const projects = await api.listProjects()
      const updated = projects.find(p => p.id === project.id)
      if (updated) onCommitted(updated)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setCommitting(false)
    }
  }

  async function registerPipelines() {
    setRegistering(true); setError('')
    try {
      await api.validateProject(project.id)   // re-uses validate endpoint as register isn't separate in api.ts
      // Actually call register:
      await fetch(`/api/projects/${project.id}/register-pipelines`, { method: 'POST' })
      setRegDone(true)
      setMsg(prev => prev + '\n✓ AzDO pipeline definitions registered (no run triggered).')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRegistering(false)
    }
  }

  return (
    <div className="step-card">
      <div className="step-header">
        <div className="step-number">Step 7 of 7</div>
        <h2 className="step-title">Commit & ship</h2>
        <p className="step-desc">
          Push the generated pipeline assets to your repository and register the AzDO pipeline definitions.
          Pipeline runs require explicit approval via the Chat agent.
        </p>
      </div>

      {alreadyCommitted && !commitDone && (
        <div className="info-box info-box-success" style={{ marginBottom: '1.5rem' }}>
          ✓ This project has already been committed (stage: <strong>{project.stage}</strong>).
        </div>
      )}

      {error && <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>{error}</div>}

      {/* Commit message */}
      <div className="field-group" style={{ marginBottom: '1.25rem' }}>
        <label className="field-label" htmlFor="commit-msg">Commit message</label>
        <textarea
          id="commit-msg"
          className="input"
          value={commitMsg}
          onChange={e => setCommitMsg(e.target.value)}
          style={{ minHeight: 72 }}
          disabled={commitDone || alreadyCommitted}
        />
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '2rem' }}>
        <button
          id="commit-btn"
          className="btn btn-primary btn-lg"
          onClick={commit}
          disabled={committing || commitDone || alreadyCommitted}
        >
          {committing
            ? <><span className="spinner" /> Committing…</>
            : commitDone || alreadyCommitted
            ? '✓ Committed'
            : '📤 Commit to GitHub / AzDO'}
        </button>

        <button
          id="register-btn"
          className="btn btn-lg"
          onClick={registerPipelines}
          disabled={registering || regDone || (!commitDone && !alreadyCommitted)}
        >
          {registering
            ? <><span className="spinner" /> Registering…</>
            : regDone
            ? '✓ Pipelines registered'
            : '📋 Register AzDO pipelines'}
        </button>
      </div>

      {msg && (
        <div className="info-box info-box-success" style={{ marginBottom: '1.5rem', whiteSpace: 'pre-line' }}>
          {msg}
        </div>
      )}

      {/* Chat CTA */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(129,140,248,0.05) 100%)',
        border: '1px solid var(--border-glow)',
        borderRadius: 'var(--radius)',
        padding: '1.25rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
        flexWrap: 'wrap',
      }}>
        <div>
          <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text)', marginBottom: '0.25rem' }}>
            🤖 Trigger a pipeline run
          </div>
          <div style={{ fontSize: '12.5px', color: 'var(--text-dim)' }}>
            Pipeline triggers require explicit approval in the Chat agent. Open it, then say:<br />
            <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent-hover)' }}>
              "trigger the training pipeline"
            </span>
          </div>
        </div>
        <button className="btn btn-primary" onClick={onOpenChat}>
          💬 Open Chat Agent
        </button>
      </div>

      {/* Summary of everything done */}
      <div style={{ marginTop: '2rem' }}>
        <div className="section-label">What was accomplished</div>
        <div style={{ display: 'grid', gap: '0.5rem' }}>
          {[
            { done: true,                              text: 'Repository registered and connected' },
            { done: ['scanned', 'evaluated', 'generated', 'validated_local', 'committed', 'operational'].includes(project.stage), text: 'Repo scanned → knowledge graph built' },
            { done: !!project.profile?.target_variable, text: 'ML profile configured (target, metrics, thresholds)' },
            { done: ['generated', 'validated_local', 'committed', 'operational'].includes(project.stage), text: 'Missing pipelines generated' },
            { done: ['validated_local', 'committed', 'operational'].includes(project.stage), text: 'Local validation passed' },
            { done: commitDone || alreadyCommitted,    text: 'Committed to repository' },
            { done: regDone,                           text: 'AzDO pipeline definitions registered' },
          ].map((item, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '13px', color: item.done ? 'var(--text)' : 'var(--text-muted)' }}>
              <span style={{ color: item.done ? 'var(--success)' : 'var(--border-strong)', fontSize: '14px', flexShrink: 0 }}>
                {item.done ? '●' : '○'}
              </span>
              {item.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
