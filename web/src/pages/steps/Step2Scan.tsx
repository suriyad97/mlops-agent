import { useState } from 'react'
import { api, type Project } from '../../api'

interface Props {
  project: Project
  onScanned: (p: Project) => void
}

export default function Step2Scan({ project, onScanned }: Props) {
  const [scanning, setScanning] = useState(false)
  const [log, setLog] = useState<string[]>([])
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)

  async function startScan() {
    setScanning(true)
    setError('')
    setLog([])
    setDone(false)

    try {
      setLog(prev => [...prev, '⟶ Opening folder picker…'])
      const { path } = await api.browsePath()
      if (!path) {
        setLog(prev => [...prev, '✖ No folder selected. Please try again.'])
        setScanning(false)
        return
      }

      setLog(prev => [...prev, `📁 Target folder: ${path}`, '⟶ Cloning repository…'])
      const result = await api.scanProject(project.id, path, project.base_branch)

      const profile = result.profile as Record<string, unknown>
      const pt = profile.project_type ?? '—'
      const tv = profile.target_variable ?? '—'
      const ep = profile.endpoint_strategy ?? '—'

      setLog(prev => [
        ...prev,
        '✓ Repository cloned successfully',
        '⟶ Building knowledge graph…',
        '✓ Knowledge graph complete',
        '⟶ Inferring project profile…',
        `  › Project type:      ${pt}`,
        `  › Target variable:   ${tv}`,
        `  › Endpoint strategy: ${ep}`,
        '',
        '✅ Scan complete — proceed to fill your profile',
      ])
      setDone(true)

      // Refresh project data and advance
      const projects = await api.listProjects()
      const updated = projects.find(p => p.id === project.id)
      if (updated) onScanned(updated)
    } catch (e) {
      setError((e as Error).message)
      setLog(prev => [...prev, `✖ Error: ${(e as Error).message}`])
    } finally {
      setScanning(false)
    }
  }

  const alreadyScanned = ['scanned', 'evaluated', 'generated', 'validated_local', 'committed', 'operational'].includes(project.stage)

  return (
    <div className="step-card">
      <div className="step-header">
        <div className="step-number">Step 2 of 7</div>
        <h2 className="step-title">Scan the repository</h2>
        <p className="step-desc">
          The platform will clone your repo locally, build a knowledge graph of all files and functions,
          and auto-infer a project profile. No Azure resources are touched at this stage.
        </p>
      </div>

      {alreadyScanned && !done && (
        <div className="info-box info-box-success" style={{ marginBottom: '1.5rem' }}>
          ✓ This project has already been scanned (stage: <strong>{project.stage}</strong>).
          You can re-scan to pick up new commits, or proceed to the next step.
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <button
          id="scan-btn"
          className="btn btn-primary btn-lg"
          onClick={startScan}
          disabled={scanning}
        >
          {scanning
            ? <><span className="spinner" /> Scanning…</>
            : alreadyScanned ? '🔄 Re-scan repository' : '🔍 Start scan'}
        </button>

        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          A native folder picker will open — choose where to clone the repo.
        </div>
      </div>

      {log.length > 0 && (
        <div className="scan-log">
          {log.map((line, i) => (
            <div key={i} style={{ color: line.startsWith('✅') ? 'var(--success)' : line.startsWith('✖') ? 'var(--danger)' : line.startsWith('  ›') ? 'var(--accent-hover)' : undefined }}>
              {line || '\u00A0'}
            </div>
          ))}
          {scanning && <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '4px' }}><span className="spinner" /><span>Processing…</span></div>}
        </div>
      )}

      {error && (
        <div className="info-box info-box-danger" style={{ marginTop: '1rem' }}>
          {error}
        </div>
      )}

      {(done || alreadyScanned) && (
        <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-success btn-lg"
            onClick={() => onScanned(project)}
          >
            Proceed to Profile →
          </button>
        </div>
      )}
    </div>
  )
}
