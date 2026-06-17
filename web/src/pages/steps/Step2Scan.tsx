import { useState } from 'react'
import { api, type Project } from '../../api'

interface Props {
  project: Project
  onScanned: (p: Project) => void
}

export default function Step2Scan({ project, onScanned }: Props) {
  const [scanning, setScanning] = useState(false)
  const [localPath, setLocalPath] = useState(project.local_repo_path || '')
  const [log, setLog] = useState<string[]>([])
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)
  const [scannedProfile, setScannedProfile] = useState<Record<string, unknown> | null>(null)

  async function handleBrowse() {
    const { path } = await api.browsePath()
    if (path) setLocalPath(path)
  }

  async function startScan() {
    if (!localPath.trim()) {
      setError('Please select or enter a local folder path to proceed.')
      return
    }
    setScanning(true)
    setError('')
    setLog([])
    setDone(false)

    try {
      setLog(prev => [...prev, `📁 Target folder: ${localPath}`, '⟶ Cloning/updating repository…'])
      const result = await api.scanProject(project.id, localPath, project.base_branch)

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
        '✅ Scan complete — review the summary below',
      ])
      setScannedProfile(profile)
      setDone(true)   // stay on this step so the user sees the repo summary
    } catch (e) {
      setError((e as Error).message)
      setLog(prev => [...prev, `✖ Error: ${(e as Error).message}`])
    } finally {
      setScanning(false)
    }
  }

  async function proceed() {
    const projects = await api.listProjects()
    const updated = projects.find(p => p.id === project.id)
    onScanned(updated ?? project)
  }

  const alreadyScanned = ['scanned', 'evaluated', 'generated', 'validated_local', 'committed', 'operational'].includes(project.stage)
  const summary = scannedProfile ?? (alreadyScanned ? (project.profile as Record<string, unknown>) : null)

  const asList = (v: unknown): string => Array.isArray(v) ? v.join(', ') : String(v ?? '—')

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

      <div className="field-group" style={{ marginBottom: '1.5rem' }}>
        <label className="field-label">Local Folder Path</label>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            className="input input-mono"
            style={{ flex: 1 }}
            placeholder="C:\Users\Suriya\Downloads\my_repo"
            value={localPath}
            onChange={e => setLocalPath(e.target.value)}
            disabled={scanning}
          />
          <button className="btn" onClick={handleBrowse} disabled={scanning}>
            Browse…
          </button>
        </div>
        <span className="field-hint">The folder where the repository should be cloned. If you enter an existing clone, it will be reused.</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <button
          id="scan-btn"
          className="btn btn-primary btn-lg"
          onClick={startScan}
          disabled={scanning || !localPath.trim()}
        >
          {scanning
            ? <><span className="spinner" /> Scanning…</>
            : alreadyScanned ? '🔄 Re-scan repository' : '🔍 Start scan'}
        </button>
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

      {/* ── Repository summary ─────────────────────────────────────────── */}
      {summary && (
        <div style={{ marginTop: '1.5rem', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '1.25rem' }}>
          <div className="section-label" style={{ marginBottom: '0.6rem' }}>📦 Repository summary</div>
          {String(summary.inference_reasoning ?? summary.reasoning ?? '') && (
            <p style={{ fontSize: '13.5px', color: 'var(--text)', lineHeight: 1.6, marginBottom: '1rem' }}>
              {String(summary.inference_reasoning ?? summary.reasoning ?? '')}
            </p>
          )}
          <div className="field-grid field-grid-2" style={{ gap: '0.75rem 1.5rem' }}>
            {([
              ['Project type', asList(summary.project_type)],
              ['Target variable', asList(summary.target_variable)],
              ['Metrics', asList(summary.metrics)],
              ['Endpoint strategy', asList(summary.endpoint_strategy)],
              ['Drift kinds', asList(summary.drift_kinds)],
              ['Inference confidence', summary.inference_confidence != null ? `${Math.round(Number(summary.inference_confidence) * 100)}%` : '—'],
            ] as [string, string][]).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', fontSize: '12.5px', borderBottom: '1px solid var(--border)', paddingBottom: '0.4rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                <span style={{ color: 'var(--text)', fontWeight: 500, textAlign: 'right' }}>{v || '—'}</span>
              </div>
            ))}
          </div>
          {Array.isArray(summary.entry_points) && (summary.entry_points as unknown[]).length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.4rem' }}>Key entry points</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {(summary.entry_points as unknown[]).slice(0, 8).map((ep, i) => (
                  <span key={i} style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', background: 'var(--surface-3)', border: '1px solid var(--border)', borderRadius: 6, padding: '0.15rem 0.5rem', color: 'var(--text-dim)' }}>
                    {String(ep)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {(done || alreadyScanned) && (
        <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-success btn-lg" onClick={proceed}>
            Proceed to Profile →
          </button>
        </div>
      )}
    </div>
  )
}
