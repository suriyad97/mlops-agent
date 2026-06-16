import { useEffect, useState } from 'react'
import { api, type InfraReport, type DiscoveredConfig, type Project } from '../../api'

interface Props {
  project: Project
  onValidated: (p: Project) => void
}

const PREREQ_STATUS: Record<string, { icon: string; label: string }> = {
  ok:             { icon: '●', label: 'Ready' },
  missing_config: { icon: '○', label: 'Not configured' },
  not_found:      { icon: '○', label: 'Not found' },
  error:          { icon: '○', label: 'Error' },
}
const statusColor = (s: string) =>
  s === 'ok' ? 'var(--success)' : s === 'missing_config' ? 'var(--warn)' : 'var(--danger)'

// ── Data-path fields (now configured here, alongside the infra prerequisites) ──
const DATA_PATH_FIELDS: { key: string; label: string; placeholder: string; strategies?: string[] }[] = [
  { key: 'training_data_path', label: 'Training data',       placeholder: 'azureml://datastores/<ds>/paths/training/' },
  { key: 'eval_data_path',     label: 'Evaluation data',     placeholder: 'optional — falls back to training data' },
  { key: 'batch_input_path',   label: 'Batch input',         placeholder: 'blob path for batch scoring input', strategies: ['batch', 'both'] },
  { key: 'batch_output_path',  label: 'Batch output',        placeholder: 'blob path where predictions land',  strategies: ['batch', 'both'] },
  { key: 'baseline_data_path', label: 'Monitoring baseline', placeholder: 'reference distribution for drift detection' },
]

// ── Deployment pipeline chain (sequenced; qa/prod run under managed identity) ──
type StageKind = 'validate' | 'pipeline' | 'promote'
interface Stage { key: string; env: 'local' | 'dev' | 'qa' | 'prod'; title: string; desc: string; kind: StageKind }

function buildStages(strategy: string): Stage[] {
  const stages: Stage[] = [
    { key: 'validate',     env: 'local', kind: 'validate', title: 'Local validation',
      desc: '4-tier check (YAML → schema → AzDO preview-run → dry-run) with self-healing, before anything runs in the cloud.' },
    { key: 'ci-dev',       env: 'dev',  kind: 'pipeline', title: 'CI — build & register environment',
      desc: 'Build the Docker image, push to ACR, and register the AML environment in dev.' },
    { key: 'promote-qa',   env: 'qa',   kind: 'promote',  title: 'Promote environment → QA',
      desc: 'Copy the validated image/environment from dev to QA. Runs under the QA managed identity in Azure DevOps.' },
    { key: 'promote-prod', env: 'prod', kind: 'promote',  title: 'Promote environment → Prod',
      desc: 'Copy from QA to Prod. Runs under the Prod managed identity — gated by approval in Azure DevOps.' },
    { key: 'train',        env: 'dev',  kind: 'pipeline', title: 'Train & register model',
      desc: 'Run the CT pipeline: train → evaluate → register the model in the AML registry.' },
  ]
  if (strategy === 'realtime' || strategy === 'both')
    stages.push({ key: 'deploy-rt', env: 'dev', kind: 'pipeline', title: 'Deploy real-time endpoint',
      desc: 'CD: create the managed online endpoint with a blue/green deployment + data collector.' })
  if (strategy === 'batch' || strategy === 'both')
    stages.push({ key: 'deploy-batch', env: 'dev', kind: 'pipeline', title: 'Deploy batch endpoint',
      desc: 'CD: create the batch endpoint + deployment for scheduled scoring.' })
  return stages
}

const ENV_BADGE: Record<string, { label: string; color: string; bg: string }> = {
  local: { label: 'LOCAL', color: '#9ca3af', bg: 'rgba(156,163,175,0.12)' },
  dev:   { label: 'DEV',   color: '#60a5fa', bg: 'rgba(96,165,250,0.12)' },
  qa:    { label: 'QA',    color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
  prod:  { label: 'PROD',  color: '#f87171', bg: 'rgba(248,113,113,0.12)' },
}

type StageStatus = 'todo' | 'active' | 'running' | 'done'

export default function Step6Validate({ project, onValidated }: Props) {
  const strategy = String((project.profile as Record<string, unknown>).endpoint_strategy ?? 'both')
  const stages = buildStages(strategy)

  // ── Infra + data paths ──
  const [infraReport,     setInfraReport]     = useState<InfraReport | null>(null)
  const [infraLoading,    setInfraLoading]    = useState(true)
  const [discoverLoading, setDiscoverLoading] = useState(false)
  const [discoverResult,  setDiscoverResult]  = useState<DiscoveredConfig | null>(null)
  const [pathsOpen,       setPathsOpen]       = useState(false)
  const [savingPaths,     setSavingPaths]     = useState(false)
  const [error,           setError]           = useState('')

  const dp0 = (project.profile as Record<string, unknown>).data_paths as Record<string, string> ?? {}
  const [paths, setPaths] = useState<Record<string, string>>(
    Object.fromEntries(DATA_PATH_FIELDS.map(f => [f.key, dp0[f.key] ?? ''])),
  )

  // ── Runner state ──
  const [status, setStatus] = useState<Record<string, StageStatus>>(
    Object.fromEntries(stages.map((s, i) => [s.key, i === 0 ? 'active' : 'todo'])),
  )
  const [runMsg, setRunMsg] = useState('')

  const doneCount = stages.filter(s => status[s.key] === 'done').length
  const progressPct = Math.round((doneCount / stages.length) * 100)
  const activeIdx = stages.findIndex(s => status[s.key] === 'active' || status[s.key] === 'running')

  function refreshInfra() {
    setInfraLoading(true)
    api.checkProjectInfra(project.id)
      .then(setInfraReport)
      .catch(e => setError((e as Error).message))
      .finally(() => setInfraLoading(false))
  }
  useEffect(refreshInfra, [project.id])   // eslint-disable-line react-hooks/exhaustive-deps

  async function discover() {
    setDiscoverLoading(true); setError('')
    try {
      const { discovered, report } = await api.discoverProjectInfra(project.id)
      setDiscoverResult(discovered); setInfraReport(report)
    } catch (e) { setError((e as Error).message) } finally { setDiscoverLoading(false) }
  }

  async function saveDataPaths() {
    setSavingPaths(true); setError('')
    try {
      const data_paths = Object.fromEntries(Object.entries(paths).filter(([, v]) => v.trim()))
      await api.patchProfile(project.id, { data_paths })
      refreshInfra()   // infra check now includes data-path readiness rows
    } catch (e) { setError((e as Error).message) } finally { setSavingPaths(false) }
  }

  function advance(key: string) {
    const idx = stages.findIndex(s => s.key === key)
    setStatus(prev => {
      const next = { ...prev, [key]: 'done' as StageStatus }
      if (idx + 1 < stages.length) next[stages[idx + 1].key] = 'active'
      return next
    })
    // when the whole chain is done, mark the project validated
    if (idx === stages.length - 1) onValidated(project)
  }

  async function runStage(s: Stage) {
    setRunMsg('')
    if (s.kind === 'validate') {
      setStatus(prev => ({ ...prev, [s.key]: 'running' }))
      try {
        await api.validateProject(project.id)
        setRunMsg('✓ Local validation complete — open the Chat panel for the full report.')
        advance(s.key)
        const ps = await api.listProjects()
        const updated = ps.find(p => p.id === project.id)
        if (updated) onValidated(updated)
      } catch (e) {
        setError((e as Error).message)
        setStatus(prev => ({ ...prev, [s.key]: 'active' }))
      }
    } else {
      // Cloud stages run as committed AzDO pipelines (qa/prod under managed identity).
      // We can't connect to them directly, so the user runs them in AzDO and marks done.
      advance(s.key)
    }
  }

  const okCount = infraReport?.checks.filter(c => c.status === 'ok').length ?? 0

  return (
    <div className="step-card-wide">
      <div className="step-header">
        <div className="step-number">Step 6 of 7</div>
        <h2 className="step-title">Validate & roll out</h2>
        <p className="step-desc">
          Confirm Azure infrastructure and data paths are ready, then walk the deployment chain stage by
          stage — CI builds the environment in dev, promote it through QA → Prod (managed identity), then
          train and deploy.
        </p>
      </div>

      {error && <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>{error}</div>}

      {/* ── Infrastructure prerequisites (incl. data paths) ─────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <div className="section-label" style={{ marginBottom: 0 }}>Infrastructure prerequisites</div>
        {infraReport && (
          <span className={`badge ${infraReport.all_ok ? 'badge-success' : 'badge-warn'}`}>
            {okCount}/{infraReport.checks.length} ready
          </span>
        )}
      </div>

      {infraLoading && (
        <div style={{ display: 'flex', gap: '0.6rem', color: 'var(--text-dim)', fontSize: '13px', alignItems: 'center', marginBottom: '1rem' }}>
          <span className="spinner" /> Checking Azure prerequisites & data paths…
        </div>
      )}

      {infraReport && (
        <div style={{ display: 'grid', gap: '0.5rem', marginBottom: '1rem' }}>
          {infraReport.checks.map(item => {
            const st = PREREQ_STATUS[item.status] ?? PREREQ_STATUS.error
            return (
              <div key={item.name} className="prereq-row">
                <span className="prereq-icon" style={{ color: statusColor(item.status) }}>{st.icon}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="prereq-name">{item.name}</div>
                  {item.detail && <div className="prereq-detail">{item.detail}</div>}
                  {item.status !== 'ok' && item.fix && <div className="prereq-fix">→ {item.fix}</div>}
                </div>
                <span className="prereq-status" style={{ color: statusColor(item.status) }}>{st.label}</span>
              </div>
            )
          })}
        </div>
      )}

      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
        {infraReport && !infraReport.all_ok && (
          <button className="btn btn-sm" onClick={discover} disabled={discoverLoading}>
            {discoverLoading ? <><span className="spinner" /> Discovering…</> : '⚡ Auto-discover Azure config'}
          </button>
        )}
        <button className="btn btn-sm" onClick={() => setPathsOpen(o => !o)}>
          {pathsOpen ? '▾ Hide data paths' : '✎ Configure data paths'}
        </button>
      </div>
      {discoverResult && (
        <div className={`info-box ${discoverResult.discovered.length > 0 ? 'info-box-success' : 'info-box-warn'}`} style={{ marginBottom: '0.75rem', fontSize: '12px' }}>
          {discoverResult.discovered.length > 0
            ? `✓ Auto-discovered: ${discoverResult.discovered.join(', ')}`
            : `Nothing found automatically.${discoverResult.errors[0] ? ' ' + discoverResult.errors[0] : ''}`}
        </div>
      )}

      {/* Data-path entry (saved to profile; verified in the infra rows above) */}
      {pathsOpen && (
        <div style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '1rem', marginBottom: '1rem' }}>
          <div className="field-grid field-grid-2">
            {DATA_PATH_FIELDS.filter(f => !f.strategies || f.strategies.includes(strategy)).map(f => (
              <div className="field-group" key={f.key}>
                <label className="field-label">{f.label}</label>
                <input className="input input-mono" placeholder={f.placeholder}
                  value={paths[f.key] ?? ''}
                  onChange={e => setPaths(prev => ({ ...prev, [f.key]: e.target.value }))} />
              </div>
            ))}
          </div>
          <button className="btn btn-primary btn-sm" style={{ marginTop: '0.85rem' }} onClick={saveDataPaths} disabled={savingPaths}>
            {savingPaths ? <><span className="spinner" /> Saving…</> : 'Save data paths & re-check'}
          </button>
        </div>
      )}

      {/* ── Deployment chain runner ─────────────────────────────────────── */}
      <div className="divider" />
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <div className="section-label" style={{ marginBottom: 0 }}>Deployment chain</div>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{doneCount}/{stages.length} stages complete</span>
      </div>

      {/* progress bar */}
      <div style={{ height: 8, background: 'var(--surface-3)', borderRadius: 9999, overflow: 'hidden', marginBottom: '1.25rem' }}>
        <div style={{
          height: '100%', width: `${progressPct}%`,
          background: 'linear-gradient(90deg, var(--accent) 0%, var(--success) 100%)',
          borderRadius: 9999, transition: 'width 0.4s ease',
        }} />
      </div>

      <div style={{ display: 'grid', gap: '0.7rem' }}>
        {stages.map((s, i) => {
          const stStatus = status[s.key]
          const env = ENV_BADGE[s.env]
          const isActive = stStatus === 'active' || stStatus === 'running'
          const isDone = stStatus === 'done'
          return (
            <div key={s.key} style={{
              display: 'flex', gap: '0.9rem', alignItems: 'flex-start',
              padding: '0.9rem 1.1rem', borderRadius: 10,
              border: `1px solid ${isActive ? 'var(--accent)' : 'var(--border)'}`,
              background: isActive ? 'var(--accent-soft)' : 'var(--surface)',
              opacity: stStatus === 'todo' ? 0.6 : 1,
              transition: 'all 0.2s ease',
            }}>
              {/* stage dot */}
              <div style={{
                width: 26, height: 26, borderRadius: '50%', flexShrink: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '12px', fontWeight: 700,
                background: isDone ? 'var(--success)' : isActive ? 'var(--accent)' : 'transparent',
                color: isDone || isActive ? '#fff' : 'var(--text-muted)',
                border: isDone || isActive ? 'none' : '1.5px solid var(--border-strong)',
              }}>
                {isDone ? '✓' : stStatus === 'running' ? <span className="spinner" style={{ width: 12, height: 12 }} /> : i + 1}
              </div>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text)' }}>{s.title}</span>
                  <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em', color: env.color, background: env.bg, border: `1px solid ${env.color}40`, borderRadius: 9999, padding: '0.1rem 0.5rem' }}>
                    {env.label}
                  </span>
                  {s.kind === 'promote' && (
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>managed identity</span>
                  )}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: 1.5 }}>{s.desc}</div>

                {isActive && (
                  <div style={{ marginTop: '0.7rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {s.kind === 'validate' ? (
                      <button className="btn btn-primary btn-sm" onClick={() => runStage(s)} disabled={stStatus === 'running'}>
                        {stStatus === 'running' ? <><span className="spinner" /> Validating…</> : '▶ Run local validation'}
                      </button>
                    ) : s.kind === 'promote' ? (
                      <>
                        <button className="btn btn-sm" onClick={() => runStage(s)}>
                          ✓ Mark {env.label} promotion done
                        </button>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)', alignSelf: 'center' }}>
                          Trigger the promotion pipeline in Azure DevOps — it runs under the {env.label} managed identity.
                        </span>
                      </>
                    ) : (
                      <>
                        <button className="btn btn-sm" onClick={() => runStage(s)}>
                          ✓ Mark stage complete
                        </button>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)', alignSelf: 'center' }}>
                          Run the committed AzDO pipeline for this stage, then mark it complete to continue.
                        </span>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {runMsg && <div className="info-box info-box-success" style={{ marginTop: '1rem' }}>{runMsg}</div>}

      {doneCount === stages.length ? (
        <div style={{ marginTop: '1.75rem', borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
          <div className="info-box info-box-success" style={{ marginBottom: '1rem' }}>
            ✓ Full deployment chain complete — environment promoted through prod, model trained and deployed.
          </div>
          <button className="btn btn-success btn-lg" onClick={() => onValidated(project)}>
            Proceed to Commit &amp; Ship →
          </button>
        </div>
      ) : (
        <p style={{ marginTop: '1.25rem', fontSize: '12px', color: 'var(--text-muted)' }}>
          Complete each stage in order — the bar advances as you go. Stage {activeIdx + 1} of {stages.length} is active.
        </p>
      )}
    </div>
  )
}
