import { useEffect, useState } from 'react'
import { api, type GenerationReport, type Project } from '../../api'

interface Props {
  project: Project
  onGenerated: (p: Project) => void
}

// ── File classification ────────────────────────────────────────────────────────

const EXT_ICON: Record<string, string> = {
  '.py':   '🐍',
  '.yml':  '📋',
  '.yaml': '📋',
  '.json': '📄',
  '.txt':  '📝',
  '.md':   '📝',
  '.sh':   '⚙️',
  '.env':  '🔑',
  '.toml': '⚙️',
  '.cfg':  '⚙️',
}

const CAPABILITY_META: Record<string, { label: string; icon: string; color: string; desc: string }> = {
  environment_lifecycle: {
    label: 'Environment & CI',
    icon: '🏗️',
    color: '#818cf8',
    desc: 'Docker/conda environment definition and CI pipeline that builds & publishes the image to ACR',
  },
  training: {
    label: 'Training Pipeline',
    icon: '🎓',
    color: '#34d399',
    desc: 'AzDO CI/CT pipeline + AML pipeline YAML + Python training/evaluation scripts',
  },
  model_lifecycle: {
    label: 'Model Lifecycle',
    icon: '📦',
    color: '#a78bfa',
    desc: 'Model registration, versioning, and promotion between AML model registry stages',
  },
  realtime_deployment: {
    label: 'Real-time Deployment',
    icon: '⚡',
    color: '#f59e0b',
    desc: 'AzDO CD pipeline + AML online endpoint + scoring script for low-latency inference',
  },
  batch_deployment: {
    label: 'Batch Deployment',
    icon: '📦',
    color: '#60a5fa',
    desc: 'AzDO CD pipeline + AML batch endpoint + batch scoring script for large-scale inference',
  },
  monitoring: {
    label: 'Drift Monitoring',
    icon: '📊',
    color: '#fb7185',
    desc: 'AzDO scheduled pipeline + AML monitoring job + PSI-based drift detection script',
  },
  retraining: {
    label: 'Retraining Pipeline',
    icon: '🔄',
    color: '#34d399',
    desc: 'Auto-triggered retraining loop with threshold guard, triggered when PSI exceeds retrain threshold',
  },
  rollback: {
    label: 'Rollback Pipeline',
    icon: '⏪',
    color: '#f87171',
    desc: 'AzDO pipeline to redeploy a previous registered model version on demand',
  },
}

const FILE_KIND_META = {
  written:    { label: 'New',        color: '#34d399', bg: 'rgba(34,211,160,0.08)',  border: 'rgba(34,211,160,0.2)',  icon: '✨', desc: 'Freshly generated from template' },
  adapter:    { label: 'Adapter',    color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.2)',  icon: '⚙️', desc: 'Thin wrapper — calls your existing code' },
  scaffold:   { label: 'Scaffold',   color: '#818cf8', bg: 'rgba(99,102,241,0.08)', border: 'rgba(99,102,241,0.2)',  icon: '📝', desc: 'TODO stub — you must implement the logic' },
  wired:      { label: 'Reused',     color: '#6b7280', bg: 'rgba(107,114,128,0.06)', border: 'rgba(107,114,128,0.15)', icon: '✓',  desc: 'Your existing script reused as-is' },
  superseded: { label: 'Superseded', color: '#f87171', bg: 'rgba(248,113,113,0.08)', border: 'rgba(248,113,113,0.2)', icon: '🗑',  desc: 'Legacy file — review and delete before commit' },
}

type FileKind = keyof typeof FILE_KIND_META

function fileKind(path: string, report: GenerationReport): FileKind {
  if (report.adapter_files.includes(path))    return 'adapter'
  if (report.scaffold_files.includes(path))   return 'scaffold'
  if (report.wired_skipped.includes(path))    return 'wired'
  if (report.superseded_files.includes(path)) return 'superseded'
  return 'written'
}

function fileExt(path: string): string {
  const dot = path.lastIndexOf('.')
  return dot >= 0 ? path.slice(dot) : ''
}

function fileIcon(path: string): string {
  return EXT_ICON[fileExt(path)] ?? '📄'
}

function filePurpose(path: string, component: string, capability: string): string {
  const p = path.toLowerCase()
  if (p.includes('azure-pipeline') || p.match(/azdopipelines\//)) {
    if (p.includes('ci')) return 'AzDO CI pipeline — builds & publishes the container image'
    if (p.includes('ct') || p.includes('train')) return 'AzDO CT pipeline — triggers AML training run'
    if (p.includes('cd') || p.includes('deploy')) return 'AzDO CD pipeline — deploys the registered model to endpoint'
    if (p.includes('monitor')) return 'AzDO scheduled pipeline — triggers drift monitoring job'
    if (p.includes('retrain')) return 'AzDO pipeline — auto-triggered retraining on drift alert'
    if (p.includes('rollback')) return 'AzDO pipeline — redeploys a previous model version'
    return 'AzDO pipeline definition'
  }
  if (p.match(/aml\/pipelines\//)) {
    if (p.includes('train')) return 'AML pipeline YAML — orchestrates data prep → train → evaluate steps'
    if (p.includes('batch')) return 'AML batch pipeline — orchestrates large-scale batch scoring'
    if (p.includes('monitor')) return 'AML monitoring pipeline — computes drift metrics and alerts'
    if (p.includes('retrain')) return 'AML retraining pipeline — re-runs training with fresh data'
    return 'AML pipeline orchestration YAML'
  }
  if (p.match(/aml\/components\//)) return 'AML component YAML — defines inputs/outputs/command for a pipeline step'
  if (p.match(/aml\/environments\//)) return 'AML environment YAML — conda/pip deps + base image for compute'
  if (p.match(/aml\/models\//)) return 'AML model asset YAML — model registration metadata'
  if (p.match(/aml\/endpoints\//)) {
    if (p.includes('batch')) return 'AML batch endpoint definition — scoring endpoint for batch jobs'
    return 'AML online endpoint definition — low-latency REST scoring endpoint'
  }
  if (p.match(/aml\/deployments\//)) return 'AML deployment YAML — maps model version to endpoint with instance config'
  if (p.endsWith('train.py') || p.endsWith('training.py')) return 'Python training script — loads data, trains, evaluates, and registers the model'
  if (p.endsWith('evaluate.py') || p.endsWith('evaluation.py')) return 'Python evaluation script — loads the model, runs metrics, and gate-checks thresholds'
  if (p.includes('score') && p.endsWith('.py')) return 'Python scoring script — init() loads the model, run() handles inference requests'
  if (p.includes('drift') && p.endsWith('.py')) return 'Python drift detection script — computes PSI between baseline and current data'
  if (p.includes('threshold') && p.endsWith('.py')) return 'Python threshold evaluation script — blocks retraining if drift is below severity threshold'
  if (p.includes('rollback') && p.endsWith('.py')) return 'Python rollback script — queries model registry and redeploys a prior version'
  if (p.endsWith('dockerfile') || p.includes('docker')) return 'Dockerfile — container image definition for the AML environment'
  if (p.endsWith('environment.yml') || p.endsWith('conda.yml')) return 'Conda environment file — Python package dependencies for training/scoring'
  if (p.endsWith('requirements.txt')) return 'pip requirements — runtime Python dependencies'
  if (p.includes('test') && p.endsWith('.py')) return 'Python unit test — validates the generated script interface'
  return `${component} — ${capability} capability`
}

// ── Component: single file row ──────────────────────────────────────────────
function FileRow({ path, kind }: { path: string; kind: FileKind }) {
  const meta = FILE_KIND_META[kind]
  const parts = path.split('/')
  const name = parts.pop() ?? path
  const dir = parts.join('/') ? parts.join('/') + '/' : ''

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.65rem',
      padding: '0.5rem 0.75rem',
      background: meta.bg,
      border: `1px solid ${meta.border}`,
      borderRadius: 7,
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: '12px',
    }}>
      <span style={{ fontSize: '14px', flexShrink: 0 }}>{fileIcon(path)}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{dir}</span>
        <span style={{ color: 'var(--text)', fontWeight: 500 }}>{name}</span>
      </div>
      <span style={{
        fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
        color: meta.color, flexShrink: 0,
      }}>
        {meta.icon} {meta.label}
      </span>
    </div>
  )
}

// ── Component: capability section ───────────────────────────────────────────
function CapabilitySection({
  capKey, files, report,
}: {
  capKey: string
  files: { path: string; component: string }[]
  report: GenerationReport
}) {
  const [open, setOpen] = useState(true)
  const meta = CAPABILITY_META[capKey] ?? {
    label: capKey, icon: '📦', color: 'var(--text-dim)', desc: '',
  }

  const counts = { written: 0, adapter: 0, scaffold: 0, wired: 0, superseded: 0 }
  for (const { path } of files) counts[fileKind(path, report)]++

  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: 10,
      overflow: 'hidden',
      background: 'var(--surface)',
    }}>
      {/* Header */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'flex-start', gap: '0.9rem',
          padding: '0.9rem 1.1rem', background: 'transparent', border: 'none', cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        <span style={{ fontSize: '20px', flexShrink: 0, marginTop: 1 }}>{meta.icon}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '14px', fontWeight: 700, color: meta.color }}>{meta.label}</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {files.length} file{files.length !== 1 ? 's' : ''}
            </span>
            {/* Kind pill summary */}
            {counts.written > 0    && <span style={{ fontSize: '10px', background: 'rgba(34,211,160,0.12)', color: '#34d399', border: '1px solid rgba(34,211,160,0.25)', borderRadius: 9999, padding: '0.1rem 0.5rem', fontWeight: 600 }}>✨ {counts.written} new</span>}
            {counts.adapter > 0    && <span style={{ fontSize: '10px', background: 'rgba(245,158,11,0.12)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.25)', borderRadius: 9999, padding: '0.1rem 0.5rem', fontWeight: 600 }}>⚙️ {counts.adapter} adapter</span>}
            {counts.scaffold > 0   && <span style={{ fontSize: '10px', background: 'rgba(99,102,241,0.12)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 9999, padding: '0.1rem 0.5rem', fontWeight: 600 }}>📝 {counts.scaffold} scaffold</span>}
            {counts.wired > 0      && <span style={{ fontSize: '10px', background: 'rgba(107,114,128,0.08)', color: '#9ca3af', border: '1px solid rgba(107,114,128,0.2)', borderRadius: 9999, padding: '0.1rem 0.5rem', fontWeight: 600 }}>✓ {counts.wired} reused</span>}
            {counts.superseded > 0 && <span style={{ fontSize: '10px', background: 'rgba(248,113,113,0.12)', color: '#f87171', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 9999, padding: '0.1rem 0.5rem', fontWeight: 600 }}>🗑 {counts.superseded} superseded</span>}
          </div>
          {meta.desc && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '0.2rem', fontFamily: 'Inter, sans-serif', fontStyle: 'italic' }}>
              {meta.desc}
            </div>
          )}
        </div>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)', flexShrink: 0, marginTop: 4 }}>
          {open ? '▾' : '▸'}
        </span>
      </button>

      {/* File table */}
      {open && (
        <div style={{ padding: '0 1rem 1rem', display: 'grid', gap: '0.35rem' }}>
          {/* Column headers */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr auto', gap: '0.5rem', padding: '0 0.5rem', marginBottom: '0.2rem' }}>
            <span style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)' }}>File</span>
            <span style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)' }}>Purpose</span>
            <span style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)' }}>Kind</span>
          </div>

          {files.map(({ path, component }) => {
            const kind = fileKind(path, report)
            const meta2 = FILE_KIND_META[kind]
            const parts = path.split('/')
            const name = parts.pop() ?? path
            const dir = parts.join('/') ? parts.join('/') + '/' : ''
            const purpose = filePurpose(path, component, capKey)

            return (
              <div key={path} style={{
                display: 'grid', gridTemplateColumns: '1fr 2fr auto',
                gap: '0.65rem', alignItems: 'start',
                padding: '0.55rem 0.75rem',
                background: meta2.bg,
                border: `1px solid ${meta2.border}`,
                borderRadius: 7,
              }}>
                {/* Filename */}
                <div style={{ fontFamily: 'JetBrains Mono, monospace', minWidth: 0 }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{dir}</div>
                  <div style={{ fontSize: '12.5px', color: 'var(--text)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span>{fileIcon(path)}</span>
                    <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</span>
                  </div>
                </div>

                {/* Purpose */}
                <div style={{ fontSize: '12px', color: 'var(--text-dim)', lineHeight: 1.5, paddingTop: '1px' }}>
                  {purpose}
                </div>

                {/* Kind badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', whiteSpace: 'nowrap' }}>
                  <span style={{
                    fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em',
                    color: meta2.color,
                    background: meta2.bg,
                    border: `1px solid ${meta2.border}`,
                    borderRadius: 9999,
                    padding: '0.15rem 0.55rem',
                  }}>
                    {meta2.icon} {meta2.label}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Superseded section ──────────────────────────────────────────────────────
function SupersededSection({ files }: { files: string[] }) {
  const [open, setOpen] = useState(false)
  if (files.length === 0) return null
  return (
    <div style={{ border: '1px solid rgba(248,113,113,0.25)', borderRadius: 10, overflow: 'hidden', background: 'rgba(248,113,113,0.04)' }}>
      <button onClick={() => setOpen(o => !o)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '0.9rem', padding: '0.9rem 1.1rem', background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left' }}>
        <span style={{ fontSize: '20px' }}>🗑</span>
        <div style={{ flex: 1 }}>
          <span style={{ fontSize: '14px', fontWeight: 700, color: '#f87171' }}>Legacy / Superseded Files</span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '0.6rem' }}>{files.length} file{files.length !== 1 ? 's' : ''}</span>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '0.1rem' }}>
            Non-standard pipeline files replaced by the generated standard set. Review and delete before committing.
          </div>
        </div>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div style={{ padding: '0 1rem 1rem', display: 'grid', gap: '0.35rem' }}>
          {files.map(f => <FileRow key={f} path={f} kind="superseded" />)}
        </div>
      )}
    </div>
  )
}

// ── Main Step ───────────────────────────────────────────────────────────────
export default function Step5Generate({ project, onGenerated }: Props) {
  const [generating, setGenerating] = useState(false)
  const [report,     setReport]     = useState<GenerationReport | null>(null)
  const [error,      setError]      = useState('')

  const alreadyGenerated = ['generated', 'validated_local', 'committed', 'operational'].includes(project.stage)

  // Load the PERSISTED generation report so the breakdown shows even after a reload or
  // when returning to this step — not just immediately after clicking Generate.
  useEffect(() => {
    if (report) return
    let cancelled = false
    ;(async () => {
      try {
        const r = await api.getGenerationReport(project.id)
        if (!cancelled && r.generated) setReport(r)
      } catch { /* nothing generated yet */ }
    })()
    return () => { cancelled = true }
  }, [project.id])  // eslint-disable-line react-hooks/exhaustive-deps

  async function generate() {
    setGenerating(true); setError(''); setReport(null)
    try {
      const r = await api.generateProject(project.id)
      setReport(r as unknown as GenerationReport)
      const projects = await api.listProjects()
      const updated = projects.find(p => p.id === project.id)
      if (updated) onGenerated(updated)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setGenerating(false)
    }
  }

  // Build capability → files map from the components list
  const capabilityFiles: Record<string, { path: string; component: string }[]> = {}
  if (report?.components) {
    for (const comp of report.components) {
      if (!capabilityFiles[comp.capability]) capabilityFiles[comp.capability] = []
      for (const f of comp.files) {
        capabilityFiles[comp.capability].push({ path: f, component: comp.component })
      }
    }
    // Also slot wired_skipped files into their capability (they have no written file entry)
    // They're already accounted for in the capability's note, so we add them as a "wired" placeholder
  }

  const capKeys = Object.keys(capabilityFiles)
  const totalFiles = report ? report.written_files.length : 0
  const totalCaps = capKeys.length

  return (
    <div className="step-card-wide">
      <div className="step-header">
        <div className="step-number">Step 5 of 7</div>
        <h2 className="step-title">Generate missing pipelines</h2>
        <p className="step-desc">
          The platform generates all missing MLOps pipeline files, AML assets, and Python scripts
          based on your profile and code checklist. Your existing source code is <strong>never modified</strong>.
        </p>
      </div>

      {/* Generate CTA */}
      {!report && (
        <>
          {alreadyGenerated && (
            <div className="info-box info-box-accent" style={{ marginBottom: '1.5rem' }}>
              ✓ Pipelines were previously generated (stage: <strong>{project.stage}</strong>).
              Re-generate to pick up profile changes.
            </div>
          )}
          <div style={{ textAlign: 'center', padding: '2.5rem 0' }}>
            <button
              id="generate-btn"
              className="btn btn-primary btn-lg pulse-glow"
              onClick={generate}
              disabled={generating}
              style={{ minWidth: 280, fontSize: '15px', padding: '0.9rem 2.2rem', borderRadius: 'var(--radius)' }}
            >
              {generating
                ? <><span className="spinner" style={{ width: 18, height: 18 }} /> Generating pipelines…</>
                : '⚡ Generate Missing Pipelines'}
            </button>
            <p style={{ marginTop: '0.85rem', fontSize: '12px', color: 'var(--text-muted)' }}>
              Files are written to your local working tree only — nothing is committed until Step 7.
            </p>
          </div>
        </>
      )}

      {error && <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>{error}</div>}

      {/* ── Results ─────────────────────────────────────────────────────────── */}
      {report && (
        <>
          {/* Summary bar */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap',
            background: 'rgba(34,211,160,0.06)', border: '1px solid rgba(34,211,160,0.2)',
            borderRadius: 'var(--radius-sm)', padding: '0.9rem 1.25rem', marginBottom: '1.75rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '22px' }}>✅</span>
              <div>
                <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--success)' }}>Generation complete</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{report.summary}</div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginLeft: 'auto' }}>
              {[
                { n: totalFiles,                   label: 'Files written',  color: 'var(--success)' },
                { n: totalCaps,                    label: 'Capabilities',   color: 'var(--accent-hover)' },
                { n: report.adapter_files.length,  label: 'Adapters',       color: '#f59e0b' },
                { n: report.scaffold_files.length, label: 'Scaffolds',      color: '#818cf8' },
                { n: report.wired_skipped.length,  label: 'Reused',         color: 'var(--text-muted)' },
              ].map(s => (
                <div key={s.label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '20px', fontWeight: 700, color: s.color }}>{s.n}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Re-generation note: created vs already-existed */}
          {report.updated_files && report.updated_files.length > 0 && (
            <div className="info-box info-box-accent" style={{ marginBottom: '1.25rem', fontSize: '12px' }}>
              ↻ <strong>{report.created_files?.length ?? 0}</strong> file(s) newly created · <strong>{report.updated_files.length}</strong> already
              existed and were overwritten with the standard version (idempotent — re-generating is safe).
            </div>
          )}

          {/* Legend */}
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1.25rem', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Legend:</span>
            {(Object.entries(FILE_KIND_META) as [FileKind, typeof FILE_KIND_META[FileKind]][]).map(([k, m]) => (
              <span key={k} style={{
                fontSize: '11px', fontWeight: 600, padding: '0.15rem 0.65rem',
                background: m.bg, border: `1px solid ${m.border}`, borderRadius: 9999,
                color: m.color, display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
              }}>
                {m.icon} {m.label} — <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}>{m.desc}</span>
              </span>
            ))}
          </div>

          {/* Capability sections */}
          <div style={{ display: 'grid', gap: '0.85rem', marginBottom: '1.5rem' }}>
            {capKeys.map(cap => (
              <CapabilitySection
                key={cap}
                capKey={cap}
                files={capabilityFiles[cap]}
                report={report}
              />
            ))}
            <SupersededSection files={report.superseded_files} />
          </div>

          {/* Action row */}
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', borderTop: '1px solid var(--border)', paddingTop: '1.25rem' }}>
            <button className="btn btn-success btn-lg" onClick={() => onGenerated(project)}>
              Proceed to Validate →
            </button>
            <button className="btn" onClick={generate} disabled={generating}>
              {generating ? <><span className="spinner" /> Re-generating…</> : '↺ Re-generate'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}
