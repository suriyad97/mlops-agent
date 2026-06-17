import { useEffect, useState } from 'react'
import { api, type ContractManifest, type ContractStage, type Project } from '../../api'

interface Props {
  project: Project
  onSignedOff: (p: Project) => void
}

function modeOf(s: ContractStage): ContractStage['mode'] {
  if (!s.present) return 'scaffold'
  if (s.meets_contract) return 'wired'
  return 'adapter'
}

const MODE_META: Record<ContractStage['mode'], { label: string; cls: string; icon: string }> = {
  wired:    { label: 'Reuse your code',   cls: 'mode-wired',    icon: '✓' },
  adapter:  { label: 'Generate adapter',  cls: 'mode-adapter',  icon: '⚙' },
  scaffold: { label: 'Generate scaffold', cls: 'mode-scaffold',  icon: '⬜' },
}

// ---------------------------------------------------------------------------
// Static config — defined at module level (not inside the component)
// ---------------------------------------------------------------------------
const GENERATED_SCRIPTS: Record<string, { label: string; files: { name: string; desc: string; type: 'py' | 'yml' }[] }> = {
  training: {
    label: 'Training',
    files: [
      { name: 'MLpipelines/training_pipeline.yml',  desc: 'AML pipeline YAML — orchestrates train + evaluate steps', type: 'yml' },
      { name: 'azdopipelines/ci-train.yml',          desc: 'Azure DevOps CI pipeline — triggers on code push', type: 'yml' },
      { name: 'azdopipelines/cd-train.yml',          desc: 'Azure DevOps CD pipeline — promotes trained model', type: 'yml' },
      { name: 'aml/environment.yml',                 desc: 'Conda environment definition for AML compute', type: 'yml' },
      { name: 'aml/register_model.py',               desc: 'Registers the trained model in AML registry', type: 'py' },
    ],
  },
  realtime_deployment: {
    label: 'Real-time Deployment',
    files: [
      { name: 'MLpipelines/realtime_deployment_pipeline.yml', desc: 'AML pipeline — builds & deploys online endpoint', type: 'yml' },
      { name: 'azdopipelines/cd-realtime.yml',                desc: 'AzDO CD pipeline — gate-guarded deployment', type: 'yml' },
      { name: 'src/score.py',                                  desc: 'Scoring entry-point (init/run contract)', type: 'py' },
    ],
  },
  batch_deployment: {
    label: 'Batch Deployment',
    files: [
      { name: 'MLpipelines/batch_deployment_pipeline.yml', desc: 'AML pipeline — batch endpoint registration & deployment', type: 'yml' },
      { name: 'azdopipelines/cd-batch.yml',                desc: 'AzDO CD pipeline for batch scoring', type: 'yml' },
      { name: 'src/batch_score.py',                        desc: 'Batch scoring entry-point (run with mini-batch)', type: 'py' },
    ],
  },
  monitoring: {
    label: 'Drift Monitoring (Data + Prediction)',
    files: [
      { name: 'src/monitoring/detect_drift.py',       desc: 'PSI-based data & prediction drift detection script', type: 'py' },
      { name: 'MLpipelines/monitoring_pipeline.yml',  desc: 'Scheduled AML pipeline — runs drift check on cron', type: 'yml' },
      { name: 'azdopipelines/cm-monitoring.yml',      desc: 'Continuous monitoring AzDO pipeline', type: 'yml' },
    ],
  },
  retraining: {
    label: 'Concept Drift & Retraining (Separate Pipeline)',
    files: [
      { name: 'src/monitoring/evaluate_thresholds.py', desc: 'Evaluates model performance vs threshold — triggers retraining if labels indicate concept drift', type: 'py' },
      { name: 'MLpipelines/retraining_pipeline.yml',   desc: 'Standalone retraining pipeline — fires when concept drift is confirmed via ground-truth labels', type: 'yml' },
      { name: 'azdopipelines/cd-retrain.yml',          desc: 'AzDO CD pipeline for gated retraining', type: 'yml' },
    ],
  },
  rollback: {
    label: 'Model Rollback',
    files: [
      { name: 'src/rollback/rollback_model.py',    desc: 'Reverts to previous production model version', type: 'py' },
      { name: 'MLpipelines/rollback_pipeline.yml', desc: 'AML pipeline — safe rollback with approval gate', type: 'yml' },
    ],
  },
}

export default function Step4Verify({ project, onSignedOff }: Props) {
  const [manifest,  setManifest]  = useState<ContractManifest | null>(null)
  const [strategy,  setStrategy]  = useState('')
  const [loading,   setLoading]   = useState(true)
  const [saving,    setSaving]    = useState(false)
  const [signedOff, setSignedOff] = useState(false)
  const [error,     setError]     = useState('')
  // Panel open/close state — MUST be at top level, before any conditional returns
  const [scriptsOpen, setScriptsOpen] = useState(false)
  const [driftOpen,   setDriftOpen]   = useState(false)

  useEffect(() => {
    api.getContract(project.id)
      .then(({ endpoint_strategy, contract }) => {
        setManifest(contract)
        setStrategy(endpoint_strategy)
        setSignedOff(false)
      })
      .catch(e => setError((e as Error).message))
      .finally(() => setLoading(false))
  }, [project.id])

  function toggleStage(i: number, field: 'present' | 'meets_contract', v: boolean) {
    setManifest(prev => {
      if (!prev) return prev
      const stages = prev.stages.map((s, j) => j === i ? { ...s, [field]: v } : s)
      return { ...prev, stages }
    })
  }

  function updatePath(i: number, v: string) {
    setManifest(prev => {
      if (!prev) return prev
      const stages = prev.stages.map((s, j) => j === i ? { ...s, detected_path: v } : s)
      return { ...prev, stages }
    })
  }

  function toggleDev(field: 'has_eda' | 'has_shap' | 'has_feature_engineering', v: boolean) {
    setManifest(prev => prev ? { ...prev, [field]: v } : prev)
  }

  async function signOff() {
    if (!manifest) return
    setSaving(true); setError('')
    try {
      const { contract } = await api.saveContract(project.id, manifest)
      setManifest(contract)
      setSignedOff(true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="step-card" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-dim)' }}>
        <span className="spinner spinner-lg" />
        Detecting your ML scripts from the knowledge graph…
      </div>
    )
  }

  return (
    <div className="step-card-wide">
      <div className="step-header">
        <div className="step-number">Step 4 of 7</div>
        <h2 className="step-title">Verify your ML scripts</h2>
        <p className="step-desc">
          We auto-detected which scripts exist in your repo. Review and confirm — the platform uses this
          to decide what to reuse, wrap, or generate from scratch. <strong>Your source code is never modified.</strong>
        </p>
        {strategy && (
          <div style={{ marginTop: '0.75rem' }}>
            <span className="badge badge-accent">Strategy: {strategy}</span>
          </div>
        )}
      </div>

      {/* ── Drift methodology callout ──────────────────────────────────────── */}
      <div style={{ marginBottom: '1.25rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
        <button
          type="button"
          onClick={() => setDriftOpen(o => !o)}
          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem 1rem', background: 'var(--surface-2)', border: 'none', cursor: 'pointer', fontSize: '13px', fontWeight: 600, color: 'var(--accent-hover)' }}
        >
          <span>📊 How drift is calculated in your pipelines</span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{driftOpen ? '▲ Hide' : '▼ Show'}</span>
        </button>
        {driftOpen && (
          <div style={{ padding: '1rem', fontSize: '12.5px', lineHeight: '1.7', color: 'var(--text-dim)', display: 'grid', gap: '0.9rem' }}>
            <div style={{ display: 'grid', gap: '0.6rem' }}>
              <div style={{ background: 'var(--surface-2)', borderRadius: 'var(--radius-xs)', padding: '0.7rem 0.9rem', borderLeft: '3px solid #60a5fa' }}>
                <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '0.25rem' }}>📉 Data Drift — <code>detect_drift.py</code></div>
                Uses <strong>PSI (Population Stability Index)</strong> per numeric feature. Compares your baseline dataset (at training time) to production inputs. Fires an alert when PSI exceeds your configured threshold (default 0.2). <em>Runs on the monitoring cron schedule.</em>
              </div>
              <div style={{ background: 'var(--surface-2)', borderRadius: 'var(--radius-xs)', padding: '0.7rem 0.9rem', borderLeft: '3px solid #a78bfa' }}>
                <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '0.25rem' }}>🎯 Prediction Drift — <code>detect_drift.py</code></div>
                Tracked in the <strong>same monitoring pipeline</strong> as data drift. Compares the distribution of model outputs (prediction scores / classes) in the baseline vs production windows using PSI. A shift in predictions without a shift in inputs suggests a covariate or distribution change.
              </div>
              <div style={{ background: 'var(--surface-2)', borderRadius: 'var(--radius-xs)', padding: '0.7rem 0.9rem', borderLeft: '3px solid #34d399' }}>
                <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: '0.25rem' }}>🧠 Concept Drift — <code>evaluate_thresholds.py</code> (Separate Pipeline)</div>
                Tracked <strong>independently</strong> in the retraining pipeline. Uses <strong>actual ground-truth labels</strong> (e.g. churn confirmed, fraud confirmed) once they arrive, comparing realized performance metrics against your configured thresholds. If performance drops below threshold, the retraining pipeline is triggered. This is inherently a lagged signal and is deliberately kept separate from the data-drift monitor.
              </div>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', paddingTop: '0.25rem' }}>
              ℹ️ The retrain threshold (PSI) you set in the profile controls when the retraining pipeline fires in response to data/prediction drift. Concept drift retraining fires on model performance degradation, not PSI.
            </div>
          </div>
        )}
      </div>

      {/* ── Generated scripts panel ────────────────────────────────────────── */}
      <div style={{ marginBottom: '1.25rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
        <button
          type="button"
          onClick={() => setScriptsOpen(o => !o)}
          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem 1rem', background: 'var(--surface-2)', border: 'none', cursor: 'pointer', fontSize: '13px', fontWeight: 600, color: 'var(--text)' }}
        >
          <span>📁 Scripts &amp; files that will be generated</span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{scriptsOpen ? '▲ Hide' : '▼ Show'}</span>
        </button>
        {scriptsOpen && (
          <div style={{ padding: '1rem', display: 'grid', gap: '1.25rem' }}>
            {Object.entries(GENERATED_SCRIPTS).map(([cap, { label, files }]) => (
              <div key={cap}>
                <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--accent-hover)', marginBottom: '0.5rem' }}>
                  {label}
                </div>
                <div style={{ display: 'grid', gap: '0.35rem' }}>
                  {files.map(f => (
                    <div key={f.name} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem', background: 'var(--surface-2)', borderRadius: 'var(--radius-xs)', padding: '0.45rem 0.7rem' }}>
                      <span style={{ fontSize: '11px', flexShrink: 0, marginTop: '1px' }}>
                        {f.type === 'py' ? '🐍' : '📄'}
                      </span>
                      <div style={{ minWidth: 0 }}>
                        <code style={{ fontSize: '11.5px', color: 'var(--accent-hover)', fontFamily: 'JetBrains Mono, monospace' }}>{f.name}</code>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '1px' }}>{f.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {error && <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>{error}</div>}

      {manifest && (
        <>
          {/* Legend */}
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
            <span className="mode-wired">✓ Reuse your code</span>
            <span className="mode-adapter">⚙ Generate adapter</span>
            <span className="mode-scaffold">⬜ Generate scaffold</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', alignSelf: 'center', marginLeft: '0.5rem' }}>
              — Auto-detected from knowledge graph. Edit as needed.
            </span>
          </div>

          {/* Checklist */}
          <div style={{ display: 'grid', gap: '0.6rem', marginBottom: '1.5rem' }}>
            {manifest.stages.map((s, i) => {
              const mode = modeOf(s)
              const meta = MODE_META[mode]
              return (
                <div key={s.stage} className={`checklist-row ${s.present ? 'checked' : ''}`}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', cursor: 'pointer', flex: 1, minWidth: 0 }}>
                      <input
                        type="checkbox"
                        id={`stage-present-${i}`}
                        checked={s.present}
                        onChange={e => toggleStage(i, 'present', e.target.checked)}
                        style={{ accentColor: 'var(--success)', width: 15, height: 15, flexShrink: 0 }}
                      />
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontSize: '13px', color: 'var(--text)', fontWeight: 500 }}>{s.display_name}</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
                          standard: {s.standard_path}
                        </div>
                      </div>
                    </label>
                    <span className={meta.cls}>{meta.icon} {meta.label}</span>
                  </div>

                  {s.present && (
                    <div style={{ paddingLeft: '1.65rem', marginTop: '0.65rem', display: 'grid', gap: '0.5rem' }}>
                      <div className="field-group">
                        <label className="field-label" htmlFor={`stage-path-${i}`}>
                          Path to your file
                        </label>
                        <input
                          id={`stage-path-${i}`}
                          className="input input-mono"
                          value={s.detected_path}
                          placeholder={s.standard_path}
                          onChange={e => updatePath(i, e.target.value)}
                        />
                      </div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '12px', color: 'var(--text-dim)', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          id={`stage-contract-${i}`}
                          checked={s.meets_contract}
                          onChange={e => toggleStage(i, 'meets_contract', e.target.checked)}
                          style={{ accentColor: 'var(--success)' }}
                        />
                        Meets execution contract&nbsp;
                        <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                          ({s.stage.includes('scoring') ? 'init() + run()' : 'CLI args --data-path / --output-dir'})
                        </span>
                      </label>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Dev-time only */}
          <div className="divider" />
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 600 }}>Dev-time only (not pipelined):</span>
            {([['has_eda', 'EDA notebook'], ['has_shap', 'SHAP / explainability'], ['has_feature_engineering', 'Feature engineering']] as const).map(([key, label]) => (
              <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '12px', color: 'var(--text-dim)', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={manifest[key]}
                  onChange={e => toggleDev(key, e.target.checked)}
                  style={{ accentColor: 'var(--text-muted)' }}
                />
                {label}
              </label>
            ))}
          </div>

          {/* Sign-off CTA */}
          {signedOff ? (
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              <div className="info-box info-box-success" style={{ flex: 1 }}>
                ✓ Checklist saved. The platform knows exactly what to generate.
              </div>
              <button className="btn btn-success btn-lg" onClick={() => onSignedOff(project)}>
                Proceed to Generate →
              </button>
            </div>
          ) : (
            <button
              id="sign-off-btn"
              className="btn btn-primary btn-lg"
              onClick={signOff}
              disabled={saving}
              style={{ minWidth: 240 }}
            >
              {saving ? <><span className="spinner" /> Saving…</> : '✅ Sign off & Proceed to Generate'}
            </button>
          )}
        </>
      )}
    </div>
  )
}
