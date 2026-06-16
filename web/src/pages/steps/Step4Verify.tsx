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

export default function Step4Verify({ project, onSignedOff }: Props) {
  const [manifest,  setManifest]  = useState<ContractManifest | null>(null)
  const [strategy,  setStrategy]  = useState('')
  const [loading,   setLoading]   = useState(true)
  const [saving,    setSaving]    = useState(false)
  const [signedOff, setSignedOff] = useState(false)
  const [error,     setError]     = useState('')

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
