import { useState } from 'react'
import { api, type Project } from '../../api'

interface Props {
  project: Project
  onSaved: (p: Project) => void
}

const ENDPOINT_OPTIONS = [
  { value: 'realtime', label: '⚡ Real-time', hint: 'Online endpoint, low-latency inference' },
  { value: 'batch',    label: '📦 Batch',     hint: 'Batch scoring on large datasets' },
  { value: 'both',     label: '⚡📦 Both',    hint: 'Deploy both endpoint types' },
]

const PROJECT_TYPES = ['regression', 'classification', 'multi-class classification', 'object detection', 'NLP', 'forecasting', 'clustering']

export default function Step3Profile({ project, onSaved }: Props) {
  const prof = project.profile as Record<string, unknown>
  const pp   = (prof.pipeline_params as Record<string, unknown>) ?? {}

  const str = (k: string, fallback = '') => {
    const v = prof[k]; return Array.isArray(v) ? v.join(', ') : String(v ?? fallback)
  }

  // Core profile fields
  const [projectType,   setProjectType]   = useState(str('project_type'))
  const [targetVar,     setTargetVar]     = useState(str('target_variable'))
  const [metrics,       setMetrics]       = useState(str('metrics'))
  const [threshold,     setThreshold]     = useState(str('threshold'))
  const [driftThresh,   setDriftThresh]   = useState(String(pp.drift_threshold ?? str('drift_threshold', '0.2')))
  const [retrainThresh, setRetrainThresh] = useState(String(pp.retrain_threshold ?? '0.25'))
  const [endpoint,      setEndpoint]      = useState(str('endpoint_strategy', 'realtime'))
  const [monCron,       setMonCron]       = useState(String(pp.monitoring_cron ?? '0 6 * * *'))


  const [saving, setSaving] = useState(false)
  const [error,  setError]  = useState('')
  const [saved,  setSaved]  = useState(false)

  async function save(e: React.FormEvent) {
    e.preventDefault()
    if (!targetVar.trim()) { setError('Target variable is required'); return }
    setSaving(true); setError(''); setSaved(false)
    try {
      const metricsArr = metrics.split(',').map(s => s.trim()).filter(Boolean)
      const pipeline_params: Record<string, unknown> = {
        ...(driftThresh   ? { drift_threshold:   parseFloat(driftThresh)   } : {}),
        ...(retrainThresh ? { retrain_threshold:  parseFloat(retrainThresh) } : {}),
        ...(monCron       ? { monitoring_cron:    monCron }                   : {}),

      }

      const updated = await api.patchProfile(project.id, {
        project_type:      projectType,
        target_variable:   targetVar,
        metrics:           metricsArr,
        threshold:         threshold ? parseFloat(threshold) : undefined,
        endpoint_strategy: endpoint,
        pipeline_params,
      })
      setSaved(true)
      onSaved(updated)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="step-card-wide">
      <div className="step-header">
        <div className="step-number">Step 3 of 7</div>
        <h2 className="step-title">Configure your ML profile</h2>
        <p className="step-desc">
          Define your model's business targets, thresholds, and deployment endpoint.
          These values are baked directly into the generated pipelines.
        </p>
      </div>

      <form onSubmit={save}>
        {/* ── Section: Model identity ─────────────────────────────────── */}
        <div className="section-label">Model identity</div>
        <div className="field-grid field-grid-2" style={{ marginBottom: '2rem' }}>
          <div className="field-group">
            <label className="field-label" htmlFor="p3-project-type">Project type</label>
            <input
              id="p3-project-type"
              className="input"
              list="project-type-list"
              placeholder="regression / classification / …"
              value={projectType}
              onChange={e => setProjectType(e.target.value)}
            />
            <datalist id="project-type-list">
              {PROJECT_TYPES.map(t => <option key={t} value={t} />)}
            </datalist>
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="p3-target">
              Target variable <span className="required">*</span>
            </label>
            <input
              id="p3-target"
              className="input"
              placeholder="e.g. survived, price, churn_label"
              value={targetVar}
              onChange={e => setTargetVar(e.target.value)}
              required
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="p3-metrics">
              Evaluation metrics <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(comma-separated)</span>
            </label>
            <input
              id="p3-metrics"
              className="input"
              placeholder="accuracy, f1, roc_auc"
              value={metrics}
              onChange={e => setMetrics(e.target.value)}
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="p3-threshold">Performance threshold</label>
            <input
              id="p3-threshold"
              type="number"
              step="0.001"
              min="0" max="1"
              className="input"
              placeholder="e.g. 0.85"
              value={threshold}
              onChange={e => setThreshold(e.target.value)}
            />
            <span className="field-hint">Min acceptable score before flagging a model as failing.</span>
          </div>
        </div>

        {/* ── Section: Drift & monitoring ─────────────────────────────── */}
        <div className="section-label">Drift & monitoring</div>
        <div className="field-grid field-grid-3" style={{ marginBottom: '2rem' }}>
          <div className="field-group">
            <label className="field-label" htmlFor="p3-drift">Drift threshold (PSI)</label>
            <input
              id="p3-drift"
              type="number"
              step="0.01"
              min="0"
              className="input"
              placeholder="0.2"
              value={driftThresh}
              onChange={e => setDriftThresh(e.target.value)}
            />
            <span className="field-hint">PSI above this → drift alert.</span>
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="p3-retrain">Retrain threshold (PSI)</label>
            <input
              id="p3-retrain"
              type="number"
              step="0.01"
              min="0"
              className="input"
              placeholder="0.25"
              value={retrainThresh}
              onChange={e => setRetrainThresh(e.target.value)}
            />
            <span className="field-hint">PSI above this → trigger retraining.</span>
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="p3-cron">Monitoring schedule (cron)</label>
            <input
              id="p3-cron"
              className="input input-mono"
              placeholder="0 6 * * *"
              value={monCron}
              onChange={e => setMonCron(e.target.value)}
            />
            <span className="field-hint">When to run the drift monitor.</span>
          </div>
        </div>

        {/* ── Section: Endpoint ───────────────────────────────────────── */}
        <div className="section-label">Deployment endpoint</div>
        <div style={{ marginBottom: '2rem' }}>
          <div className="radio-pills" style={{ marginBottom: '0.75rem' }}>
            {ENDPOINT_OPTIONS.map(opt => (
              <div className="radio-pill" key={opt.value}>
                <input
                  type="radio"
                  id={`ep-${opt.value}`}
                  name="endpoint"
                  value={opt.value}
                  checked={endpoint === opt.value}
                  onChange={() => setEndpoint(opt.value)}
                />
                <label htmlFor={`ep-${opt.value}`}>{opt.label}</label>
              </div>
            ))}
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {ENDPOINT_OPTIONS.find(o => o.value === endpoint)?.hint}
          </p>


        </div>

        {/* Data paths are configured with the infrastructure prerequisites in Step 6. */}

        {/* ── Error / saved ───────────────────────────────────────────── */}
        {error && <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>{error}</div>}
        {saved && <div className="info-box info-box-success" style={{ marginBottom: '1rem' }}>✓ Profile saved. Proceed to verify your scripts.</div>}

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button type="submit" className="btn btn-primary btn-lg" disabled={saving}>
            {saving ? <><span className="spinner" /> Saving…</> : 'Save Profile →'}
          </button>
          {saved && (
            <button type="button" className="btn btn-success btn-lg" onClick={() => onSaved(project)}>
              Proceed to Verify Scripts →
            </button>
          )}
        </div>
      </form>
    </div>
  )
}
