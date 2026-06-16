import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type ContractManifest, type ContractStage, type DiscoveredConfig, type GenerationReport, type InfraReport, type PrereqItem, type Project, type RequirementItem, type RequirementPlan } from '../api'

const PROFILE_FIELDS: { key: string; label: string; placeholder: string }[] = [
  { key: 'project_type', label: 'Project type', placeholder: 'regression / classification / …' },
  { key: 'target_variable', label: 'Target variable', placeholder: 'e.g. survived' },
  { key: 'metrics', label: 'Metrics (comma-separated)', placeholder: 'accuracy, f1' },
  { key: 'drift_kinds', label: 'Drift (comma-separated)', placeholder: 'data, concept' },
]

const ENDPOINT_OPTIONS = ['realtime', 'batch', 'both', 'none'] as const

function StageChip({ stage }: { stage: string }) {
  const tone =
    stage === 'operational'
      ? 'text-[var(--success)] border-[var(--success)]/30 bg-[var(--success)]/10'
      : ['generated', 'evaluated', 'scanned', 'validated_local', 'committed'].includes(stage)
        ? 'text-[var(--accent)] border-[var(--accent)]/30 bg-[var(--accent-soft)]'
        : 'text-[var(--text-dim)] border-[var(--border-strong)]'
  return (
    <span className={`text-[10px] font-medium uppercase tracking-wider rounded-md px-2 py-0.5 border ${tone}`}>
      {stage}
    </span>
  )
}

const STATUS_ICON: Record<PrereqItem['status'], string> = {
  ok: '●',
  missing_config: '○',
  not_found: '○',
  error: '○',
}
const STATUS_LABEL: Record<PrereqItem['status'], string> = {
  ok: 'Ready',
  missing_config: 'Not configured',
  not_found: 'Not found',
  error: 'Error',
}
const STATUS_COLOR: Record<PrereqItem['status'], string> = {
  ok: 'text-[var(--success)]',
  missing_config: 'text-[var(--warn)]',
  not_found: 'text-[var(--danger)]',
  error: 'text-[var(--danger)]',
}

function InfraPanel({
  report,
  loading,
  onDiscover,
  discoverLoading,
  discoverResult,
}: {
  report: InfraReport | null
  loading: boolean
  onDiscover?: () => void
  discoverLoading?: boolean
  discoverResult?: DiscoveredConfig | null
}) {
  if (loading) {
    return (
      <div className="mt-4 border-t border-zinc-800 pt-4">
        <p className="text-xs text-zinc-500 animate-pulse">Checking infrastructure…</p>
      </div>
    )
  }
  if (!report) return null

  const hasMissing = !report.all_ok

  return (
    <div className="mt-4 border-t border-zinc-800 pt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-zinc-200">Infrastructure Prerequisites</h3>
        <span className={`text-[11px] font-medium ${report.all_ok ? 'text-emerald-400' : 'text-amber-400'}`}>
          {report.checks.filter(c => c.status === 'ok').length}/{report.checks.length} ready
        </span>
      </div>
      <div className="grid gap-2.5">
        {report.checks.map((item) => (
          <div key={item.name} className="grid gap-0.5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2 min-w-0">
                <span className="shrink-0 text-[13px] mt-px">{STATUS_ICON[item.status]}</span>
                <div className="min-w-0">
                  <span className="text-[13px] text-zinc-100">{item.name}</span>
                  {item.detail && item.status !== 'ok' && (
                    <p className="text-[11px] text-zinc-500 truncate">{item.detail}</p>
                  )}
                </div>
              </div>
              <div className="shrink-0 text-right">
                <span className={`text-[11px] font-medium ${STATUS_COLOR[item.status]}`}>
                  {STATUS_LABEL[item.status]}
                </span>
                {item.status === 'ok' && item.detail && (
                  <p className="text-[11px] text-zinc-600 font-mono">{item.detail}</p>
                )}
                <p className="text-[10px] text-zinc-600">{item.required_for}</p>
              </div>
            </div>
            {item.status !== 'ok' && item.fix && (
              <div className="ml-6 mt-1 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2">
                <p className="text-[10px] text-zinc-400 font-semibold mb-0.5 uppercase tracking-wide">How to fix</p>
                <p className="text-[11px] text-amber-300 leading-relaxed">{item.fix}</p>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Auto-discover footer — shown when there are missing items */}
      {hasMissing && onDiscover && (
        <div className="mt-4 pt-3 border-t border-zinc-800">
          {discoverResult && (
            <div className="mb-2">
              {discoverResult.discovered.length > 0 ? (
                <p className="text-[11px] text-emerald-400">
                  ✓ Auto-discovered {discoverResult.discovered.length} setting(s): {discoverResult.discovered.join(', ')}
                  {discoverResult.errors.length > 0 && (
                    <span className="text-amber-400"> · {discoverResult.errors.length} item(s) could not be found automatically</span>
                  )}
                </p>
              ) : (
                <p className="text-[11px] text-amber-400">
                  Nothing discovered automatically.
                  {discoverResult.errors.length > 0 && ` ${discoverResult.errors[0]}`}
                </p>
              )}
            </div>
          )}
          <div className="flex items-center justify-between gap-3">
            <p className="text-[11px] text-zinc-500">
              Auto-discover reads your AzDO ARM service connection to fill in subscription, resource group, workspace, ACR, and compute — no manual .env edits needed.
            </p>
            <button
              onClick={onDiscover}
              disabled={discoverLoading}
              className="shrink-0 border border-sky-700 text-sky-400 hover:border-sky-500 hover:text-sky-300 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-wait whitespace-nowrap">
              {discoverLoading ? 'Discovering…' : '⚡ Auto-discover'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

const DATA_PATH_FIELDS: { key: string; label: string; placeholder: string }[] = [
  { key: 'training_data_path', label: 'Training data', placeholder: 'azureml://datastores/<ds>/paths/training/' },
  { key: 'eval_data_path', label: 'Evaluation data', placeholder: 'optional — falls back to training data' },
  { key: 'batch_input_path', label: 'Batch input', placeholder: 'blob path for batch scoring input' },
  { key: 'batch_output_path', label: 'Batch output', placeholder: 'blob path where predictions land' },
  { key: 'baseline_data_path', label: 'Monitoring baseline', placeholder: 'reference distribution path' },
  { key: 'current_data_path', label: 'Monitoring current', placeholder: 'optional — MDC / batch output' },
]

const PARAM_FIELDS: { key: string; label: string; type: string; placeholder: string }[] = [
  { key: 'optuna_trials', label: 'Optuna trials', type: 'number', placeholder: '20' },
  { key: 'drift_threshold', label: 'Drift PSI threshold', type: 'number', placeholder: '0.2' },
  { key: 'retrain_threshold', label: 'Retrain PSI threshold', type: 'number', placeholder: '0.25' },
  { key: 'monitoring_cron', label: 'Monitoring schedule (cron)', type: 'text', placeholder: '0 6 * * *' },
  { key: 'instance_type', label: 'Endpoint instance type', type: 'text', placeholder: 'Standard_DS2_v2' },
  { key: 'instance_count', label: 'Endpoint instance count', type: 'number', placeholder: '1' },
]

const NUMERIC_PARAMS = new Set(['optuna_trials', 'drift_threshold', 'retrain_threshold', 'instance_count'])

type ConfigDraft = { data_paths: Record<string, string>; pipeline_params: Record<string, string> }

function ConfigPanel({
  draft, saving, onField, onSave, onVerify, verifyLoading, verifyResult,
}: {
  draft: ConfigDraft
  saving: boolean
  onField: (group: 'data_paths' | 'pipeline_params', key: string, v: string) => void
  onSave: () => void
  onVerify: () => void
  verifyLoading: boolean
  verifyResult: InfraReport | null
}) {
  return (
    <div className="panel">
      <h3 className="text-[13px] font-semibold text-[var(--text)] mb-1">Pipeline configuration</h3>
      <p className="text-[11px] text-[var(--text-dim)] mb-4 max-w-2xl">
        Data paths and parameters are baked into the generated pipelines. Paste your blob/datastore
        paths and tune the knobs upfront, then Verify the paths before generating.
      </p>

      <div className="section-title mb-2">Data paths (blob / datastore)</div>
      <div className="grid md:grid-cols-2 gap-2.5 mb-5">
        {DATA_PATH_FIELDS.map((f) => (
          <label key={f.key} className="label">
            {f.label}
            <input value={draft.data_paths[f.key] ?? ''} placeholder={f.placeholder}
              onChange={(e) => onField('data_paths', f.key, e.target.value)}
              className="input input-mono" />
          </label>
        ))}
      </div>

      <div className="section-title mb-2">Parameters</div>
      <div className="grid md:grid-cols-3 gap-2.5 mb-4">
        {PARAM_FIELDS.map((f) => (
          <label key={f.key} className="label">
            {f.label}
            <input type={f.type} value={draft.pipeline_params[f.key] ?? ''} placeholder={f.placeholder}
              onChange={(e) => onField('pipeline_params', f.key, e.target.value)}
              className="input input-mono" />
          </label>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <button onClick={onSave} disabled={saving} className="btn btn-primary">
          {saving ? 'Saving…' : 'Save config'}
        </button>
        <button onClick={onVerify} disabled={verifyLoading} className="btn">
          {verifyLoading ? 'Verifying…' : 'Verify data paths'}
        </button>
      </div>

      {verifyResult && (
        <div className="mt-3 grid gap-1.5">
          {verifyResult.checks.length === 0 && (
            <p className="text-[11px] text-zinc-500">No data paths to verify for this strategy.</p>
          )}
          {verifyResult.checks.map((item) => (
            <div key={item.name} className="flex items-start gap-2 text-[12px]">
              <span className={`shrink-0 mt-px ${STATUS_COLOR[item.status]}`}>{STATUS_ICON[item.status]}</span>
              <div className="min-w-0">
                <span className="text-[var(--text)]">{item.name}</span>
                <span className={`ml-2 ${STATUS_COLOR[item.status]}`}>{item.detail}</span>
                {item.status !== 'ok' && item.fix && (
                  <p className="text-[11px] text-[var(--warn)]/80">{item.fix}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Mirror of contract.ContractStage.resolve() so the badge updates live as the user edits.
function modeOf(s: ContractStage): ContractStage['mode'] {
  if (!s.present) return 'scaffold'
  if (s.meets_contract) return 'wired'
  return 'adapter'
}

const MODE_BADGE: Record<ContractStage['mode'], { label: string; color: string; hint: string }> = {
  wired:    { label: 'Reuse your code',   color: 'bg-emerald-900 text-emerald-300 border-emerald-700',
              hint: 'pipeline points at your existing script — nothing generated' },
  adapter:  { label: 'Generate adapter',  color: 'bg-amber-900 text-amber-300 border-amber-700',
              hint: 'thin wrapper at the standard path imports your code — your file untouched' },
  scaffold: { label: 'Generate scaffold', color: 'bg-sky-900 text-sky-300 border-sky-700',
              hint: 'TODO stub at the standard path — you implement the logic' },
}

function ChecklistPanel({
  manifest, strategy, loading, saving, onToggleStage, onEditPath, onToggleDev, onSave,
}: {
  manifest: ContractManifest | null
  strategy: string
  loading: boolean
  saving: boolean
  onToggleStage: (i: number, field: 'present' | 'meets_contract', v: boolean) => void
  onEditPath: (i: number, v: string) => void
  onToggleDev: (field: 'has_eda' | 'has_shap' | 'has_feature_engineering', v: boolean) => void
  onSave: () => void
}) {
  if (loading) {
    return (
      <div className="mt-4 border-t border-zinc-800 pt-4">
        <p className="text-xs text-zinc-500 animate-pulse">Detecting your ML code…</p>
      </div>
    )
  }
  if (!manifest) return null

  return (
    <div className="mt-4 border-t border-zinc-800 pt-4">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-zinc-200">Code checklist</h3>
        <span className="text-[11px] text-zinc-500">strategy: <span className="font-mono text-zinc-400">{strategy}</span></span>
      </div>
      <p className="text-[11px] text-zinc-500 mb-3">
        We always generate the full standard MLOps block. Confirm what your repo already has —
        the rest is scaffolded. Your source code is never modified.
      </p>

      <div className="grid gap-2">
        {manifest.stages.map((s, i) => {
          const mode = modeOf(s)
          const badge = MODE_BADGE[mode]
          return (
            <div key={s.stage} className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2.5 grid gap-2">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <label className="flex items-center gap-2 text-[13px] text-zinc-100 cursor-pointer">
                  <input type="checkbox" checked={s.present}
                    onChange={(e) => onToggleStage(i, 'present', e.target.checked)}
                    className="accent-sky-500" />
                  {s.display_name}
                </label>
                <span className={`text-[10px] uppercase tracking-wide rounded-full border px-2 py-0.5 ${badge.color}`}
                  title={badge.hint}>
                  {badge.label}
                </span>
              </div>
              {s.present && (
                <div className="grid gap-1.5 pl-6">
                  <label className="text-[11px] text-zinc-400 grid gap-1">
                    Path to your code
                    <input value={s.detected_path} placeholder={s.standard_path}
                      onChange={(e) => onEditPath(i, e.target.value)}
                      className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-[12px] font-mono text-zinc-100" />
                  </label>
                  <label className="flex items-center gap-2 text-[11px] text-zinc-400 cursor-pointer">
                    <input type="checkbox" checked={s.meets_contract}
                      onChange={(e) => onToggleStage(i, 'meets_contract', e.target.checked)}
                      className="accent-emerald-500" />
                    Meets the execution contract
                    <span className="text-zinc-600">
                      ({s.stage === 'scoring_realtime' || s.stage === 'scoring_batch'
                        ? 'init() + run()'
                        : 'CLI args (--data-path / --output-dir …)'})
                    </span>
                  </label>
                </div>
              )}
              <p className="text-[10px] text-zinc-600 pl-6">→ standard path: <span className="font-mono">{s.standard_path}</span></p>
            </div>
          )
        })}
      </div>

      {/* dev-time-only declarations — acknowledged, never pipelined */}
      <div className="mt-3 flex items-center gap-4 flex-wrap">
        <span className="text-[11px] text-zinc-500">Dev-time only (not pipelined):</span>
        {([['has_eda', 'EDA'], ['has_shap', 'SHAP / explainability'], ['has_feature_engineering', 'Feature engineering']] as const).map(
          ([key, label]) => (
            <label key={key} className="flex items-center gap-1.5 text-[11px] text-zinc-400 cursor-pointer">
              <input type="checkbox" checked={manifest[key]}
                onChange={(e) => onToggleDev(key, e.target.checked)} className="accent-zinc-500" />
              {label}
            </label>
          ),
        )}
      </div>

      <div className="mt-4 flex items-center gap-2">
        <button onClick={onSave} disabled={saving}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg px-3.5 py-1.5 text-sm font-medium">
          {saving ? 'Saving…' : 'Save checklist'}
        </button>
        <span className="text-[11px] text-zinc-600">drives wired / adapter / scaffold at generation time</span>
      </div>
    </div>
  )
}

const OWNER_META: Record<RequirementItem['owner'], { icon: string; label: string; color: string }> = {
  data_scientist: { icon: '🧑‍🔬', label: 'You (DS)', color: 'text-emerald-400' },
  platform:       { icon: '🤖', label: 'Platform', color: 'text-sky-400' },
  infra:          { icon: '☁️', label: 'Infra', color: 'text-violet-400' },
}
const STATUS_META: Record<RequirementItem['status'], { icon: string; label: string }> = {
  present:      { icon: '✅', label: 'reuse your code' },
  adapter:      { icon: '🔧', label: 'adapter wraps your code' },
  scaffold:     { icon: '⬜', label: 'you must implement' },
  exists:       { icon: '✅', label: 'present' },
  will_generate:{ icon: '🔲', label: 'will generate' },
  prerequisite: { icon: '☁️', label: 'verify in Infra check' },
  user_action:  { icon: '⚠️', label: 'you must do this' },
}

function RequirementPlanPanel({ plan, loading }: { plan: RequirementPlan | null; loading: boolean }) {
  if (loading) {
    return (
      <div className="mt-4 border-t border-zinc-800 pt-4">
        <p className="text-xs text-zinc-500 animate-pulse">Resolving requirements from the endpoint…</p>
      </div>
    )
  }
  if (!plan) return null
  const dsCount = plan.items.filter((i) => i.owner === 'data_scientist').length
  const blocking = plan.items.filter((i) => i.status === 'scaffold' || i.status === 'user_action')

  return (
    <div className="mt-4 border-t border-zinc-800 pt-4">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-zinc-200">Requirement plan</h3>
        <span className="text-[11px] text-zinc-500">strategy: <span className="font-mono text-zinc-400">{plan.endpoint_strategy}</span></span>
      </div>
      <p className="text-[11px] text-zinc-500 mb-3">
        Resolved backward from the endpoint. 🧑‍🔬 your job (ML code) · 🤖 auto-generated · ☁️ infra/ops.
        <span className="text-zinc-400"> {dsCount} item(s) are yours; the platform generates the other {plan.items.length - dsCount}.</span>
      </p>

      <div className="grid gap-1">
        {plan.items.map((it) => {
          const owner = OWNER_META[it.owner]
          const status = STATUS_META[it.status]
          const isBlocking = it.status === 'scaffold' || it.status === 'user_action'
          return (
            <div key={it.order}
              className={`flex items-start gap-2 rounded-lg px-3 py-1.5 text-[12px] border
                ${isBlocking ? 'border-amber-800/60 bg-amber-950/20' : 'border-zinc-800 bg-zinc-950'}`}>
              <span className="text-zinc-600 w-5 shrink-0 text-right">{it.order}</span>
              <span className="shrink-0">{owner.icon}</span>
              <div className="min-w-0 flex-1">
                <span className="text-zinc-100">{it.requirement}</span>
                {it.detail && <span className="text-zinc-500"> — {it.detail}</span>}
                {it.deliverables.length > 0 && (
                  <span className="text-zinc-600 font-mono text-[10px] ml-1.5">{it.deliverables.join(', ')}</span>
                )}
              </div>
              <span className={`shrink-0 text-right ${owner.color}`}>{owner.label}</span>
              <span className="shrink-0 w-[150px] text-right text-zinc-400">{status.icon} {status.label}</span>
            </div>
          )
        })}
      </div>

      {blocking.length > 0 && (
        <div className="mt-3 bg-amber-950/30 border border-amber-800/50 rounded-lg px-3 py-2">
          <p className="text-[11px] text-amber-300 font-semibold mb-0.5">⚠️ Blocking a working endpoint ({blocking.length})</p>
          <p className="text-[11px] text-amber-200/80">
            {blocking.map((b) => b.requirement).join(' · ')}
          </p>
        </div>
      )}
    </div>
  )
}

function GenSummary({ report }: { report: GenerationReport }) {
  const group = (label: string, files: string[] | undefined, tone: string) =>
    files && files.length > 0 ? (
      <details className="text-[12px]">
        <summary className={`cursor-pointer select-none ${tone}`}>{label} ({files.length})</summary>
        <div className="mt-1 grid gap-0.5 pl-3 border-l border-[var(--border)]">
          {files.map((f) => <span key={f} className="font-mono text-[11px] text-[var(--text-dim)]">{f}</span>)}
        </div>
      </details>
    ) : null
  return (
    <div className="grid gap-1.5">
      <p className="text-[12px] text-[var(--success)]">{report.summary}</p>
      {group('Newly created', report.created_files, 'text-[var(--success)]')}
      {group('Already existed — overwritten', report.updated_files, 'text-[var(--text-dim)]')}
      {group('Adapters — wrap your code', report.adapter_files, 'text-[var(--warn)]')}
      {group('Scaffolds — you must implement', report.scaffold_files, 'text-[var(--accent-hover)]')}
      {group('Your scripts reused / wired', report.wired_skipped, 'text-[var(--success)]')}
      {group('Superseded — review & delete', report.superseded_files, 'text-[var(--warn)]')}
    </div>
  )
}

export default function Projects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [name, setName] = useState('')
  const [repoUrl, setRepoUrl] = useState('')
  const [pat, setPat] = useState('')
  const [formBranches, setFormBranches] = useState<string[]>([])
  const [formBranchLoading, setFormBranchLoading] = useState(false)
  const [formBranchError, setFormBranchError] = useState('')
  const [selectedFormBranch, setSelectedFormBranch] = useState('main')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // per-card state
  const [cardBranches, setCardBranches] = useState<Record<string, string[]>>({})
  const [cardBranchLoading, setCardBranchLoading] = useState<Record<string, boolean>>({})
  const [cardBranchOpen, setCardBranchOpen] = useState<Record<string, boolean>>({})
  const [selectedCardBranch, setSelectedCardBranch] = useState<Record<string, string>>({})
  const [scanning, setScanning] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [editing, setEditing] = useState<Project | null>(null)
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [strategyBusy, setStrategyBusy] = useState<Record<string, boolean>>({})
  const [infraOpen, setInfraOpen] = useState<Record<string, boolean>>({})
  const [infraLoading, setInfraLoading] = useState<Record<string, boolean>>({})
  const [infraResults, setInfraResults] = useState<Record<string, InfraReport | null>>({})
  const [infraDiscoverLoading, setInfraDiscoverLoading] = useState<Record<string, boolean>>({})
  const [infraDiscoverResult, setInfraDiscoverResult] = useState<Record<string, DiscoveredConfig | null>>({})
  const [checklistOpen, setChecklistOpen] = useState<Record<string, boolean>>({})
  const [checklistLoading, setChecklistLoading] = useState<Record<string, boolean>>({})
  const [checklistSaving, setChecklistSaving] = useState<Record<string, boolean>>({})
  const [checklistStrategy, setChecklistStrategy] = useState<Record<string, string>>({})
  const [checklistResult, setChecklistResult] = useState<Record<string, ContractManifest | null>>({})
  const [configOpen, setConfigOpen] = useState<Record<string, boolean>>({})
  const [configDraft, setConfigDraft] = useState<Record<string, ConfigDraft>>({})
  const [configSaving, setConfigSaving] = useState<Record<string, boolean>>({})
  const [configVerifyLoading, setConfigVerifyLoading] = useState<Record<string, boolean>>({})
  const [configVerifyResult, setConfigVerifyResult] = useState<Record<string, InfraReport | null>>({})
  // consolidated 3-step flow: Details · Readiness · Ship
  const [activeTab, setActiveTab] = useState<Record<string, 'details' | 'readiness' | 'ship' | null>>({})
  const [shipOpen, setShipOpen] = useState<Record<string, boolean>>({})
  const [generating, setGenerating] = useState<Record<string, boolean>>({})
  const [genResult, setGenResult] = useState<Record<string, GenerationReport | null>>({})
  const [shipBusy, setShipBusy] = useState<Record<string, string>>({})
  const [shipMsg, setShipMsg] = useState<Record<string, string>>({})
  const [commitText, setCommitText] = useState<Record<string, string>>({})
  const [planOpen, setPlanOpen] = useState<Record<string, boolean>>({})
  const [planLoading, setPlanLoading] = useState<Record<string, boolean>>({})
  const [planResult, setPlanResult] = useState<Record<string, RequirementPlan | null>>({})

  const navigate = useNavigate()
  const lastFormFetch = useRef({ url: '', pat: '' })

  const refresh = () => api.listProjects().then(setProjects).catch((e) => setError(e.message))
  useEffect(() => { refresh() }, [])

  // ── form branch loading ────────────────────────────────────────────────────
  async function loadFormBranches(url: string, token: string) {
    if (!url || !token) return
    if (lastFormFetch.current.url === url && lastFormFetch.current.pat === token) return
    lastFormFetch.current = { url, pat: token }
    setFormBranchLoading(true)
    setFormBranchError('')
    setFormBranches([])
    try {
      const { branches } = await api.getBranches(url, token)
      setFormBranches(branches)
      setSelectedFormBranch(branches.includes('main') ? 'main' : (branches[0] ?? 'main'))
    } catch (err) {
      setFormBranchError((err as Error).message)
    } finally {
      setFormBranchLoading(false)
    }
  }

  // ── per-card branch loading ────────────────────────────────────────────────
  async function openCardBranches(p: Project) {
    const alreadyOpen = cardBranchOpen[p.id]
    setCardBranchOpen((prev) => ({ ...prev, [p.id]: !alreadyOpen }))
    if (alreadyOpen || cardBranches[p.id]) return
    setCardBranchLoading((prev) => ({ ...prev, [p.id]: true }))
    try {
      const { branches } = await api.getProjectBranches(p.id)
      setCardBranches((prev) => ({ ...prev, [p.id]: branches }))
      setSelectedCardBranch((prev) => ({
        ...prev,
        [p.id]: prev[p.id] ?? (branches.includes(p.base_branch) ? p.base_branch : (branches[0] ?? 'main')),
      }))
    } catch {
      setCardBranches((prev) => ({ ...prev, [p.id]: [] }))
    } finally {
      setCardBranchLoading((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── create project ─────────────────────────────────────────────────────────
  async function create(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true); setError('')
    try {
      await api.createProject(name, repoUrl, pat, selectedFormBranch)
      setName(''); setRepoUrl(''); setPat('')
      setFormBranches([]); setSelectedFormBranch('main')
      lastFormFetch.current = { url: '', pat: '' }
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // ── scan ───────────────────────────────────────────────────────────────────
  async function scan(p: Project) {
    setScanning(p.id); setError('')
    try {
      const { path } = await api.browsePath()
      if (!path) { setScanning(null); return }
      const branch = selectedCardBranch[p.id] ?? p.base_branch ?? 'main'
      await api.scanProject(p.id, path, branch)
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setScanning(null)
    }
  }

  // ── delete ─────────────────────────────────────────────────────────────────
  async function deleteProject(id: string) {
    setError('')
    try {
      await api.deleteProject(id)
      setConfirmDelete(null)
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  // ── infrastructure check ──────────────────────────────────────────────────
  async function loadInfra(p: Project) {
    setInfraOpen((prev) => ({ ...prev, [p.id]: true }))
    if (infraResults[p.id]) return   // already fetched — just re-open
    setInfraLoading((prev) => ({ ...prev, [p.id]: true }))
    try {
      const result = await api.checkProjectInfra(p.id)
      setInfraResults((prev) => ({ ...prev, [p.id]: result }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setInfraLoading((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── auto-discover infrastructure ──────────────────────────────────────────
  async function discoverInfra(p: Project) {
    setInfraDiscoverLoading((prev) => ({ ...prev, [p.id]: true }))
    try {
      const { discovered, report } = await api.discoverProjectInfra(p.id)
      setInfraDiscoverResult((prev) => ({ ...prev, [p.id]: discovered }))
      setInfraResults((prev) => ({ ...prev, [p.id]: report }))
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setInfraDiscoverLoading((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── requirement plan ──────────────────────────────────────────────────────
  async function loadPlan(p: Project) {
    setPlanOpen((prev) => ({ ...prev, [p.id]: true }))
    if (planResult[p.id]) return
    setPlanLoading((prev) => ({ ...prev, [p.id]: true }))
    try {
      const { plan } = await api.getRequirementPlan(p.id)
      setPlanResult((prev) => ({ ...prev, [p.id]: plan }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setPlanLoading((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── code checklist (contract) ─────────────────────────────────────────────
  async function loadChecklist(p: Project) {
    setChecklistOpen((prev) => ({ ...prev, [p.id]: true }))
    if (checklistResult[p.id]) return
    setChecklistLoading((prev) => ({ ...prev, [p.id]: true }))
    try {
      const { endpoint_strategy, contract } = await api.getContract(p.id)
      setChecklistResult((prev) => ({ ...prev, [p.id]: contract }))
      setChecklistStrategy((prev) => ({ ...prev, [p.id]: endpoint_strategy }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setChecklistLoading((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  function updateStage(pid: string, i: number, field: 'present' | 'meets_contract', v: boolean) {
    setChecklistResult((prev) => {
      const m = prev[pid]; if (!m) return prev
      const stages = m.stages.map((s, j) => (j === i ? { ...s, [field]: v } : s))
      return { ...prev, [pid]: { ...m, stages } }
    })
  }
  function updatePath(pid: string, i: number, v: string) {
    setChecklistResult((prev) => {
      const m = prev[pid]; if (!m) return prev
      const stages = m.stages.map((s, j) => (j === i ? { ...s, detected_path: v } : s))
      return { ...prev, [pid]: { ...m, stages } }
    })
  }
  function updateDev(pid: string, field: 'has_eda' | 'has_shap' | 'has_feature_engineering', v: boolean) {
    setChecklistResult((prev) => {
      const m = prev[pid]; if (!m) return prev
      return { ...prev, [pid]: { ...m, [field]: v } }
    })
  }
  async function saveChecklist(p: Project) {
    const m = checklistResult[p.id]; if (!m) return
    setChecklistSaving((prev) => ({ ...prev, [p.id]: true }))
    try {
      const { contract } = await api.saveContract(p.id, m)
      setChecklistResult((prev) => ({ ...prev, [p.id]: contract }))   // re-resolved by backend
      setPlanResult((prev) => ({ ...prev, [p.id]: null }))            // plan reflects contract — refetch
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setChecklistSaving((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── pipeline configuration (data paths + parameters) ──────────────────────
  function openConfig(p: Project) {
    setConfigOpen((prev) => ({ ...prev, [p.id]: true }))
    if (configDraft[p.id]) return
    const prof = p.profile as Record<string, unknown>
    const dp = (prof.data_paths as Record<string, string>) ?? {}
    const pp = (prof.pipeline_params as Record<string, unknown>) ?? {}
    setConfigDraft((prev) => ({
      ...prev,
      [p.id]: {
        data_paths: Object.fromEntries(DATA_PATH_FIELDS.map((f) => [f.key, dp[f.key] ?? ''])),
        pipeline_params: Object.fromEntries(PARAM_FIELDS.map((f) => [f.key, String(pp[f.key] ?? '')])),
      },
    }))
  }
  function updateConfigField(pid: string, group: 'data_paths' | 'pipeline_params', key: string, v: string) {
    setConfigDraft((prev) => {
      const d = prev[pid]; if (!d) return prev
      return { ...prev, [pid]: { ...d, [group]: { ...d[group], [key]: v } } }
    })
  }
  async function saveConfig(p: Project) {
    const d = configDraft[p.id]; if (!d) return
    setConfigSaving((prev) => ({ ...prev, [p.id]: true }))
    try {
      // drop empty data paths; coerce numeric params
      const data_paths = Object.fromEntries(Object.entries(d.data_paths).filter(([, v]) => v.trim()))
      const pipeline_params: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(d.pipeline_params)) {
        if (!String(v).trim()) continue
        pipeline_params[k] = NUMERIC_PARAMS.has(k) ? Number(v) : v
      }
      await api.patchProfile(p.id, { data_paths, pipeline_params })
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setConfigSaving((prev) => ({ ...prev, [p.id]: false }))
    }
  }
  async function verifyConfig(p: Project) {
    setConfigVerifyLoading((prev) => ({ ...prev, [p.id]: true }))
    try {
      const report = await api.verifyDataPaths(p.id)
      setConfigVerifyResult((prev) => ({ ...prev, [p.id]: report }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setConfigVerifyLoading((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── endpoint strategy quick-override ──────────────────────────────────────
  async function setEndpointStrategy(p: Project, strategy: string) {
    setStrategyBusy((prev) => ({ ...prev, [p.id]: true }))
    try {
      await api.patchProfile(p.id, { endpoint_strategy: strategy })
      setChecklistResult((prev) => ({ ...prev, [p.id]: null }))   // stages differ per strategy — refetch
      setPlanResult((prev) => ({ ...prev, [p.id]: null }))        // plan depends on strategy
      setPlanOpen((prev) => ({ ...prev, [p.id]: false }))
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setStrategyBusy((prev) => ({ ...prev, [p.id]: false }))
    }
  }

  // ── profile editor ─────────────────────────────────────────────────────────
  function openEditor(p: Project) {
    setEditing(p)
    const d: Record<string, string> = {}
    for (const f of PROFILE_FIELDS) {
      const v = p.profile[f.key]
      d[f.key] = Array.isArray(v) ? v.join(', ') : String(v ?? '')
    }
    setDraft(d)
  }

  async function saveProfile() {
    if (!editing) return
    const profile: Record<string, unknown> = {}
    for (const f of PROFILE_FIELDS) {
      profile[f.key] = f.key === 'metrics' || f.key === 'drift_kinds'
        ? draft[f.key].split(',').map((s) => s.trim()).filter(Boolean)
        : draft[f.key]
    }
    // endpoint_strategy is managed by the inline chip selector, not this form
    await api.patchProfile(editing.id, profile)
    await refresh()
  }

  // ── consolidated 3-step flow ───────────────────────────────────────────────
  function closeAllPanels(pid: string) {
    setPlanOpen((prev) => ({ ...prev, [pid]: false }))
    setChecklistOpen((prev) => ({ ...prev, [pid]: false }))
    setConfigOpen((prev) => ({ ...prev, [pid]: false }))
    setInfraOpen((prev) => ({ ...prev, [pid]: false }))
    setShipOpen((prev) => ({ ...prev, [pid]: false }))
    setEditing((cur) => (cur?.id === pid ? null : cur))
  }
  function selectTab(p: Project, tab: 'details' | 'readiness' | 'ship') {
    const next = activeTab[p.id] === tab ? null : tab
    closeAllPanels(p.id)
    setActiveTab((prev) => ({ ...prev, [p.id]: next }))
    if (next === 'details') { openEditor(p); openConfig(p) }
    else if (next === 'readiness') { loadPlan(p); loadChecklist(p); loadInfra(p); loadGenReport(p) }
    else if (next === 'ship') { setShipOpen((prev) => ({ ...prev, [p.id]: true })); loadGenReport(p) }
  }
  // load the PERSISTED generation report (survives reload/navigation) if we don't have one
  async function loadGenReport(p: Project) {
    if (genResult[p.id]) return
    try {
      const r = await api.getGenerationReport(p.id)
      if (r.generated) setGenResult((prev) => ({ ...prev, [p.id]: r }))
    } catch { /* no report yet — leave empty */ }
  }
  async function generatePipelines(p: Project) {
    setGenerating((prev) => ({ ...prev, [p.id]: true }))
    try {
      const report = await api.generateProject(p.id)
      setGenResult((prev) => ({ ...prev, [p.id]: report }))
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setGenerating((prev) => ({ ...prev, [p.id]: false }))
    }
  }
  async function runValidate(p: Project) {
    setShipBusy((prev) => ({ ...prev, [p.id]: 'validate' }))
    setShipMsg((prev) => ({ ...prev, [p.id]: '' }))
    try {
      await api.validateProject(p.id)
      setShipMsg((prev) => ({ ...prev, [p.id]: 'Local validation run complete — open chat for the detailed report.' }))
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setShipBusy((prev) => ({ ...prev, [p.id]: '' }))
    }
  }
  async function runCommit(p: Project) {
    setShipBusy((prev) => ({ ...prev, [p.id]: 'commit' }))
    try {
      await api.commitProject(p.id, commitText[p.id] || 'chore: standard MLOps pipelines')
      setShipMsg((prev) => ({ ...prev, [p.id]: 'Committed & pushed. Run the pipelines on the cloud from your AzDO/GitHub project.' }))
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setShipBusy((prev) => ({ ...prev, [p.id]: '' }))
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <h1 className="text-[22px] font-semibold tracking-tight mb-1">Projects</h1>
      <p className="text-[var(--text-dim)] text-[13px] mb-7">
        Register an Azure DevOps repository, scan it, then open chat to drive the workflow.
      </p>

      {/* ── add project form ── */}
      <form onSubmit={create} className="card p-5 mb-9 grid gap-3">
        <div className="grid md:grid-cols-2 gap-3">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Project name"
            className="input" required />
          <input value={repoUrl} onChange={(e) => setRepoUrl(e.target.value)}
            onBlur={() => loadFormBranches(repoUrl, pat)}
            placeholder="https://dev.azure.com/org/_git/repo  or  https://github.com/owner/repo"
            className="input" required />
        </div>
        <input value={pat} onChange={(e) => setPat(e.target.value)} type="password"
          onBlur={() => loadFormBranches(repoUrl, pat)}
          placeholder="AzDO PAT  or  GitHub token (stored encrypted)"
          className="input" required />

        {(formBranchLoading || formBranches.length > 0 || formBranchError) && (
          <div className="grid gap-1">
            <span className="label">Branch</span>
            {formBranchLoading && <p className="text-xs text-[var(--text-muted)] animate-pulse">Loading branches…</p>}
            {formBranchError && <p className="text-xs text-[var(--danger)]">{formBranchError}</p>}
            {!formBranchLoading && formBranches.length > 0 && (
              <select value={selectedFormBranch} onChange={(e) => setSelectedFormBranch(e.target.value)}
                className="input">
                {formBranches.map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            )}
          </div>
        )}

        <button disabled={busy} className="btn btn-primary w-fit mt-1">
          {busy ? 'Creating…' : 'Add project'}
        </button>
      </form>

      {error && (
        <div className="mb-5 text-[13px] text-[var(--danger)] border border-[var(--danger)]/30 bg-[var(--danger)]/5 rounded-lg px-3.5 py-2.5">
          {error}
        </div>
      )}

      {/* ── project cards ── */}
      <div className="grid gap-4">
        {projects.map((p) => {
          const branchOpen = cardBranchOpen[p.id] ?? false
          const branches = cardBranches[p.id] ?? []
          const branchLoading = cardBranchLoading[p.id] ?? false
          const activeBranch = selectedCardBranch[p.id] ?? p.base_branch ?? 'main'
          const isDeleting = confirmDelete === p.id

          return (
            <div key={p.id} className="card p-5">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div className="min-w-0">
                  <h2 className="font-semibold text-[15px] tracking-tight">{p.name}</h2>
                  <p className="text-[12px] text-[var(--text-muted)] truncate mt-0.5">{p.repo_url}</p>
                  <div className="flex items-center gap-2.5 mt-2.5 flex-wrap">
                    <StageChip stage={p.stage} />

                    {/* branch selector */}
                    <button onClick={() => openCardBranches(p)}
                      className="text-[11px] text-[var(--text-dim)] hover:text-[var(--text)] flex items-center gap-1 transition-colors">
                      branch: <span className="font-mono text-[var(--text)]">{activeBranch}</span>
                      <span className="text-[var(--text-muted)]">{branchOpen ? '▲' : '▼'}</span>
                    </button>

                    {p.local_repo_path && (
                      <span className="text-[11px] text-[var(--text-muted)] font-mono truncate max-w-[260px]"
                        title={p.local_repo_path}>
                        {p.local_repo_path}
                      </span>
                    )}
                  </div>

                  {/* endpoint strategy selector */}
                  <div className="mt-3 flex items-center gap-2 flex-wrap">
                    <span className="text-[11px] text-[var(--text-muted)] shrink-0">Endpoint target:</span>
                    {ENDPOINT_OPTIONS.map((opt) => {
                      const current = String(p.profile.endpoint_strategy ?? 'none')
                      const active = current === opt
                      return (
                        <button
                          key={opt}
                          disabled={strategyBusy[p.id]}
                          onClick={() => setEndpointStrategy(p, opt)}
                          className={`chip ${active ? 'chip-active' : ''} ${strategyBusy[p.id] ? 'opacity-50 cursor-wait' : ''}`}
                        >
                          {opt}
                        </button>
                      )
                    })}
                    {strategyBusy[p.id] && (
                      <span className="text-[11px] text-[var(--text-muted)] animate-pulse">saving…</span>
                    )}
                  </div>

                  {/* branch dropdown (lazy) */}
                  {branchOpen && (
                    <div className="mt-2">
                      {branchLoading && <p className="text-xs text-[var(--text-muted)] animate-pulse">Loading branches…</p>}
                      {!branchLoading && branches.length > 0 && (
                        <select
                          value={activeBranch}
                          onChange={(e) => setSelectedCardBranch((prev) => ({ ...prev, [p.id]: e.target.value }))}
                          className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm">
                          {branches.map((b) => <option key={b} value={b}>{b}</option>)}
                        </select>
                      )}
                      {!branchLoading && branches.length === 0 && (
                        <p className="text-xs text-zinc-500">No branches found.</p>
                      )}
                    </div>
                  )}
                </div>

                {/* action buttons */}
                <div className="flex gap-1.5 flex-wrap items-center shrink-0">
                  {isDeleting ? (
                    <>
                      <span className="text-[13px] text-[var(--text-dim)] mr-1">Delete project and all chats?</span>
                      <button onClick={() => deleteProject(p.id)} className="btn btn-primary"
                        style={{ background: 'var(--danger)', borderColor: 'var(--danger)' }}>
                        Yes, delete
                      </button>
                      <button onClick={() => setConfirmDelete(null)} className="btn">Cancel</button>
                    </>
                  ) : (
                    <>
                      <button onClick={() => scan(p)} disabled={scanning === p.id} className="btn">
                        {scanning === p.id ? 'Picking folder…' : 'Scan'}
                      </button>
                      <span className="w-px h-5 bg-[var(--border)] mx-0.5" />
                      <button onClick={() => selectTab(p, 'details')}
                        className={`btn ${activeTab[p.id] === 'details' ? 'btn-active' : ''}`}>1 · Details</button>
                      <button onClick={() => selectTab(p, 'readiness')}
                        className={`btn ${activeTab[p.id] === 'readiness' ? 'btn-active' : ''}`}>2 · Readiness</button>
                      <button onClick={() => selectTab(p, 'ship')}
                        className={`btn ${activeTab[p.id] === 'ship' ? 'btn-active' : ''}`}>3 · Ship</button>
                      <span className="w-px h-5 bg-[var(--border)] mx-0.5" />
                      <button onClick={() => navigate(`/chat?project=${p.id}`)} className="btn btn-primary">
                        Open chat
                      </button>
                      <button onClick={() => setConfirmDelete(p.id)} className="btn btn-danger">Delete</button>
                    </>
                  )}
                </div>
              </div>

              {/* step header */}
              {activeTab[p.id] && (
                <div className="mt-4 pt-4 border-t border-[var(--border)] section-title">
                  {activeTab[p.id] === 'details' ? 'Step 1 — Project details & configuration'
                    : activeTab[p.id] === 'readiness' ? 'Step 2 — Readiness: plan, checklist & infrastructure'
                    : 'Step 3 — Ship: validate, commit & run on cloud'}
                </div>
              )}

              {/* details: profile */}
              {editing?.id === p.id && (
                <div className="mt-3 grid md:grid-cols-2 gap-3">
                  {PROFILE_FIELDS.map((f) => (
                    <label key={f.key} className="label">
                      {f.label}
                      <input value={draft[f.key] ?? ''} placeholder={f.placeholder}
                        onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
                        className="input" />
                    </label>
                  ))}
                  <div className="md:col-span-2">
                    <button onClick={saveProfile} className="btn btn-primary">Save details</button>
                  </div>
                </div>
              )}

              {/* requirement plan panel */}
              {planOpen[p.id] && (
                <RequirementPlanPanel
                  plan={planResult[p.id] ?? null}
                  loading={planLoading[p.id] ?? false}
                />
              )}

              {/* pipeline config panel */}
              {configOpen[p.id] && configDraft[p.id] && (
                <ConfigPanel
                  draft={configDraft[p.id]}
                  saving={configSaving[p.id] ?? false}
                  onField={(group, key, v) => updateConfigField(p.id, group, key, v)}
                  onSave={() => saveConfig(p)}
                  onVerify={() => verifyConfig(p)}
                  verifyLoading={configVerifyLoading[p.id] ?? false}
                  verifyResult={configVerifyResult[p.id] ?? null}
                />
              )}

              {/* code checklist panel */}
              {checklistOpen[p.id] && (
                <ChecklistPanel
                  manifest={checklistResult[p.id] ?? null}
                  strategy={checklistStrategy[p.id] ?? String(p.profile.endpoint_strategy ?? 'both')}
                  loading={checklistLoading[p.id] ?? false}
                  saving={checklistSaving[p.id] ?? false}
                  onToggleStage={(i, field, v) => updateStage(p.id, i, field, v)}
                  onEditPath={(i, v) => updatePath(p.id, i, v)}
                  onToggleDev={(field, v) => updateDev(p.id, field, v)}
                  onSave={() => saveChecklist(p)}
                />
              )}

              {/* readiness: generate */}
              {checklistOpen[p.id] && (
                <div className="panel">
                  <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div className="max-w-xl">
                      <h3 className="text-[13px] font-semibold text-[var(--text)]">Generate the standard pipelines</h3>
                      <p className="text-[11px] text-[var(--text-dim)] mt-0.5">
                        Confirm the checklist above, then generate the full standard MLOps block for this
                        endpoint strategy. Your source code is never modified.
                      </p>
                    </div>
                    <button onClick={() => generatePipelines(p)} disabled={generating[p.id]} className="btn btn-primary">
                      {generating[p.id]
                        ? 'Generating…'
                        : genResult[p.id] ? 'Re-generate pipelines' : 'Generate pipelines'}
                    </button>
                  </div>
                  {genResult[p.id] && (
                    <div className="mt-3">
                      <GenSummary report={genResult[p.id]!} />
                    </div>
                  )}
                </div>
              )}

              {/* infrastructure panel */}
              {infraOpen[p.id] && (
                <InfraPanel
                  report={infraResults[p.id] ?? null}
                  loading={infraLoading[p.id] ?? false}
                  onDiscover={() => discoverInfra(p)}
                  discoverLoading={infraDiscoverLoading[p.id] ?? false}
                  discoverResult={infraDiscoverResult[p.id] ?? null}
                />
              )}

              {/* ship */}
              {shipOpen[p.id] && (
                <div className="panel">
                  <div className="section-title mb-2">Generated files</div>
                  {genResult[p.id]
                    ? <div className="mb-4"><GenSummary report={genResult[p.id]!} /></div>
                    : <p className="text-[12px] text-[var(--text-dim)] mb-4">
                        Nothing generated yet — go to <span className="text-[var(--text)]">Readiness</span> and click Generate pipelines.
                      </p>}

                  <h3 className="text-[13px] font-semibold text-[var(--text)] mb-1">Ship to cloud</h3>
                  <p className="text-[11px] text-[var(--text-dim)] mb-3 max-w-xl">
                    Validate the generated files locally, then commit &amp; push. Pipelines run on the cloud
                    from your AzDO / GitHub project once committed.
                  </p>
                  <div className="flex items-center gap-2 flex-wrap">
                    <button onClick={() => runValidate(p)} disabled={!!shipBusy[p.id]} className="btn">
                      {shipBusy[p.id] === 'validate' ? 'Validating…' : 'Validate locally'}
                    </button>
                    <input value={commitText[p.id] ?? ''} placeholder="commit message"
                      onChange={(e) => setCommitText((prev) => ({ ...prev, [p.id]: e.target.value }))}
                      className="input" style={{ maxWidth: 300 }} />
                    <button onClick={() => runCommit(p)} disabled={!!shipBusy[p.id]} className="btn btn-primary">
                      {shipBusy[p.id] === 'commit' ? 'Committing…' : 'Commit & push'}
                    </button>
                  </div>
                  {shipMsg[p.id] && <p className="mt-2.5 text-[12px] text-[var(--text-dim)]">{shipMsg[p.id]}</p>}
                </div>
              )}
            </div>
          )
        })}
        {projects.length === 0 && (
          <p className="text-[var(--text-muted)] text-sm">No projects yet — add one above.</p>
        )}
      </div>
    </div>
  )
}
