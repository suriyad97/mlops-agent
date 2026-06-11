import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Project } from '../api'

const PROFILE_FIELDS: { key: string; label: string; placeholder: string }[] = [
  { key: 'project_type', label: 'Project type', placeholder: 'regression / classification / …' },
  { key: 'target_variable', label: 'Target variable', placeholder: 'e.g. survived' },
  { key: 'metrics', label: 'Metrics (comma-separated)', placeholder: 'accuracy, f1' },
  { key: 'endpoint_strategy', label: 'Endpoint', placeholder: 'realtime / batch / both' },
  { key: 'drift_kinds', label: 'Drift (comma-separated)', placeholder: 'data, concept' },
]

function StageChip({ stage }: { stage: string }) {
  const color =
    stage === 'operational' ? 'bg-emerald-900 text-emerald-300' :
    stage === 'evaluated'   ? 'bg-sky-900 text-sky-300' :
    stage === 'scanned'     ? 'bg-violet-900 text-violet-300' :
    'bg-zinc-800 text-zinc-300'
  return (
    <span className={`text-[11px] uppercase tracking-wide rounded-full px-2.5 py-0.5 ${color}`}>
      {stage}
    </span>
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
    await api.patchProfile(editing.id, profile)
    setEditing(null)
    await refresh()
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <h1 className="text-2xl font-semibold mb-1">Projects</h1>
      <p className="text-zinc-500 text-sm mb-6">
        Register an Azure DevOps repository, scan it, then open chat to drive the workflow.
      </p>

      {/* ── add project form ── */}
      <form onSubmit={create} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mb-8 grid gap-3">
        <div className="grid md:grid-cols-2 gap-3">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Project name"
            className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm" required />
          <input value={repoUrl} onChange={(e) => setRepoUrl(e.target.value)}
            onBlur={() => loadFormBranches(repoUrl, pat)}
            placeholder="https://dev.azure.com/org/_git/repo"
            className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm" required />
        </div>
        <input value={pat} onChange={(e) => setPat(e.target.value)} type="password"
          onBlur={() => loadFormBranches(repoUrl, pat)}
          placeholder="Azure DevOps PAT (stored encrypted)"
          className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm" required />

        {(formBranchLoading || formBranches.length > 0 || formBranchError) && (
          <div className="grid gap-1">
            <label className="text-xs text-zinc-400">Branch</label>
            {formBranchLoading && <p className="text-xs text-zinc-500 animate-pulse">Loading branches…</p>}
            {formBranchError && <p className="text-xs text-red-400">⚠️ {formBranchError}</p>}
            {!formBranchLoading && formBranches.length > 0 && (
              <select value={selectedFormBranch} onChange={(e) => setSelectedFormBranch(e.target.value)}
                className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm w-full">
                {formBranches.map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            )}
          </div>
        )}

        <button disabled={busy}
          className="bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-lg px-4 py-2 text-sm font-medium w-fit">
          {busy ? 'Creating…' : 'Add project'}
        </button>
      </form>

      {error && <p className="text-red-400 text-sm mb-4">⚠️ {error}</p>}

      {/* ── project cards ── */}
      <div className="grid gap-4">
        {projects.map((p) => {
          const branchOpen = cardBranchOpen[p.id] ?? false
          const branches = cardBranches[p.id] ?? []
          const branchLoading = cardBranchLoading[p.id] ?? false
          const activeBranch = selectedCardBranch[p.id] ?? p.base_branch ?? 'main'
          const isDeleting = confirmDelete === p.id

          return (
            <div key={p.id} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <div className="flex items-start justify-between flex-wrap gap-3">
                <div className="min-w-0">
                  <h2 className="font-semibold">{p.name}</h2>
                  <p className="text-xs text-zinc-500 truncate">{p.repo_url}</p>
                  <div className="flex items-center gap-2 mt-2 flex-wrap">
                    <StageChip stage={p.stage} />

                    {/* branch selector */}
                    <button onClick={() => openCardBranches(p)}
                      className="text-[11px] text-sky-400 hover:text-sky-300 flex items-center gap-1">
                      branch: <span className="font-mono">{activeBranch}</span>
                      <span className="text-zinc-500">{branchOpen ? '▲' : '▼'}</span>
                    </button>

                    {p.local_repo_path && (
                      <span className="text-[11px] text-zinc-600 font-mono truncate max-w-[260px]"
                        title={p.local_repo_path}>
                        {p.local_repo_path}
                      </span>
                    )}
                  </div>

                  {/* branch dropdown (lazy) */}
                  {branchOpen && (
                    <div className="mt-2">
                      {branchLoading && <p className="text-xs text-zinc-500 animate-pulse">Loading branches…</p>}
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
                <div className="flex gap-2 flex-wrap items-center shrink-0">
                  {isDeleting ? (
                    <>
                      <span className="text-sm text-zinc-400">Delete project and all chats?</span>
                      <button onClick={() => deleteProject(p.id)}
                        className="bg-red-700 hover:bg-red-600 rounded-lg px-3 py-1.5 text-sm font-medium">
                        Yes, delete
                      </button>
                      <button onClick={() => setConfirmDelete(null)}
                        className="border border-zinc-700 hover:border-zinc-500 rounded-lg px-3 py-1.5 text-sm">
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      <button onClick={() => scan(p)} disabled={scanning === p.id}
                        className="border border-zinc-700 hover:border-sky-500 hover:text-sky-400 rounded-lg px-3.5 py-1.5 text-sm disabled:opacity-50">
                        {scanning === p.id ? 'Picking folder…' : 'Scan'}
                      </button>
                      <button onClick={() => openEditor(p)}
                        className="border border-zinc-700 hover:border-zinc-500 rounded-lg px-3.5 py-1.5 text-sm">
                        Profile
                      </button>
                      <button onClick={() => navigate(`/chat?project=${p.id}`)}
                        className="bg-sky-600 hover:bg-sky-500 rounded-lg px-3.5 py-1.5 text-sm font-medium">
                        Open chat
                      </button>
                      <button onClick={() => setConfirmDelete(p.id)}
                        className="border border-red-900 hover:border-red-600 hover:text-red-400 text-zinc-500 rounded-lg px-3 py-1.5 text-sm">
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* profile editor */}
              {editing?.id === p.id && (
                <div className="mt-4 border-t border-zinc-800 pt-4 grid md:grid-cols-2 gap-3">
                  {PROFILE_FIELDS.map((f) => (
                    <label key={f.key} className="text-xs text-zinc-400 grid gap-1">
                      {f.label}
                      <input value={draft[f.key] ?? ''} placeholder={f.placeholder}
                        onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
                        className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100" />
                    </label>
                  ))}
                  <div className="flex gap-2 items-end">
                    <button onClick={saveProfile}
                      className="bg-emerald-600 hover:bg-emerald-500 rounded-lg px-3.5 py-1.5 text-sm">Save</button>
                    <button onClick={() => setEditing(null)}
                      className="border border-zinc-700 rounded-lg px-3.5 py-1.5 text-sm">Cancel</button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
        {projects.length === 0 && (
          <p className="text-zinc-600 text-sm">No projects yet — add one above.</p>
        )}
      </div>
    </div>
  )
}
