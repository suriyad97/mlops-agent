import { useState } from 'react'
import { api, type Project } from '../../api'

interface Props {
  onCreated: (project: Project) => void
}

export default function Step1Register({ onCreated }: Props) {
  const [name, setName] = useState('')
  const [repoUrl, setRepoUrl] = useState('')
  const [pat, setPat] = useState('')
  const [branch, setBranch] = useState('main')
  const [branches, setBranches] = useState<string[]>([])
  const [loadingBranches, setLoadingBranches] = useState(false)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')

  async function fetchBranches() {
    if (!repoUrl || !pat) return
    setLoadingBranches(true)
    setError('')
    try {
      const { branches: b } = await api.getBranches(repoUrl, pat)
      setBranches(b)
      setBranch(b.includes('main') ? 'main' : b[0] ?? 'main')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoadingBranches(false)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!name || !repoUrl || !pat) { setError('All fields are required'); return }
    setCreating(true)
    setError('')
    try {
      const project = await api.createProject(name, repoUrl, pat, branch)
      onCreated(project)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="step-card">
      <div className="step-header">
        <div className="step-number">Step 1 of 7</div>
        <h2 className="step-title">Register your repository</h2>
        <p className="step-desc">
          Connect an Azure DevOps or GitHub repository. Your PAT is encrypted at rest and never logged.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="field-grid" style={{ marginBottom: '1.5rem' }}>
          <div className="field-group">
            <label className="field-label">
              Project name <span className="required">*</span>
            </label>
            <input
              id="reg-name"
              className="input"
              placeholder="e.g. Credit Risk Model"
              value={name}
              onChange={e => setName(e.target.value)}
              required
            />
          </div>

          <div className="field-group">
            <label className="field-label">
              Repository URL <span className="required">*</span>
            </label>
            <input
              id="reg-repo-url"
              className="input input-mono"
              placeholder="https://dev.azure.com/org/project/_git/repo"
              value={repoUrl}
              onChange={e => setRepoUrl(e.target.value)}
              required
            />
          </div>

          <div className="field-group">
            <label className="field-label">
              Personal Access Token (PAT) <span className="required">*</span>
            </label>
            <input
              id="reg-pat"
              type="password"
              className="input input-mono"
              placeholder="••••••••••••••••••••"
              value={pat}
              onChange={e => setPat(e.target.value)}
              required
            />
            <span className="field-hint">
              Needs Code (Read) + Build (Read) permissions. Stored encrypted, never logged.
            </span>
          </div>

          <div className="field-group">
            <label className="field-label">Branch</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {branches.length > 0 ? (
                <select
                  id="reg-branch"
                  className="input"
                  value={branch}
                  onChange={e => setBranch(e.target.value)}
                  style={{ flex: 1 }}
                >
                  {branches.map(b => <option key={b} value={b}>{b}</option>)}
                </select>
              ) : (
                <input
                  id="reg-branch-manual"
                  className="input"
                  placeholder="main"
                  value={branch}
                  onChange={e => setBranch(e.target.value)}
                  style={{ flex: 1 }}
                />
              )}
              <button
                type="button"
                className="btn btn-sm"
                onClick={fetchBranches}
                disabled={loadingBranches || !repoUrl || !pat}
              >
                {loadingBranches ? <span className="spinner" /> : 'Load branches'}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        <button type="submit" className="btn btn-primary btn-lg" disabled={creating}>
          {creating ? <><span className="spinner" /> Registering…</> : 'Register Project →'}
        </button>
      </form>
    </div>
  )
}
