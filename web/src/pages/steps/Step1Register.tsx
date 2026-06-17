import { useState, useMemo } from 'react'
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
  const [scanLog, setScanLog] = useState<string[]>([])

  const isLocalPath = useMemo(() => {
    const trimmed = repoUrl.trim()
    if (!trimmed) return false
    return !trimmed.startsWith('http://') && !trimmed.startsWith('https://') && !trimmed.startsWith('git@')
  }, [repoUrl])

  async function handleBrowse() {
    const { path } = await api.browsePath()
    if (path) setRepoUrl(path)
  }

  async function fetchBranches() {
    if (!repoUrl) return
    if (!isLocalPath && !pat) return
    
    setLoadingBranches(true)
    setError('')
    try {
      // Pass 'local_placeholder' for the pat so it passes any backend validation schema for getBranches
      const { branches: b } = await api.getBranches(repoUrl, isLocalPath ? "local_placeholder" : pat)
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
    if (!name || !repoUrl) { setError('Name and Repository URL are required'); return }
    if (!isLocalPath && !pat) { setError('PAT is required for remote repositories'); return }
    
    setCreating(true)
    setError('')
    setScanLog([])
    try {
      setScanLog(['⟶ Registering project…'])
      let project = await api.createProject(name, repoUrl, isLocalPath ? "local_placeholder" : pat, branch)
      
      if (isLocalPath) {
        setScanLog(prev => [...prev, '✓ Project registered', `📁 Target folder: ${repoUrl}`, '⟶ Scanning local repository…'])
        await api.scanProject(project.id, repoUrl, branch)
        
        // Refresh project data to get the updated stage and profile
        const projects = await api.listProjects()
        const updated = projects.find(p => p.id === project.id)
        if (updated) {
          project = updated
          setScanLog(prev => [...prev, '✅ Scan complete!'])
        }
      }
      
      onCreated(project)
    } catch (e) {
      setError((e as Error).message)
      setCreating(false) // Only stop creating on error, otherwise we navigate away
    }
  }

  return (
    <div className="step-card">
      <div className="step-header">
        <div className="step-number">Step 1 of 7</div>
        <h2 className="step-title">Register your repository</h2>
        <p className="step-desc">
          Connect an Azure DevOps/GitHub repository, or provide a local folder path.
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
              disabled={creating}
            />
          </div>

          <div className="field-group">
            <label className="field-label">
              Repository URL or Local Path <span className="required">*</span>
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                id="reg-repo-url"
                className="input input-mono"
                style={{ flex: 1 }}
                placeholder="https://... or C:\path\to\repo"
                value={repoUrl}
                onChange={e => setRepoUrl(e.target.value)}
                required
                disabled={creating}
              />
              <button type="button" className="btn" onClick={handleBrowse} disabled={creating}>
                Browse…
              </button>
            </div>
          </div>

          <div className="field-group" style={{ opacity: isLocalPath ? 0.5 : 1 }}>
            <label className="field-label">
              Personal Access Token (PAT) {!isLocalPath && <span className="required">*</span>}
            </label>
            <input
              id="reg-pat"
              type="password"
              className="input input-mono"
              placeholder={isLocalPath ? "Not required for local paths" : "••••••••••••••••••••"}
              value={pat}
              onChange={e => setPat(e.target.value)}
              required={!isLocalPath}
              disabled={isLocalPath || creating}
            />
            {!isLocalPath && (
              <span className="field-hint">
                Needs Code (Read) + Build (Read) permissions. Stored encrypted.
              </span>
            )}
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
                  disabled={creating}
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
                  disabled={creating}
                />
              )}
              <button
                type="button"
                className="btn btn-sm"
                onClick={fetchBranches}
                disabled={loadingBranches || !repoUrl || (!isLocalPath && !pat) || creating}
              >
                {loadingBranches ? <span className="spinner" /> : 'Load branches'}
              </button>
            </div>
          </div>
        </div>

        {scanLog.length > 0 && (
          <div className="scan-log" style={{ marginBottom: '1rem' }}>
            {scanLog.map((line, i) => (
              <div key={i} style={{ color: line.startsWith('✅') ? 'var(--success)' : undefined }}>
                {line}
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="info-box info-box-danger" style={{ marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        <button type="submit" className="btn btn-primary btn-lg" disabled={creating}>
          {creating ? <><span className="spinner" /> Processing…</> : isLocalPath ? 'Register & Scan →' : 'Register Project →'}
        </button>
      </form>
    </div>
  )
}
