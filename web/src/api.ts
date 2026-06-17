export type PrereqItem = {
  name: string
  status: 'ok' | 'missing_config' | 'not_found' | 'error'
  detail: string
  required_for: string
  fix: string
}

export type InfraReport = {
  checks: PrereqItem[]
  all_ok: boolean
}

export type DiscoveredConfig = {
  subscription_id: string
  resource_group: string
  acr_name: string
  aml_workspace: string
  aml_compute_target: string
  discovered: string[]   // profile key names that were auto-discovered
  errors: string[]       // non-fatal errors during discovery
}

export type DiscoverResult = {
  discovered: DiscoveredConfig
  report: InfraReport
}

export type AzureInventory = {
  subscription_id: string
  resource_groups: string[]
  workspaces: { name: string; resource_group: string }[]
  acrs: { name: string; resource_group: string }[]
  service_connections: string[]
  errors: string[]
}

export type ContractStage = {
  stage: string
  display_name: string
  standard_path: string
  capability: string
  present: boolean
  detected_path: string
  detected_symbol: string
  meets_contract: boolean
  mode: 'wired' | 'adapter' | 'scaffold'
  note: string
}

export type ContractManifest = {
  stages: ContractStage[]
  has_eda: boolean
  has_shap: boolean
  has_feature_engineering: boolean
}

export type ContractResult = {
  endpoint_strategy: string
  contract: ContractManifest
}

export type GeneratedComponent = {
  capability: string
  component: string
  files: string[]
  adapted: boolean
  note: string
}

export type GenerationReport = {
  generated?: boolean
  components: GeneratedComponent[]
  written_files: string[]
  created_files?: string[]
  updated_files?: string[]
  adapter_files: string[]
  scaffold_files: string[]
  wired_skipped: string[]
  superseded_files: string[]
  summary: string
}

export type RequirementItem = {
  order: number
  requirement: string
  owner: 'data_scientist' | 'platform' | 'infra'
  status: 'present' | 'adapter' | 'scaffold' | 'exists' | 'will_generate' | 'prerequisite' | 'user_action'
  deliverables: string[]
  detail: string
}

export type RequirementPlan = {
  endpoint_strategy: string
  items: RequirementItem[]
}

export type RequirementPlanResult = {
  plan: RequirementPlan
  markdown: string
}

export type Project = {
  id: string
  name: string
  repo_url: string
  stage: string
  profile: Record<string, unknown>
  local_repo_path: string
  base_branch: string
}

export type Thread = { id: string; title: string }
export type AgentStep = { tool: string; args: Record<string, unknown>; result?: string; progress?: string[] }
export type Message = { role: 'user' | 'assistant'; content: string; steps?: AgentStep[] }

export type StreamEvent =
  | { type: 'thinking'; content: string }
  | { type: 'step'; tool: string; args: Record<string, unknown> }
  | { type: 'observation'; tool: string; result: string }
  | { type: 'final'; content: string }
  | { type: 'error'; message: string }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    let errMsg = `HTTP ${response.status}`
    if (data.detail) {
      if (typeof data.detail === 'string') {
        errMsg = data.detail
      } else if (Array.isArray(data.detail)) {
        errMsg = data.detail.map((e: any) => `${e.loc?.join('.')}: ${e.msg}`).join(', ')
      } else {
        errMsg = JSON.stringify(data.detail)
      }
    }
    throw new Error(errMsg)
  }
  return response.json()
}

export const api = {
  listProjects: () => request<Project[]>('/api/projects'),
  createProject: (name: string, repo_url: string, pat: string, base_branch: string) =>
    request<Project>('/api/projects', { method: 'POST', body: JSON.stringify({ name, repo_url, pat, base_branch }) }),
  browsePath: () => request<{ path: string }>('/api/browse'),
  getBranches: (repo_url: string, pat: string) =>
    request<{ branches: string[] }>('/api/repos/branches', { method: 'POST', body: JSON.stringify({ repo_url, pat }) }),
  deleteProject: (id: string) => request<{ deleted: string }>(`/api/projects/${id}`, { method: 'DELETE' }),
  getProjectBranches: (id: string) => request<{ branches: string[] }>(`/api/projects/${id}/branches`),
  scanProject: (id: string, local_path: string, branch: string) =>
    request<{ profile: Record<string, unknown> }>(`/api/projects/${id}/scan`, { method: 'POST', body: JSON.stringify({ local_path, branch }) }),
  patchProfile: (id: string, profile: Record<string, unknown>) =>
    request<Project>(`/api/projects/${id}/profile`, { method: 'PATCH', body: JSON.stringify({ profile }) }),
  getRequirementPlan: (id: string) => request<RequirementPlanResult>(`/api/projects/${id}/requirement-plan`),
  getContract: (id: string) => request<ContractResult>(`/api/projects/${id}/contract`),
  saveContract: (id: string, contract: ContractManifest) =>
    request<{ saved: boolean; contract: ContractManifest }>(
      `/api/projects/${id}/contract`, { method: 'PUT', body: JSON.stringify({ contract }) }),
  checkProjectInfra: (id: string) => request<InfraReport>(`/api/projects/${id}/infra-check`),
  verifyDataPaths: (id: string) => request<InfraReport>(`/api/projects/${id}/verify-data-paths`),
  discoverProjectInfra: (id: string) => request<DiscoverResult>(`/api/projects/${id}/infra-discover`, { method: 'POST' }),
  getAzureInventory: (id: string) => request<AzureInventory>(`/api/projects/${id}/azure-inventory`),
  generateProject: (id: string) =>
    request<GenerationReport>(`/api/projects/${id}/generate`, { method: 'POST', body: JSON.stringify({}) }),
  getGenerationReport: (id: string) =>
    request<GenerationReport>(`/api/projects/${id}/generation-report`),
  validateProject: (id: string) => request<Record<string, unknown>>(`/api/projects/${id}/validate`, { method: 'POST' }),
  commitProject: (id: string, message: string) =>
    request<Record<string, unknown>>(`/api/projects/${id}/commit`, { method: 'POST', body: JSON.stringify({ message }) }),
  getReports: (id: string, kind?: string) =>
    request<{ kind: string; payload: Record<string, unknown> }[]>(
      `/api/projects/${id}/reports${kind ? `?kind=${kind}` : ''}`,
    ),
  listThreads: (projectId: string) => request<Thread[]>(`/api/projects/${projectId}/threads`),
  createThread: (projectId: string) =>
    request<Thread>(`/api/projects/${projectId}/threads`, { method: 'POST' }),
  renameThread: (threadId: string, title: string) =>
    request<Thread>(`/api/threads/${threadId}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteThread: (threadId: string) => request(`/api/threads/${threadId}`, { method: 'DELETE' }),
  getMessages: (threadId: string) => request<Message[]>(`/api/threads/${threadId}/messages`),
  sendMessage: (threadId: string, content: string) =>
    request<Message>(`/api/threads/${threadId}/messages`, { method: 'POST', body: JSON.stringify({ content }) }),
  sendMessageStream: async (threadId: string, content: string, onEvent: (e: StreamEvent) => void) => {
    const res = await fetch(`/api/threads/${threadId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop() ?? ''
      for (const part of parts) {
        const line = part.trim()
        if (line.startsWith('data:')) {
          const payload = line.slice(5).trim()
          if (payload) {
            try { onEvent(JSON.parse(payload) as StreamEvent) } catch { /* ignore partial */ }
          }
        }
      }
    }
  },
}
