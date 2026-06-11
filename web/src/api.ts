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
    const detail = await response.json().catch(() => ({}))
    throw new Error(detail.detail ?? `HTTP ${response.status}`)
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
  getReports: (id: string, kind?: string) =>
    request<{ kind: string; payload: Record<string, unknown> }[]>(
      `/api/projects/${id}/reports${kind ? `?kind=${kind}` : ''}`,
    ),
  listThreads: (projectId: string) => request<Thread[]>(`/api/projects/${projectId}/threads`),
  createThread: (projectId: string) =>
    request<Thread>(`/api/projects/${projectId}/threads`, { method: 'POST' }),
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
