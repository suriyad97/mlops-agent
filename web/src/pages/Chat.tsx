import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, type AgentStep, type Message, type Project, type Thread } from '../api'

const TOOL_LABELS: Record<string, string> = {
  get_report: 'Reading report',
  repo_graph_digest: 'Analyzing repo graph',
  what_does_file_do: 'Inspecting file',
  who_references_file: 'Tracing references',
  generate_missing_assets: 'Generating assets',
  validate_generated_assets: 'Validating files',
  commit_generated_assets: 'Committing files',
  register_generated_pipelines: 'Registering pipelines',
  list_pipelines: 'Listing pipelines',
  list_pipeline_runs: 'Listing pipeline runs',
  get_pipeline_logs: 'Fetching pipeline logs',
  trigger_pipeline: 'Triggering pipeline',
  list_pull_requests: 'Listing pull requests',
  list_repositories: 'Listing repositories',
  get_ml_workspace: 'Reading ML workspace',
  list_aml_jobs: 'Listing AML jobs',
  list_aml_compute: 'Listing compute targets',
  list_aml_environments: 'Listing environments',
  start_remediation: 'Starting remediation',
  submit_aml_pipeline: 'Submitting AML pipeline',
  get_background_job: 'Checking job status',
  list_background_jobs: 'Listing background jobs',
  get_aml_job_logs: 'Fetching AML logs',
  get_generation_plan: 'Building generation plan',
  watch_pipeline_run: 'Monitoring pipeline run',
  endpoint_requirement_plan: 'Building requirement plan',
  check_pipeline_readiness: 'Checking pipeline readiness',
  check_file_on_disk: 'Verifying file on disk',
  check_infrastructure_prerequisites: 'Checking infrastructure prerequisites',
}
const toolLabel = (t: string) => TOOL_LABELS[t] ?? t

function argSummary(args: Record<string, unknown>): string {
  const entries = Object.entries(args)
  if (entries.length === 0) return ''
  return entries.map(([k, v]) => `${k}: ${String(v).slice(0, 40)}`).join(', ')
}

function StepRow({ step, running }: { step: AgentStep; running?: boolean }) {
  return (
    <div className="flex items-start gap-2 text-[13px]">
      <span className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${running ? 'bg-sky-400 animate-pulse' : 'bg-emerald-400'}`} />
      <div className="min-w-0 flex-1">
        <span className="text-zinc-200">{toolLabel(step.tool)}</span>
        {argSummary(step.args) && (
          <span className="text-zinc-500 ml-1.5 font-mono text-[11px]">{argSummary(step.args)}</span>
        )}
        {/* Per-component progress lines (e.g. from generate_assets) */}
        {step.progress && step.progress.length > 0 && (
          <div className="mt-1.5 grid gap-0.5 border-l-2 border-zinc-700 pl-2.5">
            {step.progress.map((line, i) => (
              <div key={i} className="text-[11px] text-emerald-400 font-mono">{line}</div>
            ))}
            {running && (
              <div className="text-[11px] text-sky-400 animate-pulse">…generating</div>
            )}
          </div>
        )}
        {step.result && (
          <div className="text-zinc-500 text-[11px] mt-0.5 truncate max-w-[480px]">↳ {step.result}</div>
        )}
      </div>
    </div>
  )
}

function StepsPanel({ steps, thinking, live }: { steps: AgentStep[]; thinking?: string; live?: boolean }) {
  if (steps.length === 0 && !thinking) return null
  const body = (
    <div className="grid gap-1.5">
      {thinking && <p className="text-[13px] text-zinc-400 italic whitespace-pre-wrap font-mono">{thinking}</p>}
      {steps.map((s, i) => (
        <StepRow key={i} step={s} running={live && i === steps.length - 1 && !s.result} />
      ))}
    </div>
  )
  if (live) {
    return (
      <div className="bg-zinc-900/70 border border-zinc-700/60 rounded-xl px-3.5 py-3 max-w-[85%]">
        <div className="text-[11px] uppercase tracking-wide text-zinc-500 mb-2 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" /> Agent working
        </div>
        {body}
      </div>
    )
  }
  return (
    <details className="mb-1.5 text-zinc-400">
      <summary className="cursor-pointer text-[12px] text-zinc-500 hover:text-zinc-300 select-none">
        🔧 {steps.length} step{steps.length === 1 ? '' : 's'}
      </summary>
      <div className="mt-2 pl-1 border-l border-zinc-700/60 ml-1">{body}</div>
    </details>
  )
}

function Bubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] ${isUser ? '' : 'w-full'}`}>
        {!isUser && message.steps && message.steps.length > 0 && (
          <StepsPanel steps={message.steps} />
        )}
        <div className={`inline-block rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed ${
          isUser ? 'bg-sky-600 text-white rounded-br-sm'
                 : 'bg-zinc-800 text-zinc-100 rounded-bl-sm border border-zinc-700/60'}`}>
          {isUser
            ? <span className="whitespace-pre-wrap">{message.content}</span>
            : <div className="md-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>}
        </div>
      </div>
    </div>
  )
}

export default function Chat({ embeddedProjectId }: { embeddedProjectId?: string }) {
  const [params, setParams] = useSearchParams()
  const [projects, setProjects] = useState<Project[]>([])
  const [threads, setThreads] = useState<Thread[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [activeThread, setActiveThread] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [liveSteps, setLiveSteps] = useState<AgentStep[]>([])
  const [liveThinking, setLiveThinking] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // When embedded, use the prop directly; otherwise fall back to query param
  const embedded = !!embeddedProjectId
  const projectId = embeddedProjectId ?? params.get('project') ?? ''

  useEffect(() => {
    if (embedded) return  // don't load all projects in embedded mode
    api.listProjects().then((ps) => {
      setProjects(ps)
      if (!projectId && ps.length > 0) setParams({ project: ps[0].id })
    })
  }, [])

  const loadThreads = useCallback(async (pid: string) => {
    const ts = await api.listThreads(pid)
    setThreads(ts)
    return ts
  }, [])

  useEffect(() => {
    if (!projectId) return
    setActiveThread(null)
    setMessages([])
    loadThreads(projectId).then((ts) => {
      if (ts.length > 0) selectThread(ts[0].id)
    })
  }, [projectId])

  async function selectThread(threadId: string) {
    setActiveThread(threadId)
    setMessages(await api.getMessages(threadId))
  }

  async function newThread() {
    const t = await api.createThread(projectId)
    await loadThreads(projectId)
    setActiveThread(t.id)
    setMessages([])
  }

  async function removeThread(threadId: string) {
    await api.deleteThread(threadId)
    const ts = await loadThreads(projectId)
    if (activeThread === threadId) {
      if (ts.length > 0) selectThread(ts[0].id)
      else { setActiveThread(null); setMessages([]) }
    }
  }

  async function send(e: React.FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || busy) return
    let threadId = activeThread
    if (!threadId) {
      const t = await api.createThread(projectId)
      threadId = t.id
      setActiveThread(t.id)
    }
    setMessages((m) => [...m, { role: 'user', content: text }])
    setInput('')
    setBusy(true)
    setLiveSteps([])
    setLiveThinking('')

    const steps: AgentStep[] = []
    let finalContent = ''
    try {
      await api.sendMessageStream(threadId, text, (e) => {
        switch (e.type) {
          case 'thinking':
            // Append to the last step's progress list (shows as sub-lines under the step)
            if (steps.length > 0) {
              const last = steps[steps.length - 1]
              last.progress = [...(last.progress ?? []), e.content]
              setLiveSteps([...steps])
            } else {
              setLiveThinking(e.content)
            }
            break
          case 'step':
            steps.push({ tool: e.tool, args: e.args, progress: [] })
            setLiveSteps([...steps])
            break
          case 'observation': {
            // attach result to the most recent step for this tool that lacks one
            for (let i = steps.length - 1; i >= 0; i--) {
              if (steps[i].tool === e.tool && !steps[i].result) { steps[i].result = e.result; break }
            }
            setLiveSteps([...steps])
            break
          }
          case 'final':
            finalContent = e.content
            break
          case 'error':
            finalContent = `⚠️ **Error:** ${e.message}`
            break
        }
      })
      setMessages((m) => [...m, { role: 'assistant', content: finalContent || '(no response)', steps }])
      loadThreads(projectId) // refresh titles
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', content: `⚠️ **Request failed:** ${(err as Error).message}`, steps }])
    } finally {
      setBusy(false)
      setLiveSteps([])
      setLiveThinking('')
    }
  }

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  // Embedded mode: compact single-column layout (no project selector, no thread sidebar)
  if (embedded) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
        {/* Thread strip */}
        <div style={{ padding: '0.5rem 0.75rem', borderBottom: '1px solid var(--border)', display: 'flex', gap: '0.5rem', alignItems: 'center', overflowX: 'auto', flexShrink: 0 }}>
          <button className="btn btn-sm btn-primary" onClick={newThread} disabled={!projectId}>＋ New thread</button>
          {threads.map(t => (
            <button key={t.id}
              className={`btn btn-sm ${activeThread === t.id ? 'btn-active' : ''}`}
              style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              onClick={() => selectThread(t.id)}
            >
              {t.title}
            </button>
          ))}
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--text-muted)', fontSize: '13px' }}>
              Ask the agent anything about this project.<br />
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Try: "show the gap report" · "generate missing files" · "trigger the training pipeline"
              </span>
            </div>
          )}
          {messages.map((m, i) => <Bubble key={i} message={m} />)}
          {busy && (
            <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
              {liveSteps.length > 0 || liveThinking
                ? <StepsPanel steps={liveSteps} thinking={liveThinking} live />
                : <div style={{ background: 'var(--surface-3)', border: '1px solid var(--border)', borderRadius: 12, padding: '0.6rem 1rem', fontSize: '13px', color: 'var(--text-dim)', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <span className="spinner" /> Thinking…
                  </div>
              }
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{ padding: '0.75rem', borderTop: '1px solid var(--border)', flexShrink: 0 }}>
          <form onSubmit={send} style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={busy || !projectId}
              placeholder='e.g. "show the gap report"'
              className="input"
              style={{ flex: 1, fontSize: '13px' }}
            />
            <button type="submit" disabled={busy || !input.trim() || !projectId} className="btn btn-primary">
              Send
            </button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100vh-57px)]">
      {/* threads sidebar */}
      <aside className="w-64 border-r border-zinc-800 flex flex-col shrink-0">
        <div className="p-3 border-b border-zinc-800 grid gap-2">
          <select value={projectId} onChange={(e) => setParams({ project: e.target.value })}
            className="bg-zinc-900 border border-zinc-700 rounded-lg px-2.5 py-2 text-sm w-full">
            {projects.length === 0 && <option value="">No projects</option>}
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <button onClick={newThread} disabled={!projectId}
            className="bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-lg px-3 py-2 text-sm font-medium">
            + New thread
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {threads.map((t) => (
            <div key={t.id}
              className={`group flex items-center justify-between rounded-lg px-3 py-2 text-sm cursor-pointer mb-1 ${
                activeThread === t.id ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-900'}`}
              onClick={() => selectThread(t.id)}>
              <span className="truncate">{t.title}</span>
              <button onClick={(e) => { e.stopPropagation(); removeThread(t.id) }}
                className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 ml-2">✕</button>
            </div>
          ))}
          {projectId && threads.length === 0 && (
            <p className="text-zinc-600 text-xs px-3 py-2">No threads yet.</p>
          )}
        </div>
      </aside>

      {/* conversation */}
      <section className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 overflow-y-auto px-4 md:px-8 py-6">
          <div className="max-w-3xl mx-auto flex flex-col gap-4">
            {messages.length === 0 && (
              <div className="text-center mt-16 text-zinc-500">
                <h2 className="text-xl font-semibold text-zinc-300">
                  {projectId ? 'Start the conversation' : 'Create a project first'}
                </h2>
                <p className="text-sm mt-2">
                  {projectId
                    ? 'Ask for the scan report, request file generation, or query pipelines. Every AzDO pipeline run requires your approval.'
                    : 'Go to Projects and register a repository.'}
                </p>
              </div>
            )}
            {messages.map((m, i) => <Bubble key={i} message={m} />)}
            {busy && (
              <div className="flex justify-start">
                {liveSteps.length > 0 || liveThinking
                  ? <StepsPanel steps={liveSteps} thinking={liveThinking} live />
                  : (
                    <div className="bg-zinc-900/70 border border-zinc-700/60 rounded-xl px-4 py-3 flex items-center gap-2 text-[13px] text-zinc-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" /> Thinking…
                    </div>
                  )}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </main>
        <footer className="border-t border-zinc-800 px-4 md:px-8 py-4">
          <form onSubmit={send} className="max-w-3xl mx-auto flex gap-3">
            <input value={input} onChange={(e) => setInput(e.target.value)} disabled={busy || !projectId}
              placeholder={projectId ? 'e.g. "show the gap report" or "generate the missing files"' : 'Create a project first'}
              className="flex-1 bg-zinc-900 border border-zinc-700 focus:border-sky-500 outline-none rounded-xl px-4 py-3 text-[15px] placeholder:text-zinc-600" />
            <button type="submit" disabled={busy || !input.trim() || !projectId}
              className="bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-xl px-5 font-medium text-[15px]">
              Send
            </button>
          </form>
        </footer>
      </section>
    </div>
  )
}
