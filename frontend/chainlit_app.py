"""
MLOps Agent — Chainlit UI
=========================
Features:
  • Chat with streaming LLM tokens
  • Tool calls rendered as collapsible cl.Step blocks
  • Human-in-the-loop via cl.AskUserMessage
  • On-demand dashboard card (type "status" or "show dashboard")
  • Repo settings in sidebar via cl.ChatSettings
  • Quick action buttons via cl.Action
"""

import json
import httpx
import chainlit as cl

BACKEND_URL = "http://localhost:8001"
THREAD_ID   = "mlops-session-1"


# ── Startup ───────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Show welcome message and quick-action buttons on first load."""

    # Set default repo settings
    await cl.ChatSettings(
        [
            cl.input_widget.TextInput(
                id="repo_name",
                label="Azure Repo Name",
                placeholder="my-ml-repo",
                initial="",
            ),
            cl.input_widget.TextInput(
                id="repo_branch",
                label="Branch",
                placeholder="main",
                initial="main",
            ),
        ]
    ).send()

    cl.user_session.set("repo_name", "")
    cl.user_session.set("repo_branch", "main")

    # Welcome card
    welcome = """### ⚡ MLOps Agent — Azure ML Workspace Controller

I can help you manage your entire ML lifecycle on Azure ML.

**Try asking:**
- *"Scan my repo and generate all pipelines"*
- *"What's the status of the last training job?"*
- *"Run the inference pipeline"*
- *"Show me the latest drift report"*
- *"Why did the last job fail?"*
- *"Compare challenger vs champion"*
- Type **`status`** or **`show dashboard`** to see a live metrics snapshot."""

    actions = [
        cl.Action(name="scan_repo",      payload={"cmd": "scan_repo"},      label="🔍 Scan Repo"),
        cl.Action(name="run_training",   payload={"cmd": "run_training"},   label="🏋️ Run Training"),
        cl.Action(name="drift_report",   payload={"cmd": "drift_report"},   label="📊 Drift Report"),
        cl.Action(name="run_inference",  payload={"cmd": "run_inference"},  label="🔮 Run Inference"),
        cl.Action(name="show_dashboard", payload={"cmd": "show_dashboard"}, label="📈 Dashboard"),
    ]

    await cl.Message(content=welcome, actions=actions).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    cl.user_session.set("repo_name",   settings.get("repo_name", ""))
    cl.user_session.set("repo_branch", settings.get("repo_branch", "main"))


# ── Quick Action Buttons ──────────────────────────────────────────────────────

@cl.action_callback("scan_repo")
async def on_scan_repo(action: cl.Action):
    await run_agent("Scan the repository and tell me about the ML project")

@cl.action_callback("run_training")
async def on_run_training(action: cl.Action):
    await run_agent("Run the CT pipeline")

@cl.action_callback("drift_report")
async def on_drift_report(action: cl.Action):
    await run_agent("Show me the latest drift monitoring report")

@cl.action_callback("run_inference")
async def on_run_inference(action: cl.Action):
    await run_agent("Run the batch inference pipeline")

@cl.action_callback("show_dashboard")
async def on_show_dashboard(action: cl.Action):
    await render_dashboard()


# ── Main Message Handler ──────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip().lower()

    # Dashboard shortcut
    if text in ("status", "show dashboard", "dashboard", "show status"):
        await render_dashboard()
        return

    await run_agent(message.content.strip())


# ── Agent Runner ──────────────────────────────────────────────────────────────

async def run_agent(user_text: str):
    """
    Calls POST /chat on the FastAPI backend, streams SSE events,
    and renders them as Chainlit steps + a final message.
    """
    repo_name   = cl.user_session.get("repo_name",   "")
    repo_branch = cl.user_session.get("repo_branch", "main")

    # The final agent reply text (accumulated from llm_token events)
    agent_reply   = ""
    # Stack: current open cl.Step (for tool calls)
    current_step: cl.Step | None = None
    # Parent step grouping per node
    node_step: cl.Step | None    = None

    # Outer streaming message (LLM response)
    reply_msg = cl.Message(content="")
    await reply_msg.send()

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/chat",
                json={
                    "message":     user_text,
                    "thread_id":   THREAD_ID,
                    "repo_name":   repo_name,
                    "repo_branch": repo_branch,
                },
            ) as response:

                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data: "):
                        continue

                    try:
                        event = json.loads(raw_line[6:])
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    # ── LLM token → stream into reply message ────────────────
                    if etype == "llm_token":
                        token = event.get("content", "")
                        agent_reply += token
                        await reply_msg.stream_token(token)

                    # ── Node start → open a parent step ─────────────────────
                    elif etype == "node_start":
                        node_name = event.get("node", "agent")
                        node_step = cl.Step(
                            name=f"▶ {node_name}",
                            type="run",
                            show_input=False,
                        )
                        await node_step.__aenter__()

                    # ── Tool call → open a child step ────────────────────────
                    elif etype == "tool_call":
                        tool_name  = event.get("tool", "tool")
                        tool_input = event.get("input", "")
                        current_step = cl.Step(
                            name=f"🔧 {tool_name}",
                            type="tool",
                            show_input=True,
                        )
                        current_step.input = tool_input
                        await current_step.__aenter__()

                    # ── Tool result → close child step ───────────────────────
                    elif etype == "tool_result":
                        tool_output = event.get("output", "")
                        if current_step:
                            current_step.output = tool_output
                            await current_step.__aexit__(None, None, None)
                            current_step = None

                    # ── Node end → close parent step ─────────────────────────
                    elif etype == "node_end":
                        if node_step:
                            await node_step.__aexit__(None, None, None)
                            node_step = None

                    # ── Interrupt → human-in-the-loop ────────────────────────
                    elif etype == "interrupt":
                        interrupt_msg = event.get("message", "The agent needs your decision.")
                        await handle_interrupt(interrupt_msg)

                    # ── Error ────────────────────────────────────────────────
                    elif etype == "error":
                        err = event.get("message", "Unknown error")
                        await cl.Message(
                            content=f"❌ **Error:** {err}",
                            author="System",
                        ).send()

                    # ── Done ─────────────────────────────────────────────────
                    elif etype == "done":
                        break

    except httpx.ConnectError:
        await reply_msg.update()
        await cl.Message(
            content=(
                "⚠️ **Cannot reach the backend.** "
                "Make sure the FastAPI server is running:\n"
                "```bash\nuvicorn backend.main:app --port 8001\n```"
            ),
            author="System",
        ).send()
        return

    except Exception as e:
        await cl.Message(content=f"❌ Unexpected error: {e}", author="System").send()
        return

    # Finalise streaming message
    if not agent_reply:
        reply_msg.content = "✅ Done."
    await reply_msg.update()

    # Re-attach quick action buttons after every reply
    actions = [
        cl.Action(name="scan_repo",      payload={"cmd": "scan_repo"},      label="🔍 Scan Repo"),
        cl.Action(name="run_training",   payload={"cmd": "run_training"},   label="🏋️ Run Training"),
        cl.Action(name="drift_report",   payload={"cmd": "drift_report"},   label="📊 Drift Report"),
        cl.Action(name="run_inference",  payload={"cmd": "run_inference"},  label="🔮 Run Inference"),
        cl.Action(name="show_dashboard", payload={"cmd": "show_dashboard"}, label="📈 Dashboard"),
    ]
    await cl.Message(content="", actions=actions, author="System").send()


# ── Human-in-the-Loop ─────────────────────────────────────────────────────────

async def handle_interrupt(prompt: str):
    """
    Presents an approval prompt and resumes the graph with the user's decision.
    """
    res = await cl.AskActionMessage(
        content=f"⏸ **Agent needs your approval**\n\n{prompt}",
        actions=[
            cl.Action(name="approve", payload={"value": "yes"}, label="✅ Approve / Yes"),
            cl.Action(name="reject",  payload={"value": "no"},  label="❌ Reject / No"),
        ],
        timeout=300,
    ).send()

    decision = res.get("payload", {}).get("value", "no") if res else "no"

    # Post decision to backend and stream the resumed execution
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/feedback",
                json={"thread_id": THREAD_ID, "decision": decision},
            ) as response:
                resume_msg = cl.Message(content="")
                await resume_msg.send()
                resumed_text = ""

                async for raw_line in response.aiter_lines():
                    if not raw_line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(raw_line[6:])
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "llm_token":
                        token = event.get("content", "")
                        resumed_text += token
                        await resume_msg.stream_token(token)

                    elif etype == "tool_call":
                        step = cl.Step(name=f"🔧 {event.get('tool','tool')}", type="tool", show_input=True)
                        step.input = event.get("input", "")
                        await step.__aenter__()
                        await step.__aexit__(None, None, None)

                    elif etype == "tool_result":
                        step = cl.Step(name=f"✓ {event.get('tool','tool')}", type="tool")
                        step.output = event.get("output", "")
                        await step.__aenter__()
                        await step.__aexit__(None, None, None)

                    elif etype == "done":
                        break

                if not resumed_text:
                    resume_msg.content = "✅ Decision recorded. Resuming..."
                await resume_msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ Failed to send decision: {e}", author="System").send()


# ── Dashboard Card ────────────────────────────────────────────────────────────

async def render_dashboard():
    """Fetch /status and render a rich HTML dashboard card."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BACKEND_URL}/status/{THREAD_ID}")
            status = r.json() if r.status_code == 200 else {}
    except Exception:
        status = {}

    pipeline_runs   = status.get("pipeline_runs",         [])
    champion        = status.get("champion_model")
    challenger      = status.get("challenger_model")
    dd              = status.get("data_drift_score")
    pd_             = status.get("prediction_drift_score")
    cd              = status.get("concept_drift_score")

    # ── Pipeline rows ────────────────────────────────────────────────
    def badge(s: str) -> str:
        sl = s.lower()
        if sl in ("running", "inprogress"):
            return f'<span style="background:#1a3a2a;color:#34d399;padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;">● Running</span>'
        elif sl == "completed":
            return f'<span style="background:#1a2d3a;color:#38bdf8;padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;">✓ Completed</span>'
        elif sl == "failed":
            return f'<span style="background:#3a1a1a;color:#f87171;padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;">✗ Failed</span>'
        return f'<span style="background:#2a2a1a;color:#fbbf24;padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;">○ {s}</span>'

    if pipeline_runs:
        pipeline_html = "".join(
            f"""<div style="display:flex;justify-content:space-between;align-items:center;
                           padding:8px 0;border-bottom:1px solid #1e293b;">
                  <div>
                    <div style="color:#e2e8f0;font-size:0.82rem;">{r.get('display_name','Pipeline')}</div>
                    <div style="color:#475569;font-size:0.7rem;font-family:monospace;">{r.get('job_id','')}</div>
                  </div>
                  {badge(r.get('status','Unknown'))}
                </div>"""
            for r in pipeline_runs[-5:]
        )
    else:
        pipeline_html = '<div style="color:#475569;font-size:0.82rem;padding:8px 0;">No pipeline runs yet</div>'

    # ── Model cards ─────────────────────────────────────────────────
    def model_card(model: dict, role: str, color: str, label_color: str) -> str:
        metrics = model.get("metrics", {})
        metric_str = " &nbsp;|&nbsp; ".join(
            f'<span style="color:#94a3b8;">{k}:</span> <b style="color:#e2e8f0;">{v}</b>'
            for k, v in list(metrics.items())[:3]
        )
        return f"""<div style="border:1px solid {color};border-radius:8px;padding:10px;margin:6px 0;">
                     <div style="color:{label_color};font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">{role}</div>
                     <div style="color:#e2e8f0;font-size:0.85rem;margin-top:3px;">{model.get('name')} v{model.get('version')}</div>
                     <div style="font-size:0.75rem;margin-top:4px;">{metric_str}</div>
                   </div>"""

    model_html = ""
    if champion:
        model_html += model_card(champion, "CHAMPION", "#16a34a", "#34d399")
    if challenger:
        model_html += model_card(challenger, "CHALLENGER", "#d97706", "#fbbf24")
    if not champion and not challenger:
        model_html = '<div style="color:#475569;font-size:0.82rem;padding:8px 0;">No models registered yet</div>'

    # ── Drift rows ───────────────────────────────────────────────────
    def drift_row(score, threshold: float, label: str) -> str:
        if score is None:
            return f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b;"><span style="color:#475569;font-size:0.8rem;">{label}</span><span style="color:#475569;font-size:0.8rem;">—</span></div>'
        color = "#f87171" if score > threshold * 1.5 else "#fbbf24" if score > threshold else "#34d399"
        icon  = "🔴" if score > threshold * 1.5 else "🟡" if score > threshold else "🟢"
        return f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b;"><span style="color:#94a3b8;font-size:0.8rem;">{label}</span><span style="color:{color};font-size:0.8rem;">{icon} {score:.2%}</span></div>'

    drift_html = (
        drift_row(dd,  0.15, "Data Drift") +
        drift_row(pd_, 0.10, "Prediction Drift") +
        drift_row(cd,  0.05, "Concept Drift")
    )

    # ── Section header helper ────────────────────────────────────────
    def section(title: str) -> str:
        return f'<div style="font-size:0.75rem;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:0.06em;margin:14px 0 8px;padding-bottom:5px;border-bottom:1px solid #2d3748;">{title}</div>'

    card = f"""
<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:20px;font-family:'Inter',sans-serif;max-width:640px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
    <span style="font-size:1.1rem;font-weight:700;color:#7c3aed;">⚡ MLOps Dashboard</span>
    <span style="font-size:0.72rem;color:#34d399;">● Live</span>
  </div>
  <div style="color:#475569;font-size:0.72rem;margin-bottom:12px;">Thread: {THREAD_ID}</div>

  {section("🔄 Recent Pipelines")}
  {pipeline_html}

  {section("🏆 Model Registry")}
  {model_html}

  {section("📊 Drift Monitor")}
  <div style="background:#131720;border-radius:6px;padding:6px 10px;">
    {drift_html}
  </div>
</div>
"""

    await cl.Message(content=card, author="Dashboard").send()
