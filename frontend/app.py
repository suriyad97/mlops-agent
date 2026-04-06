"""
MLOps Agent — Streamlit UI
============================
Three-panel dashboard:
  Left   : Pipeline status + model registry + drift scores
  Center : Live agent action feed (streams every tool call in real time)
  Right  : Chat interface to talk to the agent
"""

import json
import time
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"
THREAD_ID = "mlops-session-1"

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLOps Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    /* Panel cards */
    .panel-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Status badges */
    .badge-running   { background:#1a3a2a; color:#34d399; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-completed { background:#1a2d3a; color:#38bdf8; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-failed    { background:#3a1a1a; color:#f87171; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .badge-pending   { background:#2a2a1a; color:#fbbf24; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

    /* Event feed items */
    .event-node_start  { border-left: 3px solid #34d399; padding-left:10px; margin:4px 0; font-size:0.82rem; }
    .event-tool_call   { border-left: 3px solid #38bdf8; padding-left:10px; margin:4px 0; font-size:0.82rem; }
    .event-tool_result { border-left: 3px solid #7c3aed; padding-left:10px; margin:4px 0; font-size:0.82rem; }
    .event-interrupt   { border-left: 3px solid #fbbf24; padding-left:10px; margin:4px 0; font-size:0.82rem; }
    .event-error       { border-left: 3px solid #f87171; padding-left:10px; margin:4px 0; font-size:0.82rem; }

    /* Metric numbers */
    .metric-value { font-size:1.4rem; font-weight:700; color:#e2e8f0; }
    .metric-label { font-size:0.75rem; color:#64748b; margin-top:2px; }
    .metric-green { color:#34d399; }
    .metric-amber { color:#fbbf24; }
    .metric-red   { color:#f87171; }

    /* Section headers */
    .section-header {
        font-size:0.8rem; font-weight:700; color:#7c3aed;
        text-transform:uppercase; letter-spacing:0.05em;
        margin-bottom:10px; padding-bottom:6px;
        border-bottom:1px solid #2d3748;
    }

    /* Chat messages */
    .chat-user   { background:#2d1f4e; border-radius:8px; padding:10px 14px; margin:6px 0; font-size:0.85rem; }
    .chat-agent  { background:#1a1f2e; border-radius:8px; padding:10px 14px; margin:6px 0; font-size:0.85rem; border:1px solid #2d3748; }
    .chat-label  { font-size:0.7rem; font-weight:700; margin-bottom:4px; }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "event_feed" not in st.session_state:
    st.session_state.event_feed = []
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False
if "approval_context" not in st.session_state:
    st.session_state.approval_context = ""


# ── Helper Functions ──────────────────────────────────────────────────────────

def fetch_status() -> dict:
    """Fetch current pipeline/model/drift status from backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/status/{THREAD_ID}", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_badge(status: str) -> str:
    status_lower = status.lower()
    if status_lower in ("running", "inprogress"):
        return f'<span class="badge-running">● Running</span>'
    elif status_lower == "completed":
        return f'<span class="badge-completed">✓ Completed</span>'
    elif status_lower == "failed":
        return f'<span class="badge-failed">✗ Failed</span>'
    else:
        return f'<span class="badge-pending">○ {status}</span>'


def drift_color(score, threshold) -> str:
    if score is None:
        return "metric-label"
    if score > threshold * 1.5:
        return "metric-red"
    elif score > threshold:
        return "metric-amber"
    return "metric-green"


def stream_chat(message: str, repo_name: str = "", repo_branch: str = "main"):
    """Send a message and stream the response events into session state."""
    st.session_state.agent_running = True
    st.session_state.chat_history.append({"role": "user", "content": message})
    st.session_state.event_feed = []

    agent_response = ""

    try:
        with requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "message": message,
                "thread_id": THREAD_ID,
                "repo_name": repo_name,
                "repo_branch": repo_branch,
            },
            stream=True,
            timeout=300,
        ) as response:
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    raw = line[6:].decode("utf-8")
                    try:
                        event = json.loads(raw)
                        event_type = event.get("type", "")

                        if event_type == "llm_token":
                            agent_response += event.get("content", "")

                        elif event_type in ("tool_call", "tool_result", "node_start", "node_end", "interrupt", "error"):
                            st.session_state.event_feed.append(event)

                        elif event_type == "done":
                            break

                    except json.JSONDecodeError:
                        pass

    except Exception as e:
        st.session_state.event_feed.append({"type": "error", "message": str(e)})

    if agent_response:
        st.session_state.chat_history.append({"role": "agent", "content": agent_response})

    st.session_state.agent_running = False


def send_approval(decision: str):
    """Send human approval/rejection back to the agent."""
    try:
        with requests.post(
            f"{BACKEND_URL}/feedback",
            json={"thread_id": THREAD_ID, "decision": decision},
            stream=True,
            timeout=300,
        ) as response:
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    raw = line[6:].decode("utf-8")
                    try:
                        event = json.loads(raw)
                        if event.get("type") in ("tool_call", "tool_result", "node_start"):
                            st.session_state.event_feed.append(event)
                        elif event.get("type") == "llm_token":
                            pass
                        elif event.get("type") == "done":
                            break
                    except Exception:
                        pass
        st.session_state.awaiting_approval = False
        st.session_state.chat_history.append({"role": "user", "content": f"[{decision.upper()}]"})
    except Exception as e:
        st.error(f"Failed to send approval: {e}")


# ── Layout ────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1f2e,#2d3748);padding:16px 24px;border-radius:10px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;">
    <div>
        <span style="font-size:1.4rem;font-weight:700;color:#7c3aed;">⚡ MLOps Agent</span>
        <span style="font-size:0.8rem;color:#64748b;margin-left:12px;">Azure ML Workspace Controller</span>
    </div>
    <span style="font-size:0.75rem;color:#34d399;">● Connected</span>
</div>
""", unsafe_allow_html=True)

# Three-column layout
col_left, col_center, col_right = st.columns([1.2, 1.5, 1.3])

# ── LEFT: Dashboard ───────────────────────────────────────────────────────────
with col_left:
    status = fetch_status()

    # Repo input at top
    st.markdown('<div class="section-header">🔗 Repo Settings</div>', unsafe_allow_html=True)
    repo_name = st.text_input("Azure Repo Name", placeholder="my-ml-repo", label_visibility="collapsed")
    repo_branch = st.text_input("Branch", value="main", label_visibility="collapsed")

    st.markdown("---")

    # Pipelines
    st.markdown('<div class="section-header">🔄 Pipelines</div>', unsafe_allow_html=True)
    pipeline_runs = status.get("pipeline_runs", [])
    if pipeline_runs:
        for run in pipeline_runs[-6:]:
            badge = get_badge(run.get("status", "Unknown"))
            st.markdown(f"""
            <div class="panel-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.82rem;color:#e2e8f0;">{run.get('display_name','Pipeline')}</span>
                    {badge}
                </div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:4px;">{run.get('job_id','')}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#64748b;font-size:0.82rem;padding:8px;">No pipeline runs yet</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Models
    st.markdown('<div class="section-header">🏆 Models</div>', unsafe_allow_html=True)
    champion = status.get("champion_model")
    challenger = status.get("challenger_model")

    if champion:
        metrics = champion.get("metrics", {})
        metric_str = " | ".join([f"{k}: {v}" for k, v in list(metrics.items())[:2]])
        st.markdown(f"""
        <div class="panel-card" style="border-color:#16a34a;">
            <div style="color:#34d399;font-size:0.75rem;font-weight:700;">CHAMPION</div>
            <div style="color:#e2e8f0;font-size:0.85rem;margin-top:4px;">{champion.get('name')} v{champion.get('version')}</div>
            <div style="color:#64748b;font-size:0.72rem;margin-top:2px;">{metric_str}</div>
        </div>
        """, unsafe_allow_html=True)

    if challenger:
        metrics = challenger.get("metrics", {})
        metric_str = " | ".join([f"{k}: {v}" for k, v in list(metrics.items())[:2]])
        st.markdown(f"""
        <div class="panel-card" style="border-color:#d97706;">
            <div style="color:#fbbf24;font-size:0.75rem;font-weight:700;">CHALLENGER</div>
            <div style="color:#e2e8f0;font-size:0.85rem;margin-top:4px;">{challenger.get('name')} v{challenger.get('version')}</div>
            <div style="color:#64748b;font-size:0.72rem;margin-top:2px;">{metric_str}</div>
        </div>
        """, unsafe_allow_html=True)

    if not champion and not challenger:
        st.markdown('<div style="color:#64748b;font-size:0.82rem;padding:8px;">No models registered yet</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Drift Scores
    st.markdown('<div class="section-header">📊 Drift Monitor</div>', unsafe_allow_html=True)
    dd = status.get("data_drift_score")
    pd_ = status.get("prediction_drift_score")
    cd = status.get("concept_drift_score")

    thresholds = {"data": 0.15, "prediction": 0.10, "concept": 0.05}

    def drift_indicator(score, threshold, label):
        if score is None:
            return f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b;"><span style="color:#64748b;font-size:0.8rem;">{label}</span><span style="color:#64748b;font-size:0.8rem;">—</span></div>'
        color = "#f87171" if score > threshold * 1.5 else "#fbbf24" if score > threshold else "#34d399"
        icon = "🔴" if score > threshold * 1.5 else "🟡" if score > threshold else "🟢"
        return f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b;"><span style="color:#94a3b8;font-size:0.8rem;">{label}</span><span style="color:{color};font-size:0.8rem;">{icon} {score:.2%}</span></div>'

    st.markdown(f"""
    <div class="panel-card">
        {drift_indicator(dd, thresholds["data"], "Data Drift")}
        {drift_indicator(pd_, thresholds["prediction"], "Prediction Drift")}
        {drift_indicator(cd, thresholds["concept"], "Concept Drift")}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

# ── CENTER: Agent Action Feed ─────────────────────────────────────────────────
with col_center:
    st.markdown('<div class="section-header">⚡ Live Agent Actions</div>', unsafe_allow_html=True)

    # Approval gate — shows when agent hits an interrupt node
    if st.session_state.awaiting_approval:
        st.markdown(f"""
        <div style="background:#2e1f0c;border:1px solid #d97706;border-radius:8px;padding:14px;margin-bottom:12px;">
            <div style="color:#fbbf24;font-weight:700;font-size:0.85rem;">⏸ Waiting for your decision</div>
            <div style="color:#94a3b8;font-size:0.82rem;margin-top:6px;">{st.session_state.approval_context}</div>
        </div>
        """, unsafe_allow_html=True)
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✓ Approve / Yes", use_container_width=True, type="primary"):
                send_approval("yes")
                st.rerun()
        with col_no:
            if st.button("✗ Reject / No", use_container_width=True):
                send_approval("no")
                st.rerun()

    # Event feed
    event_container = st.container()
    with event_container:
        if not st.session_state.event_feed:
            st.markdown("""
            <div style="color:#64748b;font-size:0.82rem;text-align:center;padding:40px 20px;">
                Agent actions will appear here in real time.<br>
                Type a command in the chat to get started.
            </div>
            """, unsafe_allow_html=True)
        else:
            for event in st.session_state.event_feed:
                event_type = event.get("type", "")

                if event_type == "node_start":
                    st.markdown(f'<div class="event-node_start"><span style="color:#34d399;">▶</span> <strong>{event.get("node","")}</strong></div>', unsafe_allow_html=True)

                elif event_type == "tool_call":
                    st.markdown(f"""
                    <div class="event-tool_call">
                        <span style="color:#38bdf8;">🔧</span>
                        <strong>{event.get("tool","")}</strong>
                        <div style="color:#64748b;font-size:0.75rem;margin-top:2px;font-family:monospace;">{event.get("input","")}</div>
                    </div>
                    """, unsafe_allow_html=True)

                elif event_type == "tool_result":
                    st.markdown(f"""
                    <div class="event-tool_result">
                        <span style="color:#c4b5fd;">✓</span>
                        <strong>{event.get("tool","")}</strong> result
                        <div style="color:#64748b;font-size:0.75rem;margin-top:2px;font-family:monospace;">{event.get("output","")[:150]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                elif event_type == "interrupt":
                    st.markdown(f'<div class="event-interrupt"><span style="color:#fbbf24;">⏸</span> <strong>Awaiting human input</strong></div>', unsafe_allow_html=True)
                    st.session_state.awaiting_approval = True
                    st.session_state.approval_context = event.get("message", "")

                elif event_type == "error":
                    st.markdown(f'<div class="event-error"><span style="color:#f87171;">✗</span> Error: {event.get("message","")}</div>', unsafe_allow_html=True)

    if st.session_state.agent_running:
        st.markdown('<div style="color:#7c3aed;font-size:0.8rem;margin-top:8px;animation:pulse 1s infinite;">⚡ Agent is working...</div>', unsafe_allow_html=True)

# ── RIGHT: Chat ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-header">💬 Chat with Agent</div>', unsafe_allow_html=True)

    # Chat history
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="color:#64748b;font-size:0.82rem;padding:16px;">
                <strong style="color:#c4b5fd;">Try asking:</strong><br><br>
                • "Scan my repo and generate all pipelines"<br>
                • "What's the status of the last training job?"<br>
                • "Run the inference pipeline"<br>
                • "Show me the latest drift report"<br>
                • "Why did the last job fail?"<br>
                • "Compare challenger vs champion"
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-user">
                        <div class="chat-label" style="color:#c4b5fd;">You</div>
                        <div style="color:#e2e8f0;">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-agent">
                        <div class="chat-label" style="color:#38bdf8;">⚡ Agent</div>
                        <div style="color:#e2e8f0;">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Input box
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder="Ask anything or give a command...",
            label_visibility="collapsed",
            height=80,
        )
        submitted = st.form_submit_button(
            "Send ⚡",
            use_container_width=True,
            disabled=st.session_state.agent_running,
        )

    if submitted and user_input.strip():
        stream_chat(user_input.strip(), repo_name, repo_branch)
        st.rerun()

    # Quick action buttons
    st.markdown('<div class="section-header" style="margin-top:12px;">⚡ Quick Actions</div>', unsafe_allow_html=True)
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("🔍 Scan Repo", use_container_width=True):
            stream_chat("Scan the repository and tell me about the ML project", repo_name, repo_branch)
            st.rerun()
        if st.button("🏋️ Run Training", use_container_width=True):
            stream_chat("Run the CT pipeline", repo_name, repo_branch)
            st.rerun()
    with col_q2:
        if st.button("📊 Drift Report", use_container_width=True):
            stream_chat("Show me the latest drift monitoring report", repo_name, repo_branch)
            st.rerun()
        if st.button("🔮 Run Inference", use_container_width=True):
            stream_chat("Run the batch inference pipeline", repo_name, repo_branch)
            st.rerun()
