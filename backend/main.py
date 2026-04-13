"""
FastAPI Backend — MLOps Agent
==============================
Serves the LangGraph agent and streams events to the Chainlit frontend
via Server-Sent Events (SSE).

Run on port 8001:  uvicorn backend.main:app --port 8001 --reload
"""

import json
import asyncio
from typing import AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from backend/)
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlops_agent.graph import build_graph
from mlops_agent.state import MLOpsState

app = FastAPI(title="MLOps Agent API", version="1.0.0")

# Allow Chainlit (localhost:8000) and any local dev origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph graph once at startup
graph = build_graph()


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    repo_name: str = ""
    repo_branch: str = "main"


class HumanFeedbackRequest(BaseModel):
    thread_id: str
    decision: str        # e.g. "yes", "no", "promote", "keep_champion"
    extra_context: str = ""


# ── SSE Stream Helper ─────────────────────────────────────────────────────────

async def event_stream(thread_id: str, message: str, repo_name: str, repo_branch: str) -> AsyncGenerator[str, None]:
    """
    Runs the LangGraph agent and yields SSE-formatted events for each
    step so the Streamlit frontend can render them in real time.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: MLOpsState = {
        "messages": [{"role": "user", "content": message}],
        "repo_name": repo_name or None,
        "repo_branch": repo_branch,
    }

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event", "")
            data = event.get("data", {})
            name = event.get("name", "")

            # Filter and format events to send to the UI
            if event_type == "on_chain_start" and name not in ("LangGraph",):
                payload = {"type": "node_start", "node": name, "message": f"Starting: {name}"}
                yield f"data: {json.dumps(payload)}\n\n"

            elif event_type == "on_tool_start":
                tool_input = data.get("input", {})
                payload = {"type": "tool_call", "tool": name, "input": str(tool_input)[:200]}
                yield f"data: {json.dumps(payload)}\n\n"

            elif event_type == "on_tool_end":
                tool_output = str(data.get("output", ""))[:300]
                payload = {"type": "tool_result", "tool": name, "output": tool_output}
                yield f"data: {json.dumps(payload)}\n\n"

            elif event_type == "on_chat_model_stream":
                chunk = data.get("chunk", {})
                content = getattr(chunk, "content", "")
                if content:
                    payload = {"type": "llm_token", "content": content}
                    yield f"data: {json.dumps(payload)}\n\n"

            elif event_type == "on_chain_end" and name not in ("LangGraph",):
                payload = {"type": "node_end", "node": name}
                yield f"data: {json.dumps(payload)}\n\n"

            await asyncio.sleep(0)  # yield control to event loop

    except Exception as e:
        payload = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(payload)}\n\n"

    # Signal stream completion
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Streams agent events back as SSE.
    """
    return StreamingResponse(
        event_stream(req.thread_id, req.message, req.repo_name, req.repo_branch),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
async def human_feedback(req: HumanFeedbackRequest):
    """
    Resumes an interrupted graph with the human's decision.
    Called when the user clicks approve/reject in the UI.
    """
    config = {"configurable": {"thread_id": req.thread_id}}

    feedback_message = req.decision
    if req.extra_context:
        feedback_message += f". Additional context: {req.extra_context}"

    update = {
        "messages": [{"role": "user", "content": feedback_message}],
        "human_review_approved": req.decision.lower() in ("yes", "approve", "proceed", "promote"),
        "human_promotion_decision": req.decision.lower(),
    }

    await graph.aupdate_state(config, update)

    return StreamingResponse(
        event_stream(req.thread_id, feedback_message, "", "main"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    """
    Returns the current state snapshot for a given thread.
    Used by the Streamlit dashboard to refresh pipeline/model/drift panels.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    if snapshot and snapshot.values:
        state = snapshot.values
        return {
            "pipeline_runs": state.get("pipeline_runs", []),
            "champion_model": state.get("champion_model"),
            "challenger_model": state.get("challenger_model"),
            "data_drift_score": state.get("data_drift_score"),
            "prediction_drift_score": state.get("prediction_drift_score"),
            "concept_drift_score": state.get("concept_drift_score"),
            "current_job_id": state.get("current_job_id"),
            "current_job_status": state.get("current_job_status"),
            "drift_threshold_breached": state.get("drift_threshold_breached"),
        }
    return {}


@app.get("/health")
async def health():
    return {"status": "ok"}
