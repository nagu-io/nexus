"""
NEXUS FastAPI backend — serves the React dashboard.
Endpoints: /chat, /chat/history, /status, /reflect, /agents, /runtime/*, /ws
"""

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nexus.config import config
from nexus.memory.conversation_store import ConversationStore
from nexus.reflect.reflect_score import ReflectScore
from nexus.router.mind_router import MindRouter
from nexus.runtime.context_reducer import BaseContextReducer, ContextReductionResult
from nexus.runtime.event_bus import runtime_event_bus
from nexus.runtime.insights import RuntimeInsights

app = FastAPI(title="NEXUS API", version="0.2.0")
router = MindRouter()
scorer = ReflectScore()
runtime_insights = RuntimeInsights()
conversation_store = ConversationStore()
context_reducer = router.context_reducer

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# WebSocket connection manager
# ------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections for live dashboard updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: dict[str, Any]):
        """Send an event to all connected clients."""
        dead = []
        message = json.dumps(event)
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)

    @property
    def count(self) -> int:
        return len(self.active_connections)


ws_manager = ConnectionManager()
runtime_event_bus.subscribe(ws_manager.broadcast)


# ------------------------------------------------------------------
# Chat endpoints
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat(req: ChatRequest):
    """Route a message through AEON Mind Router, persist, and return response."""
    # Save user message
    conversation_store.save_message(req.session_id, "user", req.message)
    history = conversation_store.get_context(req.session_id, limit=8)
    prompt = _compose_chat_prompt(history=history, latest_message=req.message)

    # Broadcast user message event
    await ws_manager.broadcast({
        "type": "chat_message",
        "role": "user",
        "content": req.message,
        "session_id": req.session_id,
    })

    # Route through AEON
    await ws_manager.broadcast({"type": "agent_started", "status": "routing"})
    result = await router.route(prompt, return_meta=True)
    reduction = result.get("context_reduction")

    if reduction is not None:
        await ws_manager.broadcast(
            {
                "type": "context_reduced",
                "scope": "chat",
                "session_id": req.session_id,
                **reduction,
            }
        )

    response_data = {
        "response": result["response"],
        "agent": result["agent"],
        "route": result["final_route"],
        "initial_route": result["initial_route"],
        "was_rerouted": result["was_rerouted"],
        "warning": result["warning"],
        "reflect_score": result["reflect_score"],
        "reflect_verdict": result["reflect_verdict"],
        "reflect_action": result["reflect_action"],
        "session_id": req.session_id,
        "context_reduction": reduction,
    }

    # Save assistant response
    conversation_store.save_message(
        req.session_id, "assistant", result["response"],
        metadata={
            "agent": result["agent"],
            "route": result["final_route"],
            "initial_route": result["initial_route"],
            "reflect_score": result["reflect_score"],
            "reflect_verdict": result["reflect_verdict"],
            "reflect_action": result["reflect_action"],
            "warning": result["warning"],
            "was_rerouted": result["was_rerouted"],
            "context_reduction": reduction,
        },
    )

    # Broadcast response event
    await ws_manager.broadcast({
        "type": "chat_response",
        "role": "assistant",
        "content": result["response"],
        "agent": result["agent"],
        "reflect_score": result["reflect_score"],
        "session_id": req.session_id,
        "context_reduction": reduction,
    })

    return response_data


@app.get("/chat/history")
async def chat_history(session_id: str = "default", limit: int = 50):
    """Return conversation history for a session."""
    messages = conversation_store.get_history(session_id, limit=min(limit, 200))
    return {"session_id": session_id, "messages": messages, "total": len(messages)}


@app.get("/chat/sessions")
async def chat_sessions(limit: int = 20):
    """Return recent chat sessions."""
    sessions = conversation_store.list_sessions(limit=min(limit, 100))
    return {"sessions": sessions}


@app.get("/chat/search")
async def chat_search(q: str, limit: int = 20):
    """Full-text search across all conversations."""
    if not q.strip():
        return {"results": [], "query": q}
    results = conversation_store.search(q, limit=min(limit, 50))
    return {"results": results, "query": q}


@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and all its messages."""
    deleted = conversation_store.delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


# ------------------------------------------------------------------
# WebSocket endpoint
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Live event stream for the dashboard.

    Events broadcasted:
    - ``chat_message`` — user sent a message
    - ``chat_response`` — assistant response ready
    - ``agent_started`` — agent is processing
    - ``tool_executed`` — a tool call completed
    - ``execution_output`` — streaming code execution output
    - ``workflow_complete`` — workflow finished
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; handle client messages if needed
            data = await websocket.receive_text()
            # Clients can send ping/subscribe messages
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "connections": ws_manager.count,
                    }))
            except (json.JSONDecodeError, TypeError):
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    finally:
        ws_manager.disconnect(websocket)


# ------------------------------------------------------------------
# System endpoints
# ------------------------------------------------------------------

@app.get("/status")
async def status():
    """Return full NEXUS system status."""
    import httpx
    from nexus.config import config

    ollama_ok = False
    models = []
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{config.ollama_base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except Exception:
        pass

    return {
        "online": True,
        "model": config.nexus_model,
        "ollama": ollama_ok,
        "ollama_models": models,
        "supabase": bool(config.supabase_url),
        "groq": bool(config.groq_api_key),
        "canaryvaults": bool(config.canaryvaults_api_key),
        "route_stats": dict(router.stats),
        "reflect_stats": dict(router.reflect_stats),
        "ws_connections": ws_manager.count,
        "conversation_count": conversation_store.message_count(),
        "thresholds": {
            "routing_complexity": config.routing_complexity_threshold,
            "reflect_warn": config.reflect_warn_threshold,
            "reflect_block": config.reflect_block_threshold,
        },
        "context_reduction": {
            "enabled": bool(context_reducer is not None),
            "backend": context_reducer.backend_name if context_reducer is not None else "disabled",
            "threshold_chars": config.context_reduction_threshold_chars,
            "target_chars": config.context_reduction_target_chars,
            "model": config.context_reduction_model,
        },
    }


@app.get("/reflect/benchmark")
async def run_benchmark():
    """Run ReflectScore benchmark."""
    result = await scorer.run_benchmark(n_samples=300)
    return result


@app.get("/agents")
async def list_agents():
    """List available agents."""
    return {
        "agents": [
            {"name": "coding", "description": "Code generation and debugging"},
            {"name": "research", "description": "Web search, page scraping, and summarization"},
            {"name": "memory", "description": "Persistent memory via Supabase"},
            {"name": "file", "description": "Safe file system operations"},
            {"name": "canary", "description": "CanaryRAG + CanaryVaults integration"},
        ]
    }


@app.get("/runtime/overview")
async def runtime_overview(limit: int = 8):
    """Return recent runtime history plus skill-memory and cache metrics."""
    bounded_limit = max(1, min(int(limit), 25))
    return runtime_insights.overview(limit=bounded_limit)


@app.get("/runtime/runs/{workflow_id}")
async def runtime_run_detail(workflow_id: str):
    """Return one full runtime trace with a derived summary."""
    payload = runtime_insights.run_detail(workflow_id)
    if not payload:
        raise HTTPException(status_code=404, detail="workflow trace not found")
    return payload


def _compose_chat_prompt(*, history: list[dict[str, str]], latest_message: str) -> str:
    """Inject recent local conversation context into the next routed prompt."""
    prior = history[:-1] if history else []
    if not prior:
        return latest_message
    rendered = []
    for item in prior[-6:]:
        role = str(item.get("role", "user")).upper()
        rendered.append(f"{role}: {item.get('content', '')}")
    return (
        "Conversation history:\n"
        + "\n".join(rendered)
        + "\n\nLatest user request:\n"
        + latest_message
    )


def _prepare_chat_prompt(
    *,
    history: list[dict[str, str]],
    latest_message: str,
    reducer: BaseContextReducer | None,
) -> tuple[str, ContextReductionResult | None]:
    """Compose chat context, then reduce it if it exceeds the prompt budget."""
    prompt = _compose_chat_prompt(history=history, latest_message=latest_message)
    if reducer is None:
        return prompt, None
    reduction = reducer.reduce(
        prompt,
        metadata={
            "scope": "chat",
            "history_messages": max(0, len(history) - 1),
        },
    )
    if reduction.reduced:
        return reduction.text, reduction
    return prompt, None
