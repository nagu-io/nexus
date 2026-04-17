"""
NEXUS FastAPI backend — serves the React dashboard.
Endpoints: /chat, /chat/history, /workspace/*, /status, /reflect, /agents, /runtime/*, /ws
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nexus.blueprint_generator import BlueprintGenerator
from nexus.compiler.planner_engine import PlannerEngine
from nexus.config import config
from nexus.hive.runtime import HiveRuntime
from nexus.intent_parser import IntentParser
from nexus.memory.conversation_store import ConversationStore
from nexus.orchestrator import Orchestrator
from nexus.reflect.reflect_score import ReflectScore
from nexus.router.mind_router import MindRouter
from nexus.router.provider_runtime import (
    active_local_model_label,
    configured_local_backend,
    preferred_cloud_model,
    preferred_cloud_provider,
)
from nexus.runtime.context_reducer import BaseContextReducer, ContextReductionResult
from nexus.runtime.event_bus import runtime_event_bus
from nexus.runtime.file_tool import FileTool
from nexus.runtime.insights import RuntimeInsights
from nexus.runtime.model_control import ModelControlCenter
from nexus.runtime.project_mode import ProjectModeManager

app = FastAPI(title="NEXUS API", version="0.2.0")
router = MindRouter()
scorer = ReflectScore()
hive_runtime = HiveRuntime()
runtime_insights = RuntimeInsights()
model_control = ModelControlCenter(
    config=config,
    repo_root=Path(os.getenv("NEXUS_APP_ROOT", Path.cwd())),
)
conversation_store = ConversationStore()
context_reducer = router.context_reducer

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
workspace_route_count = 0
HIDDEN_WORKSPACE_NAMES = {".nexus", ".nexus-dev"}
ENV_FILE_PATH = Path(os.getenv("NEXUS_ENV_PATH", Path.cwd() / ".env")).expanduser()


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _provider_settings_payload() -> dict[str, Any]:
    provider = preferred_cloud_provider(config)
    return {
        "active_provider": provider or "none",
        "active_model": preferred_cloud_model(config) if provider else "",
        "openrouter": {
            "configured": bool(config.openrouter_api_key),
            "masked_key": _mask_secret(config.openrouter_api_key),
            "model": config.openrouter_model,
            "base_url": config.openrouter_base_url,
        },
        "anthropic": {
            "configured": bool(config.anthropic_api_key),
        },
    }


def _persist_env_values(updates: dict[str, str]) -> None:
    lines = []
    if ENV_FILE_PATH.exists():
        lines = ENV_FILE_PATH.read_text(encoding="utf-8", errors="replace").splitlines()

    for key, value in updates.items():
        entry = f"{key}={value}"
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[index] = entry
                replaced = True
                break
        if not replaced:
            lines.append(entry)

    ENV_FILE_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Chat endpoints
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    workspace_root: str | None = None
    workspace_mode: bool = False
    execution_mode: str = "stable"


class WorkspaceFileUpdate(BaseModel):
    root: str
    path: str
    content: str


class OpenRouterSettingsUpdate(BaseModel):
    api_key: str | None = None
    model: str | None = None
    clear_api_key: bool = False


class LocalRuntimeSettingsUpdate(BaseModel):
    backend: str | None = None
    local_model_dir: str | None = None
    launch_model: str | None = None


class HiveDemoRequest(BaseModel):
    prompt: str
    intent: str = "coding"


class CompressionRequest(BaseModel):
    bits: int = 4


@app.post("/chat")
async def chat(req: ChatRequest):
    """Route a message through AEON Mind Router, persist, and return response."""
    global workspace_route_count
    try:
        # Save user message
        conversation_store.save_message(req.session_id, "user", req.message)

        # Broadcast user message event
        await ws_manager.broadcast({
            "type": "chat_message",
            "role": "user",
            "content": req.message,
            "session_id": req.session_id,
        })

        # Route through AEON
        await ws_manager.broadcast({"type": "agent_started", "status": "routing"})
        if req.workspace_mode and req.workspace_root:
            response_data = await _chat_via_workspace_execution(req)
        elif _is_hive_chat_request(req.message):
            response_data = await _chat_via_hive(req)
        else:
            history = conversation_store.get_context(req.session_id, limit=8)
            prompt = _compose_chat_prompt(history=history, latest_message=req.message)
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
                "execution": _hive_execution_payload(result.get("hive_details")) if result["final_route"] == "hive" else None,
                "workspace_root": req.workspace_root,
            }

        if response_data["route"] == "workspace":
            workspace_route_count += 1

        # Save assistant response
        conversation_store.save_message(
            req.session_id, "assistant", response_data["response"],
            metadata={
                "agent": response_data["agent"],
                "route": response_data["route"],
                "initial_route": response_data["initial_route"],
                "reflect_score": response_data["reflect_score"],
                "reflect_verdict": response_data["reflect_verdict"],
                "reflect_action": response_data["reflect_action"],
                "warning": response_data["warning"],
                "was_rerouted": response_data["was_rerouted"],
                "context_reduction": response_data.get("context_reduction"),
                "execution": response_data.get("execution"),
                "workspace_root": response_data.get("workspace_root"),
            },
        )

        # Broadcast response event
        await ws_manager.broadcast({
            "type": "chat_response",
            "role": "assistant",
            "content": response_data["response"],
            "agent": response_data["agent"],
            "reflect_score": response_data["reflect_score"],
            "session_id": req.session_id,
            "context_reduction": response_data.get("context_reduction"),
            "execution": response_data.get("execution"),
        })

        return response_data
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {error}") from error


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


@app.get("/workspace")
async def workspace_listing(root: str | None = None, path: str = ""):
    """Browse one local workspace directory for the dashboard editor."""
    try:
        workspace_root = _resolve_workspace_root(root)
        directory = _resolve_workspace_path(workspace_root, path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory}")
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    current_path = "" if directory == workspace_root else directory.relative_to(workspace_root).as_posix()
    parent_path = None
    if current_path:
        parent_path = str(Path(current_path).parent).replace("\\", "/")
        if parent_path == ".":
            parent_path = ""

    return {
        "root": str(workspace_root),
        "root_name": workspace_root.name or str(workspace_root),
        "current_path": current_path,
        "parent_path": parent_path,
        "items": _list_workspace_items(workspace_root, directory),
    }


@app.get("/workspace/file")
async def workspace_file(root: str | None = None, path: str = ""):
    """Load one file from the chosen workspace root."""
    if not path.strip():
        raise HTTPException(status_code=400, detail="A file path is required")

    try:
        workspace_root = _resolve_workspace_root(root)
        tool = FileTool(allowed_roots=[workspace_root])
        result = tool.execute(
            {
                "tool": "file_tool",
                "action": "read_file",
                "arguments": {"path": path},
            }
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or result.get("summary") or "Could not read file")
        resolved_path = Path(result["path"]).resolve()
        relative_path = resolved_path.relative_to(workspace_root).as_posix()
        return {
            "root": str(workspace_root),
            "path": relative_path,
            "name": resolved_path.name,
            "content": result["content"],
            "chars": result.get("chars", len(result["content"])),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.put("/workspace/file")
async def save_workspace_file(req: WorkspaceFileUpdate):
    """Write one file inside the chosen workspace root."""
    try:
        workspace_root = _resolve_workspace_root(req.root)
        tool = FileTool(allowed_roots=[workspace_root])
        result = tool.execute(
            {
                "tool": "file_tool",
                "action": "write_file",
                "arguments": {
                    "path": req.path,
                    "content": req.content,
                },
            }
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or result.get("summary") or "Could not write file")
        resolved_path = Path(result["path"]).resolve()
        relative_path = resolved_path.relative_to(workspace_root).as_posix()
        payload = {
            "type": "file_saved",
            "root": str(workspace_root),
            "path": relative_path,
            "chars": result.get("chars", len(req.content)),
            "summary": result.get("summary"),
            "edit_preview": result.get("edit_preview"),
        }
        await ws_manager.broadcast(payload)
        return {
            "ok": True,
            "root": str(workspace_root),
            "path": relative_path,
            "summary": result.get("summary"),
            "chars": result.get("chars", len(req.content)),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
        "route_stats": {
            **dict(router.stats),
            "workspace": workspace_route_count,
        },
        "online": True,
        "model": active_local_model_label(config),
        "local_backend": configured_local_backend(config),
        "workspace_root": str(Path.cwd().resolve()),
        "ollama": ollama_ok,
        "ollama_models": models,
        "supabase": bool(config.supabase_url),
        "groq": bool(config.groq_api_key),
        "anthropic": bool(config.anthropic_api_key),
        "openrouter": bool(config.openrouter_api_key),
        "cloud_provider": preferred_cloud_provider(config) or "none",
        "cloud_model": preferred_cloud_model(config) if preferred_cloud_provider(config) else "",
        "canaryvaults": bool(config.canaryvaults_api_key),
        "reflect_stats": dict(router.reflect_stats),
        "ws_connections": ws_manager.count,
        "conversation_count": conversation_store.message_count(),
        "hive": hive_runtime.status(),
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


@app.get("/settings/providers")
async def provider_settings():
    """Return dashboard-editable provider settings metadata."""
    return _provider_settings_payload()


@app.put("/settings/providers/openrouter")
async def update_openrouter_settings(req: OpenRouterSettingsUpdate):
    """Persist OpenRouter settings for the local dashboard workspace."""
    next_key = config.openrouter_api_key
    if req.clear_api_key:
        next_key = ""
    elif req.api_key is not None and req.api_key.strip():
        next_key = req.api_key.strip()

    next_model = config.openrouter_model or "openrouter/auto"
    if req.model is not None and req.model.strip():
        next_model = req.model.strip()

    _persist_env_values(
        {
            "OPENROUTER_API_KEY": next_key,
            "OPENROUTER_MODEL": next_model,
        }
    )

    config.openrouter_api_key = next_key
    config.openrouter_model = next_model

    return {
        "ok": True,
        "summary": "OpenRouter settings saved.",
        **_provider_settings_payload(),
    }


@app.get("/models/control-center")
async def models_control_center():
    """Return desktop model-management data for local runtime and packaging flows."""
    return model_control.overview()


@app.put("/settings/local-runtime")
async def update_local_runtime_settings(req: LocalRuntimeSettingsUpdate):
    """Persist local runtime preferences for the desktop app."""
    updates: dict[str, str] = {}

    if req.backend is not None and req.backend.strip():
        backend = req.backend.strip().lower()
        if backend not in {"ollama", "adapter"}:
            raise HTTPException(status_code=400, detail="Local backend must be 'ollama' or 'adapter'.")
        updates["NEXUS_LOCAL_BACKEND"] = backend

    if req.local_model_dir is not None and req.local_model_dir.strip():
        updates["NEXUS_LOCAL_MODEL_DIR"] = req.local_model_dir.strip()

    if req.launch_model is not None and req.launch_model.strip():
        updates["NEXUS_MODEL"] = req.launch_model.strip()

    if updates:
        _persist_env_values(updates)

    try:
        overview = model_control.update_runtime(
            backend=req.backend,
            local_model_dir=req.local_model_dir,
            launch_model=req.launch_model,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "ok": True,
        "summary": "Local runtime settings saved.",
        "overview": overview,
    }


@app.post("/models/compress/launch")
async def compress_launch_model(req: CompressionRequest):
    """Run CompressX for the active launch model and return the refreshed overview."""
    try:
        return model_control.compress_launch_model(bits=req.bits)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
            {"name": "hive", "description": "Experimental distributed-search simulation with trust gating"},
        ]
    }


@app.get("/hive/status")
async def hive_status():
    """Return the experimental NEXUS Hive status snapshot."""
    return hive_runtime.status()


@app.post("/hive/demo")
async def hive_demo(req: HiveDemoRequest):
    """Run one Hive planning and consensus simulation."""
    try:
        return await hive_runtime.demo(req.prompt, intent=req.intent)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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


def _is_hive_chat_request(message: str) -> bool:
    raw = str(message or "").strip().lower()
    return raw.startswith("/hive") or raw.startswith("hive:") or raw.startswith("simulate hive ")


def _parse_hive_chat_request(message: str) -> tuple[str, str]:
    raw = str(message or "").strip()
    lowered = raw.lower()
    intent = "coding"
    prompt = raw
    if lowered.startswith("/hive"):
        prompt = raw[5:].strip(" :")
    elif lowered.startswith("hive:"):
        prompt = raw[5:].strip()
    elif lowered.startswith("simulate hive "):
        prompt = raw[14:].strip()

    lowered_prompt = prompt.lower()
    if lowered_prompt.startswith("research "):
        intent = "research"
        prompt = prompt[9:].strip()
    elif lowered_prompt.startswith("design "):
        intent = "design"
        prompt = prompt[7:].strip()
    elif lowered_prompt.startswith("memory "):
        intent = "memory"
        prompt = prompt[7:].strip()
    elif lowered_prompt.startswith("canary "):
        intent = "canary"
        prompt = prompt[7:].strip()

    return intent, prompt or "build me a full authentication system"


async def _chat_via_hive(req: ChatRequest) -> dict[str, Any]:
    """Run a Hive simulation and summarize it as a chat response."""
    intent, prompt = _parse_hive_chat_request(req.message)
    result = await hive_runtime.demo(prompt, intent=intent)
    winner = result.get("winner")
    lines = [
        f"NEXUS Hive simulated a distributed {result['task']['strategy']} run for: {result['task']['prompt']}",
        f"Selected nodes: {', '.join(result['plan']['selected_nodes']) or 'none'}",
        f"Responses: {result['responded_nodes']}, canary coverage: {result['plan']['canary_sample_size']}, blocked: {len(result['blocked_nodes'])}",
    ]
    if winner:
        lines.append(
            f"Winner: {winner['node_id']} with ReflectScore {winner['reflect_score']:.2f} "
            f"({winner['reflect_verdict']}) and network score {winner['network_score']:.2f}."
        )
        if result.get("assembled_output"):
            lines.append(f"Assembly: {result['assembled_output']}")
        else:
            lines.append(f"Best answer: {winner['output']}")
    if result["assembly_candidates"]:
        lines.append(
            "Assembly set: "
            + ", ".join(
                f"{candidate['node_id']} ({candidate['network_score']:.2f})"
                for candidate in result["assembly_candidates"]
            )
        )
    if result.get("canary_results"):
        failed = [item["node_id"] for item in result["canary_results"] if not item["passed"]]
        lines.append(
            "Canary checks: "
            + ("all passed" if not failed else f"failed on {', '.join(failed)}")
        )
    if result.get("note"):
        lines.append(result["note"])

    return {
        "response": "\n".join(lines),
        "agent": "hive",
        "route": "hive",
        "initial_route": "hive",
        "was_rerouted": False,
        "warning": None if not winner or winner["reflect_action"] == "serve" else "Hive winner is medium risk. Review before using it as ground truth.",
        "reflect_score": winner["reflect_score"] if winner else None,
        "reflect_verdict": winner["reflect_verdict"] if winner else None,
        "reflect_action": winner["reflect_action"] if winner else None,
        "session_id": req.session_id,
        "context_reduction": None,
        "execution": _hive_execution_payload(result),
        "workspace_root": req.workspace_root,
    }


def _hive_execution_payload(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize Hive runtime metadata into the dashboard execution shape."""
    if not result:
        return None
    winner = result.get("winner") or {}
    return {
        "workflow_id": result.get("task", {}).get("task_id"),
        "status": "completed" if winner else "blocked",
        "execution_mode": "hive",
        "final_confidence": winner.get("network_score"),
        "project_root": None,
        "touched_files": [],
        "documentation": {},
        "selected_nodes": result.get("plan", {}).get("selected_nodes", []),
        "blocked_nodes": result.get("blocked_nodes", []),
        "assembly_sources": result.get("assembly_sources", []),
        "assembled_output": result.get("assembled_output"),
        "canary_results": result.get("canary_results", []),
        "envelopes": result.get("envelopes", []),
    }


async def _chat_via_workspace_execution(req: ChatRequest) -> dict[str, Any]:
    """Run the message as a real project-mode execution against the selected repo."""
    workspace_root = _resolve_workspace_root(req.workspace_root)
    result = await _execute_workspace_goal(
        goal=req.message,
        workspace_root=workspace_root,
        execution_mode=req.execution_mode,
    )

    execution = {
        "workflow_id": result["workflow_id"],
        "status": result["status"],
        "execution_mode": result["execution_mode"],
        "final_confidence": result["final_confidence"],
        "project_root": str(workspace_root),
        "touched_files": list(result["touched_files"]),
        "documentation": dict(result.get("documentation") or {}),
    }
    return {
        "response": result["final_output"] or "Workspace execution finished without a textual summary.",
        "agent": result["primary_intent"],
        "route": "workspace",
        "initial_route": "workspace",
        "was_rerouted": False,
        "warning": None,
        "reflect_score": None,
        "reflect_verdict": None,
        "reflect_action": None,
        "session_id": req.session_id,
        "context_reduction": None,
        "execution": execution,
        "workspace_root": str(workspace_root),
    }


async def _execute_workspace_goal(
    *,
    goal: str,
    workspace_root: Path,
    execution_mode: str = "stable",
) -> dict[str, Any]:
    """Execute a goal inside a selected repo using the compiler + orchestrator path."""
    parser = IntentParser()
    planner = PlannerEngine()
    generator = BlueprintGenerator()
    project_manager = ProjectModeManager()
    project_context = project_manager.prepare(
        project_dir=workspace_root,
        goal=goal,
        execution_mode=execution_mode,
    )
    orchestrator = Orchestrator(
        execution_mode=execution_mode,
        project_context=project_context,
        environment_memory=project_manager.environment_memory,
    )

    intent = parser.parse(goal, project_context=project_context)
    plan = planner.plan(intent, project_context=project_context)
    blueprint = generator.generate(plan)
    result = await orchestrator.run_blueprint(blueprint)
    touched_files = _derive_touched_files(result.get("executions", []), workspace_root)
    return {
        "workflow_id": result["workflow_id"],
        "status": result["status"],
        "final_output": result.get("final_output", ""),
        "execution_mode": result.get("execution_mode", execution_mode),
        "final_confidence": result.get("final_confidence"),
        "documentation": result.get("documentation", {}),
        "touched_files": touched_files,
        "primary_intent": intent.primary_intent,
        "blueprint": blueprint.to_dict(),
    }


def _resolve_workspace_root(raw_root: str | None) -> Path:
    """Resolve and validate the selected workspace root for the file editor."""
    candidate = Path(str(raw_root).strip()).expanduser() if raw_root and str(raw_root).strip() else Path.cwd()
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Folder not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected a folder but received a file: {resolved}")
    return resolved


def _resolve_workspace_path(workspace_root: Path, raw_path: str | None) -> Path:
    """Resolve a child path and keep it inside the selected workspace root."""
    relative = str(raw_path or "").strip()
    candidate = workspace_root if not relative else (workspace_root / relative)
    resolved = candidate.resolve()
    if resolved != workspace_root and workspace_root not in resolved.parents:
        raise PermissionError(f"Access denied: {resolved} is outside {workspace_root}")
    return resolved


def _list_workspace_items(workspace_root: Path, directory: Path) -> list[dict[str, Any]]:
    """Return a shallow directory listing for the dashboard file browser."""
    items: list[dict[str, Any]] = []
    for child in sorted(directory.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
        if child.name in HIDDEN_WORKSPACE_NAMES:
            continue
        stat = child.stat()
        relative = child.relative_to(workspace_root).as_posix()
        items.append(
            {
                "name": child.name,
                "path": relative,
                "type": "directory" if child.is_dir() else "file",
                "size": None if child.is_dir() else stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return items


def _derive_touched_files(executions: list[dict[str, Any]], workspace_root: Path) -> list[str]:
    """Extract touched file paths from orchestrator execution payloads."""
    touched: list[str] = []
    for execution in executions:
        observation = execution.get("observation", {}) or {}
        for action in observation.get("tool_actions", []) or []:
            raw_path = action.get("path")
            if not raw_path:
                continue
            try:
                candidate = Path(str(raw_path))
                if not candidate.is_absolute():
                    candidate = workspace_root / candidate
                resolved = candidate.resolve()
                if resolved == workspace_root or workspace_root in resolved.parents:
                    touched.append(resolved.relative_to(workspace_root).as_posix())
            except Exception:
                continue
    deduped: list[str] = []
    seen: set[str] = set()
    for item in touched:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
