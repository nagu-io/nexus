"""
NEXUS FastAPI backend — serves the React dashboard.
Endpoints: /chat, /status, /reflect, /agents
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nexus.reflect.reflect_score import ReflectScore
from nexus.router.mind_router import MindRouter

app = FastAPI(title="NEXUS API", version="0.1.0")
router = MindRouter()
scorer = ReflectScore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat(req: ChatRequest):
    """Route a message through AEON Mind Router and return response."""
    result = await router.route(req.message, return_meta=True)

    return {
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
    }


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
        "thresholds": {
            "routing_complexity": config.routing_complexity_threshold,
            "reflect_warn": config.reflect_warn_threshold,
            "reflect_block": config.reflect_block_threshold,
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
            {"name": "research", "description": "Web search and summarization"},
            {"name": "memory", "description": "Persistent memory via Supabase"},
            {"name": "file", "description": "Safe file system operations"},
            {"name": "canary", "description": "CanaryRAG + CanaryVaults integration"},
        ]
    }
