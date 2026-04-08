"""
NEXUS configuration — loads from .env and provides typed config object.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class NexusConfig(BaseModel):
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")

    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # CanaryVaults
    canaryvaults_api_url: str = os.getenv("CANARYVAULTS_API_URL", "https://canaryvaults.com/api")
    canaryvaults_api_key: str = os.getenv("CANARYVAULTS_API_KEY", "")

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # NEXUS settings
    nexus_model: str = os.getenv("NEXUS_MODEL", "phi3:mini")
    routing_complexity_threshold: float = float(
        os.getenv("NEXUS_ROUTER_COMPLEXITY_THRESHOLD", os.getenv("NEXUS_REFLECT_THRESHOLD", "0.5"))
    )
    reflect_threshold: float = float(os.getenv("NEXUS_REFLECT_THRESHOLD", "0.5"))
    reflect_warn_threshold: float = float(os.getenv("NEXUS_REFLECT_WARN_THRESHOLD", "0.3"))
    reflect_block_threshold: float = float(os.getenv("NEXUS_REFLECT_BLOCK_THRESHOLD", "0.6"))
    data_dir: Path = Path(os.getenv("NEXUS_DATA_DIR", "~/.nexus")).expanduser()
    context_reduction_enabled: bool = _env_bool("NEXUS_CONTEXT_REDUCTION_ENABLED", True)
    context_reduction_backend: str = os.getenv("NEXUS_CONTEXT_REDUCTION_BACKEND", "heuristic")
    context_reduction_threshold_chars: int = int(os.getenv("NEXUS_CONTEXT_REDUCTION_THRESHOLD_CHARS", "12000"))
    context_reduction_target_chars: int = int(os.getenv("NEXUS_CONTEXT_REDUCTION_TARGET_CHARS", "6000"))
    context_reduction_model: str = os.getenv("NEXUS_CONTEXT_REDUCTION_MODEL", "")

    class Config:
        arbitrary_types_allowed = True


config = NexusConfig()

# Ensure data dir exists
config.data_dir.mkdir(parents=True, exist_ok=True)
