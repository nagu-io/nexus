"""Generate ollama_export.py with correct Go-template braces."""
from pathlib import Path

LB = "{" * 2 + " "   # {{ 
RB = " " + "}" * 2   # }}
TQ = '"' * 3          # """

code = f'''from __future__ import annotations
import os, subprocess, types
from pathlib import Path

# ── COMPATIBILITY SHIM ──
import torch
for _a in [f"int{{i}}" for i in range(1,8)] + [f"uint{{i}}" for i in range(1,8)] + ["float8_e4m3fn","float8_e5m2","float8_e4m3fnuz","float8_e5m2fnuz"]:
    if not hasattr(torch, _a): setattr(torch, _a, torch.uint8)
if not hasattr(torch.utils, "_pytree"):
    torch.utils._pytree = types.ModuleType("_pytree")
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda x: x
# ── END SHIM ──

from rich.console import Console
from unsloth import FastLanguageModel

console = Console()
MERGED_MODEL_DIR = Path("nexus_model")
GGUF_BASENAME = "nexus_model-unsloth.Q4_K_M.gguf"
UNSLOTH_LLAMA_CPP_PATH = r"D:\\unsloth\\llama.cpp"

MODELFILE_BODY = (
    'TEMPLATE {TQ}<|system|>\\n'
    '{LB}.System{RB}<|end|>\\n'
    '
