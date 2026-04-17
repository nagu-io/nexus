"""One-shot generator: creates modelfile_template.txt and ollama_export.py.
Run this once:  py -3.11 nexus/train/_gen_files.py
"""
import pathlib

HERE = pathlib.Path(__file__).parent

# ── Write the Modelfile template ──────
LB = chr(123) * 2   # {{
RB = chr(125) * 2   # }}
TQ = chr(34) * 3    # triple-quote

modelfile_lines = [
    "FROM __MODEL_SOURCE__",
    f"TEMPLATE {TQ}<|system|>",
    f"{LB} .System {RB}<|end|>",
    "
