"""Helper script that writes ollama_export.py with the correct template syntax."""
from pathlib import Path

# The Modelfile template for Phi-3 with proper Go template braces
MODELFILE_BODY_LINES = [
    'TEMPLATE """<|system|>',
    "{{" + " .System " + "}}<|end|>",
    "
