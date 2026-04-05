# Cold Install Report

Date: 2026-04-03

Environment:

- OS: Windows
- Python: 3.11
- Workspace: `D:\nexus`
- Fresh virtual environment: `.tmp-launch-venv`

## Commands Run

```powershell
python -m venv .tmp-launch-venv
.\.tmp-launch-venv\Scripts\python -m pip install -e .
.\.tmp-launch-venv\Scripts\python -m nexus.cli --help
.\.tmp-launch-venv\Scripts\nexus.exe doctor
.\.tmp-launch-venv\Scripts\python -c "import uvicorn; print(uvicorn.__version__)"
```

API smoke test:

```powershell
@'
from fastapi.testclient import TestClient
from nexus.api import app
client = TestClient(app)
print(client.get('/status').status_code)
print(client.get('/agents').status_code)
print(client.post('/chat', json={'message': 'What is 2+2?'}).status_code)
'@ | .\.tmp-launch-venv\Scripts\python -
```

## Results

- Editable install succeeded in a fresh virtual environment
- `nexus` CLI entrypoint was installed and ran successfully
- `nexus doctor` ran successfully in the fresh environment
- `uvicorn` is now present in the base install
- FastAPI `/status`, `/agents`, and `/chat` returned `200`

## Expected Warnings

The cold install still reported missing runtime services in this environment:

- Ollama not running
- Groq API key not set
- Anthropic API key not set
- Supabase not set
- CanaryVaults API key not set

Those are environment/setup warnings, not package install failures.
